"""Audita lacunas e planeja observáveis no interior da região do artigo.

As faixas são um domínio de busca, não uma classificação física presumida.
O plano inclui testemunhas com nucleação já aceita, falhas locais e uma malha
regular. A avaliação posterior volta a resolver fases, nucleação e observáveis.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from Articles.collect_data import json_safe
from Articles.figures.data import coordinate, gw_valid, load_dataset
from Articles.figures.frontiers.planning import common_plan, separate_output
from Articles.figures.points import write_plan
from Articles.figures.publication.analysis import (
    envelopes,
    fit_boundary_guides,
    value_valid,
)
from Articles.figures.publication.inputs import (
    load_original,
    load_received,
)

REQUIRED = ("Tn_GeV", "Tc_GeV", "alpha_trace", "beta_over_H", "vev_over_Tc")


def complete_observables(row):
    """Tn positivo não implica que alpha, beta e o campo crítico foram calculados."""
    return gw_valid(row) and all(value_valid(row, field) for field in REQUIRED)


def complete_transition_set(transitions):
    """O leitor usa a primeira transição aceita da história térmica."""
    primary = min(
        (
            t
            for t in transitions
            if t.get("nucleation_accepted") and t.get("trantype") == 1
        ),
        key=lambda t: t.get("transition_index", 0),
        default=None,
    )
    return primary is not None and complete_observables(primary)


def diagnostic_reason(row):
    if not row:
        return "not_calculated"
    if row.get("source_conflict"):
        return "conflicting_sources"
    if row.get("nucleation_accepted"):
        if row.get("alpha_trace") is None:
            return "nucleated_observables_not_calculated"
        if not value_valid(row, "alpha_trace"):
            return "alpha_unresolved"
        if not value_valid(row, "beta_over_H"):
            return "beta_unresolved"
        if not all(value_valid(row, field) for field in ("Tc_GeV", "vev_over_Tc")):
            return "critical_observables_missing"
        return "resolved"
    if row.get("status") in ("no_first_order_found", "no_nucleation_found"):
        return "no_accepted_nucleation_in_saved_evaluation"
    return "numerical_failure_or_timeout"


def coverage_audit(dataset, *, threshold=10):
    """Contagens de dados, sem interpretar a área branca como exclusão física."""
    cases = []
    for case in dataset.cases:
        rows = [r for r in dataset.rows if float(r["m8_GeV"]) == case]
        selected = [r for r in rows if r.get("boundary_decision") == "yes"]
        bins = []
        for lo, hi in (
            (500, 600),
            (600, 700),
            (700, 800),
            (800, 900),
            (900, 1100),
            (1100, 2001),
        ):
            known = [r for r in rows if lo <= float(r["m6_GeV"]) < hi and gw_valid(r)]
            bins.append(
                {
                    "m6_min_GeV": lo,
                    "m6_max_GeV": hi,
                    "GW_points": len(known),
                    "LISA_positive_samples": sum(
                        r.get("LISA_rho_sw", 0) >= threshold for r in known
                    ),
                    "maximum_sampled_LISA_SNR": max(
                        (r.get("LISA_rho_sw", 0) for r in known), default=None
                    ),
                }
            )
        cases.append(
            {
                "m8_GeV": case,
                "points": len(rows),
                "selected": len(selected),
                "selected_diagnostics": dict(
                    Counter(diagnostic_reason(r) for r in selected)
                ),
                "low_mass_detection": bins,
            }
        )
    return {
        "cases": cases,
        "LISA_SNR_threshold": threshold,
        "white_region_meaning": "missing_local_support_or_unresolved_calculation_not_automatic_non_nucleation",
    }


def coverage_targets(
    dataset,
    *,
    mass_step=50.0,
    low_mass_step=25.0,
    C_step=0.04,
    low_mass_max=900.0,
    C_max=4.05,
    repair_existing=True,
):
    """Malha regular, reparo de pontos existentes e amostragem fina junto ao topo.

    Pontos com física resolvida são reutilizados. Mesmo as testemunhas negativas
    antigas dentro da guia são reavaliadas: a primeira busca pode ter perdido uma
    fase. Isso permite testar, em vez de assumir, a conectividade da região.
    """
    if not all(
        math.isfinite(x) and x > 0
        for x in (mass_step, low_mass_step, C_step, low_mass_max, C_max)
    ):
        raise ValueError("Passos e limites devem ser positivos e finitos.")
    if not dataset.columns:
        raise ValueError(
            "A malha exige os brackets originais; carregue os bancos ou as exportações completas."
        )
    windows = fit_boundary_guides(envelopes(dataset))
    by_key = {coordinate(r): r for r in dataset.rows}
    targets = {}

    def add(case, mass, coupling, role):
        key = (case, mass, round(float(coupling), 12))
        if not 0 <= key[2] <= C_max:
            return
        old = by_key.get(key)
        if old and complete_observables(old):
            return
        item = targets.setdefault(
            key,
            {
                "m8_GeV": case,
                "m6_GeV": mass,
                "C": key[2],
                "roles": [],
                "previous_diagnostic": diagnostic_reason(old),
            },
        )
        if role not in item["roles"]:
            item["roles"].append(role)

    for case in dataset.cases:
        band = sorted(
            (r for r in windows if r["m8_GeV"] == case and math.isfinite(r["m6_GeV"])),
            key=lambda r: r["m6_GeV"],
        )
        if not band:
            continue
        masses = np.array([r["m6_GeV"] for r in band])
        lower = np.array([r["lower_C"] for r in band])
        upper = np.array([r["upper_C"] for r in band])
        selected_masses = {
            float(m)
            for m in masses
            if math.isclose(m / mass_step, round(m / mass_step), abs_tol=1e-8)
            or m <= low_mass_max
            and math.isclose(m / low_mass_step, round(m / low_mass_step), abs_tol=1e-8)
        }
        selected_masses.update((float(masses[0]), float(masses[-1])))
        for row in dataset.rows:
            case_r, mass, coupling = coordinate(row)
            if case_r != case or coupling > C_max:
                continue
            lo, hi = (
                (float(np.interp(mass, masses, edge)) for edge in (lower, upper))
                if math.isfinite(mass)
                else (0, -1)
            )
            if repair_existing and row.get("boundary_decision") == "yes":
                add(case, mass, coupling, "repair_nucleated_point")
            elif (
                repair_existing
                and math.isfinite(mass)
                and lo <= coupling <= hi
                and diagnostic_reason(row) != "resolved"
            ):
                add(case, mass, coupling, "recheck_interior_evaluation")
        for mass in sorted(selected_masses):
            lo = max(0, float(np.interp(mass, masses, lower)) - 0.02)
            hi = min(C_max, float(np.interp(mass, masses, upper)) + 0.02)
            if lo > hi:
                continue
            count = max(1, math.ceil((hi - lo) / C_step))
            for coupling in np.linspace(lo, hi, count + 1):
                add(case, mass, coupling, "interior_mesh")
            # A faixa do LISA é mais estreita que a malha do interior.
            top = float(np.interp(mass, masses, upper))
            for offset in (-0.08, -0.06, -0.04, -0.025, -0.015, -0.005, 0, 0.01):
                add(case, mass, top + offset, "upper_reach_strip")
    points = sorted(targets.values(), key=lambda r: (-r["m8_GeV"], r["m6_GeV"], r["C"]))
    return points


def pilot_points(points):
    """Quatro dificuldades diferentes por cenário; o piloto testa o fluxo físico."""
    result = []
    for case in sorted({float(r["m8_GeV"]) for r in points}, reverse=True):
        pool = [p for p in points if float(p["m8_GeV"]) == case]
        chosen = []
        predicates = (
            lambda p: p["previous_diagnostic"] == "beta_unresolved",
            lambda p: (
                p["previous_diagnostic"] == "nucleated_observables_not_calculated"
                and "repair_nucleated_point" in p["roles"]
                and float(p["m6_GeV"]) >= 900
            ),
            lambda p: (
                "upper_reach_strip" in p["roles"] and 600 <= float(p["m6_GeV"]) <= 800
            ),
            lambda p: (
                "interior_mesh" in p["roles"] and 900 <= float(p["m6_GeV"]) <= 1200
            ),
        )
        for predicate in predicates:
            found = next((p for p in pool if predicate(p) and p not in chosen), None)
            if found:
                chosen.append(found)
        for p in pool:
            if len(chosen) >= 4:
                break
            if p not in chosen:
                chosen.append(p)
        result.extend(chosen)
    return result


def build_coverage_plan(dataset, roots, **kwargs):
    points = coverage_targets(dataset, **kwargs)
    base = common_plan(dataset, roots)
    base.update(
        schema="article_coverage_plan_v2",
        cases_GeV=dataset.cases,
        policy="sample_observables_interior_and_recheck_holes",
        points=points,
        sampling=kwargs,
        source_audit=coverage_audit(dataset),
        numerics={
            "policy": "central_step_ladder_v1",
            "maximum_steps_GeV": [0.5, 1.0, 0.25],
            "margin_fraction": 0.1,
            "minimum_step_GeV": 0.005,
            "phase_recovery": True,
        },
    )
    return json_safe(base)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "plan"):
        p = sub.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--boundaries", type=Path)
        source.add_argument("--exports", type=Path)
        p.add_argument("--frontiers", type=Path)
        p.add_argument("--full-table", type=Path)
        p.add_argument("--observables", type=Path, action="append", default=[])
        p.add_argument("--reach", type=Path, action="append", default=[])
        p.add_argument("--output", type=Path, required=True)
        if name == "plan":
            p.add_argument("--pilot-output", type=Path)
            p.add_argument("--mass-step", type=float, default=50.0)
            p.add_argument("--low-mass-step", type=float, default=25.0)
            p.add_argument("--C-step", type=float, default=0.04)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--task-index", type=int, default=0)
    run.add_argument("--shards-per-scenario", type=int, default=4)
    run.add_argument("--point-timeout", type=float, default=900)
    run.add_argument("--max-points", type=int)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--reuse", type=Path, action="append", default=[])
    run.add_argument("--dry-run", action="store_true")
    for command in ("status", "check-pilot"):
        progress = sub.add_parser(command)
        progress.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from Articles.figures.publication.coverage_execution import run_shard

            return run_shard(args)
        if args.command in ("status", "check-pilot"):
            paths = sorted(args.output.glob("m8-*/interior_state.json"))
            if not paths:
                raise ValueError("Nenhum checkpoint encontrado.")
            for path in paths:
                state = json.loads(path.read_text(encoding="utf-8"))
                print(
                    f"{path.parent.name}: {state['state']}; {state['points_checked']}/{state['points_planned']} pontos; {len(state['unknown_points'])} inconclusivos"
                )
                if args.command == "check-pilot" and (
                    state["state"] not in ("complete", "complete_with_unresolved")
                    or state["points_checked"] != state["points_planned"]
                ):
                    raise ValueError("O piloto ainda não concluiu todas as avaliações.")
            if args.command == "check-pilot":
                data = load_dataset(args.output)
                if len(data.cases) != 3 or len(paths) != 3:
                    raise ValueError("O piloto deve conter três cenários completos.")
                for case in data.cases:
                    count = sum(
                        complete_observables(r)
                        for r in data.rows
                        if float(r["m8_GeV"]) == case
                    )
                    if not count:
                        raise ValueError(
                            f"m8={case:g}: nenhum conjunto completo de observáveis; revise o piloto antes da coleta grande."
                        )
                    print(f"m8={case:g}: {count} pontos completos no piloto.")
            return 0
        if args.boundaries:
            dataset, _ = load_original(args.boundaries, args.observables, args.reach)
        elif args.frontiers and args.full_table:
            dataset, _ = load_received(args.exports, args.frontiers, args.full_table)
        else:
            raise ValueError("Exportações exigem --frontiers e --full-table.")
        if args.command == "audit":
            args.output.mkdir(parents=True, exist_ok=True)
            report = coverage_audit(dataset)
            (args.output / "coverage_audit.json").write_text(
                json.dumps(json_safe(report), indent=2), encoding="utf-8"
            )
            print(json.dumps(json_safe(report), indent=2))
            return 0
        if not args.boundaries:
            raise ValueError(
                "O plano executável exige os manifestos originais via --boundaries."
            )
        roots = [args.boundaries, *args.observables, *args.reach]
        plan = build_coverage_plan(
            dataset,
            roots,
            mass_step=args.mass_step,
            low_mass_step=args.low_mass_step,
            C_step=args.C_step,
        )
        separate_output(args.output, plan)
        write_plan(args.output, plan)
        if args.pilot_output:
            pilot = {**plan, "points": pilot_points(plan["points"]), "pilot": True}
            separate_output(args.pilot_output, pilot)
            write_plan(args.pilot_output, pilot)
            print(f"Piloto: {len(pilot['points'])} pontos; {args.pilot_output}")
        print(f"Plano completo: {len(plan['points'])} pontos; {args.output}")
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
