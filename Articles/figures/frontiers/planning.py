"""Planos pequenos e imutáveis: observáveis nas bordas e cortes para o LISA."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from Articles.collect_data import (
    M8_SCENARIOS_GEV,
    Settings,
    json_safe,
    validate_campaign,
)
from Articles.figures.data import coordinate
from Articles.figures.frontiers.data import edge_samples, grouped_rows
from Articles.figures.frontiers.execution import (
    checkpoint,
    execution_manifest,
    live_store,
)


def common_plan(dataset, roots):
    if not dataset.campaigns:
        raise ValueError(
            "Um plano físico exige os bancos/manifestos originais, não apenas os CSVs."
        )
    reference = next(
        (
            item["manifest"]
            for item in dataset.campaigns
            if item["manifest"].get("grid", {}).get("stage") == "boundaries"
        ),
        dataset.campaigns[-1]["manifest"],
    )
    settings = {
        **asdict(Settings()),
        **reference["settings"],
        "beta_check": True,
        "recovery_attempts": 0,
    }
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "reference_manifest": reference,
        "settings": settings,
        "source_roots": [str(Path(root).resolve()) for root in roots],
        "source_fingerprints": [
            item["manifest"].get("fingerprint") for item in dataset.campaigns
        ],
    }


def selected_masses(dataset, case, step, additional=()):
    if not math.isfinite(step) or step <= 0:
        raise ValueError("mass-step deve ser positivo e finito.")
    available = sorted(
        {
            float(row["m6_GeV"])
            for row in dataset.rows
            if float(row["m8_GeV"]) == case and row["boundary_decision"] == "yes"
        }
    )
    finite = [mass for mass in available if math.isfinite(mass)]
    chosen = {
        mass
        for mass in available
        if math.isinf(mass)
        or math.isclose(mass / step, round(mass / step), abs_tol=1e-8)
    }
    if finite:
        chosen.update((finite[0], finite[-1]))
    for mass in additional:
        if mass not in available:
            raise ValueError(
                f"m6={mass:g} não tem amostra selecionada no cenário m8={case:g}."
            )
        chosen.add(mass)
    return sorted(chosen, reverse=True)


def build_edge_plan(dataset, roots, *, mass_step=5, interior_checks=0):
    """Duas testemunhas por massa; pontos extras de interior só sob opção."""
    if interior_checks < 0:
        raise ValueError("interior-checks deve ser não negativo.")
    edges = edge_samples(dataset)
    allowed = {
        case: set(selected_masses(dataset, case, mass_step)) for case in dataset.cases
    }
    by_coordinate = {coordinate(row): row for row in dataset.rows}
    targets, evidence = {}, []
    for row in edges:
        case, mass, coupling = coordinate(row)
        if mass not in allowed[case]:
            continue
        evidence.append(
            {
                key: row.get(key)
                for key in (
                    "m6_GeV",
                    "m8_GeV",
                    "C",
                    "point_id",
                    "edge",
                    "edge_certified",
                    "edge_original_reason",
                )
            }
        )
        if not row.get("gw_resolved"):
            target = targets.setdefault(
                (case, mass, coupling),
                {"m6_GeV": mass, "m8_GeV": case, "C": coupling, "roles": []},
            )
            target["roles"].append(row["edge"])
    if interior_checks:
        for (case, mass), rows in grouped_rows(dataset).items():
            selected = [row for row in rows if row["boundary_decision"] == "yes"]
            if mass not in allowed[case] or len(selected) < 2:
                continue
            for coupling in np.linspace(
                selected[0]["C"], selected[-1]["C"], interior_checks + 2
            )[1:-1]:
                coupling = round(float(coupling), 12)
                old = by_coordinate.get((case, mass, coupling))
                if old is None or not old.get("gw_resolved"):
                    targets[(case, mass, coupling)] = {
                        "m6_GeV": mass,
                        "m8_GeV": case,
                        "C": coupling,
                        "roles": ["interior_check"],
                    }
    return json_safe(
        {
            **common_plan(dataset, roots),
            "schema": "article_point_plan_v1",
            "policy": "sampled_C_edges_not_alpha_beta_global_extrema",
            "mass_step_GeV": mass_step,
            "interior_checks": interior_checks,
            "edge_targets": evidence,
            "points": sorted(
                targets.values(),
                key=lambda row: (-row["m8_GeV"], -row["m6_GeV"], row["C"]),
            ),
        }
    )


def build_lisa_plan(
    dataset,
    roots,
    *,
    mass_step=100,
    benchmark_masses=(1000.0, 1500.0),
    C_step=0.05,
    C_tolerance=0.01,
    years=4,
    threshold=10,
):
    """O limiar exige cortes EM C; não é inferido de apenas duas bordas."""
    if (
        not all(
            math.isfinite(x) and x > 0 for x in (C_step, C_tolerance, years, threshold)
        )
        or C_tolerance > C_step
    ):
        raise ValueError(
            "Passo C, tolerância, tempo e SNR devem ser positivos; tolerância <= passo C."
        )
    cuts = []
    edges = {
        (float(row["m8_GeV"]), float(row["m6_GeV"]), row["edge"]): row
        for row in edge_samples(dataset)
    }
    reuse = defaultdict(list)
    groups = grouped_rows(dataset)
    for case in dataset.cases:
        masses = selected_masses(dataset, case, mass_step, benchmark_masses)
        # Benchmarks e caso puro primeiro, para que o piloto seja informativo.
        masses.sort(key=lambda mass: (mass not in (*benchmark_masses, math.inf), -mass))
        for mass in masses:
            rows = [
                row for row in groups[(case, mass)] if row["boundary_decision"] == "yes"
            ]
            inside_low, inside_high = rows[0]["C"], rows[-1]["C"]
            low, high = inside_low, inside_high
            lower, upper = edges[(case, mass, "lower")], edges[(case, mass, "upper")]
            if (
                lower.get("edge_outside_C") is not None
                and 0 < low - lower["edge_outside_C"] <= 0.2
            ):
                low = lower["edge_outside_C"]
            if (
                upper.get("edge_outside_C") is not None
                and 0 < upper["edge_outside_C"] - high <= 0.2
            ):
                high = upper["edge_outside_C"]
            grid = (
                np.linspace(low, high, max(1, math.ceil((high - low) / C_step)) + 1)
                if high > low
                else [low]
            )
            grid = [*grid, inside_low, inside_high]
            for row in groups[(case, mass)]:
                if low <= row["C"] <= high and row.get("gw_resolved"):
                    grid.append(row["C"])
                    reuse[row["source_folder"]].append({"C": row["C"], "m6_GeV": mass})
            cuts.append(
                {
                    "m6_GeV": mass,
                    "m8_GeV": case,
                    "C_min": low,
                    "C_max": high,
                    "prior_selected_C_min": inside_low,
                    "prior_selected_C_max": inside_high,
                    "grid_C": sorted({round(float(x), 12) for x in grid}),
                    "domain": "selected_C_envelope_plus_nearest_outward_witness_within_0.2_not_global_search",
                }
            )
    return json_safe(
        {
            **common_plan(dataset, roots),
            "schema": "article_lisa_cuts_v1",
            "cuts": cuts,
            "mass_step_GeV": mass_step,
            "benchmark_masses_GeV": list(benchmark_masses),
            "C_step": C_step,
            "C_tolerance": C_tolerance,
            "observation_years": years,
            "snr_threshold": threshold,
            "channel": "sound_waves_only_PIS",
            "reuse_sources": [
                {"folder": folder, "coordinates": coordinates}
                for folder, coordinates in sorted(reuse.items())
            ],
            "resolution_limit": "islands_between_C_samples_or_mass_cuts_are_not_excluded",
        }
    )


def separate_output(output, plan):
    output = Path(output).resolve()
    for source in plan.get("source_roots", []):
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                "O destino novo deve ser separado das campanhas de origem."
            )


def partition(items, task_index, shards):
    if shards < 1 or not 0 <= task_index < 3 * shards:
        raise ValueError(
            "Índice de tarefa deve estar entre 0 e 3*shards-per-scenario-1."
        )
    scenario, part = divmod(task_index, shards)
    case = M8_SCENARIOS_GEV[scenario]
    matching = [item for item in items if float(item["m8_GeV"]) == case]
    masses = sorted({float(item["m6_GeV"]) for item in matching}, reverse=True)
    assigned = {mass for index, mass in enumerate(masses) if index % shards == part}
    return (
        scenario,
        part,
        [item for item in matching if float(item["m6_GeV"]) in assigned],
    )


def run_edge_shard(args):
    """Cada job recebe massas inteiras e usa o Store/processo limitado v4."""
    content = args.plan.read_bytes()
    plan = json.loads(content)
    if plan.get("schema") != "article_point_plan_v1":
        raise ValueError("Use o plano criado por plan-edges.")
    separate_output(args.output, plan)
    scenario, part, points = partition(
        plan["points"], args.task_index, args.shards_per_scenario
    )
    output = args.output / f"m8-{scenario}-part-{part}"
    case = M8_SCENARIOS_GEV[scenario]
    settings, manifest = execution_manifest(
        plan, content, args, case, "article_edge_observables_v1"
    )
    validate_campaign(output, manifest)
    if args.dry_run or not points:
        print(
            f"Tarefa {args.task_index}: {len(points)} pontos; configuração compatível; nenhum cálculo/pasta nova."
        )
        return 0
    report = {
        "state": "running",
        "kind": "edge_observables",
        "points_planned": len(points),
        "points_checked": 0,
        "unknown_points": [],
        "plan_sha256": hashlib.sha256(content).hexdigest(),
    }
    with live_store(args, output, settings, manifest, case, report) as store:
        for point in points:
            result = store.evaluate(float(point["C"]), float(point["m6_GeV"]))
            report["points_checked"] += 1
            if (
                result["decision"] == "unknown"
                or result["status"] == "observables_unresolved"
            ):
                report["unknown_points"].append(point)
            checkpoint(store, report)
        report["state"] = (
            "complete_with_unresolved" if report["unknown_points"] else "complete"
        )
    print(f"{report['state']}: {store.new_points} pontos novos em {output}", flush=True)
    return 3 if report["state"] == "complete_with_unresolved" else 0
