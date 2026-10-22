# ruff: noqa: N999
"""Bordas e alcance do LISA: python -m Articles.figures.frontiers --help."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from Articles.collect_data import json_safe
from Articles.figures.__main__ import write_csv
from Articles.figures.frontiers.data import (
    alpha_valid,
    beta_valid,
    edge_samples,
    load_exports,
    load_sources,
    strength_crossings,
)
from Articles.figures.frontiers.planning import (
    build_edge_plan,
    build_lisa_plan,
    run_edge_shard,
)
from Articles.figures.frontiers.reach import run_lisa_shard
from Articles.figures.points import write_plan
from Articles.figures.render import FigureWriter


def render(args):
    from Articles.figures.frontiers.render import (
        interior_edge_violations,
        plot_C_dependence,
        plot_clean_window,
        plot_edge_thermodynamics,
        plot_endpoint_spectra,
        plot_lisa_window,
        reach_profiles,
    )

    if (
        not all(
            math.isfinite(x)
            for x in (
                *args.C_limits,
                *args.m6_limits,
                args.observation_years,
                args.snr_threshold,
                args.maximum_mass_gap,
                args.maximum_C_gap,
                args.C_tolerance,
            )
        )
        or args.C_limits[0] >= args.C_limits[1]
        or args.m6_limits[0] >= args.m6_limits[1]
        or min(
            args.observation_years,
            args.snr_threshold,
            args.maximum_mass_gap,
            args.maximum_C_gap,
            args.C_tolerance,
        )
        <= 0
    ):
        raise ValueError(
            "Limites crescentes e tempo/limiar/lacunas/tolerância positivos e finitos são obrigatórios."
        )
    if args.dpi < 72 or not 1 <= len(args.benchmark_masses) <= 3:
        raise ValueError("dpi>=72; escolha de uma a três massas de benchmark.")
    roots = [
        *([args.exports] if args.exports else [args.boundaries]),
        *args.observables,
        *args.reach,
    ]
    for source in roots:
        source, output = source.resolve(), args.output.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                "As figuras novas devem ter destino separado dos dados e das figuras v1."
            )
    if args.exports:
        if args.observables or args.reach:
            raise ValueError(
                "Para unir campanhas novas use --boundaries com --observables/--reach no CHE."
            )
        dataset = load_exports(args.exports, args.observation_years)
    else:
        dataset = load_sources(
            args.boundaries,
            [*args.observables, *args.reach],
            years=args.observation_years,
        )
    edges = edge_samples(dataset)
    crossings = strength_crossings(dataset) + strength_crossings(dataset, "vev_over_Tc")
    writer = FigureWriter(args.output, args.formats, args.dpi)
    if dataset.audit.get("synthetic_data"):
        writer.preview_notice = "SYNTHETIC VALIDATION DATA - STYLE PREVIEW"
    elif dataset.audit.get("unverified_sources"):
        writer.preview_notice = "UNVERIFIED SOURCE - DIAGNOSTIC PREVIEW"
    profiles = reach_profiles(dataset, args.reach, args)
    print(
        f"{len(dataset.rows)} pontos; {len(edges)} testemunhas de borda; "
        f"alpha em {sum(alpha_valid(row) for row in edges)} e beta em {sum(beta_valid(row) for row in edges)} testemunhas.",
        flush=True,
    )
    plot_clean_window(dataset, edges, crossings, writer, args)
    plot_edge_thermodynamics(dataset, edges, writer, args)
    plot_C_dependence(dataset, writer, args)
    plot_lisa_window(dataset, edges, profiles, writer, args)
    benchmarks, spectra = plot_endpoint_spectra(dataset, profiles, writer, args)
    violations = interior_edge_violations(dataset, edges)
    write_csv(args.output / "edge_observables.csv", edges)
    write_csv(args.output / "strength_reference.csv", crossings)
    write_csv(args.output / "lisa_endpoint_benchmarks.csv", benchmarks)
    write_csv(args.output / "spectrum_samples.csv", spectra)
    write_csv(args.output / "interior_outside_edge_range.csv", violations)
    (args.output / "lisa_profiles.json").write_text(
        json.dumps(json_safe(list(profiles.values())), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_audit": dataset.audit,
        "analysis_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "edge_witnesses": len(edges),
        "edge_coordinates": len(
            {(row["m8_GeV"], row["m6_GeV"], row["C"]) for row in edges}
        ),
        "edge_alpha_available": sum(alpha_valid(row) for row in edges),
        "edge_beta_available": sum(beta_valid(row) for row in edges),
        "interior_outside_edge_range": len(violations),
        "captions": writer.captions,
        "figure_files": [path.name for path in writer.files],
        "assumptions": {
            "observation_years": args.observation_years,
            "snr_threshold": args.snr_threshold,
            "channel": "sound_waves_only_PIS",
            "PIS_reference": "https://arxiv.org/abs/2002.04615",
            "C_display_limits": args.C_limits,
            "m6_display_limits_GeV": args.m6_limits,
            "strength_line": "numerical_saved_value_interpolation_not_analytic_full_potential",
            "strength_reference": args.strength_reference,
            "maximum_mass_gap_GeV_for_reach": args.maximum_mass_gap,
            "maximum_C_gap_for_saved_samples": args.maximum_C_gap,
            "edge_values_are_global_extrema": False,
            "LISA_threshold_uses_global_noise_minimum": False,
        },
    }
    (args.output / "frontier_report.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# Frontier figures",
        "",
        "These figures are an additive analysis. Original v1 outputs remain separate.",
        "",
    ]
    for name, caption in writer.captions.items():
        lines += [f"## {name}", "", caption, ""]
    (args.output / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    if args.bundle:
        with zipfile.ZipFile(
            args.output / "review_bundle_frontiers.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(args.output.iterdir()):
                if path.suffix in (".png", ".json", ".csv", ".md") and path.is_file():
                    archive.write(path, arcname=path.name)
    print(f"Figuras de bordas: {args.output.resolve()}", flush=True)
    return 0


def make_plan(args):
    dataset = load_sources(args.boundaries, args.observables, details=False)
    roots = [args.boundaries, *args.observables]
    if args.command == "plan-edges":
        plan = build_edge_plan(
            dataset,
            roots,
            mass_step=args.mass_step,
            interior_checks=args.interior_checks,
        )
        print(
            f"Plano: {len(plan['points'])} pontos novos; {len(plan['edge_targets'])} testemunhas de borda."
        )
    else:
        plan = build_lisa_plan(
            dataset,
            roots,
            mass_step=args.mass_step,
            benchmark_masses=args.benchmark_masses,
            C_step=args.C_step,
            C_tolerance=args.C_tolerance,
            years=args.observation_years,
            threshold=args.snr_threshold,
        )
        print(
            f"Plano: {len(plan['cuts'])} cortes; {sum(len(cut['grid_C']) for cut in plan['cuts'])} coordenadas de malha antes do refinamento/reuso."
        )
    if args.beta_step is not None:
        if not math.isfinite(args.beta_step) or args.beta_step <= 0:
            raise ValueError("beta-step deve ser positivo e finito.")
        plan["settings"]["beta_step"] = args.beta_step
    plan["numerics"] = {
        "policy": "metastable_margin_v1" if args.beta_policy == "margin" else "fixed",
        "margin_fraction": 0.1,
        "minimum_step_GeV": 0.005,
    }
    write_plan(args.output, plan)
    print(f"Plano imutável salvo em {args.output.resolve()}")
    return 0


def status(args):
    paths = sorted(args.output.glob("m8-*/interior_state.json"))
    if not paths:
        raise ValueError("Nenhum checkpoint de bordas/LISA encontrado nesse destino.")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{path.parent.name}: {report['state']}; "
            f"{report.get('points_checked', len(report.get('cuts', [])))}/"
            f"{report.get('points_planned', report.get('cuts_planned', '?'))} "
            f"{'pontos' if report.get('kind') == 'edge_observables' else 'cortes'}; "
            f"{len(report.get('unknown_points', []))} inconclusivos"
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    figures = commands.add_parser("render", help="Figuras novas, sem calcular física.")
    sources = figures.add_mutually_exclusive_group(required=True)
    sources.add_argument("--boundaries", type=Path)
    sources.add_argument(
        "--exports", type=Path, help="Pasta de CSVs/figure_report v1 já baixada."
    )
    figures.add_argument("--observables", type=Path, action="append", default=[])
    figures.add_argument("--reach", type=Path, action="append", default=[])
    figures.add_argument("--output", type=Path, required=True)
    figures.add_argument("--C-limits", type=float, nargs=2, default=(0.0, 4.1))
    figures.add_argument("--m6-limits", type=float, nargs=2, default=(500.0, 2000.0))
    figures.add_argument(
        "--strength-reference", choices=("Tn", "Tc", "none"), default="Tn"
    )
    figures.add_argument(
        "--benchmark-masses", type=float, nargs="+", default=(1000.0, 1500.0)
    )
    figures.add_argument("--observation-years", type=float, default=4.0)
    figures.add_argument("--snr-threshold", type=float, default=10.0)
    figures.add_argument("--maximum-mass-gap", type=float, default=100.01)
    figures.add_argument("--maximum-C-gap", type=float, default=0.0500001)
    figures.add_argument("--C-tolerance", type=float, default=0.01)
    figures.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "svg", "png"),
    )
    figures.add_argument("--dpi", type=int, default=300)
    figures.add_argument("--bundle", action="store_true")
    figures.set_defaults(action=render)
    for name in ("plan-edges", "plan-lisa"):
        plan = commands.add_parser(name, help="Plano imutável sem cálculos físicos.")
        plan.add_argument("--boundaries", type=Path, required=True)
        plan.add_argument("--observables", type=Path, action="append", default=[])
        plan.add_argument("--output", type=Path, required=True)
        plan.add_argument(
            "--mass-step", type=float, default=5.0 if name == "plan-edges" else 100.0
        )
        plan.add_argument(
            "--beta-policy",
            choices=("margin", "fixed"),
            default="margin",
            help="margin limita h a 10%% da faixa metastável local; fixed reproduz o passo fixo v4.",
        )
        plan.add_argument(
            "--beta-step",
            type=float,
            help="Passo máximo h em GeV; herda o valor da campanha (0.5 no v4).",
        )
        if name == "plan-edges":
            plan.add_argument("--interior-checks", type=int, default=0)
        else:
            plan.add_argument(
                "--benchmark-masses", type=float, nargs="+", default=(1000.0, 1500.0)
            )
            plan.add_argument("--C-step", type=float, default=0.05)
            plan.add_argument("--C-tolerance", type=float, default=0.01)
            plan.add_argument("--observation-years", type=float, default=4.0)
            plan.add_argument("--snr-threshold", type=float, default=10.0)
        plan.set_defaults(action=make_plan)
    for name, action in (("run-edges", run_edge_shard), ("run-lisa", run_lisa_shard)):
        run = commands.add_parser(
            name, help="Executa uma parte usando o núcleo e checkpoints v4."
        )
        run.add_argument("--plan", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--work-dir", type=Path, required=True)
        run.add_argument("--task-index", type=int, default=0)
        run.add_argument("--shards-per-scenario", type=int, default=4)
        run.add_argument("--point-timeout", type=float, default=900.0)
        run.add_argument("--max-points", type=int)
        run.add_argument("--retry-failed", action="store_true")
        run.add_argument("--dry-run", action="store_true")
        if name == "run-lisa":
            run.add_argument("--reuse", type=Path, action="append", default=[])
        run.set_defaults(action=action)
    progress = commands.add_parser(
        "status", help="Somente leitura dos estados publicados."
    )
    progress.add_argument("--output", type=Path, required=True)
    progress.set_defaults(action=status)
    args = parser.parse_args(argv)
    try:
        return args.action(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
