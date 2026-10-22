# ruff: noqa: N999
"""Figuras do artigo, plano de pontos selecionados e execução desse plano.

python -m Articles.figures render --help
python -m Articles.figures plan --help
python -m Articles.figures points --help
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from Articles.collect_data import canonical_json, json_safe
from Articles.figures.data import load_dataset
from Articles.figures.geometry import bracket_columns
from Articles.figures.points import plan_from_args, run_points


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["point_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list, tuple)) else json_safe(value) for key, value in row.items()})


def render(args):
    from Articles.figures.render import (
        FigureWriter,
        plot_detectors,
        plot_observables,
        plot_quality,
        plot_relations,
        plot_spectra,
        plot_windows,
    )

    if args.C_limits[0] >= args.C_limits[1] or args.snr_threshold <= 0 or args.observation_years <= 0:
        raise ValueError("Intervalo C crescente, limiar SNR e tempo positivos são obrigatórios.")
    if args.max_spectra < 1 or args.dpi < 72:
        raise ValueError("max-spectra deve ser positivo e dpi>=72.")
    sources = [*([args.boundaries] if args.boundaries else []), *args.observables]
    if not sources:
        raise ValueError("Informe --boundaries.")
    for source in sources:
        source = source.resolve()
        if args.output.resolve() == source or args.output.resolve() in source.parents or source in args.output.resolve().parents:
            raise ValueError("O destino das figuras não pode substituir ou conter uma campanha de entrada.")
    dataset = load_dataset(args.boundaries, args.observables, years=args.observation_years,
                           read_details=not args.no_details, allow_running=args.allow_running_preview,
                           allow_unverified=args.allow_unverified)
    if args.require_gw and not dataset.audit["resolved_GW_points"]:
        raise ValueError("Não há observáveis GW resolvidos; calcule o plano de pontos ou a etapa interior.")
    args.output.mkdir(parents=True, exist_ok=True)
    writer = FigureWriter(args.output, args.formats, args.dpi)
    if dataset.audit["synthetic_data"]:
        writer.preview_notice = "SYNTHETIC VALIDATION DATA - STYLE PREVIEW"
    elif dataset.audit["unverified_sources"] or args.allow_running_preview:
        writer.preview_notice = "UNVERIFIED / RUNNING SOURCE - DIAGNOSTIC PREVIEW"
    print(f"Lidos {len(dataset.rows)} pontos; {dataset.audit['resolved_GW_points']} com GW; {dataset.audit['critical_ratios_available']} razões em Tc.", flush=True)
    plot_windows(dataset, writer, args.C_limits)
    plot_quality(dataset, writer)
    plot_observables(dataset, writer, args.C_limits)
    plot_detectors(dataset, writer, args.C_limits, args.snr_threshold, args.observation_years)
    benchmarks, samples = plot_spectra(dataset, writer, args.snr_threshold, args.observation_years, args.max_spectra)
    pairs = plot_relations(dataset, writer, args.C_limits)
    write_csv(args.output / "points_analysis.csv", dataset.rows)
    write_csv(args.output / "all_saved_transitions.csv", dataset.transitions)
    write_csv(args.output / "boundary_brackets.csv", [{"m8_GeV": case, **column} for case, columns in dataset.columns.items() for column in bracket_columns(columns)])
    write_csv(args.output / "benchmarks.csv", [{"benchmark": f"B{i+1}", **row} for i, row in enumerate(benchmarks)])
    write_csv(args.output / "spectrum_samples.csv", samples)
    write_csv(args.output / "degeneracy_candidates.csv", pairs)
    outside = [row for row in dataset.rows if row["boundary_decision"] == "yes" and not args.C_limits[0] <= row["C"] <= args.C_limits[1]]
    write_csv(args.output / "selected_outside_zoom.csv", outside)
    dataset.audit.update(created_utc=datetime.now(UTC).isoformat(), captions=writer.captions,
                         detector_threshold=args.snr_threshold, C_display_limits=args.C_limits,
                         selected_points_outside_zoom=len(outside),
                         figure_files=[path.name for path in writer.files])
    reference = {"references": ["https://arxiv.org/abs/2502.09593", "https://arxiv.org/abs/1802.02168", "https://arxiv.org/abs/2002.04615"],
                 "critical_C_pure_measure": 4.14, "critical_C_interpretation": "literature reference for Lambda=1000 GeV, not a universal bound in combined models",
                 "vTc_source": "saved critical VEV or unambiguous critical-transition record; never v(Tn)",
                 "SNR_interpretation": "sound-wave channel only, PIS template; not total-SNR inference"}
    dataset.audit.update(reference)
    (args.output / "figure_report.json").write_text(json.dumps(json_safe(dataset.audit), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# Captions and data availability", "", f"Saved points: {len(dataset.rows)}. Resolved GW points: {dataset.audit['resolved_GW_points']}.", ""]
    for name, caption in writer.captions.items():
        lines += [f"## {name}", "", caption, ""]
    if not dataset.audit["resolved_GW_points"]:
        lines += ["Alpha/beta, detector windows and spectra are not available from the boundary-only campaign. Empty panels are explicit data requests, not scientific zeroes.", ""]
    (args.output / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    if args.bundle:
        bundle = args.output / "review_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output.iterdir()):
                if path.is_file() and path.suffix in (".png", ".csv", ".json", ".md"):
                    archive.write(path, arcname=path.name)
        print(f"Pacote de revisão (sem SQLite/detalhes): {bundle}", flush=True)
    print(f"Figuras e tabelas: {args.output.resolve()}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    figures = commands.add_parser("render", help="Somente figuras/tabelas a partir dos dados salvos.")
    figures.add_argument("--boundaries", type=Path, required=True)
    figures.add_argument("--observables", type=Path, action="append", default=[])
    figures.add_argument("--output", type=Path, required=True)
    figures.add_argument("--C-limits", type=float, nargs=2, default=(0.0, 4.3))
    figures.add_argument("--observation-years", type=float, default=4.0)
    figures.add_argument("--snr-threshold", type=float, default=10.0)
    figures.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("pdf", "svg", "png"))
    figures.add_argument("--dpi", type=int, default=300)
    figures.add_argument("--max-spectra", type=int, default=6)
    figures.add_argument("--no-details", action="store_true")
    figures.add_argument("--allow-running-preview", action="store_true")
    figures.add_argument("--allow-unverified", action="store_true", help="Prévia para fontes sem hashes; registrada no relatório.")
    figures.add_argument("--require-gw", action="store_true")
    figures.add_argument("--bundle", action="store_true")
    figures.set_defaults(action=render)
    plan = commands.add_parser("plan", help="Plano imutável de poucos pontos para obter observáveis ausentes.")
    plan.add_argument("--boundaries", type=Path, required=True)
    plan.add_argument("--observables", type=Path, action="append", default=[])
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--mass-samples", type=int, default=3)
    plan.add_argument("--C-samples", type=int, default=3)
    plan.add_argument("--include-controls", action="store_true")
    plan.add_argument("--point", type=float, nargs=3, action="append", metavar=("M6", "C", "M8"),
                      help="Se informado, calcula somente estes pontos explícitos (GeV, adimensional, GeV). Repetível.")
    plan.add_argument("--allow-unverified", action="store_true")
    plan.set_defaults(action=plan_from_args)
    points = commands.add_parser("points", help="Calcula apenas as coordenadas do plano; nunca escreve nas fontes.")
    points.add_argument("--plan", type=Path, required=True)
    points.add_argument("--output", type=Path, required=True)
    points.add_argument("--work-dir", type=Path, required=True)
    points.add_argument("--scenario-index", type=int, choices=range(3))
    points.add_argument("--point-timeout", type=float, default=900.0)
    points.add_argument("--max-points", type=int)
    points.add_argument("--retry-failed", action="store_true")
    points.add_argument("--dry-run", action="store_true")
    points.set_defaults(action=run_points)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return args.action(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
