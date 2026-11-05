# ruff: noqa: N999
"""Seis figuras do artigo: python -m Articles.figures.publication --help."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from Articles.figures.publication.analysis import (
    availability,
    detector_cuts,
    envelopes,
    fit_boundary_guides,
    match_spectra,
    refinement_requests,
)
from Articles.figures.publication.inputs import (
    add_half_step_comparison,
    load_original,
    load_received,
)


def render(args):
    from Articles.figures.publication.products import write_products
    from Articles.figures.publication.rendering import (
        PublicationWriter,
        plot_atlases,
        plot_detector_contours,
        plot_matched_spectra,
        plot_windows,
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
                args.maximum_peak_distance,
                args.boundary_smoothness,
            )
        )
        or args.C_limits[0] >= args.C_limits[1]
        or args.m6_limits[0] <= 0
        or args.m6_limits[0] >= args.m6_limits[1]
        or min(
            args.observation_years,
            args.snr_threshold,
            args.maximum_mass_gap,
            args.maximum_C_gap,
            args.maximum_peak_distance,
            args.boundary_smoothness,
        )
        <= 0
        or args.dpi < 72
    ):
        raise ValueError("Limites crescentes e parâmetros positivos/finitos; dpi>=72.")
    roots = [
        p
        for p in (
            args.exports,
            args.frontiers,
            args.full_table,
            args.boundaries,
            *args.observables,
            *args.reach,
        )
        if p is not None
    ]
    for source in roots:
        source, output = source.resolve(), args.output.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Destino das figuras deve ser separado das entradas.")
    if args.output.exists() and any(args.output.iterdir()) and not args.replace_figures:
        raise ValueError(
            "Destino não vazio. Escolha outra pasta ou use --replace-figures para regenerar esta edição."
        )
    if args.exports:
        if not args.frontiers:
            raise ValueError("Informe --frontiers com o pacote da segunda leva.")
        if not args.full_table and not args.preview:
            raise ValueError(
                "A versão completa exige --full-table: o ZIP de bordas omitiu os observáveis interiores. --preview é somente diagnóstico."
            )
        if args.observables or args.reach:
            raise ValueError(
                "Use --boundaries para ler bancos ou --exports/--frontiers/--full-table para ler tabelas."
            )
        dataset, profiles = load_received(
            args.exports,
            args.frontiers,
            args.full_table,
            years=args.observation_years,
            threshold=args.snr_threshold,
        )
    else:
        if args.frontiers or args.full_table:
            raise ValueError("--frontiers e --full-table exigem --exports.")
        dataset, profiles = load_original(
            args.boundaries,
            args.observables,
            args.reach,
            years=args.observation_years,
            threshold=args.snr_threshold,
        )
    if len(dataset.cases) != 3:
        raise ValueError("As seis figuras exigem os três cenários m8 da campanha.")
    if any(
        row.get("Lambda_GeV") is not None and not math.isclose(row["Lambda_GeV"], 1000)
        for row in dataset.rows
    ):
        raise ValueError(
            "Esta edição fixa Lambda=1000 GeV; não misture outra escala nos mapas."
        )
    if dataset.audit.get("conflicting_coordinates") or dataset.audit.get(
        "unverified_sources"
    ):
        raise ValueError(
            "Há conflito ou fonte não verificada; resolva a auditoria antes das figuras do artigo."
        )
    if any(
        s.get("state") in ("running", "initializing")
        for s in dataset.audit.get("sources", [])
    ):
        raise ValueError("Há fonte em execução; aguarde o encerramento dos jobs.")
    writer = PublicationWriter(args.output, args.formats, args.dpi)
    if dataset.audit.get("synthetic_data"):
        writer.preview_notice = "SYNTHETIC VALIDATION DATA — NOT RESEARCH RESULTS"
    elif args.preview:
        writer.preview_notice = "INCOMPLETE DATA TRANSFER — DIAGNOSTIC PREVIEW"
    add_half_step_comparison(
        dataset, years=args.observation_years, threshold=args.snr_threshold
    )
    windows = envelopes(dataset)
    if args.boundary_guide == "bracketed":
        windows = fit_boundary_guides(windows, smoothness=args.boundary_smoothness)
    coverage = availability(dataset, threshold=args.snr_threshold)
    print(
        f"{len(dataset.rows)} pontos; {sum(r['GW_valid'] for r in coverage)} GW válidos.",
        flush=True,
    )
    plot_windows(dataset, windows, writer, args)
    print("Figura 1 concluída; construindo os mapas.", flush=True)
    meshes = plot_atlases(dataset, windows, writer, args)
    print("Figuras 2–4 concluídas; construindo alcance dos detectores.", flush=True)
    meshes += plot_detector_contours(dataset, windows, writer, args)
    print("Figura 5 concluída; construindo os espectros.", flush=True)
    benchmarks, spectra = match_spectra(
        dataset,
        threshold=args.snr_threshold,
        maximum_peak_distance=args.maximum_peak_distance,
    )
    plot_matched_spectra(benchmarks, spectra, dataset.cases, writer, args)
    print("Figura 6 concluída; salvando tabelas e ZIP.", flush=True)
    cuts = detector_cuts(dataset, profiles, threshold=args.snr_threshold)
    cuts += detector_cuts(
        dataset, profiles, threshold=args.snr_threshold, half_step=True
    )
    requests = refinement_requests(dataset, profiles, threshold=args.snr_threshold)
    write_products(
        dataset,
        profiles,
        windows,
        benchmarks,
        spectra,
        cuts,
        requests,
        coverage,
        meshes,
        writer,
        args,
    )
    print(f"Seis figuras, legendas e tabelas: {args.output.resolve()}", flush=True)
    print(
        f"{len(requests)} avaliações pontuais prioritárias listadas; nenhum cálculo submetido.",
        flush=True,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--exports", type=Path, help="Pasta/ZIP das tabelas da primeira leva."
    )
    source.add_argument(
        "--boundaries", type=Path, help="Bancos originais no CHE, lidos sem modificar."
    )
    parser.add_argument(
        "--frontiers", type=Path, help="Pasta/ZIP das figuras/tabelas da segunda leva."
    )
    parser.add_argument(
        "--full-table",
        type=Path,
        help="CSV/ZIP all_points_analysis com todos os observáveis.",
    )
    parser.add_argument("--observables", type=Path, action="append", default=[])
    parser.add_argument("--reach", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--C-limits", type=float, nargs=2, default=(0.0, 4.05))
    parser.add_argument("--m6-limits", type=float, nargs=2, default=(500.0, 2000.0))
    parser.add_argument("--maximum-mass-gap", type=float, default=100.01)
    parser.add_argument("--maximum-C-gap", type=float, default=0.081)
    parser.add_argument("--maximum-peak-distance", type=float, default=0.16)
    parser.add_argument(
        "--boundary-guide", choices=("bracketed", "raw"), default="bracketed"
    )
    parser.add_argument(
        "--boundary-smoothness",
        type=float,
        default=1000.0,
        help="Peso da curvatura da guia; nunca relaxa restrições dos pontos.",
    )
    parser.add_argument("--observation-years", type=float, default=4.0)
    parser.add_argument("--snr-threshold", type=float, default=10.0)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "svg", "png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--replace-figures",
        action="store_true",
        help="Regenera somente a edição no destino informado.",
    )
    args = parser.parse_args(argv)
    try:
        return render(args)
    except (ValueError, OSError, KeyError, ArithmeticError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
