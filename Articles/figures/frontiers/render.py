"""Figuras da reunião: duas bordas legíveis, observáveis e alcance do LISA.

Não há suavização para esconder buracos, valores extremos inferidos só de
duas bordas ou limite de detecção obtido do mínimo global da sensibilidade.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from Articles.figures.data import finite
from Articles.figures.frontiers.data import (
    alpha_valid,
    beta_valid,
    grouped_rows,
    valid_runs,
)
from Articles.figures.frontiers.reach import summarize_cut
from Articles.figures.geometry import adjacent_runs
from Articles.figures.render import CASE_COLORS, DETECTOR_COLORS, mass_label
from Articles.plot_results import reconstruct_spectrum

GREEN = "#14866D"


def axes_style(ax, case, args):
    ax.set(
        xlabel=r"$m_6$ [GeV]", ylabel=r"$C$", xlim=args.m6_limits, ylim=args.C_limits
    )
    ax.set_title(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
    ax.grid(alpha=0.12, lw=0.5)


def panels(writer, cases, height=4.0):
    fig, axes = writer.plt.subplots(
        1,
        len(cases),
        figsize=(4.15 * len(cases), height),
        squeeze=False,
        layout="constrained",
    )
    return fig, axes[0]


def draw_edge_bands(ax, edges, *, outline=GREEN, max_gap=5.01):
    """Desenha cada lado separadamente; nunca une as duas bordas na vertical."""

    def continuous_runs(records):
        for adjacent in adjacent_runs(records, maximum_gap=max_gap):
            current = []
            for row in adjacent:
                if current and (
                    abs(row["C"] - current[-1]["C"]) > 0.08
                    or finite(row.get("edge_outside_C"))
                    and finite(current[-1].get("edge_outside_C"))
                    and abs(row["edge_outside_C"] - current[-1]["edge_outside_C"])
                    > 0.08
                ):
                    yield current
                    current = []
                current.append(row)
            if current:
                yield current

    for side in ("lower", "upper"):
        records = [row for row in edges if row["edge"] == side]
        for run in continuous_runs(records):
            ax.plot(
                [float(row["m6_GeV"]) for row in run],
                [row["C"] for row in run],
                color=outline,
                lw=0.8,
                alpha=0.85,
            )
        for certified in (True, False):
            bands = [
                row
                for row in records
                if row["edge_certified"] == certified
                and finite(row.get("edge_outside_C"))
                and abs(row["C"] - row["edge_outside_C"]) <= 0.2
            ]
            for run in continuous_runs(bands):
                if len(run) < 2:
                    continue
                ax.fill_between(
                    [float(row["m6_GeV"]) for row in run],
                    [row["C"] for row in run],
                    [row["edge_outside_C"] for row in run],
                    facecolor="#F1DAA8" if certified else "#EEEEEE",
                    edgecolor="#B59861" if certified else "#999999",
                    hatch=None if certified else "////",
                    linewidth=0.25,
                    alpha=0.9,
                    zorder=1,
                )


def draw_strength(ax, crossings, case, field):
    """Conecta apenas níveis numericamente apoiados em massas vizinhas."""
    groups = defaultdict(list)
    for row in crossings:
        if (
            row["m8_GeV"] == case
            and row["field"] == field
            and math.isfinite(row["m6_GeV"])
        ):
            groups[row["m6_GeV"]].append(row)
    tracks = defaultdict(list)
    for rows in groups.values():
        for index, row in enumerate(sorted(rows, key=lambda item: item["C"])):
            tracks[index].append(row)
    for rows in tracks.values():
        for run in adjacent_runs(rows):
            # Um salto grande em C pode trocar a identidade do ramo.
            current = []
            for row in run:
                if current and abs(row["C"] - current[-1]["C"]) > 0.15:
                    ax.plot(
                        [r["m6_GeV"] for r in current],
                        [r["C"] for r in current],
                        color="#222222",
                        ls="--",
                        lw=1.1,
                        marker="." if len(current) == 1 else None,
                        markersize=2,
                        zorder=5,
                    )
                    current = []
                current.append(row)
            ax.plot(
                [r["m6_GeV"] for r in current],
                [r["C"] for r in current],
                color="#222222",
                ls="--",
                lw=1.1,
                marker="." if len(current) == 1 else None,
                markersize=2,
                zorder=5,
            )


def plot_clean_window(dataset, edges, crossings, writer, args):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = panels(writer, dataset.cases)
    for ax, case in zip(axes, dataset.cases, strict=True):
        current = [row for row in edges if float(row["m8_GeV"]) == case]
        draw_edge_bands(ax, current)
        rows = [
            row
            for row in dataset.rows
            if float(row["m8_GeV"]) == case
            and row["boundary_decision"] == "yes"
            and math.isfinite(float(row["m6_GeV"]))
            and not row.get("source_conflict")
        ]
        ax.scatter(
            [float(row["m6_GeV"]) for row in rows],
            [row["C"] for row in rows],
            s=3,
            color=GREEN,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        if args.strength_reference != "none":
            draw_strength(
                ax,
                crossings,
                case,
                "vev_over_Tn" if args.strength_reference == "Tn" else "vev_over_Tc",
            )
        axes_style(ax, case, args)
    handles = [
        Line2D(
            [],
            [],
            color=GREEN,
            marker="o",
            ls="none",
            markersize=3,
            label="Calculated strong-transition points",
        ),
        Patch(facecolor="#F1DAA8", label="Resolved local edge bracket"),
        Patch(
            facecolor="#EEEEEE",
            edgecolor="#999999",
            hatch="////",
            label="Unresolved edge interval",
        ),
    ]
    if args.strength_reference != "none":
        symbol = "n" if args.strength_reference == "Tn" else "c"
        handles.append(
            Line2D(
                [],
                [],
                color="#333333",
                ls="--",
                label=rf"Numerical $v(T_{symbol})/T_{symbol}=1$, where bracketed",
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    writer.save(
        fig,
        "E01_clean_transition_window",
        "Green markers are calculated selected points only. Lines join the outermost selected C samples in adjacent mass columns; they are not a proof of connectedness. Only thin edge intervals are filled: resolved original brackets in sand, unresolved intervals hatched. No vertical connection between the two edges and no failure/non-nucleation markers. Dashed strength reference is a numerical interpolation between saved values surrounding 1, never an analytic solution of the full potential; no line is drawn across missing values or large gaps. White space and omitted failures are not exclusions; the original v1 coverage plot and tables remain the audit.",
    )


def plot_edge_thermodynamics(dataset, edges, writer, args):
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    fig, axes = writer.plt.subplots(
        2,
        len(dataset.cases),
        figsize=(4.2 * len(dataset.cases), 6.5),
        squeeze=False,
        layout="constrained",
    )
    fields = (
        ("alpha_trace", r"$\alpha_{\rm trace}$", alpha_valid),
        ("beta_over_H", r"$\beta/H_n$", beta_valid),
    )
    norm = Normalize(*args.C_limits)
    artist = None
    for j, case in enumerate(dataset.cases):
        for i, (field, label, valid) in enumerate(fields):
            ax = axes[i, j]
            for side, marker, style in (("lower", "o", "--"), ("upper", "^", "-")):
                rows = [
                    row
                    for row in edges
                    if float(row["m8_GeV"]) == case
                    and row["edge"] == side
                    and math.isfinite(float(row["m6_GeV"]))
                    and valid(row)
                ]
                for run in adjacent_runs(rows):
                    ax.plot(
                        [row["m6_GeV"] for row in run],
                        [row[field] for row in run],
                        color="#555555",
                        ls=style,
                        lw=0.6,
                    )
                if rows:
                    artist = ax.scatter(
                        [row["m6_GeV"] for row in rows],
                        [row[field] for row in rows],
                        c=[row["C"] for row in rows],
                        cmap="viridis",
                        norm=norm,
                        s=9,
                        marker=marker,
                        linewidths=0,
                        rasterized=True,
                        zorder=3,
                    )
            valid_values = [
                row[field]
                for row in edges
                if math.isfinite(float(row["m6_GeV"])) and valid(row)
            ]
            ax.set(
                xlabel=r"$m_6$ [GeV]", ylabel=label, xlim=args.m6_limits, yscale="log"
            )
            if valid_values:
                ax.set_ylim(min(valid_values) / 1.3, max(valid_values) * 1.3)
            if not any(
                float(row["m8_GeV"]) == case
                and valid(row)
                and math.isfinite(float(row["m6_GeV"]))
                for row in edges
            ):
                ax.text(
                    0.5,
                    0.5,
                    "Edge observables not yet available",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            ax.grid(alpha=0.15)
            if i == 0:
                ax.set_title(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
    if artist is not None:
        fig.colorbar(
            artist,
            ax=list(axes.flat),
            label=r"$C$ at the sampled edge",
            shrink=0.85,
            pad=0.02,
        )
    fig.legend(
        handles=[
            Line2D([], [], color="#555555", marker="o", ls="--", label="Lower C edge"),
            Line2D([], [], color="#555555", marker="^", ls="-", label="Upper C edge"),
        ],
        loc="outside lower center",
        ncol=2,
    )
    writer.save(
        fig,
        "E02_alpha_beta_at_edges",
        "Alpha and beta/H evaluated at the lower and upper sampled C edges, with C itself encoded by colour. These are edge values, not guaranteed global minima/maxima over the interior. Alpha is retained when thermodynamics succeeded but beta failed; beta with quality flags is omitted. Missing mass columns break curves. Infinite-m6 controls are exported separately rather than placed at an arbitrary mass coordinate.",
    )


def plot_C_dependence(dataset, writer, args):
    fig, axes = writer.plt.subplots(
        2,
        len(dataset.cases),
        figsize=(4.2 * len(dataset.cases), 6.3),
        squeeze=False,
        layout="constrained",
    )
    for j, case in enumerate(dataset.cases):
        for i, (field, label, valid) in enumerate(
            (
                ("alpha_trace", r"$\alpha_{\rm trace}$", alpha_valid),
                ("beta_over_H", r"$\beta/H_n$", beta_valid),
            )
        ):
            ax = axes[i, j]
            has_data = False
            for mass, color in zip(
                args.benchmark_masses,
                CASE_COLORS[: len(args.benchmark_masses)],
                strict=True,
            ):
                rows = [
                    row
                    for row in dataset.rows
                    if float(row["m8_GeV"]) == case and float(row["m6_GeV"]) == mass
                ]
                supported = [row for row in rows if valid(row)]
                for run in valid_runs(rows, valid, args.maximum_C_gap):
                    ax.plot(
                        [row["C"] for row in run],
                        [row[field] for row in run],
                        color=color,
                        lw=0.8,
                    )
                if supported:
                    has_data = True
                    ax.scatter(
                        [row["C"] for row in supported],
                        [row[field] for row in supported],
                        color=color,
                        s=7,
                        label=rf"$m_6={mass:g}$ GeV",
                        rasterized=True,
                    )
            ax.set(xlabel=r"$C$", ylabel=label, xlim=args.C_limits, yscale="log")
            ax.grid(alpha=0.15)
            if i == 0:
                ax.set_title(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
            if has_data:
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Selected C cuts required",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
    writer.save(
        fig,
        "E03_alpha_beta_along_C",
        "Saved alpha and beta/H along C at the two benchmark masses. These cuts test the assumption that edge values bound the interior; no monotonicity is imposed. Lines require adjacent valid samples within the declared C gap; failed/missing samples interrupt the line.",
    )


def reach_profiles(dataset, report_roots, args):
    """Estados refinados quando presentes; na ausência, somente dados salvos."""
    profiles = {}
    rows_by_coordinate = {
        (float(row["m8_GeV"]), float(row["m6_GeV"]), round(row["C"], 12)): row
        for row in dataset.rows
    }
    for root in report_roots:
        for path in sorted(Path(root).glob("m8-*/interior_state.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("kind") != "lisa_cuts":
                continue
            if (
                report["observation_years"] != args.observation_years
                or report["snr_threshold"] != args.snr_threshold
            ):
                raise ValueError(
                    "O relatório LISA usa outro tempo/limiar. Use os valores do plano ou crie outro plano de alcance."
                )
            for cut in report["cuts"]:
                if cut.get("state") not in ("complete", "complete_with_unresolved"):
                    continue
                samples = []
                for sample in cut["samples"]:
                    row = rows_by_coordinate.get(
                        (
                            float(cut["m8_GeV"]),
                            float(cut["m6_GeV"]),
                            round(sample["C"], 12),
                        )
                    )
                    if row is None:
                        raise ValueError(
                            "Relatório LISA sem o respectivo ponto na união de dados; confira a pasta --reach."
                        )
                    samples.append(
                        {**sample, "decision": "unknown", "rho_sw": None}
                        if row.get("source_conflict")
                        else sample
                    )
                profiles[(float(cut["m8_GeV"]), float(cut["m6_GeV"]))] = summarize_cut(
                    cut,
                    samples,
                    report["C_tolerance"],
                    maximum_gap=report["C_step"] + 1e-9,
                )
    for (case, mass), rows in grouped_rows(dataset).items():
        if (case, mass) in profiles or not any(row.get("gw_resolved") for row in rows):
            continue
        samples = [
            {
                "C": row["C"],
                "point_id": row["point_id"],
                "rho_sw": row.get("LISA_rho_sw"),
                "decision": (
                    "yes" if row["LISA_rho_sw"] >= args.snr_threshold else "no"
                )
                if row.get("gw_resolved") and not row.get("source_conflict")
                else "unknown",
            }
            for row in rows
        ]
        profiles[(case, mass)] = summarize_cut(
            {"m8_GeV": case, "m6_GeV": mass, "source": "saved_samples_only"},
            samples,
            args.C_tolerance,
            maximum_gap=args.maximum_C_gap,
        )
    return profiles


def plot_lisa_window(dataset, edges, profiles, writer, args):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = panels(writer, dataset.cases)
    for ax, case in zip(axes, dataset.cases, strict=True):
        draw_edge_bands(
            ax,
            [row for row in edges if float(row["m8_GeV"]) == case],
            outline="#AAAAAA",
        )
        cuts = sorted(
            [
                profile
                for (m8, mass), profile in profiles.items()
                if m8 == case and math.isfinite(mass)
            ],
            key=lambda row: float(row["m6_GeV"]),
        )
        for cut in cuts:
            for low, high in cut["intervals"]:
                ax.plot([float(cut["m6_GeV"])] * 2, [low, high], color=GREEN, lw=1.5)
        for left, right in pairwise(cuts):
            if (
                float(right["m6_GeV"]) - float(left["m6_GeV"]) > args.maximum_mass_gap
                or left["state"] != "complete"
                or right["state"] != "complete"
                or len(left["intervals"]) != len(right["intervals"])
            ):
                continue
            for a, b in zip(left["intervals"], right["intervals"], strict=True):
                if max(a[0], b[0]) > min(a[1], b[1]):
                    continue
                ax.fill_between(
                    [float(left["m6_GeV"]), float(right["m6_GeV"])],
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=GREEN,
                    alpha=0.2,
                    linewidth=0,
                )
        detected = [
            row
            for row in dataset.rows
            if float(row["m8_GeV"]) == case
            and math.isfinite(float(row["m6_GeV"]))
            and row.get("gw_resolved")
            and not row.get("source_conflict")
            and row.get("LISA_rho_sw", -1) >= args.snr_threshold
        ]
        ax.scatter(
            [row["m6_GeV"] for row in detected],
            [row["C"] for row in detected],
            s=7,
            color=GREEN,
            linewidths=0,
            rasterized=True,
        )
        if not detected:
            ax.text(
                0.5,
                0.12,
                "Additional GW cuts required",
                ha="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        axes_style(ax, case, args)
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, alpha=0.2, label="Sample-supported acoustic reach"),
            Line2D(
                [],
                [],
                color=GREEN,
                marker="o",
                ls="none",
                label="Calculated detectable point",
            ),
            Patch(
                facecolor="#EEEEEE",
                edgecolor="#999999",
                hatch="////",
                label="Unresolved transition edge",
            ),
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.suptitle(
        rf"LISA acoustic PIS: $t_{{obs}}={args.observation_years:g}$ yr, $\rho_{{sw}}\geq {args.snr_threshold:g}$"
    )
    writer.save(
        fig,
        "E04_lisa_parameter_window",
        "LISA acoustic-channel reach. Positive intervals require actual resolved SNR samples. Area is connected between mass cuts only if both cuts are complete, have matched overlapping interval components and are within the declared mass gap. Unknowns break the area; disconnected intervals are retained. The search is limited to the prior selected-C envelope and the plan resolution; this does not certify all possible detectable islands or the full GW SNR.",
    )


def plot_endpoint_spectra(dataset, profiles, writer, args):
    from Articles.core_imports import (
        bbo_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        lisa_sensitivity_s_pis,
    )

    frequency = np.geomspace(1e-5, 10, 900)
    groups = [("E05_GW_pure_measure", math.inf, [math.inf])]
    groups += [
        (f"E{6 + i:02d}_GW_m6_{mass:g}", mass, dataset.cases)
        for i, mass in enumerate(args.benchmark_masses)
    ]
    exported, samples = [], []
    for name, mass, cases in groups:
        fig, axes = writer.plt.subplots(
            2,
            len(cases),
            figsize=(max(6.5, 4.3 * len(cases)), 6.5),
            squeeze=False,
            layout="constrained",
        )
        for j, case in enumerate(cases):
            ax, peak_ax = axes[0, j], axes[1, j]
            current = sorted(
                [
                    row
                    for row in dataset.rows
                    if float(row["m8_GeV"]) == case
                    and float(row["m6_GeV"]) == mass
                    and row.get("gw_resolved")
                    and not row.get("source_conflict")
                    and row.get("LISA_rho_sw", -1) >= args.snr_threshold
                ],
                key=lambda row: row["C"],
            )
            chosen = (
                [current[0]]
                + (
                    [current[-1]]
                    if current and current[-1]["point_id"] != current[0]["point_id"]
                    else []
                )
                if current
                else []
            )
            for row, label, color in zip(
                chosen,
                ("lowest sampled C", "highest sampled C"),
                ("#0072B2", "#D55E00"),
            ):
                spectrum = reconstruct_spectrum(
                    row, frequency, row.get("spectrum_defaults") or {}
                )
                legend = rf"$C={row['C']:.4g}$, $\rho_{{sw}}={row['LISA_rho_sw']:.3g}$"
                ax.loglog(
                    frequency, spectrum["total"], color=color, lw=1.3, label=legend
                )
                ax.loglog(frequency, spectrum["sw"], color=color, lw=0.8, ls=":")
                peak_ax.scatter(
                    [row["f_sw_peak_Hz"]],
                    [row["omega_sw_peak_h2"]],
                    color=color,
                    s=22,
                    label=rf"$C={row['C']:.4g}$",
                    zorder=4,
                )
                cut = profiles.get((case, mass), {})
                exported.append(
                    {
                        **row,
                        "figure": name,
                        "endpoint_role": label,
                        "cut_state": cut.get("state", "not_scanned"),
                        "endpoint_interpretation": "lowest_or_highest_detectable_C_sample_not_certified_global_extremum",
                    }
                )
                samples.extend(
                    {
                        "figure": name,
                        "point_id": row["point_id"],
                        "C": row["C"],
                        "m6_GeV": mass,
                        "m8_GeV": case,
                        "f_Hz": float(f),
                        "omega_total_h2": float(total),
                        "omega_sw_h2": float(sound),
                    }
                    for f, total, sound in zip(
                        frequency, spectrum["total"], spectrum["sw"], strict=True
                    )
                )
            for detector, sensitivity, color in zip(
                ("LISA", "DECIGO", "BBO"),
                (
                    lisa_sensitivity_s_pis,
                    decigo_sensitivity_s_pis,
                    bbo_sensitivity_s_pis,
                ),
                DETECTOR_COLORS,
                strict=True,
            ):
                peak_ax.loglog(
                    frequency,
                    args.snr_threshold
                    / math.sqrt(args.observation_years)
                    * sensitivity(frequency * 1000),
                    color=color,
                    lw=1,
                    label=detector,
                )
            if chosen:
                ax.legend(fontsize=8, loc="best")
                maximum = max(row["omega_sw_peak_h2"] for row in chosen)
                ax.set_ylim(maximum * 1e-7, maximum * 10)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No resolved LISA-detectable sample yet",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            ax.set(
                xlabel=r"$f$ [Hz]",
                ylabel=r"$h^2\Omega_{GW}(f)$",
                title=rf"$m_8={mass_label(case)}$ GeV",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            peak_ax.set(
                xlabel=r"$f_{sw,peak}$ [Hz]",
                ylabel=r"$h^2\Omega_{sw,peak}$",
                ylim=(1e-19, 1e-7),
            )
            peak_ax.legend(fontsize=7, loc="best")
            for axis in (ax, peak_ax):
                axis.grid(alpha=0.15)
        fig.suptitle(
            rf"$m_6={mass_label(mass)}$ GeV; {args.observation_years:g} yr, acoustic SNR threshold {args.snr_threshold:g}"
        )
        writer.save(
            fig,
            name,
            "Upper panels: total (solid) and acoustic (dotted) spectra for the lowest and highest CALCULATED LISA-detectable C at the stated masses. These are sampled endpoints, not a proof of global extrema. Lower panels: their acoustic peaks versus frequency-dependent PIS reach, scaled by SNR threshold/sqrt(observation years). Detector PIS curves are not pointwise noise against full spectra. The pure-measure figure has both m6 and m8 infinite. Benchmark coordinates, SNR, point IDs and search status are exported.",
        )
    return exported, samples


def interior_edge_violations(dataset, edges):
    """Se dados interiores já contrariam a hipótese de extremos nas bordas."""
    endpoints = defaultdict(dict)
    for row in edges:
        endpoints[(float(row["m8_GeV"]), float(row["m6_GeV"]))][row["edge"]] = row
    result = []
    for row in dataset.rows:
        sides = endpoints.get((float(row["m8_GeV"]), float(row["m6_GeV"])), {})
        if len(sides) != 2 or not sides["lower"]["C"] < row["C"] < sides["upper"]["C"]:
            continue
        for field, valid in (("alpha_trace", alpha_valid), ("beta_over_H", beta_valid)):
            if valid(row) and all(valid(item) for item in sides.values()):
                values = [item[field] for item in sides.values()]
                if row[field] < min(values) * (1 - 1e-8) or row[field] > max(values) * (
                    1 + 1e-8
                ):
                    result.append(
                        {
                            "point_id": row["point_id"],
                            "m6_GeV": row["m6_GeV"],
                            "m8_GeV": row["m8_GeV"],
                            "C": row["C"],
                            "field": field,
                            "value": row[field],
                            "minimum_edge_value": min(values),
                            "maximum_edge_value": max(values),
                        }
                    )
    return result
