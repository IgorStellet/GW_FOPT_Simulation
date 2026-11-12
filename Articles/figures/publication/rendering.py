"""Apresentação das seis figuras: tipografia, cores e eixos compartilhados."""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Articles.figures.publication.analysis import (
    envelope_curves,
    field_mesh,
    value_valid,
)
from Articles.figures.publication.detectors import lisa_plis_h2
from Articles.figures.render import (
    CASE_COLORS,
    DETECTOR_COLORS,
    FigureWriter,
    mass_label,
)

LABELS = {
    "Tn_GeV": r"$T_n$ [GeV]",
    "alpha_trace": r"$\alpha_{\mathrm{tr}}$",
    "beta_over_H": r"$\beta/H_n$",
    "vev_over_Tc": r"$v(T_c)/T_c$",
}


class PublicationWriter(FigureWriter):
    """Usa o salvamento atômico existente; cada figura possui uma legenda."""

    def __init__(self, folder, formats=("pdf", "svg", "png"), dpi=300):
        super().__init__(folder, formats, dpi)
        self.plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "dejavuserif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "axes.linewidth": 0.75,
                "lines.linewidth": 1.35,
                "savefig.pad_inches": 0.06,
            }
        )


def style_axis(ax, args, *, label=True):
    ax.set(xlim=args.m6_limits, ylim=args.C_limits)
    if label:
        ax.set(xlabel=r"$m_6$ [GeV]", ylabel=r"$C$")
    ax.tick_params(direction="in", top=False, right=False)
    ax.grid(False)


def draw_envelope(
    ax, records, *, color="0.55", filled=False, strength=False, linewidth=1
):
    for x, low, high in envelope_curves(records, strength_axis=strength):
        if filled:
            ax.fill_between(
                x, low, high, color=color, alpha=0.13, linewidth=0, zorder=1
            )
        ax.plot(x, low, color=color, linewidth=linewidth, zorder=3)
        ax.plot(x, high, color=color, linewidth=linewidth, zorder=3)


def plot_windows(dataset, window_rows, writer, args):
    fig, ax = writer.plt.subplots(figsize=(7.2, 4.7), layout="constrained")
    for case, color in zip(dataset.cases, CASE_COLORS, strict=True):
        records = [row for row in window_rows if row["m8_GeV"] == case]
        draw_envelope(
            ax, records, color=color, filled=True, strength=True, linewidth=1.45
        )
        # A origem é um controle isolado. Não se preenche o intervalo não
        # amostrado entre m6=infinito e a maior massa finita da campanha.
        pure = next((r for r in records if math.isinf(r["m6_GeV"])), None)
        if pure:
            for side in ("lower_C", "upper_C"):
                ax.plot([0, 0.15], [pure[side]] * 2, color=color, ls="--", linewidth=1)
    ax.set(xlabel=r"$(1\,\mathrm{TeV}/m_6)^2$", ylabel=r"$C$", ylim=args.C_limits)
    ax.set_xlim(-0.035, (1000 / args.m6_limits[0]) ** 2 + 0.04)
    ax.tick_params(direction="in")
    handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.65,
            label=rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$",
        )
        for case, color in zip(dataset.cases, CASE_COLORS, strict=True)
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.text(
        0.98,
        0.98,
        r"$\Lambda=1000\,\mathrm{GeV}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    writer.save(
        fig,
        "Fig01_mechanism_compensation",
        "Selected-transition envelopes for the three dimension-eight mass scales at Lambda=1000 GeV. "
        "Selection requires an accepted S3(Tn)/Tn approximately 140 and |phi_low(Tn)|/Tn>1. "
        + (
            "Curves are regularized boundary guides constrained by the actual inside/outside witnesses. "
            "Regularization does not relax any witness inequality; constraints, raw samples and fitted values are exported. "
            if args.boundary_guide == "bracketed"
            else "Curves interpolate the raw inside witnesses, which are one-sided bounds rather than measured boundaries. "
        )
        + "Shape-preserving PCHIP interpolation connects contiguous 5 GeV mass columns. "
        "There is no extrapolation or vertical joining of the two edges. "
        "Shading is an interpolated guide, not a certification of a connected nucleating interior. "
        "Dashed stubs at zero inverse mass mark the isolated m6=infinity controls; the unsampled gap "
        "to m6=2000 GeV is not filled. Smaller m6 corresponds to a larger dimension-six coefficient.",
    )


def _case_rows(dataset, case, args):
    return [
        row
        for row in dataset.rows
        if float(row["m8_GeV"]) == case
        and args.m6_limits[0] <= float(row["m6_GeV"]) <= args.m6_limits[1]
        and args.C_limits[0] <= row["C"] <= args.C_limits[1]
    ]


def _mesh(rows, field, args, audits, case):
    tri, values, report = field_mesh(
        rows,
        field,
        maximum_mass_gap=args.maximum_mass_gap,
        maximum_C_gap=args.maximum_C_gap,
    )
    audits.append({"m8_GeV": case, **report})
    return tri, values


def has_area(tri):
    return tri is not None and np.any(~tri.mask)


def snr_contour(
    ax, tri, values, threshold, *, color="#222222", width=1.3, linestyle="solid"
):
    if not has_area(tri):
        return
    valid_vertices = np.unique(tri.triangles[~tri.mask])
    level = math.log10(threshold)
    if np.min(values[valid_vertices]) < level < np.max(values[valid_vertices]):
        return ax.tricontour(
            tri,
            values,
            levels=[level],
            colors=[color],
            linewidths=width,
            linestyles=linestyle,
            zorder=6,
        )


def plot_atlases(dataset, window_rows, writer, args):
    """Escalas idênticas entre cenários, sem pontos sobre o campo interpolado."""
    reports = []
    bounds = {}
    for field in LABELS:
        values = [
            row[field]
            for row in dataset.rows
            if value_valid(row, field)
            and math.isfinite(float(row["m6_GeV"]))
            and args.m6_limits[0] <= row["m6_GeV"] <= args.m6_limits[1]
            and args.C_limits[0] <= row["C"] <= args.C_limits[1]
        ]
        if values:
            low, high = min(values), max(values)
            bounds[field] = (low * 0.99, high * 1.01) if low == high else (low, high)
    for index, case in enumerate(dataset.cases):
        rows = _case_rows(dataset, case, args)
        records = [row for row in window_rows if row["m8_GeV"] == case]
        lisa_tri, lisa_values = _mesh(rows, "LISA_rho_sw", args, reports, case)
        half_tri, half_values = (None, None)
        if getattr(args, "diagnostics", False):
            half_tri, half_values = _mesh(
                rows, "LISA_rho_sw_half_step", args, reports, case
            )
        fig, axes = writer.plt.subplots(2, 2, figsize=(7.4, 6.3), layout="constrained")
        for panel, (ax, (field, label)) in enumerate(
            zip(axes.flat, LABELS.items(), strict=True)
        ):
            tri, values = _mesh(rows, field, args, reports, case)
            logarithmic = field in ("alpha_trace", "beta_over_H")
            if has_area(tri) and field in bounds:
                low, high = bounds[field]
                norm = LogNorm(low, high) if logarithmic else Normalize(low, high)
                shown = np.power(10.0, values) if logarithmic else values
                artist = ax.tripcolor(
                    tri,
                    shown,
                    shading="gouraud",
                    cmap="viridis",
                    norm=norm,
                    rasterized=True,
                )
                bar = fig.colorbar(artist, ax=ax, pad=0.025, fraction=0.047)
                bar.ax.tick_params(labelsize=8)
                if field == "vev_over_Tc":
                    vertices = np.unique(tri.triangles[~tri.mask])
                    if np.min(values[vertices]) < 1 < np.max(values[vertices]):
                        contour = ax.tricontour(
                            tri,
                            values,
                            levels=[1],
                            colors="black",
                            linestyles="--",
                            linewidths=1.15,
                            zorder=7,
                        )
                        contour.set_path_effects(
                            [
                                path_effects.Stroke(linewidth=3, foreground="white"),
                                path_effects.Normal(),
                            ]
                        )
                        ax.clabel(
                            contour, fmt={1: r"$\xi_c=1$"}, fontsize=8, inline=True
                        )
            else:
                ax.text(
                    0.5,
                    0.45,
                    "Insufficient local coverage",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=9,
                    color="0.4",
                )
            draw_envelope(ax, records, linewidth=0.65, color="0.65")
            # Contorno em duas cores para ser legível sobre todo o colormap.
            snr_contour(
                ax, lisa_tri, lisa_values, args.snr_threshold, color="white", width=2.7
            )
            snr_contour(
                ax,
                lisa_tri,
                lisa_values,
                args.snr_threshold,
                color="#9C2542",
                width=1.35,
            )
            snr_contour(
                ax,
                half_tri,
                half_values,
                args.snr_threshold,
                color="#9C2542",
                width=0.8,
                linestyle="dashed",
            )
            style_axis(ax, args)
            ax.set_title(f"({chr(97 + panel)}) " + label, loc="left", fontsize=11)
        fig.suptitle(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$", fontsize=13)
        handles = [
            Line2D(
                [],
                [],
                color="#9C2542",
                lw=1.35,
                label=rf"LISA: $\rho_{{\rm sw}}={args.snr_threshold:g}$, ${args.observation_years:g}$ yr",
            ),
            Line2D(
                [],
                [],
                color="black",
                ls="--",
                lw=1.15,
                label=r"$\xi_c\equiv v(T_c)/T_c=1$",
            ),
        ]
        if getattr(args, "diagnostics", False):
            handles.append(
                Line2D(
                    [],
                    [],
                    color="#9C2542",
                    lw=0.8,
                    ls="--",
                    label=r"LISA with $\beta(h/2)$",
                ),
            )
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        writer.save(
            fig,
            f"Fig0{index + 2}_observables_m8_{index}",
            f"Thermodynamic atlas at m8={mass_label(case)} GeV: (a) nucleation temperature, "
            "(b) positive trace-anomaly strength, (c) beta/Hn, and (d) the critical-temperature field ratio. "
            "Colour normalizations are shared by all three mass scenarios; alpha and beta use logarithmic scales. "
            "Only local triangles with mass span at most 100.01 GeV and C span at most 0.081 are coloured. "
            "Failed additional evaluations mask adjacent triangles; boundary-only records in which an observable "
            "was never calculated do not supply a value for that observable. White regions have no supported "
            "interpolant and are not physical exclusions. The burgundy contour denotes the LISA acoustic-channel "
            f"SNR={args.snr_threshold:g} level for {args.observation_years:g} years, using the same PIS prescription "
            "as the scan. It is drawn only where resolved SNR values bracket the threshold. Grey curves are "
            "boundary guides. The black dashed contour with a white halo marks xi_c=v(Tc)/Tc=1, "
            "evaluated from the critical phases only where the data bracket unity. "
            "The selection continues to use the field ratio at Tn. "
            + (
                "A thin burgundy dashed contour shows the saved beta(h/2) diagnostic. "
                if getattr(args, "diagnostics", False)
                else "Numerical h/2 diagnostics are retained in the tables. "
            ),
        )
    return reports


def _fill_reach(ax, tri, values, threshold, color):
    if not has_area(tri):
        return
    good = np.unique(tri.triangles[~tri.mask])
    level = math.log10(threshold)
    maximum = float(np.max(values[good]))
    if maximum > level:
        return ax.tricontourf(
            tri,
            values,
            levels=[level, max(maximum + 1, level + 1)],
            colors=[color],
            alpha=0.75,
            zorder=2,
        )


def _label_observable_contours(ax, tri, values, field):
    if not has_area(tri):
        return
    good = np.unique(tri.triangles[~tri.mask])
    low, high = float(np.min(values[good])), float(np.max(values[good]))
    levels = (
        [level for level in (-4, -3, -2, -1, 0) if low < level < high]
        if field == "alpha_trace"
        else [level for level in (2, 3, 4, 5) if low < level < high]
    )
    if len(levels):
        lines = ax.tricontour(
            tri,
            values,
            levels=levels,
            colors="0.22",
            linestyles="--",
            linewidths=0.65,
            zorder=5,
        )
        for level, segments in zip(lines.levels, lines.allsegs, strict=True):
            segments = [segment for segment in segments if len(segment) > 2]
            if segments:
                segment = max(segments, key=lambda s: np.ptp(s[:, 0]))
                point = segment[len(segment) * 2 // 3]
                ax.clabel(
                    lines,
                    levels=[level],
                    manual=[tuple(point)],
                    fontsize=7,
                    inline=True,
                    fmt="%g",
                    inline_spacing=3,
                )


def plot_detector_contours(dataset, window_rows, writer, args):
    """Análogo à figura de referência: faixas instrumentais + isolinhas físicas."""
    fig, axes = writer.plt.subplots(
        3, 2, figsize=(7.4, 8.2), sharex=True, sharey=True, layout="constrained"
    )
    audits = []
    for column, case in enumerate(dataset.cases):
        rows = _case_rows(dataset, case, args)
        records = [row for row in window_rows if row["m8_GeV"] == case]
        meshes = {
            name: _mesh(rows, name + "_rho_sw", args, audits, case)
            for name in ("LISA", "DECIGO", "BBO")
        }
        half_tri, half_values = (None, None)
        if getattr(args, "diagnostics", False):
            half_tri, half_values = _mesh(
                rows, "LISA_rho_sw_half_step", args, audits, case
            )
        for row_index, field in enumerate(("alpha_trace", "beta_over_H")):
            ax = axes[column, row_index]
            draw_envelope(ax, records, filled=True, color="0.7", linewidth=0.65)
            # Desenha a sensibilidade mais ampla primeiro para permitir que a
            # região LISA permaneça visível quando as áreas se sobrepõem.
            for name, color in (
                ("BBO", "#F0D35B"),
                ("DECIGO", "#EDAB67"),
                ("LISA", "#D96870"),
            ):
                _fill_reach(ax, *meshes[name], args.snr_threshold, color)
            tri, values = _mesh(rows, field, args, audits, case)
            _label_observable_contours(ax, tri, values, field)
            snr_contour(
                ax,
                half_tri,
                half_values,
                args.snr_threshold,
                color="#9C2542",
                width=0.75,
                linestyle="dashed",
            )
            style_axis(ax, args, label=False)
            if column == 2:
                ax.set_xlabel(r"$m_6$ [GeV]")
            if row_index == 0:
                ax.set_ylabel(r"$C$")
            label = (
                r"$\log_{10}\alpha_{\rm tr}$"
                if field == "alpha_trace"
                else r"$\log_{10}(\beta/H_n)$"
            )
            ax.text(
                0.96,
                0.07,
                label,
                transform=ax.transAxes,
                fontsize=9,
                ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
            ax.set_title(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$", fontsize=10)
            if row_index == 0 and any(
                r["m6_GeV"] >= 1200 and math.isfinite(r["m6_GeV"]) for r in records
            ):
                zoom = ax.inset_axes([0.48, 0.22, 0.47, 0.27])
                draw_envelope(zoom, records, color="0.65", linewidth=0.6)
                for name, color in (
                    ("BBO", "#F0D35B"),
                    ("DECIGO", "#EDAB67"),
                    ("LISA", "#D96870"),
                ):
                    _fill_reach(zoom, *meshes[name], args.snr_threshold, color)
                snr_contour(
                    zoom,
                    half_tri,
                    half_values,
                    args.snr_threshold,
                    color="#9C2542",
                    width=0.6,
                    linestyle="dashed",
                )
                upper = [r["upper_C"] for r in records if 1200 <= r["m6_GeV"] <= 2000]
                zoom.set(
                    xlim=(1200, 2000),
                    ylim=(min(upper) - 0.13, max(upper) + 0.035),
                    xticks=(1200, 1600, 2000),
                )
                zoom.tick_params(labelsize=6, direction="in", pad=1)
                for spine in zoom.spines.values():
                    spine.set_visible(True)
                    spine.set_color("0.55")
                    spine.set_linewidth(0.5)
    handles = [
        Patch(facecolor=color, label=name)
        for name, color in (
            ("LISA", "#D96870"),
            ("DECIGO", "#EDAB67"),
            ("BBO", "#F0D35B"),
        )
    ]
    if getattr(args, "diagnostics", False):
        handles.append(
            Line2D(
                [], [], color="#9C2542", ls="--", lw=0.75, label=r"LISA: $\beta(h/2)$"
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    writer.save(
        fig,
        "Fig05_detector_reach_and_contours",
        "Acoustic-channel detector reach in the (m6,C) plane at fixed Lambda=1000 GeV. Rows correspond "
        "to the three m8 scenarios. Dashed contours in the left/right columns show log10(alpha_trace) and "
        "log10(beta/Hn), respectively; labels are logarithms, not the untransformed observables. "
        "Left-panel insets magnify the high-mass reach bands. "
        + (
            "Burgundy dashed lines show the saved beta(h/2) diagnostic. "
            if getattr(args, "diagnostics", False)
            else "The saved h/2 diagnostic is retained in the tables. "
        )
        + f"Instrumental shading denotes interpolated acoustic SNR>={args.snr_threshold:g} for "
        f"{args.observation_years:g} years using the historical Schmitz PIS fits, with BBO drawn first, "
        "then DECIGO, then LISA. These are overlapping reach regions, not mutually exclusive classes. "
        "Only supported local triangles are used and unknown evaluations remain holes. LISA crossings were "
        "refined by the campaign; DECIGO/BBO boundaries are post-processed at the available sampling "
        "resolution and do not inherit the LISA refinement tolerance. Grey envelopes are guides through "
        "selected samples; unshaded regions need not be excluded.",
    )
    return audits


def plot_matched_spectra(benchmarks, spectra_rows, cases, writer, args):
    from Articles.core_imports import (
        bbo_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        lisa_sensitivity_s_pis,
    )

    by_group = defaultdict(list)
    for row in benchmarks:
        by_group[row["group"]].append(row)
    groups = sorted(by_group)
    fig, axes = writer.plt.subplots(
        2, max(1, len(groups)), figsize=(7.8, 6.1), squeeze=False, layout="constrained"
    )
    if not groups:
        for ax in axes.flat:
            ax.text(
                0.5,
                0.5,
                "No validated matched triplet",
                ha="center",
                transform=ax.transAxes,
            )
        writer.save(
            fig,
            "Fig06_matched_GW_signals",
            "No three-scenario matched detectable spectrum satisfies the declared selection tolerance.",
        )
        return
    colors = dict(zip(cases, CASE_COLORS, strict=True))
    styles = ("-", "--", ":")
    markers = ("o", "s", "^")
    for column, group in enumerate(groups):
        top, bottom = axes[:, column]
        rows = by_group[group]
        for index, row in enumerate(rows):
            samples = [r for r in spectra_rows if r["benchmark"] == row["benchmark"]]
            frequency = [r["f_Hz"] for r in samples]
            top.loglog(
                frequency,
                [r["omega_total_h2"] for r in samples],
                color=colors[row["m8_GeV"]],
                ls=styles[index],
                lw=1.65,
                label=rf"{row['benchmark']}: $\rho_{{\rm sw}}={row['LISA_rho_sw']:.1f}$",
            )
            if getattr(args, "diagnostics", False):
                top.fill_between(
                    frequency,
                    [
                        min(r["omega_total_h2"], r["omega_total_h2_beta_half_step"])
                        for r in samples
                    ],
                    [
                        max(r["omega_total_h2"], r["omega_total_h2_beta_half_step"])
                        for r in samples
                    ],
                    color=colors[row["m8_GeV"]],
                    alpha=0.12,
                    linewidth=0,
                )
            if getattr(args, "diagnostics", False):
                bottom.plot(
                    [row["f_sw_peak_Hz"], row["f_sw_peak_Hz_beta_half_step"]],
                    [row["omega_sw_peak_h2"], row["omega_sw_peak_h2_beta_half_step"]],
                    color=colors[row["m8_GeV"]],
                    lw=0.9,
                    alpha=0.7,
                )
            bottom.scatter(
                row["f_sw_peak_Hz"],
                row["omega_sw_peak_h2"],
                facecolor="none",
                marker=markers[index],
                s=(76, 54, 36)[index],
                edgecolor=colors[row["m8_GeV"]],
                lw=1.15,
                zorder=5,
            )
        lisa_frequency = np.geomspace(3e-5, 0.3, 600)
        top.loglog(
            lisa_frequency,
            lisa_plis_h2(
                lisa_frequency,
                years=args.observation_years,
                threshold=args.snr_threshold,
            ),
            color="0.2",
            lw=1.25,
            alpha=0.8,
            label=rf"LISA PLIS, ${args.observation_years:g}$ yr",
            zorder=2,
        )
        top.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{\rm GW}(f)$", xlim=(3e-5, 0.3))
        peak_max = max(r["omega_sw_peak_h2"] for r in rows)
        top.set_ylim(peak_max * 1e-5, peak_max * 5)
        top.set_title(
            "(a) Fixed $m_6$" if group == "A" else "(b) Different $m_6$", loc="left"
        )
        top.legend(frameon=False, loc="upper right", fontsize=8)
        frequency = np.geomspace(1e-4, 0.1, 500)
        for name, sensitivity, color in zip(
            ("LISA", "DECIGO", "BBO"),
            (lisa_sensitivity_s_pis, decigo_sensitivity_s_pis, bbo_sensitivity_s_pis),
            DETECTOR_COLORS,
            strict=True,
        ):
            reach = (
                args.snr_threshold
                / math.sqrt(args.observation_years)
                * sensitivity(1000 * frequency)
            )
            bottom.loglog(frequency, reach, color=color, lw=1, label=name)
        bottom.set(
            xlabel=r"$f_{\rm sw,peak}$ [Hz]",
            ylabel=r"$h^2\Omega_{\rm sw,peak}$",
            xlim=(1e-4, 0.1),
        )
        minimum = min(r["omega_sw_peak_h2"] for r in rows)
        maximum = max(r["omega_sw_peak_h2"] for r in rows)
        bottom.set_ylim(min(minimum / 20, 2e-15), max(maximum * 20, 2e-11))
        bottom.legend(frameon=False, loc="lower left", fontsize=8)
        for ax in (top, bottom):
            ax.tick_params(direction="in", which="both")
            ax.grid(alpha=0.12, which="major", linewidth=0.4)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                ls=style,
                label=rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$",
            )
            for case, color, style in zip(cases, CASE_COLORS, styles, strict=True)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    writer.save(
        fig,
        "Fig06_matched_GW_signals",
        "Nearby predicted GW signals from different m8 scenarios. Group A compares the same m6; group B "
        "allows different m6. Benchmarks pass the LISA threshold with both saved beta(h) and beta(h/2). They are chosen by "
        "minimizing distance in (log10 f_sw,peak, log10 h2Omega_sw,peak) within the declared low-mass ranges. "
        "No direction of parameter displacement is imposed. Upper panels show total reconstructed spectra using the "
        "stored efficiencies and a LISA power-law-integrated sensitivity (PLIS) from the Robson-Cornish-Liu "
        "reference noise, including its four-year Galactic foreground fit. This graphical reference does not "
        "replace the campaign's acoustic PIS classification. Lower panels place acoustic peaks against frequency-dependent acoustic "
        f"PIS reach for SNR={args.snr_threshold:g} and {args.observation_years:g} years. "
        "PIS curves belong to this peak plane and are not pointwise noise curves for the upper spectra. "
        + (
            "Faint bands and peak displacements show the saved numerical step comparison. "
            if getattr(args, "diagnostics", False)
            else "Numerical step comparisons remain available in the tables. "
        )
        + "Table 1 lists parameters, thermodynamics, both SNR values and spectral distances. Close curves demonstrate "
        "similar predicted signals, not statistical indistinguishability in an instrumental likelihood.",
    )
