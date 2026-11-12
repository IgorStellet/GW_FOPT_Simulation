"""Figuras vetoriais do artigo. Todas as quantidades físicas vêm dos dados."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from Articles.figures.data import DETECTOR_NAMES, finite
from Articles.figures.geometry import (
    adjacent_runs,
    bracket_columns,
    detector_intervals,
    supported_triangulation,
)
from Articles.plot_results import _pyplot, reconstruct_spectrum

CASE_COLORS = ("#0072B2", "#D55E00", "#009E73")
DETECTOR_COLORS = ("#0072B2", "#CC79A7", "#D55E00")


def mass_label(value):
    return r"\infty" if math.isinf(float(value)) else f"{float(value):.1f}"


class FigureWriter:
    def __init__(self, folder, formats=("pdf", "svg", "png"), dpi=300):
        self.folder, self.formats, self.dpi = Path(folder), tuple(formats), dpi
        self.folder.mkdir(parents=True, exist_ok=True)
        self.plt = _pyplot(self.folder)
        self.plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
            "axes.titlesize": 11, "legend.fontsize": 8, "xtick.labelsize": 9,
            "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        })
        self.files, self.captions = [], {}
        self.preview_notice = None

    def save(self, figure, name, caption):
        if self.preview_notice:
            figure.text(0.5, 1.01, self.preview_notice, ha="center", color="#A04040", fontsize=9)
        for extension in self.formats:
            target = self.folder / f"{name}.{extension}"
            temporary = target.with_suffix(target.suffix + ".tmp")
            figure.savefig(temporary, format=extension, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            temporary.replace(target)
            self.files.append(target)
        self.captions[name] = caption
        self.plt.close(figure)


def _panels(writer, cases, height=3.8):
    fig, axes = writer.plt.subplots(1, len(cases), figsize=(4.15 * len(cases), height), squeeze=False, layout="constrained")
    return fig, axes[0]


def _axes(ax, case, c_limits):
    ax.set(xlabel=r"$m_6$ [GeV]", ylabel=r"$C$", ylim=c_limits, xlim=(500, 2000))
    ax.set_title(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
    ax.grid(alpha=0.13, linewidth=0.5)


def _selected_rows(dataset, case):
    return [row for row in dataset.rows if float(row["m8_GeV"]) == case]


def draw_window(ax, columns, *, color="#0072B2", strength_axis=False):
    """Faixa interna para colunas bounded; outras ficam como testemunhas."""
    rows = bracket_columns(columns)
    valid = [row for row in rows if row["state"] == "bounded" and finite(row["lower_inside"]) and finite(row["upper_inside"])]

    def transform(mass):
        return (1000 / mass) ** 2 if strength_axis else mass

    for run in adjacent_runs(valid):
        x = [transform(row["m6_GeV"]) for row in run]
        lower, upper = [row["lower_inside"] for row in run], [row["upper_inside"] for row in run]
        if len(run) > 1:
            ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, lower, color=color, lw=1.25)
            ax.plot(x, upper, color=color, lw=1.25)
        else:
            ax.plot([x[0]] * 2, [lower[0], upper[0]], color=color, lw=1.5)
        for side in ("lower", "upper"):
            # Se a busca toca C=0, não há testemunha externa nem fronteira
            # física certificada. Só se desenha o ponto interno nesse bound.
            lo = [row[f"{side}_inside"] for row in run]
            out = [row[f"{side}_outside"] if row[f"{side}_state"] == "bracketed" else np.nan for row in run]
            ax.fill_between(x, lo, out, color="#E69F00", alpha=0.35, linewidth=0)
    unresolved = [row for row in rows if row["state"].startswith("unresolved") and math.isfinite(row["m6_GeV"])]
    for side in ("lower", "upper"):
        samples = [row for row in unresolved if finite(row[f"{side}_inside"])]
        ax.scatter([transform(row["m6_GeV"]) for row in samples], [row[f"{side}_inside"] for row in samples],
                   facecolors="none", edgecolors="#8B5BA5", s=8, linewidths=0.65, zorder=4)
    reference = next((row for row in rows if math.isinf(row["m6_GeV"])), None)
    if reference:
        for side in ("lower", "upper"):
            inside = reference[f"{side}_inside"]
            if not finite(inside):
                continue
            if strength_axis:
                ax.scatter([0], [inside], color=color, marker="s", s=20, zorder=5)
            else:
                ax.axhline(inside, color=color, lw=0.8, linestyle="--", alpha=0.5)


def plot_windows(dataset, writer, c_limits):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = _panels(writer, dataset.cases, 4.0)
    for ax, case, color in zip(axes, dataset.cases, CASE_COLORS[:len(dataset.cases)], strict=True):
        draw_window(ax, dataset.columns.get(case, []), color=color)
        _axes(ax, case, c_limits)
    handles = [Patch(facecolor="#0072B2", alpha=0.16, label="Inner interval; connected-band assumption"),
               Patch(facecolor="#E69F00", alpha=0.4, label="Verified boundary bracket"),
               Line2D([], [], marker="o", color="#8B5BA5", markerfacecolor="none", linestyle="none", label="Unresolved edge: inside witness only"),
               Line2D([], [], color="#555555", linestyle="--", label=r"$m_6\to\infty$: inside witnesses")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle(r"$\Lambda=1000\,\mathrm{GeV}$; accepted $S_3/T\simeq140$; $v(T_n)/T_n>1$")
    writer.save(fig, "01_parameter_windows", "Selection windows in physical mass coordinates. Filled bands connect only adjacent bounded columns (5 GeV); orange strips are actual inside/outside brackets. Open symbols are positive witnesses at unresolved edges, not certified boundaries. Dashed horizontal lines show the m6=infinity witnesses for the same m8. White space is not an exclusion. The default selection is |phi_low(Tn)|/Tn>1 and accepted S3/T≈140; lower edges must not be relabelled v(Tc)/Tc=1.")
    fig, ax = writer.plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for case, color in zip(dataset.cases, CASE_COLORS[:len(dataset.cases)], strict=True):
        draw_window(ax, dataset.columns.get(case, []), color=color, strength_axis=True)
        ax.plot([], [], color=color, label=rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
    ax.set(xlabel=r"$(1\,\mathrm{TeV}/m_6)^2$", ylabel=r"$C$", ylim=c_limits)
    ax.legend(loc="best")
    ax.grid(alpha=0.15)
    writer.save(fig, "02_mechanism_compensation", "Overlay in the operator-strength coordinate (1 TeV/m6)^2. Zero turns off the dimension-six operator. The gap between zero and the first finite mass is not filled. Lower C at larger operator strength visualizes compensation, without identifying unknown regions as exclusions.")


def plot_quality(dataset, writer):
    from matplotlib.lines import Line2D

    fig, axes = _panels(writer, dataset.cases, 4.0)
    styles = (("yes", "#009E73", ".", "Selected strong transition"),
              ("weak_nucleation", "#E69F00", ".", "Weak nucleation"),
              ("no_nucleation_found", "#D55E00", ".", "No accepted nucleation found"),
              ("no_first_order_found", "#888888", ".", "No first-order transition found"),
              ("unknown", "#8B5BA5", "x", "Unresolved"))
    for ax, case in zip(axes, dataset.cases, strict=True):
        for decision, color, marker, _ in styles:
            def category(row):
                if row["boundary_decision"] != "no":
                    return row["boundary_decision"]
                return "weak_nucleation" if row["selection_reason"] == "weak_nucleation" else row["point_status"]

            rows = [row for row in _selected_rows(dataset, case) if math.isfinite(float(row["m6_GeV"])) and category(row) == decision]
            ax.scatter([float(row["m6_GeV"]) for row in rows], [row["C"] for row in rows],
                       s=4 if decision != "unknown" else 8, marker=marker, color=color, alpha=0.55, linewidths=0.5, rasterized=True)
        _axes(ax, case, (0, 10))
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
                        for _, color, marker, label in styles], loc="outside lower center", ncol=3)
    writer.save(fig, "A1_calculated_coverage", "Full C=0..10 coverage, including points hidden by the publication zoom. Outside selection includes weak nucleation and absence of accepted nucleation; it does not mean detector exclusion. Unknowns remain separate. Markers alone correspond to evaluated points.")


def _field_bounds(rows, field, c_limits, logarithmic=False):
    """Uma mesma escala de cores para comparar os três cenários."""
    values = [row[field] for row in rows if finite(row.get(field)) and not row.get("source_conflict")
              and math.isfinite(float(row["m6_GeV"])) and c_limits[0] <= row["C"] <= c_limits[1]
              and (not logarithmic or row[field] > 0)
              and (field not in ("alpha_trace", "beta_over_H") or row.get("gw_resolved"))]
    return (min(values), max(values)) if values else None


def _colored_field(ax, rows, field, *, logarithmic=False, levels=None, C_limits=None, bounds=None):
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    if C_limits:
        rows = [row for row in rows if C_limits[0] <= row["C"] <= C_limits[1]]

    good = [row for row in rows if finite(row.get(field)) and not row.get("source_conflict") and math.isfinite(float(row["m6_GeV"]))
            and (not logarithmic or row[field] > 0)]
    if not good:
        ax.text(0.5, 0.5, "Not available in saved data", ha="center", va="center", transform=ax.transAxes, color="#666666")
        return None
    values = np.array([row[field] for row in good])
    low, high = bounds if bounds is not None else (float(values.min()), float(values.max()))
    if low == high:
        low, high = (low / 1.01, high * 1.01) if logarithmic else (low - 0.01, high + 0.01)
    norm = LogNorm(low, high) if logarithmic else Normalize(low, high)
    cmap = "viridis"
    if field == "delta_V0_GeV4" and low < 0 < high:
        norm = TwoSlopeNorm(0, vmin=low, vmax=high)
        cmap = "RdBu_r"
    tri, all_values = supported_triangulation(rows, field)
    artist = None
    if tri is not None and not np.all(tri.mask):
        artist = ax.tripcolor(tri, all_values, shading="gouraud", cmap=cmap, norm=norm, rasterized=True)
        if levels:
            valid_levels = [level for level in levels if low < level < high]
            if valid_levels:
                lines = ax.tricontour(tri, all_values, levels=valid_levels, colors="white", linewidths=0.8)
                ax.clabel(lines, fontsize=7, inline=True)
    scatter = ax.scatter([float(row["m6_GeV"]) for row in good], [row["C"] for row in good],
                         c=values, s=5, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
    return artist or scatter


def plot_observables(dataset, writer, c_limits):
    fields = (("Tn_GeV", r"$T_n$ [GeV]", False), ("alpha_trace", r"$\alpha_{\rm trace}$", True),
              ("beta_over_H", r"$\beta/H_n$", True), ("vev_over_Tc", r"$v(T_c)/T_c$", False))
    for index, case in enumerate(dataset.cases):
        fig, axes = writer.plt.subplots(2, 2, figsize=(8.1, 6.7), layout="constrained")
        rows = _selected_rows(dataset, case)
        for ax, (field, label, logarithmic) in zip(axes.flat, fields, strict=True):
            # Alpha/beta exigem observáveis resolvidos, não apenas nucleação.
            available = [{**row, field: row.get(field) if row.get("gw_resolved") or field not in ("alpha_trace", "beta_over_H") else None} for row in rows]
            artist = _colored_field(ax, available, field, logarithmic=logarithmic, C_limits=c_limits,
                                    bounds=_field_bounds(dataset.rows, field, c_limits, logarithmic),
                                    levels=(0.5, 1, 1.5, 2, 3) if field == "vev_over_Tc" else None)
            _axes(ax, case, c_limits)
            ax.set_title(label)
            if artist is not None:
                fig.colorbar(artist, ax=ax, pad=0.02)
        fig.suptitle(rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$")
        writer.save(fig, f"03_observables_m8_{index}", "Four saved observables: Tn, signed trace-anomaly alpha, beta/H and critical field ratio. Interpolation is local only: triangles touching missing/unknown values or spanning >5.01 GeV in m6 or >0.081 in C are masked. Grey/white gaps have no inferred value. Unavailable GW observables require explicit selected-point or interior calculations, not zeros.")
    for field, label, filename, log in (
        ("vev_over_Tn", r"$v(T_n)/T_n$", "A2_strength_at_nucleation", False),
        ("vev_over_Tc", r"$v(T_c)/T_c$", "04_strength_at_criticality", False),
        ("supercooling_fraction", r"$1-T_n/T_c$", "A3_supercooling", False),
        ("delta_V0_GeV4", r"$V(v,0)-V(0,0)$ [GeV$^4$]", "A4_zero_temperature_vacua", False),
    ):
        fig, axes = _panels(writer, dataset.cases, 4.0)
        for ax, case in zip(axes, dataset.cases, strict=True):
            artist = _colored_field(ax, _selected_rows(dataset, case), field, logarithmic=log, C_limits=c_limits,
                                    bounds=_field_bounds(dataset.rows, field, c_limits, log),
                                    levels=(0,) if field == "delta_V0_GeV4" else (0.5, 1, 1.5, 2, 3) if "vev" in field else None)
            _axes(ax, case, c_limits)
            if artist is not None:
                fig.colorbar(artist, ax=ax, label=label, pad=0.02)
        writer.save(fig, filename, f"{label}. Quantities are read from saved transitions/critical transitions or zero-temperature diagnostics. v(Tc) and v(Tn) are distinct. The zero-temperature zero contour only compares the origin and electroweak vacuum, not all possible minima; C=4.14 of the pure-measure model is not imposed on combined models.")


def plot_detectors(dataset, writer, c_limits, threshold, years):
    from matplotlib.lines import Line2D

    fig, axes = _panels(writer, dataset.cases, 4.0)
    for ax, case in zip(axes, dataset.cases, strict=True):
        draw_window(ax, dataset.columns.get(case, []), color="#999999")
        by_mass = defaultdict(list)
        for row in _selected_rows(dataset, case):
            if math.isfinite(float(row["m6_GeV"])):
                by_mass[float(row["m6_GeV"])].append(row)
        for detector, color in zip(DETECTOR_NAMES, DETECTOR_COLORS, strict=True):
            for mass, rows in by_mass.items():
                for low, high in detector_intervals(rows, f"{detector}_rho_sw", threshold):
                    ax.plot([mass, mass], [low, high], color=color, lw=1.2, alpha=0.7)
            selected = [row for rows in by_mass.values() for row in rows
                        if row.get("gw_resolved") and row.get(f"{detector}_rho_sw", -1) >= threshold]
            ax.scatter([float(row["m6_GeV"]) for row in selected], [row["C"] for row in selected], s=5, color=color, alpha=0.6, rasterized=True)
        _axes(ax, case, c_limits)
    handles = [Line2D([], [], color=color, lw=2, label=name) for name, color in zip(DETECTOR_NAMES, DETECTOR_COLORS, strict=True)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, title=rf"Acoustic PIS: $t_{{obs}}={years:g}$ yr, $\rho_{{sw}}\geq {threshold:g}$")
    writer.save(fig, "05_detector_windows", "Acoustic-channel PIS reach under Schmitz (2021) detector assumptions. rho_sw=sqrt(t_obs/yr) Omega_sw,peak/Omega_PIS^s. Coloured intervals require adjacent resolved GW samples separated by at most 0.041 in C; multiple intervals are retained and unknown gaps are not closed. Grey bands are the strong-nucleation reference. This is not the total GW SNR or an experimental discovery guarantee.")


def benchmark_rows(dataset, maximum=6):
    """Seleção determinística cobrindo cenários e amplitudes; nunca usa inválidos."""
    selected = []
    for case in dataset.cases:
        candidates = sorted((row for row in _selected_rows(dataset, case) if row.get("gw_resolved")), key=lambda row: row["omega_sw_peak_h2"])
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    remaining = sorted((row for row in dataset.rows if row.get("gw_resolved") and row not in selected), key=lambda row: row["omega_sw_peak_h2"], reverse=True)
    for row in remaining:
        if len(selected) >= maximum:
            break
        if all((float(row["m8_GeV"]), float(row["m6_GeV"])) != (float(old["m8_GeV"]), float(old["m6_GeV"])) for old in selected):
            selected.append(row)
    return selected[:maximum]


def plot_spectra(dataset, writer, threshold, years, maximum=6):
    from Articles.core_imports import (
        bbo_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        lisa_sensitivity_s_pis,
    )

    benchmarks = benchmark_rows(dataset, maximum)
    frequency = np.geomspace(1e-6, 10, 1000)
    fig, (ax, peak_ax) = writer.plt.subplots(1, 2, figsize=(11.2, 4.1), layout="constrained")
    samples = []
    for i, row in enumerate(benchmarks):
        color = writer.plt.get_cmap("tab10")(i)
        label = f"B{i+1}"
        spectrum = reconstruct_spectrum(row, frequency, row["spectrum_defaults"])
        ax.loglog(frequency, spectrum["total"], color=color, lw=1.4, label=label)
        ax.loglog(frequency, spectrum["sw"], color=color, lw=0.75, ls=":")
        peak_ax.scatter(row["f_sw_peak_Hz"], row["omega_sw_peak_h2"], color=color, s=30, label=label, zorder=4)
        for f, total, sound in zip(frequency, spectrum["total"], spectrum["sw"], strict=True):
            samples.append({"benchmark": label, "point_id": row["point_id"], "f_Hz": float(f), "omega_total_h2": float(total), "omega_sw_h2": float(sound)})
    for name, sensitivity, color in zip(DETECTOR_NAMES, (lisa_sensitivity_s_pis, decigo_sensitivity_s_pis, bbo_sensitivity_s_pis), DETECTOR_COLORS, strict=True):
        peak_ax.loglog(frequency, threshold / math.sqrt(years) * sensitivity(frequency * 1000), color=color, lw=1.3, label=name)
    if not benchmarks:
        ax.text(0.5, 0.5, "GW observables not yet available\nRun the selected-point plan or interior stage", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(ncol=2)
    ax.set(xlabel=r"$f$ [Hz]", ylabel=r"$h^2\Omega_{GW}(f)$", title="Selected spectra: total and acoustic (dotted)")
    peak_ax.set(xlabel=r"$f_{sw,peak}$ [Hz]", ylabel=r"$h^2\Omega_{sw,peak}$", title=rf"Acoustic PIS reach: {years:g} yr, $\rho_{{sw}}={threshold:g}$")
    peak_ax.legend(fontsize=8)
    peak_ax.set_ylim(1e-20, max(1e-7, max((row["omega_sw_peak_h2"] for row in benchmarks), default=1e-10) * 5))
    for axis in (ax, peak_ax):
        axis.grid(alpha=0.15, which="major")
    writer.save(fig, "06_gw_benchmarks", "Left: reconstructed total (solid) and sound-wave (dotted) spectra using the recorded efficiencies. Right: acoustic peak amplitudes compared with threshold/sqrt(years) times the published PIS curves. PIS curves are never plotted as pointwise detector noise against the full spectrum. Benchmark coordinates and transition indices are exported, not chosen silently.")
    return benchmarks, samples


def plot_relations(dataset, writer, c_limits):
    from scipy.spatial import cKDTree

    good = [row for row in dataset.rows if row.get("gw_resolved")]
    fig, axes = writer.plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    for case, color in zip(dataset.cases, CASE_COLORS[:len(dataset.cases)], strict=True):
        rows = [row for row in good if float(row["m8_GeV"]) == case]
        label = rf"$m_8={mass_label(case)}\,\mathrm{{GeV}}$"
        axes[0].scatter([row["alpha_trace"] for row in rows], [row["beta_over_H"] for row in rows], color=color, s=7, alpha=0.5, label=label, rasterized=True)
        axes[1].scatter([row["f_sw_peak_Hz"] for row in rows], [row["omega_sw_peak_h2"] for row in rows], color=color, s=7, alpha=0.5, rasterized=True)
    pairs = []
    for i, case in enumerate(dataset.cases):
        left = [row for row in good if float(row["m8_GeV"]) == case]
        right = [row for row in good if float(row["m8_GeV"]) in dataset.cases[i+1:]]
        if not left or not right:
            continue
        features = lambda rows: np.log10([[row["f_sw_peak_Hz"], row["omega_sw_peak_h2"]] for row in rows])
        distances, indices = cKDTree(features(right)).query(features(left))
        for row, distance, index in zip(left, distances, indices, strict=True):
            other = right[index]
            pairs.append({"left_point_id": row["point_id"], "right_point_id": other["point_id"],
                          "left_m6_GeV": row["m6_GeV"], "right_m6_GeV": other["m6_GeV"],
                          "left_m8_GeV": row["m8_GeV"], "right_m8_GeV": other["m8_GeV"],
                          "left_C": row["C"], "right_C": other["C"], "peak_distance_dex": float(distance)})
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.15)
    axes[0].set(xlabel=r"$\alpha_{trace}$", ylabel=r"$\beta/H_n$", title="Thermodynamic correlations")
    axes[1].set(xlabel=r"$f_{sw,peak}$ [Hz]", ylabel=r"$h^2\Omega_{sw,peak}$", title="Peak-space coincidences across models")
    if good:
        axes[0].legend()
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Resolved GW observables required", ha="center", transform=ax.transAxes)
    writer.save(fig, "07_correlations_and_degeneracy", "Cross-scenario correlations and near coincidences in log10(f_peak), log10(Omega_peak). The candidate-pair table ranks Euclidean distances in dex; this is not a detector likelihood, a parameter posterior, or a demonstration of degeneracy of the entire spectrum.")
    fig, axes = _panels(writer, dataset.cases, 4.0)
    for ax, case in zip(axes, dataset.cases, strict=True):
        rows = [row for row in _selected_rows(dataset, case) if math.isfinite(float(row["m6_GeV"])) and row["boundary_decision"] == "yes"]
        for verified, color, label in ((False, "#BBBBBB", "Controls unavailable / not both outside"), (True, "#009E73", "Combined only; both pure controls outside")):
            subset = [row for row in rows if row["cooperation_verified"] == verified]
            ax.scatter([float(row["m6_GeV"]) for row in subset], [row["C"] for row in subset], s=5, color=color, label=label, rasterized=True)
        _axes(ax, case, c_limits)
    axes[0].legend(fontsize=7)
    writer.save(fig, "08_verified_cooperation", "Green markers require three actual calculations at matched coordinates: combined selected, C=0 polynomial control outside selection, and m6=m8=infinity measure control outside selection at the same C. Missing or failed controls never establish cooperation. This compares the strong-transition criterion, not collider invisibility or GW non-detectability.")
    return sorted(pairs, key=lambda row: row["peak_distance_dex"])[:20]
