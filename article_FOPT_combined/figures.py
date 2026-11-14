"""Seis figuras principais, atlas suplementar e auditoria da interpolação.

Uma célula do mapa usa quatro pontos vizinhos calculados na grade (m6,u).
Se um vértice está pendente ou falhou, a célula não é colorida. Não há uma
triangulação global capaz de atravessar regiões não nucleadas ou preencher
um resultado ausente. A figura final exige auditoria; --draft permite inspeção.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from . import core, detectors
from .boundaries import column_path
from .observables import plan
from .storage import atomic_json, load_campaign, point_records, read_json, write_csv

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
DETECTOR_COLORS = {"LISA": "#D95459", "DECIGO": "#EDAB53", "BBO": "#F4DA69"}
FIELDS = (
    ("Tn_GeV", r"$T_n$ [GeV]"),
    ("alpha_trace", r"$\alpha_{\rm tr}$"),
    ("beta_over_H", r"$\beta/H_n$"),
    ("vev_over_Tc", r"$v(T_c)/T_c$"),
)


def mass_label(mass):
    """Representação LaTeX curta dos cenários, incluindo operador desligado."""
    return r"\infty" if math.isinf(float(mass)) else f"{float(mass):.1f}"


def number(value):
    """Retorna número finito ou None; valores ausentes continuam ausentes nos mapas."""
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def valid_value(row, field):
    """Uma transição selecionada pode ter Tn válido e beta pendente, separadamente."""
    value = number(row.get(field))
    if (field == "beta_over_H" or field.endswith("_rho_sw")) and row.get(
        "beta_stability_passed"
    ) is False:
        return False
    return (
        row.get("decision") == "yes"
        and row.get("nucleation_accepted")
        and value is not None
        and value > 0
    )


def flatten_records(output, *, years):
    """Lê os resultados mais recentes de observáveis e calcula SNRs sem novos bounces."""
    records, rows = {}, []
    for record in point_records(output, "observables"):
        row = {"point_id": record["point_id"], **record["parameters"], **record["summary"]}
        row["m6_GeV"] = math.inf if row["m6_GeV"] == "inf" else float(row["m6_GeV"])
        row["m8_GeV"] = math.inf if row["m8_GeV"] == "inf" else float(row["m8_GeV"])
        if (
            valid_value(row, "alpha_trace")
            and valid_value(row, "beta_over_H")
            and number(row.get("f_sw_peak_Hz"))
        ):
            for detector in detectors.PIS:
                row[f"{detector}_rho_sw"] = detectors.acoustic_snr(row, detector, years=years)
        records[row["point_id"]] = row
        rows.append(row)
    return records, rows


@dataclass
class Surface:
    """Triângulos locais de células completas e valores transformados (log ou linear)."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    triangles: np.ndarray
    logarithmic: bool
    omitted_cells: int

    def triangulation(self):
        """Materializa a geometria Matplotlib apenas quando há área calculada suficiente."""
        import matplotlib.tri as mtri

        return mtri.Triangulation(self.x, self.y, self.triangles) if len(self.triangles) else None


def build_surface(planned, records, case, field):
    """Interpola somente entre quatro vizinhos válidos do plano, sem pontos validation."""
    logarithmic = field in ("alpha_trace", "beta_over_H") or field.endswith("_rho_sw")
    grid = [
        p
        for p in planned["points"]
        if p["kind"] == "grid" and p["m8_GeV"] == case and math.isfinite(p["m6_GeV"])
    ]
    masses = sorted(
        m for m in planned.get("observable_masses", {p["m6_GeV"] for p in grid}) if math.isfinite(m)
    )
    fractions = sorted({p["fraction"] for p in grid})
    nodes, x, y, values, validity = {}, [], [], [], []
    for point in grid:
        key = point["m6_GeV"], point["fraction"]
        if key in nodes:
            continue
        nodes[key] = len(x)
        x.append(point["m6_GeV"])
        y.append(point["C"])
        row = records.get(point["point_id"], {})
        valid = valid_value(row, field)
        value = row[field] if valid else 1.0
        values.append(math.log10(value) if logarithmic else value)
        validity.append(valid)
    triangles, omitted = [], 0
    for left, right in pairwise(masses):
        for bottom, top in pairwise(fractions):
            keys = ((left, bottom), (right, bottom), (right, top), (left, top))
            if not all(k in nodes for k in keys):
                omitted += 1
                continue
            a, b, c, d = (nodes[k] for k in keys)
            if not all(validity[i] for i in (a, b, c, d)) or y[a] == y[d] or y[b] == y[c]:
                omitted += 1
                continue
            # Ordem anti-horária para o interpolador linear de Matplotlib.
            triangles.extend(((a, b, c), (a, c, d)))
    return Surface(
        np.asarray(x),
        np.asarray(y),
        np.asarray(values),
        np.asarray(triangles, dtype=int).reshape(-1, 3),
        logarithmic,
        omitted,
    )


def coverage_report(output, campaign, planned, records):
    """Audita bordas, dados ausentes e validação independente da interpolação em C."""
    missing, failed, edge_issues = [], [], []
    for ci, case in enumerate(campaign.m8_values):
        for mi, mass in enumerate(campaign.masses()):
            path = column_path(output, ci, mi)
            if not path.exists():
                edge_issues.append({"m8_GeV": case, "m6_GeV": mass, "state": "not_calculated"})
            else:
                column = read_json(path)
                if column["state"] not in ("bounded", "no_seed_on_grid"):
                    edge_issues.append({"m8_GeV": case, "m6_GeV": mass, "state": column["state"]})
    for point in planned["points"]:
        row = records.get(point["point_id"])
        fields = [field for field, label in FIELDS]
        absent = [field for field in fields if not valid_value(row or {}, field)]
        if absent:
            request = {
                **point,
                "missing_fields": ";".join(absent),
                "status": (row or {}).get("status", "not_calculated"),
            }
            (missing if row is None else failed).append(request)
    from matplotlib.tri import LinearTriInterpolator

    checks, interpolators = [], {}
    for point in planned["points"]:
        if point["kind"] != "validation":
            continue
        actual = records.get(point["point_id"], {})
        nearby = sorted(
            (
                p
                for p in planned["points"]
                if p["kind"] == "grid"
                and p["m6_GeV"] == point["m6_GeV"]
                and p["m8_GeV"] == point["m8_GeV"]
            ),
            key=lambda p: p["fraction"],
        )
        left = next((p for p in reversed(nearby) if p["fraction"] < point["fraction"]), None)
        right = next((p for p in nearby if p["fraction"] > point["fraction"]), None)
        if math.isinf(point["m6_GeV"]) and (left is None or right is None):
            continue
        for field, _label in (*FIELDS, *((f"{name}_rho_sw", name) for name in detectors.PIS)):
            if not valid_value(actual, field):
                continue
            logarithmic = field in ("alpha_trace", "beta_over_H") or field.endswith("_rho_sw")
            transform = math.log10 if logarithmic else float
            if math.isfinite(point["m6_GeV"]):
                key = point["m8_GeV"], field
                if key not in interpolators:
                    surface = build_surface(planned, records, point["m8_GeV"], field)
                    tri = surface.triangulation()
                    interpolators[key] = (
                        LinearTriInterpolator(tri, surface.values) if tri is not None else None
                    )
                interpolator = interpolators[key]
                estimate = (
                    interpolator(point["m6_GeV"], point["C"])
                    if interpolator is not None
                    else np.ma.masked
                )
                if np.ma.is_masked(estimate):
                    checks.append(
                        {
                            "point_id": point["point_id"],
                            "m8_GeV": point["m8_GeV"],
                            "m6_GeV": point["m6_GeV"],
                            "C": point["C"],
                            "field": field,
                            "error": None,
                            "unit": "unsupported",
                            "needs_refinement": True,
                        }
                    )
                    continue
                estimate = float(estimate)
            else:
                a, b = records.get(left["point_id"], {}), records.get(right["point_id"], {})
                if not all(valid_value(r, field) for r in (a, b)):
                    continue
                fraction = (point["fraction"] - left["fraction"]) / (
                    right["fraction"] - left["fraction"]
                )
                estimate = (1 - fraction) * transform(a[field]) + fraction * transform(b[field])
            error = abs(transform(actual[field]) - estimate)
            relative = error if logarithmic else error / max(abs(actual[field]), 1e-12)
            checks.append(
                {
                    "point_id": point["point_id"],
                    "m8_GeV": point["m8_GeV"],
                    "m6_GeV": point["m6_GeV"],
                    "C": point["C"],
                    "field": field,
                    "error": relative,
                    "unit": "dex" if logarithmic else "relative",
                    "needs_refinement": relative > (0.15 if logarithmic else 0.1),
                }
            )
    return {
        "boundary_issues": edge_issues,
        "selection_contradictions": [
            p for p in failed if records[p["point_id"]].get("decision") == "no"
        ],
        "missing_points": missing,
        "failed_points": failed,
        "interpolation_checks": checks,
        "interpolation_refinement_needed": any(c["needs_refinement"] for c in checks),
        "counts": {
            "planned": len(planned["points"]),
            "saved_observables": len(records),
            "missing": len(missing),
            "failed_or_incomplete": len(failed),
            "boundary_issues": len(edge_issues),
        },
        "ready": not edge_issues
        and not missing
        and not failed
        and not any(c["needs_refinement"] for c in checks),
    }


def boundary_segments(output, campaign, case_index):
    """Separa sequências de colunas delimitadas; PCHIP não atravessa coluna pendente."""
    groups, group = [], []
    for mi, mass in reversed(list(enumerate(campaign.masses()))):
        if not math.isfinite(mass):
            continue
        path = column_path(output, case_index, mi)
        row = read_json(path) if path.exists() else {}
        if row.get("state") == "bounded":
            group.append(row)
        elif group:
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    return groups


def match_benchmarks(rows, cases, *, threshold, maximum_distance=0.2):
    """Seleciona sinais próximos em log(f_peak,Omega_peak), priorizando massas menores.

    Grupo A usa o controle C puro quando disponível; B usa m6<=1000 GeV.
    Nenhuma proximidade é presumida: o raio máximo é salvo junto aos benchmarks.
    """
    valid = [
        r
        for r in rows
        if r.get("observables_complete")
        and valid_value(r, "alpha_trace")
        and valid_value(r, "beta_over_H")
        and r.get("LISA_rho_sw", 0) >= threshold
    ]
    groups, used = [], set()
    for name, pure in (("A", True), ("B", False)):
        pools = []
        for index, case in enumerate(cases):
            pool = [
                r
                for r in valid
                if r["m8_GeV"] == case
                and (
                    (index == 0 and pure and math.isinf(r["m6_GeV"]))
                    or (
                        not (index == 0 and pure)
                        and math.isfinite(r["m6_GeV"])
                        and r["m6_GeV"] <= (1300 if pure else 1000)
                    )
                )
            ]
            pools.append(pool)
        if not all(pools):
            continue
        coordinates = [
            np.log10([[r["f_sw_peak_Hz"], r["omega_sw_peak_h2"]] for r in pool]) for pool in pools
        ]
        trees = [cKDTree(x) for x in coordinates[1:]]
        candidates = []
        for index, anchor in enumerate(pools[0]):
            selected, distances = [anchor], []
            for pool, tree in zip(pools[1:], trees, strict=True):
                distance, chosen = tree.query(coordinates[0][index])
                selected.append(pool[int(chosen)])
                distances.append(float(distance))
            keys = tuple(r["point_id"] for r in selected)
            if max(distances, default=0) <= maximum_distance and keys not in used:
                finite_masses = [r["m6_GeV"] for r in selected if math.isfinite(r["m6_GeV"])]
                candidates.append(
                    (
                        (
                            max(finite_masses),
                            sum(r["C"] for r in selected),
                            max(distances, default=0),
                        ),
                        selected,
                        keys,
                        max(distances, default=0),
                    )
                )
        if candidates:
            _, selected, keys, distance = min(candidates, key=lambda candidate: candidate[0])
            used.add(keys)
            groups.append(
                {
                    "group": name,
                    "maximum_log_peak_distance": distance,
                    "tolerance_dex": maximum_distance,
                    "benchmarks": [
                        dict(row, benchmark=f"{name}{i + 1}") for i, row in enumerate(selected)
                    ],
                }
            )
    return groups


class Writer:
    """Estilo único, PDF vetorial, PNG e legendas em inglês para a colaboração."""

    def __init__(self, destination, dpi, draft):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self.plt, self.destination, self.dpi, self.draft = plt, Path(destination), dpi, draft
        self.captions = []
        self.products = []
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.facecolor": "white",
            }
        )

    def save(self, figure, name, caption):
        """Marca rascunhos e salva duas representações da mesma figura."""
        if self.draft:
            existing = figure._suptitle.get_text() if figure._suptitle else ""
            figure.suptitle(
                "DRAFT: incomplete or unvalidated sampling" + ("\n" + existing if existing else ""),
                fontsize=8,
            )
        for extension in ("pdf", "png"):
            figure.savefig(
                self.destination / f"{name}.{extension}", dpi=self.dpi, bbox_inches="tight"
            )
            self.products.append(f"{name}.{extension}")
        self.captions.append(f"**{name}.** {caption}\n")
        self.plt.close(figure)


def draw_windows(writer, output, campaign, cases, name="Fig01_mechanism_compensation"):
    """Figura 1: compensação de mecanismos, com faixas contínuas e sem pontos."""
    from matplotlib.lines import Line2D

    figure, ax = writer.plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for case, color in zip(cases, COLORS, strict=False):
        ci = campaign.m8_values.index(case)
        for group in boundary_segments(output, campaign, ci):
            if len(group) < 2:
                continue
            x = np.array([(1000 / float(r["m6_GeV"])) ** 2 for r in group])
            order = np.argsort(x)
            x = x[order]
            xx = np.linspace(x[0], x[-1], max(30, len(x) * 3))
            lower = PchipInterpolator(x, np.array([r["C_low_inside"] for r in group])[order])(xx)
            upper = PchipInterpolator(x, np.array([r["C_high_inside"] for r in group])[order])(xx)
            ax.fill_between(xx, lower, upper, color=color, alpha=0.13, linewidth=0)
            ax.plot(xx, lower, color=color, lw=1.6)
            ax.plot(xx, upper, color=color, lw=1.6)
        pure_path = column_path(output, ci, 0)
        if pure_path.exists():
            pure = read_json(pure_path)
            if pure["C_low_inside"] is not None:
                ax.plot(
                    [0, 0],
                    [pure["C_low_inside"], pure["C_high_inside"]],
                    color=color,
                    lw=3,
                    alpha=0.8,
                )
    ax.set(xlabel=r"$(1\,\mathrm{TeV}/m_6)^2$", ylabel=r"$C$", ylim=campaign.C_range)
    ax.legend(
        handles=[
            Line2D([], [], color=color, label=rf"$m_8={mass_label(case)}$ GeV")
            for case, color in zip(cases, COLORS, strict=False)
        ],
        frameon=False,
    )
    ax.grid(alpha=0.12)
    writer.save(
        figure,
        name,
        "Selected first-order-transition windows at Lambda=1000 GeV. "
        "Boundaries interpolate calculated inside witnesses using shape-preserving PCHIP; "
        "their bracketing uncertainty is tabulated separately. Selection requires accepted "
        "S3(Tn)/Tn=140 within the declared tolerance, v(Tn)/Tn>1 and V(v,0)<V(0,0). "
        "Shading assumes a connected interval between its two edges. Lines at zero inverse mass "
        "show the m6=infinity controls; the gap to the finite mass scan is left unfilled.",
    )


def contour(ax, surface, levels, **kwargs):
    """Desenha somente níveis presentes na área interpolada e calculada."""
    tri = surface.triangulation()
    if tri is None:
        return None
    used = np.unique(surface.triangles)
    lo, hi = min(surface.values[used]), max(surface.values[used])
    levels = [level for level in levels if lo < level < hi]
    return ax.tricontour(tri, surface.values, levels=levels, **kwargs) if levels else None


def shared_limits(planned, records, cases):
    """Mesmas escalas de cor em todos os atlas; nenhuma seleção por percentis."""
    limits = {}
    training_ids = {p["point_id"] for p in planned["points"] if p["kind"] == "grid"}
    for field, _label in FIELDS:
        values = [
            r[field]
            for identifier, r in records.items()
            if identifier in training_ids and r["m8_GeV"] in cases and valid_value(r, field)
        ]
        if field in ("alpha_trace", "beta_over_H"):
            values = list(np.log10(values))
        lo, hi = (min(values), max(values)) if values else (0, 1)
        if hi <= lo:
            hi = lo + max(abs(lo) * 0.01, 0.01)
        limits[field] = lo, hi
    return limits


def draw_atlas(writer, planned, records, campaign, case, name, limits, threshold):
    """Mapas Tn, alpha, beta/H e v(Tc)/Tc, com o contorno acústico LISA."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    figure, axes = writer.plt.subplots(
        2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True, layout="constrained"
    )
    lisa = build_surface(planned, records, case, "LISA_rho_sw")
    for ax, (field, label) in zip(axes.flat, FIELDS, strict=True):
        surface = build_surface(planned, records, case, field)
        tri = surface.triangulation()
        if tri is not None:
            lo, hi = limits[field]
            image = ax.tricontourf(
                tri, surface.values, levels=np.linspace(lo, hi, 30), cmap="viridis", extend="both"
            )
            bar = figure.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
            if surface.logarithmic:
                ticks = [
                    math.log10(mantissa) + exponent
                    for exponent in range(math.floor(lo), math.ceil(hi) + 1)
                    for mantissa in (1, 2, 5)
                    if lo <= math.log10(mantissa) + exponent <= hi
                ]
                if len(ticks) < 2:
                    ticks = list(np.linspace(lo, hi, 3))
                ticks = ticks[:: max(1, math.ceil(len(ticks) / 5))]
                bar.set_ticks(ticks)
                bar.ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, pos: f"{10**value:.2g}")
                )
            else:
                bar.locator = MaxNLocator(nbins=4)
                bar.formatter = FuncFormatter(lambda value, pos: f"{value:.1f}")
                bar.update_ticks()
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient calculated support",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="0.4",
            )
        contour(ax, lisa, [math.log10(threshold)], colors="white", linewidths=1.6)
        contour(ax, lisa, [math.log10(threshold)], colors="black", linewidths=0.8)
        if field == "vev_over_Tc":
            critical = contour(ax, surface, [1], colors="#E8A530", linewidths=1.3, linestyles="--")
            if critical is not None:
                ax.clabel(critical, fmt={1: r"$v_c/T_c=1$"}, fontsize=7)
        ax.set(
            title=label,
            xlim=campaign.m6_range[:2]
            if campaign.m6_range[0] != campaign.m6_range[1]
            else (campaign.m6_range[0] - 10, campaign.m6_range[1] + 10),
            ylim=campaign.C_range,
        )
        ax.grid(alpha=0.08)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$m_6$ [GeV]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$C$")
    figure.suptitle(
        rf"$m_8={mass_label(case)}$ GeV; black contour: LISA acoustic SNR={threshold:g}",
        fontsize=10,
    )
    writer.save(
        figure,
        name,
        f"Thermodynamic atlas for m8={case:g} GeV. Panels show Tn, signed trace-anomaly alpha, beta/Hn "
        "and v(Tc)/Tc. Colour scales are shared by all selected scenarios; alpha and beta are interpolated "
        "logarithmically. The outlined contour marks the LISA acoustic threshold. "
        "The critical-strength contour is drawn only where v(Tc)/Tc=1 is supported by the calculated field. "
        "Every coloured cell has four calculated neighbouring vertices; validation data and missing-cell "
        "requests accompany the figures. No analytic critical-strength boundary is assumed.",
    )


def draw_detectors(writer, planned, records, campaign, cases, threshold):
    """Figura 5: regiões detectáveis mais isolinhas de alpha e beta/H."""
    from matplotlib.patches import Patch

    figure, axes = writer.plt.subplots(
        len(cases),
        2,
        figsize=(7.2, 2.8 * len(cases)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for i, case in enumerate(cases):
        for j, field in enumerate(("alpha_trace", "beta_over_H")):
            ax = axes[i, j]
            has_data = False
            for detector in ("BBO", "DECIGO", "LISA"):
                surface = build_surface(planned, records, case, f"{detector}_rho_sw")
                tri = surface.triangulation()
                if tri is None:
                    continue
                has_data = True
                upper = float(max(surface.values[np.unique(surface.triangles)]))
                if upper > math.log10(threshold):
                    ax.tricontourf(
                        tri,
                        surface.values,
                        levels=[math.log10(threshold), upper + 1],
                        colors=[DETECTOR_COLORS[detector]],
                        alpha=0.7,
                    )
            physical = build_surface(planned, records, case, field)
            tri = physical.triangulation()
            if tri is not None:
                used = physical.values[np.unique(physical.triangles)]
                levels = np.arange(math.ceil(min(used)), math.floor(max(used)) + 1)
                lines = contour(ax, physical, levels, colors="0.2", linestyles="--", linewidths=0.8)
                if lines is not None:
                    ax.clabel(lines, fmt=lambda x: rf"$10^{{{x:g}}}$", fontsize=7)
            if not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient calculated support",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.4",
                )
            ax.set_title(
                rf"$m_8={mass_label(case)}$ GeV; "
                + (r"$\alpha_{\rm tr}$" if j == 0 else r"$\beta/H_n$")
            )
            ax.set(
                xlim=campaign.m6_range[:2]
                if campaign.m6_range[0] != campaign.m6_range[1]
                else (campaign.m6_range[0] - 10, campaign.m6_range[1] + 10),
                ylim=campaign.C_range,
            )
            ax.grid(alpha=0.1)
        axes[i, 0].set_ylabel(r"$C$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$m_6$ [GeV]")
    figure.legend(
        handles=[Patch(color=color, label=name) for name, color in DETECTOR_COLORS.items()],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    writer.save(
        figure,
        "Fig05_detector_reach_and_contours",
        f"Acoustic-channel detector reach at SNR>={threshold:g}. Left and right columns overlay "
        "alpha_trace and beta/Hn contours, labelled by their values. The reach follows the calculated "
        "peak amplitude and frequency with published peak-integrated sensitivity fits. "
        "Each detector is tested throughout the sampled band, including its lower and upper sides; "
        "detectability is not assumed monotonic in C. White space outside calculated support carries "
        "no exclusion claim. Observation time and the missing-support audit are recorded in figure_report.json.",
    )


def draw_spectra(writer, groups, cases, years, threshold):
    """Figura 6: espectros com PLIS LISA acima e picos/PIS dos detectores abaixo."""
    from matplotlib.lines import Line2D

    figure, axes = writer.plt.subplots(
        2, max(1, len(groups)), figsize=(7.4, 5.8), squeeze=False, layout="constrained"
    )
    samples = []
    frequency = np.geomspace(1e-6, 1, 1000)
    if not groups:
        for ax in axes.flat:
            ax.text(
                0.5,
                0.5,
                "No matched detectable benchmark set\nwithin the declared tolerance",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
    for index, group in enumerate(groups):
        top, bottom = axes[:, index]
        top.loglog(
            frequency,
            detectors.lisa_plis_h2(frequency, years=years, threshold=threshold),
            color="0.3",
            lw=1.2,
            label="LISA PLIS",
        )
        for detector in detectors.PIS:
            bottom.loglog(
                frequency,
                detectors.pis_h2(frequency, detector, years=years, threshold=threshold),
                color=DETECTOR_COLORS[detector],
                lw=1.2,
            )
        for row in group["benchmarks"]:
            ci = cases.index(row["m8_GeV"])
            spectrum = core.sound_spectrum(frequency, row)
            top.loglog(frequency, spectrum, color=COLORS[ci], ls=("-", "--", "-.", ":")[ci], lw=1.6)
            bottom.scatter(
                row["f_sw_peak_Hz"], row["omega_sw_peak_h2"], s=30, color=COLORS[ci], zorder=4
            )
            samples.extend(
                {"benchmark": row["benchmark"], "f_Hz": float(f), "omega_sw_h2": float(omega)}
                for f, omega in zip(frequency, spectrum, strict=True)
            )
        top.set(
            title=f"Matched group {group['group']}",
            xlabel=r"$f$ [Hz]",
            ylabel=r"$h^2\Omega_{\rm sw}(f)$",
            ylim=(1e-23, 1e-7),
            xlim=(1e-5, 1),
        )
        top.legend(loc="upper left", frameon=False, fontsize=7)
        bottom.legend(
            handles=[
                Line2D([], [], color=color, label=name) for name, color in DETECTOR_COLORS.items()
            ],
            loc="upper right",
            frameon=False,
            fontsize=7,
        )
        bottom.set(
            xlabel=r"$f_{\rm sw,peak}$ [Hz]",
            ylabel=r"$h^2\Omega_{\rm sw,peak}$",
            ylim=(1e-19, 1e-7),
            xlim=(1e-5, 1),
        )
        for ax in (top, bottom):
            ax.grid(alpha=0.12, which="major")
    figure.legend(
        handles=[
            Line2D([], [], color=COLORS[i], label=rf"$m_8={mass_label(case)}$ GeV")
            for i, case in enumerate(cases)
        ],
        loc="outside lower center",
        ncol=min(4, len(cases)),
        frameon=False,
    )
    writer.save(
        figure,
        "Fig06_matched_GW_signals",
        "Nearby acoustic GW predictions from distinct m8 scenarios. Group A compares a C-only control "
        "with combined mechanisms, when available; group B prioritises m6<=1000 GeV. "
        "Sets are selected within a Euclidean distance of 0.2 dex in (log10 f_peak,log10 peak amplitude). "
        "All benchmarks individually pass the acoustic LISA threshold. Upper panels include the LISA "
        "power-law-integrated sensitivity as a visual reference; lower panels compare peaks to the "
        "peak-integrated LISA, DECIGO and BBO sensitivities. Exact coordinates, SNRs and matching distances "
        "are tabulated; no detector-level parameter degeneracy is inferred from peak proximity alone.",
    )
    return samples


def render(args):
    """Audita primeiro; publicação requer suporte completo, rascunho permite diagnosticar."""
    if not all(math.isfinite(x) and x > 0 for x in (args.years, args.snr)) or args.dpi < 72:
        raise ValueError("Tempo/limiar positivos e finitos; dpi>=72.")
    output, destination = args.output.resolve(), args.destination.resolve()
    if output == destination or output in destination.parents or destination in output.parents:
        raise ValueError("Pasta das figuras deve ficar separada da campanha.")
    if destination.exists() and any(destination.iterdir()) and not args.replace:
        raise ValueError(
            "Destino não vazio; use --replace para regenerar apenas os produtos gráficos."
        )
    campaign, manifest = load_campaign(output, verify=False)
    cases = tuple(args.m8 or campaign.m8_values[:3])
    if (
        len(cases) != 3
        or len(set(cases)) != 3
        or any(case not in campaign.m8_values for case in cases)
        or not math.isinf(cases[0])
    ):
        raise ValueError(
            "Escolha três cenários do manifesto, começando por m8=inf como controle C puro; "
            "os demais ganham atlas suplementar."
        )
    refined_path = output / "plans" / "observables-refined.json"
    planned = plan(output, campaign, allow_partial=args.draft, refine=refined_path.exists())
    records, rows = flatten_records(output, years=args.years)
    report = coverage_report(output, campaign, planned, records)
    destination.mkdir(parents=True, exist_ok=True)
    old_products_path = destination / "figure_products.json"
    old_products = read_json(old_products_path) if old_products_path.exists() else []
    atomic_json(destination / "coverage_audit.json", report)
    write_csv(
        destination / "point_requests.csv", report["missing_points"] + report["failed_points"]
    )
    write_csv(destination / "interpolation_validation.csv", report["interpolation_checks"])
    groups = match_benchmarks(rows, cases, threshold=args.snr)
    report.update(
        campaign_fingerprint=manifest["fingerprint"],
        years=args.years,
        snr_threshold=args.snr,
        primary_m8=cases,
        supplementary_m8=[case for case in campaign.m8_values if case not in cases],
        benchmark_groups=len(groups),
        source="sound waves only; published PIS fits; 1 yr/SNR1 rescaled",
        primary_figures=6,
        boundaries="PCHIP guides with tabulated brackets; connected-band assumption",
        draft=args.draft,
        generation_complete=False,
    )
    report["ready"] = report["ready"] and len(groups) == 2
    atomic_json(destination / "figure_report.json", report)
    write_csv(destination / "points_analysis.csv", rows)
    write_csv(
        destination / "boundary_brackets.csv",
        [read_json(p) for p in sorted((output / "boundaries").glob("m8-*/column-*.json"))],
    )
    write_csv(
        destination / "benchmarks.csv",
        [
            dict(row, group=g["group"], matching_distance_dex=g["maximum_log_peak_distance"])
            for g in groups
            for row in g["benchmarks"]
        ],
    )
    if not args.draft and not report["ready"]:
        raise ValueError(
            "Auditoria ainda tem bordas/pontos pendentes, interpolação a refinar ou benchmarks insuficientes. "
            "Veja coverage_audit.json e point_requests.csv; use --draft para diagnóstico."
        )
    writer = Writer(destination, args.dpi, args.draft)
    draw_windows(writer, output, campaign, cases)
    limits = shared_limits(planned, records, campaign.m8_values)
    for index, case in enumerate(cases):
        draw_atlas(
            writer,
            planned,
            records,
            campaign,
            case,
            f"Fig0{index + 2}_observables",
            limits,
            args.snr,
        )
    draw_detectors(writer, planned, records, campaign, cases, args.snr)
    samples = draw_spectra(writer, groups, cases, args.years, args.snr)
    for index, case in enumerate(campaign.m8_values):
        if case not in cases:
            draw_atlas(
                writer,
                planned,
                records,
                campaign,
                case,
                f"Supplement_m8_{index}_observables",
                limits,
                args.snr,
            )
    if len(campaign.m8_values) > len(cases):
        draw_windows(
            writer, output, campaign, campaign.m8_values, name="Supplement_all_scenarios_windows"
        )
    write_csv(destination / "spectrum_samples.csv", samples)
    (destination / "CAPTIONS.md").write_text(
        "# Figure captions\n\n" + "\n".join(writer.captions), encoding="utf-8"
    )
    # Só retira produtos conhecidos da edição anterior depois de concluir a
    # nova. Não há limpeza recursiva nem alteração dos dados da campanha.
    for name in set(old_products) - set(writer.products):
        path = destination / name
        if (
            path.parent.resolve() == destination
            and path.suffix in (".pdf", ".png")
            and path.name.startswith(("Fig0", "Supplement_m8_", "Supplement_all_scenarios_windows"))
            and path.is_file()
        ):
            path.unlink()
    atomic_json(old_products_path, writer.products)
    report.update(generation_complete=True, figure_products=writer.products)
    atomic_json(destination / "figure_report.json", report)
    print(f"Seis figuras principais e atlas suplementares em {destination}")
