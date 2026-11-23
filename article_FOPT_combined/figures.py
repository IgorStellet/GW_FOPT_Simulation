"""Seis figuras principais, atlas suplementar e auditoria da interpolação.

Os mapas usam triângulos dentro das células vizinhas da grade (m6,u).
Os vértices são dados calculados. Pontos adicionais refinam as arestas das
células, e as amostras de validação ficam reservadas à auditoria.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, pairwise, product
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from . import core, detectors
from .boundaries import column_path
from .model import Parameters
from .observables import plan
from .storage import atomic_json, load_campaign, point_id, point_records, read_json, write_csv

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
            and all(
                (number(row.get(field)) or 0) > 0 for field in ("f_sw_peak_Hz", "omega_sw_peak_h2")
            )
        ):
            for detector in detectors.PIS:
                row[f"{detector}_rho_sw"] = detectors.acoustic_snr(row, detector, years=years)
        records[row["point_id"]] = row
        rows.append(row)
    return records, rows


@dataclass
class Surface:
    """Triângulos com suporte local e valores transformados (log ou linear)."""

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


def case_samples(planned, records, case, *, include_validation=False):
    """Separa os nós da grade e os pontos extras dentro de colunas amostradas."""
    grid = [p for p in planned["points"] if p["kind"] == "grid" and p["m8_GeV"] == case]
    reserved = {
        p["point_id"] for p in planned["points"] if p["kind"] == "grid" or not include_validation
    }
    limits = {}
    for p in grid:
        mass, C = p["m6_GeV"], p["C"]
        low, high = limits.get(mass, (C, C))
        limits[mass] = min(low, C), max(high, C)
    extra = [
        row
        for identifier, row in records.items()
        if identifier not in reserved
        and row.get("m8_GeV") == case
        and row.get("m6_GeV") in limits
        and limits[row["m6_GeV"]][0] < row["C"] < limits[row["m6_GeV"]][1]
    ]
    return grid, extra


def supported_cell(corners, extra, x, y, valid):
    """Escolhe a diagonal com mais área sustentada por amostras válidas."""
    if len(corners) < 3:
        return []
    if len(corners) == 3:
        options = ((tuple(corners),),)
    else:
        a, b, c, d = corners
        options = (((a, b, c), (a, c, d)), ((a, b, d), (b, c, d)))

    def supported(triangles):
        out, area = [], 0.0
        for triangle in triangles:
            a, b, c = triangle
            # Cada triângulo de duas colunas tem uma aresta vertical. Cortes
            # adicionais subdividem essa aresta sem passar pela célula vizinha.
            if x[a] != x[b]:
                a, b, c = (b, c, a) if x[b] == x[c] else (c, a, b)
            bottom, top = sorted((a, b), key=y.__getitem__)
            chain = [
                bottom,
                *(i for i in extra.get(x[a], ()) if y[bottom] < y[i] < y[top]),
                top,
            ]
            for p, q in pairwise(chain):
                if not all(valid[i] for i in (p, q, c)) or y[p] == y[q]:
                    continue
                area += (y[q] - y[p]) * abs(x[c] - x[p])
                out.append((p, c, q) if x[p] < x[c] else (p, q, c))
        return area, out

    return max(map(supported, options), key=lambda option: option[0])[1]


def build_surface(planned, records, case, field, *, include_validation=False):
    """Usa os triângulos válidos de cada célula, incluindo refinamentos locais."""
    logarithmic = field in ("alpha_trace", "beta_over_H") or field.endswith("_rho_sw")
    grid, additional = case_samples(planned, records, case, include_validation=include_validation)
    grid = [p for p in grid if math.isfinite(p["m6_GeV"])]
    masses = sorted(
        m for m in planned.get("observable_masses", {p["m6_GeV"] for p in grid}) if math.isfinite(m)
    )
    fractions = sorted({p["fraction"] for p in grid})
    nodes, x, y, values, validity = {}, [], [], [], []

    def add_node(point):
        index = len(x)
        x.append(point["m6_GeV"])
        y.append(point["C"])
        row = records.get(point["point_id"], {})
        valid = valid_value(row, field)
        value = row[field] if valid else 1.0
        values.append(math.log10(value) if logarithmic else value)
        validity.append(valid)
        return index

    for point in grid:
        key = point["m6_GeV"], point["fraction"]
        if key not in nodes:
            nodes[key] = add_node(point)
    extra = {}
    for point in sorted(additional, key=lambda r: r["C"]):
        if math.isfinite(point["m6_GeV"]):
            extra.setdefault(point["m6_GeV"], []).append(add_node(point))
    triangles, omitted = [], 0
    for left, right in pairwise(masses):
        for bottom, top in pairwise(fractions):
            keys = ((left, bottom), (right, bottom), (right, top), (left, top))
            corners = [nodes[k] for k in keys if k in nodes]
            supported = supported_cell(corners, extra, x, y, validity)
            if not supported:
                omitted += 1
            triangles.extend(supported)
    return Surface(
        np.asarray(x),
        np.asarray(y),
        np.asarray(values),
        np.asarray(triangles, dtype=int).reshape(-1, 3),
        logarithmic,
        omitted,
    )


def lisa_refinement_requests(planned, records, cases, *, threshold, tolerance):
    """Solicita o ponto médio de cada cruzamento LISA ainda largo em C."""
    requests = []
    for case in cases:
        grid, extra = case_samples(planned, records, case, include_validation=True)
        columns = {}
        for point in (*grid, *extra):
            columns.setdefault(point["m6_GeV"], {})[point["C"]] = records.get(point["point_id"], {})
        for mass, samples in sorted(columns.items()):
            for (low, a), (high, b) in pairwise(sorted(samples.items())):
                if high - low <= tolerance or not all(
                    valid_value(row, "LISA_rho_sw") for row in (a, b)
                ):
                    continue
                if (a["LISA_rho_sw"] >= threshold) == (b["LISA_rho_sw"] >= threshold):
                    continue
                params = Parameters(mass, case, (low + high) / 2)
                identifier = point_id(params)
                if identifier not in records:
                    requests.append(
                        {
                            "point_id": identifier,
                            "m6_GeV": mass,
                            "m8_GeV": case,
                            "C": params.C,
                            "C_left": low,
                            "C_right": high,
                            "LISA_rho_left": a["LISA_rho_sw"],
                            "LISA_rho_right": b["LISA_rho_sw"],
                        }
                    )
    return requests


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


def boundary_columns(output, campaign, case_index):
    """Lê as colunas finitas preservando posições onde falta a fronteira."""
    columns = []
    for mi, mass in reversed(list(enumerate(campaign.masses()))):
        if not math.isfinite(mass):
            continue
        path = column_path(output, case_index, mi)
        columns.append(read_json(path) if path.exists() else {})
    return columns


def boundary_segments(columns, supported):
    """Agrupa colunas vizinhas para que a interpolação respeite as lacunas."""
    group = []
    for row in columns:
        if supported(row):
            group.append(row)
        elif group:
            yield group
            group = []
    if group:
        yield group


def connected_witnesses(row):
    """Aceita uma faixa entre testemunhas sem exclusão calculada no interior."""
    lo, hi = (number(row.get(key)) for key in ("C_low_inside", "C_high_inside"))
    return (
        lo is not None
        and hi is not None
        and lo < hi
        and row.get("state") != "reentry_detected"
        and not any(s["decision"] == "no" and lo < s["C"] < hi for s in row.get("samples", ()))
    )


def edge_is_resolved(row, side):
    """Consulta cada bracket separadamente, inclusive quando o outro é inconclusivo."""
    edge = row.get(side)
    if edge is None:
        return row.get("state") == "bounded"
    return edge["state"] == "bracketed" or (
        side == "lower" and edge["state"] == "range_limit" and edge.get("inside_C") == 0
    )


def draw_boundary_guides(ax, columns, x_coordinate, color, *, fill_alpha=0.15, show_edges=True):
    """Desenha a faixa entre testemunhas e distingue a qualidade de cada borda."""

    def interpolated(columns, fields):
        ordered = sorted(columns, key=x_coordinate)
        x = np.array([x_coordinate(col) for col in ordered])
        xx = np.linspace(x[0], x[-1], max(30, len(x) * 3))
        yy = [PchipInterpolator(x, [col.get(field) for col in ordered])(xx) for field in fields]
        return xx, yy

    for group in boundary_segments(columns, connected_witnesses):
        if len(group) > 1:
            x, (lo, hi) = interpolated(group, ("C_low_inside", "C_high_inside"))
            ax.fill_between(x, lo, hi, where=lo <= hi, color=color, alpha=fill_alpha, linewidth=0)
    if not show_edges:
        return
    for side, key in (("lower", "C_low_inside"), ("upper", "C_high_inside")):
        for group in boundary_segments(
            columns,
            lambda r, field=key, edge=side: (
                number(r.get(field)) is not None and edge_is_resolved(r, edge)
            ),
        ):
            if len(group) > 1:
                x, (y,) = interpolated(group, (key,))
                ax.plot(x, y, color=color, lw=1.4)
        # Une testemunhas apenas quando pelo menos uma das duas bordas
        # vizinhas ainda está aberta. Os brackets resolvidos ficam sólidos.
        for group in boundary_segments(columns, connected_witnesses):
            if len(group) < 2:
                continue
            ordered = sorted(group, key=x_coordinate)
            x, (y,) = interpolated(ordered, (key,))
            positions = np.array([x_coordinate(col) for col in ordered])
            incomplete = np.array(
                [
                    not (edge_is_resolved(a, side) and edge_is_resolved(b, side))
                    for a, b in pairwise(ordered)
                ]
            )
            interval = (np.searchsorted(positions, x, side="right") - 1).clip(
                0, len(incomplete) - 1
            )
            ax.plot(x, np.where(incomplete[interval], y, np.nan), color=color, lw=1.3, ls="--")


def nearest_peak_match(pools, tolerance, *, prefer_different_masses=False):
    """Compara pares de picos calculados dentro da tolerância declarada."""
    if not all(pools):
        return None
    peaks = [np.log10([[r["f_sw_peak_Hz"], r["omega_sw_peak_h2"]] for r in pool]) for pool in pools]
    trees = [cKDTree(coordinates) for coordinates in peaks[1:]]
    best = None
    for anchor, coordinate in enumerate(peaks[0]):
        neighbours = [
            tree.query_ball_point(coordinate, tolerance, return_sorted=True) for tree in trees
        ]
        for indices in product((anchor,), *neighbours):
            selected = [pool[i] for pool, i in zip(pools, indices, strict=True)]
            masses = {r["m6_GeV"] for r in selected if math.isfinite(r["m6_GeV"])}
            couplings = [r["C"] for r in selected]
            preference = (
                prefer_different_masses and len(masses) == 1,
                max(masses, default=math.inf),
                min(couplings) - max(couplings) if prefer_different_masses else 0.0,
            )
            if best and preference > best[0][:-1]:
                continue
            coordinates = [points[i] for points, i in zip(peaks, indices, strict=True)]
            distance = max(math.dist(a, b) for a, b in combinations(coordinates, 2))
            if distance > tolerance:
                continue
            score = (*preference, distance)
            if best is None or score < best[0]:
                best = score, selected, distance
    return (best[1], best[2]) if best else None


def match_benchmarks(rows, cases, *, threshold, maximum_distance=0.2):
    """Prefere o controle puro em A; busca massas finitas e variadas em B."""
    valid = [
        r
        for r in rows
        if r.get("observables_complete")
        and valid_value(r, "alpha_trace")
        and valid_value(r, "beta_over_H")
        and (number(r.get("LISA_rho_sw")) or 0) >= threshold
        and (number(r.get("f_sw_peak_Hz")) or 0) > 0
        and (number(r.get("omega_sw_peak_h2")) or 0) > 0
    ]
    pools = [[r for r in valid if r["m8_GeV"] == case] for case in cases]
    finite = [[r for r in pool if math.isfinite(r["m6_GeV"])] for pool in pools]
    match_a = nearest_peak_match(
        [[r for r in pools[0] if math.isinf(r["m6_GeV"])]]
        + [[r for r in pool if r["m6_GeV"] <= 1300] for pool in finite[1:]],
        maximum_distance,
    )
    selection_a = "Measure-only reference"
    if match_a is None:
        masses = set.intersection(*({r["m6_GeV"] for r in pool} for pool in finite))
        for mass in sorted(masses):
            match_a = nearest_peak_match(
                [[r for r in pool if r["m6_GeV"] == mass] for pool in finite],
                maximum_distance,
            )
            if match_a:
                selection_a = f"Fixed m6 = {mass:g} GeV"
                break
    used = {r["point_id"] for r in match_a[0]} if match_a else set()
    remaining = [[r for r in pool if r["point_id"] not in used] for pool in finite]
    match_b = nearest_peak_match(
        [[r for r in pool if r["m6_GeV"] <= 1000] for pool in remaining],
        maximum_distance,
        prefer_different_masses=True,
    )
    if match_b is None:
        match_b = nearest_peak_match(remaining, maximum_distance, prefer_different_masses=True)
    groups = []
    for name, selection, match in (("A", selection_a, match_a), ("B", "Finite m6", match_b)):
        if match:
            selected, distance = match
            groups.append(
                {
                    "group": name,
                    "selection": selection,
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
                "font.family": "serif",
                "mathtext.fontset": "stix",
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
    """Figura 1: compensação de mecanismos com guias de cada borda."""
    from matplotlib.lines import Line2D

    figure, ax = writer.plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for case, color in zip(cases, COLORS, strict=False):
        ci = campaign.m8_values.index(case)
        draw_boundary_guides(
            ax,
            boundary_columns(output, campaign, ci),
            lambda row: (1000 / float(row["m6_GeV"])) ** 2,
            color,
        )
        pure_path = column_path(output, ci, 0)
        if pure_path.exists():
            pure = read_json(pure_path)
            if connected_witnesses(pure):
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
        ]
        + [Line2D([], [], color="0.35", ls="--", label="Inside witnesses")],
        frameon=False,
    )
    ax.grid(alpha=0.12)
    writer.save(
        figure,
        name,
        "Selected first-order-transition windows at Lambda=1000 GeV. "
        "Solid curves use resolved boundaries and dashed curves connect inside witnesses "
        "where a boundary is inconclusive. Brackets are tabulated separately. Selection requires "
        "S3(Tn)/Tn=140 within the declared tolerance, v(Tn)/Tn>1 and V(v,0)<V(0,0). "
        "Shading joins inside witnesses under a connected-interval assumption. "
        "Columns with a detected re-entry or an excluded interior sample interrupt the shading. "
        "Lines at zero inverse mass "
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
    plotted_ids = {
        r["point_id"]
        for case in cases
        for collection in case_samples(planned, records, case, include_validation=True)
        for r in collection
    }
    for field, _label in FIELDS:
        values = [
            r[field]
            for identifier, r in records.items()
            if identifier in plotted_ids and valid_value(r, field)
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
    lisa = build_surface(planned, records, case, "LISA_rho_sw", include_validation=True)
    for ax, (field, label) in zip(axes.flat, FIELDS, strict=True):
        surface = build_surface(planned, records, case, field, include_validation=True)
        tri = surface.triangulation()
        if tri is not None:
            lo, hi = limits[field]
            image = ax.tricontourf(
                tri, surface.values, levels=np.linspace(lo, hi, 40), cmap="viridis"
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
                ticks = MaxNLocator(nbins=5).tick_values(lo, hi)
                bar.set_ticks([value for value in ticks if lo <= value <= hi])
                bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
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
            contour(ax, surface, [1], colors="white", linewidths=2.4, linestyles="--")
            critical = contour(ax, surface, [1], colors="0.15", linewidths=1.2, linestyles="--")
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
        "and v(Tc)/Tc. The three main atlases share colour scales; alpha and beta are interpolated "
        "logarithmically. The outlined contour marks the LISA acoustic threshold. "
        "The critical-strength contour is drawn only where v(Tc)/Tc=1 is supported by the calculated field. "
        "Each coloured triangle is supported by three calculated neighbours. Validation data and "
        "requests for additional points accompany the figures.",
    )


def draw_detectors(writer, output, planned, records, campaign, cases, threshold):
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
        columns = boundary_columns(output, campaign, campaign.m8_values.index(case))
        reach = {
            detector: build_surface(
                planned, records, case, f"{detector}_rho_sw", include_validation=True
            )
            for detector in ("BBO", "DECIGO", "LISA")
        }
        for j, field in enumerate(("alpha_trace", "beta_over_H")):
            ax = axes[i, j]
            draw_boundary_guides(
                ax,
                columns,
                lambda row: float(row["m6_GeV"]),
                "0.55",
                fill_alpha=0.08,
                show_edges=False,
            )
            has_data = any(len(surface.triangles) for surface in reach.values())
            for detector, surface in reach.items():
                tri = surface.triangulation()
                if tri is None:
                    continue
                upper = float(max(surface.values[np.unique(surface.triangles)]))
                if upper > math.log10(threshold):
                    ax.tricontourf(
                        tri,
                        surface.values,
                        levels=[math.log10(threshold), upper + 1],
                        colors=[DETECTOR_COLORS[detector]],
                        alpha=0.7,
                    )
            physical = build_surface(planned, records, case, field, include_validation=True)
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
        "Grey shading connects the inside witnesses of the transition window. "
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
    styles = ("-", "--", "-.", ":")
    benchmarks = [row for group in groups for row in group["benchmarks"]]
    if benchmarks:
        frequencies = [row["f_sw_peak_Hz"] for row in benchmarks]
        amplitudes = [row["omega_sw_peak_h2"] for row in benchmarks]
        display_frequency = (
            max(frequency[0], min(frequencies) / 25),
            min(frequency[-1], max(frequencies) * 25),
        )
        lisa = detectors.lisa_plis_h2(frequency, years=years, threshold=threshold)
        display_amplitude = (min(amplitudes) / 3e4, max(amplitudes) * 300)
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
            lisa,
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
            top.loglog(frequency, spectrum, color=COLORS[ci], ls=styles[ci], lw=1.6)
            bottom.scatter(
                row["f_sw_peak_Hz"], row["omega_sw_peak_h2"], s=30, color=COLORS[ci], zorder=4
            )
            samples.extend(
                {"benchmark": row["benchmark"], "f_Hz": float(f), "omega_sw_h2": float(omega)}
                for f, omega in zip(frequency, spectrum, strict=True)
            )
        top.set(
            title=f"{group['group']}  {group['selection']}",
            xlabel=r"$f$ [Hz]",
            ylabel=r"$h^2\Omega_{\rm sw}(f)$",
            ylim=display_amplitude,
            xlim=display_frequency,
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
            Line2D([], [], color=COLORS[i], ls=styles[i], label=rf"$m_8={mass_label(case)}$ GeV")
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
        "with combined mechanisms when a match exists, with a common finite m6 as the fallback. "
        "Group B prioritises distinct finite masses at or below 1000 GeV and a wider spread in C. "
        "Every pair within a set satisfies a Euclidean distance of at most 0.2 dex in "
        "(log10 f_peak, log10 peak amplitude). "
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
    lisa_requests = lisa_refinement_requests(
        planned, records, cases, threshold=args.snr, tolerance=campaign.C_tolerance / 2
    )
    report["lisa_refinement_points"] = len(lisa_requests)
    report["lisa_bracket_target_C"] = campaign.C_tolerance / 2
    destination.mkdir(parents=True, exist_ok=True)
    old_products_path = destination / "figure_products.json"
    old_products = read_json(old_products_path) if old_products_path.exists() else []
    atomic_json(destination / "coverage_audit.json", report)
    write_csv(
        destination / "point_requests.csv", report["missing_points"] + report["failed_points"]
    )
    write_csv(destination / "lisa_point_requests.csv", lisa_requests)
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
            dict(
                row,
                group=g["group"],
                selection=g["selection"],
                matching_distance_dex=g["maximum_log_peak_distance"],
            )
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
    limits = shared_limits(planned, records, cases)
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
    draw_detectors(writer, output, planned, records, campaign, cases, args.snr)
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
                shared_limits(planned, records, (case,)),
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
