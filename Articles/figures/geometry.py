"""Geometria dos plots: lacunas e falhas continuam sendo lacunas e falhas."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from Articles.figures.data import finite


def adjacent_runs(items, *, mass_key="m6_GeV", maximum_gap=5.01):
    """Não liga uma coluna resolvida à próxima atravessando colunas ausentes."""
    runs = []
    for item in sorted(items, key=lambda row: float(row[mass_key])):
        mass = float(item[mass_key])
        if not math.isfinite(mass):
            continue
        if not runs or mass - float(runs[-1][-1][mass_key]) > maximum_gap:
            runs.append([])
        runs[-1].append(item)
    return runs


def bracket_columns(columns):
    result = []
    for column in columns:
        lower, upper = column.get("lower") or {}, column.get("upper") or {}
        result.append({
            "m6_GeV": float(column["m6_GeV"]), "state": column["state"],
            "lower_inside": lower.get("inside_C"), "lower_outside": lower.get("outside_C"),
            "upper_inside": upper.get("inside_C"), "upper_outside": upper.get("outside_C"),
            "lower_state": lower.get("state"), "upper_state": upper.get("state"),
            "lower_kind": lower.get("outside_kind"), "upper_kind": upper.get("outside_kind"),
            "lower_reason": lower.get("reason"), "upper_reason": upper.get("reason"),
        })
    return result


def supported_triangulation(rows, field, *, max_mass_gap=5.01, max_C_gap=0.081):
    """Triangula também os pontos inválidos, mascarando seus triângulos.

    Remover antes os inválidos fecharia seus buracos por interpolação. O teste
    de comprimentos impede preencher o interior que a busca adaptativa saltou.
    """
    import matplotlib.tri as mtri

    points = [row for row in rows if math.isfinite(float(row["m6_GeV"]))]
    if len(points) < 3:
        return None, None
    x = np.array([float(row["m6_GeV"]) for row in points])
    y = np.array([float(row["C"]) for row in points])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    coordinates = np.column_stack((x / max_mass_gap, y / max_C_gap))
    if np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0), tol=1e-10) < 2:
        # Três ou mais benchmarks ao longo de um corte não definem área.
        return None, None
    values = np.array([row.get(field) if finite(row.get(field)) and not row.get("source_conflict") else np.nan for row in points])
    # Delaunay em unidades da resolução, sem depender da razão GeV/C.
    normalized = mtri.Triangulation(x / max_mass_gap, y / max_C_gap)
    triangles = normalized.triangles
    bad = (~np.isfinite(values[triangles])).any(axis=1)
    bad |= np.ptp(x[triangles], axis=1) > max_mass_gap
    bad |= np.ptp(y[triangles], axis=1) > max_C_gap
    triangulation = mtri.Triangulation(x, y, triangles=triangles, mask=bad)
    return triangulation, np.nan_to_num(values)


def detector_intervals(rows, field, threshold, max_C_gap=0.041):
    """Limiar interpolado só entre dois vizinhos com GW resolvido.

    Retorna todos os intervalos, sem supor detecção monótona em C. Um ponto
    desconhecido ou um salto grande quebra o intervalo; não vira não detecção.
    """
    intervals = []
    for left, right in pairwise(sorted(rows, key=lambda row: row["C"])):
        if right["C"] - left["C"] > max_C_gap:
            continue
        if not all(row.get("gw_resolved") and finite(row.get(field)) for row in (left, right)):
            continue
        a, b = left[field] - threshold, right[field] - threshold
        if max(a, b) < 0:
            continue
        low, high = left["C"], right["C"]
        if a * b < 0:
            crossing = low + (high - low) * (-a) / (b - a)
            low, high = (crossing, high) if a < 0 else (low, crossing)
        intervals.append((low, high))
    return intervals
