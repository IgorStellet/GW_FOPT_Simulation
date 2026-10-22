"""Dados e geometria das bordas: observação, bracket e lacuna são distintos.

Os CSVs exportados permitem revisar figuras sem transferir SQLite/detalhes.
Os planos físicos exigem o banco/manifesto original, disponível no CHE.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from Articles.figures.data import (
    Dataset,
    coordinate,
    detector_metrics,
    finite,
    load_dataset,
)


def read_table(path):
    """Restaura tipos dos CSVs da análise v1, sem converter ausência em zero."""

    def value(key, text):
        if text == "":
            return None
        if key.endswith("_id"):
            return text
        if text in ("True", "False"):
            return text == "True"
        if text.startswith(("{", "[")):
            return json.loads(text)
        try:
            return float(text)
        except ValueError:
            return text

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [
            {key: value(key, text) for key, text in row.items()}
            for row in csv.DictReader(stream)
        ]


def load_exports(folder, years=4.0):
    """Lê a exportação v1; calcula apenas métricas algébricas dos detectores."""
    folder = Path(folder).resolve()
    report = json.loads((folder / "figure_report.json").read_text(encoding="utf-8"))
    rows = read_table(folder / "points_analysis.csv")
    if len({coordinate(row) for row in rows}) != len(rows):
        raise ValueError(
            "CSV com coordenadas duplicadas; use a união auditada das campanhas originais."
        )
    columns = defaultdict(list)
    for item in read_table(folder / "boundary_brackets.csv"):
        column = {"m6_GeV": item["m6_GeV"], "state": item["state"]}
        for side in ("lower", "upper"):
            column[side] = {
                "inside_C": item.get(side + "_inside"),
                "outside_C": item.get(side + "_outside"),
                "state": item.get(side + "_state"),
                "outside_kind": item.get(side + "_kind"),
                "reason": item.get(side + "_reason"),
            }
        columns[float(item["m8_GeV"])].append(column)
    for row in rows:
        if row.get("gw_resolved") and not row.get("source_conflict"):
            row.update(detector_metrics(row, row.get("spectrum_defaults") or {}, years))
    names = (
        "figure_report.json",
        "points_analysis.csv",
        "boundary_brackets.csv",
        "all_saved_transitions.csv",
    )
    audit = {
        **report,
        "input_kind": "exported_v1_tables",
        "source_export_folder": str(folder),
        "input_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in names
        },
    }
    return Dataset(
        rows, read_table(folder / "all_saved_transitions.csv"), dict(columns), [], audit
    )


def load_sources(boundaries, observables=(), *, years=4.0, details=True):
    """Prioriza observáveis adicionais em empates, usando o leitor v1 intacto."""
    sources = [*reversed(observables), boundaries]
    return load_dataset(sources[0], sources[1:], years=years, read_details=details)


def grouped_rows(dataset):
    groups = defaultdict(list)
    for row in dataset.rows:
        groups[coordinate(row)[:2]].append(row)
    return {key: sorted(rows, key=lambda row: row["C"]) for key, rows in groups.items()}


def edge_samples(dataset):
    """Extremos EM C dos pontos selecionados em cada massa, não de alpha/beta.

    Uma coluna inconclusiva pode ter testemunhas positivas válidas. Elas são
    mantidas, mas o bracket só é certificado quando coincide com o bracket
    resolvido original e tem uma testemunha negativa efetivamente calculada.
    """
    columns = {
        (case, float(column["m6_GeV"])): column
        for case, records in dataset.columns.items()
        for column in records
    }
    result = []
    for key, rows in sorted(grouped_rows(dataset).items(), key=lambda item: item[0]):
        selected = [
            row
            for row in rows
            if row["boundary_decision"] == "yes" and not row.get("source_conflict")
        ]
        if not selected:
            continue
        column = columns.get(key, {})
        for side, witness in (("lower", selected[0]), ("upper", selected[-1])):
            saved = column.get(side) or {}
            outside = (
                [row for row in rows if row["C"] < witness["C"]]
                if side == "lower"
                else [row for row in rows if row["C"] > witness["C"]]
            )
            neighbour = (
                (outside[-1] if side == "lower" else outside[0]) if outside else None
            )
            saved_outside = next(
                (row for row in rows if row["C"] == saved.get("outside_C")), None
            )
            certified = (
                saved.get("state") == "bracketed"
                and saved.get("inside_C") == witness["C"]
                and saved_outside is not None
                and saved_outside["boundary_decision"] == "no"
            )
            if certified:
                neighbour = saved_outside
            result.append(
                {
                    **witness,
                    "edge": side,
                    "edge_C": witness["C"],
                    "edge_certified": certified,
                    "edge_column_state": column.get("state", "not_recorded"),
                    "edge_original_inside_C": saved.get("inside_C"),
                    "edge_original_reason": saved.get("reason"),
                    "edge_outside_C": neighbour["C"] if neighbour else None,
                    "edge_outside_decision": neighbour["boundary_decision"]
                    if neighbour
                    else None,
                    "edge_outside_point_id": neighbour["point_id"]
                    if neighbour
                    else None,
                    "edge_interpretation": "outermost_selected_sample_in_C_not_global_observable_extremum",
                }
            )
    return result


def alpha_valid(row):
    """Alpha já calculado permanece utilizável se apenas beta falhou."""
    return (
        bool(row.get("nucleation_accepted"))
        and not row.get("source_conflict")
        and finite(row.get("alpha_trace"))
        and row["alpha_trace"] > 0
    )


def beta_valid(row):
    """Não apresenta uma derivada marcada como sensível ao passo."""
    return (
        bool(row.get("nucleation_accepted"))
        and not row.get("source_conflict")
        and finite(row.get("beta_over_H"))
        and row["beta_over_H"] > 0
        and not row.get("quality_flags")
    )


def strength_crossings(dataset, field="vev_over_Tn", maximum_C_gap=0.081):
    """Interpola o nível 1 entre dois valores salvos, sem novos bounces.

    Não é solução analítica do potencial. Não atravessa falhas ou mudanças
    de história térmica conhecidas. Mantém múltiplos cruzamentos por massa.
    """
    crossings = []
    for (case, mass), rows in grouped_rows(dataset).items():
        seen = set()
        for left, right in pairwise(rows):
            if right["C"] - left["C"] > maximum_C_gap:
                continue
            if not all(
                row.get("nucleation_accepted")
                and not row.get("source_conflict")
                and finite(row.get(field))
                for row in (left, right)
            ):
                continue
            if any(
                left.get(key) != right.get(key)
                for key in ("high_phase", "low_phase", "transition_index")
            ):
                continue
            a, b = left[field] - 1, right[field] - 1
            if a == b or a * b > 0:
                continue
            coupling = left["C"] + (right["C"] - left["C"]) * (-a) / (b - a)
            if round(coupling, 12) in seen:
                continue
            seen.add(round(coupling, 12))
            crossings.append(
                {
                    "m6_GeV": mass,
                    "m8_GeV": case,
                    "C": coupling,
                    "C_left": left["C"],
                    "C_right": right["C"],
                    "field": field,
                    "left_point_id": left["point_id"],
                    "right_point_id": right["point_id"],
                    "interpretation": "numerical_level_1_interpolated_between_saved_values",
                }
            )
    return crossings


def merge_intervals(intervals, tolerance=1e-10):
    result = []
    for low, high in sorted(intervals):
        if result and low <= result[-1][1] + tolerance:
            result[-1][1] = max(high, result[-1][1])
        else:
            result.append([low, high])
    return result


def valid_runs(rows, valid, maximum_gap):
    """Mantém amostras inválidas na sequência para que quebrem a curva."""
    runs, current = [], []
    for row in sorted(rows, key=lambda item: item["C"]):
        if not valid(row) or current and row["C"] - current[-1]["C"] > maximum_gap:
            if current:
                runs.append(current)
            current = []
        if valid(row):
            current.append(row)
    if current:
        runs.append(current)
    return runs
