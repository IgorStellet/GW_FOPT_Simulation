"""Busca por continuação; nenhuma rotina física é reimplementada aqui.

O avaliador retorna 'yes', 'no' ou 'unknown'. Só 'no' conta para encerrar
uma direção. A sequência determinística pode ser reconstruída do SQLite
após qualquer interrupção, sem recalcular pontos confirmados.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from uuid import uuid4

from Articles.combined_model import ModelParameters


class PointBudgetReached(Exception):
    """Pausa normal de um lote; não é uma classificação física."""


class UnresolvedPoint(RuntimeError):
    """Não se pode decidir a próxima poda a partir deste ponto."""


@dataclass(frozen=True)
class AdaptivePlan:
    m6_range_GeV: tuple = (500.0, 2000.0, 5.0)
    C_range: tuple = (0.0, 10.0, 0.02)
    m8_GeV: tuple = (math.inf, 840.8964152537145, 668.740304976422)
    seed_C: float = 3.83
    seed_m6_GeV: float = math.inf
    patience: int = 3
    on_unresolved: str = "continue"
    C_anchors: tuple = (1.0, 3.2)
    # Limites somente do piloto, aplicados DEPOIS de verificar a semente.
    column_limit: int | None = None
    C_steps_limit: int | None = None

    def axes(self):
        from Articles.collect_data import inclusive_axis

        lo, hi, step = self.C_range
        if not (0 <= lo <= self.seed_C <= hi <= 10):
            raise ValueError("Exige 0<=C_min<=seed-C<=C_max<=10.")
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Passo de C deve ser positivo e finito.")
        masses = inclusive_axis(*self.m6_range_GeV)
        if masses[0] <= 0 or self.patience < 1:
            raise ValueError("Exige m6>0 e patience>=1.")
        masses = (math.inf, *reversed(masses))
        if self.seed_m6_GeV not in masses:
            raise ValueError("seed-m6 deve ser inf ou pertencer à grade de m6.")
        if not self.m8_GeV or any(math.isnan(m) or m <= 0 for m in self.m8_GeV):
            raise ValueError("Cenários m8 devem ser positivos ou inf.")
        for name in ("column_limit", "C_steps_limit"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} deve ser positivo.")
        if self.on_unresolved not in ("continue", "stop"):
            raise ValueError("on_unresolved deve ser continue ou stop.")
        if any(not math.isfinite(c) or not 0 <= c <= 10 for c in self.C_anchors):
            raise ValueError("Âncoras de C devem ser finitas e estar em [0,10].")
        a, b, h, seed = (Decimal(str(x)) for x in (lo, hi, step, self.seed_C))
        # O passo é ancorado na semente: 3.83, 3.81, ...; inclui 0 e 10,
        # mesmo quando o último intervalo é menor que h.
        below = [seed - i * h for i in range(1, int((seed - a) / h) + 1)]
        above = [seed + i * h for i in range(1, int((b - seed) / h) + 1)]
        cs = tuple(sorted({float(x) for x in [a, b, seed, *below, *above]}
                          | {float(c) for c in self.C_anchors if lo <= c <= hi}))
        return masses, cs

    def identity(self):
        return {"strategy": "adaptive_v3_audited_unknowns", **asdict(self), "Lambda_GeV": 1000.0}


def nucleation_decision(row, transitions):
    """Beta/GW não resolvidos não apagam uma nucleação já aceita."""
    if any(t.get("trantype") == 1 and t.get("nucleation_accepted") for t in transitions):
        return "yes"
    if row["status"] in ("no_first_order_found", "no_nucleation_found"):
        return "no"
    return "unknown"


def trace_scan(plan, evaluate, report=None):
    """Uma faixa de C por m6: primeiro verifica semente, depois aplica poda.

    Toda nova massa repete a busca da primeira nucleação, sem patience.
    Descemos C a partir da faixa anterior; se necessário, procuramos também
    acima até C_max, evitando supor monotonicidade entre massas. Uma coluna
    só conta como vazia depois de esgotar seu eixo C inteiro sem nucleação.
    """
    masses, cs = plan.axes()
    report = report if report is not None else {}
    report.update(state="running", slices=[], unresolved_points=[],
                  assumption="connected_C_band_after_verified_seed")
    initial_i = masses.index(plan.seed_m6_GeV)
    initial_j = cs.index(plan.seed_C)
    decisions = {}

    def ask(j, i, m8):
        key = (j, i, m8)
        if key not in decisions:
            params = ModelParameters(m6_GeV=masses[i], m8_GeV=m8, C=cs[j])
            decision = evaluate(params)
            if decision not in ("yes", "no", "unknown"):
                raise ValueError(f"Decisão inválida do avaliador: {decision!r}")
            if decision == "unknown" and plan.on_unresolved == "stop":
                raise UnresolvedPoint(
                    f"Ponto inconclusivo C={cs[j]:g}, m6={masses[i]:g}, m8={m8:g}. "
                    "Nenhuma região foi podada a partir dele. Consulte details e "
                    "retome com --retry-failed; para mudar código/configuração, use outra pasta."
                )
            if decision == "unknown":
                report["unresolved_points"].append({"C": cs[j], "m6_GeV": masses[i], "m8_GeV": m8})
            decisions[key] = decision
        return decisions[key]

    def scan_column(i, start_j, section):
        m8 = section["m8_GeV"]
        column = {"m6_GeV": masses[i], "initial_C": cs[start_j], "bootstrap": [],
                  "hits_C": [], "C_stops": [], "unknown_C": [], "state": "searching_seed"}
        section["columns"].append(column)
        seed_j = None
        searches = ((range(start_j, -1, -1), "decreasing_C"),
                    (range(start_j + 1, len(cs)), "increasing_C_fallback"))
        for candidates, name in searches:
            for j in candidates:
                d = ask(j, i, m8)
                column["bootstrap"].append({"C": cs[j], "decision": d, "direction": name})
                if d == "unknown":
                    column["unknown_C"].append(cs[j])
                if d == "yes":
                    seed_j = j
                    break
            if seed_j is not None:
                break
        if seed_j is None:
            column.update(state="unresolved_no_seed" if column["unknown_C"] else "no_nucleation_on_grid",
                          C_search_complete=True)
            return None
        column["verified_seed_C"] = cs[seed_j]
        column["state"] = "expanding_band"
        hits = {seed_j}
        column["hits_C"] = [cs[seed_j]]
        for direction, name in ((1, "increasing_C"), (-1, "decreasing_C")):
            trail, steps = [], 0
            j = seed_j + direction
            reason = "bound"
            while 0 <= j < len(cs):
                if plan.C_steps_limit is not None and steps >= plan.C_steps_limit:
                    reason = "pilot_step_limit"
                    break
                d = ask(j, i, m8)
                steps += 1
                if d == "yes":
                    hits.add(j)
                    column["hits_C"] = [cs[k] for k in sorted(hits)]
                    trail = []
                elif d == "no":
                    trail.append(cs[j])
                else:
                    if cs[j] not in column["unknown_C"]:
                        column["unknown_C"].append(cs[j])
                    trail = []  # Uma falha quebra a sequência de ausências verificadas.
                j += direction
                if len(trail) >= plan.patience:
                    reason = "consecutive_no_nucleation"
                    break
            next_unknown = j
            # O bootstrap pode já ter calculado os C acima da fronteira.
            # Não os rotulamos como pulados ao encerrar a expansão da faixa.
            while 0 <= next_unknown < len(cs) and (next_unknown, i, m8) in decisions:
                next_unknown += direction
            column["C_stops"].append({
                "direction": name, "reason": reason, "last_no_C": trail,
                "first_after_walk_C": cs[j] if 0 <= j < len(cs) else None,
                "first_skipped_C": cs[next_unknown] if 0 <= next_unknown < len(cs) else None,
            })
        column["state"] = "pilot_window" if any(
            s["reason"] == "pilot_step_limit" for s in column["C_stops"]
        ) else "complete"
        if column["unknown_C"]:
            column["state"] += "_with_unresolved"
        # Reutilizamos o maior C observado, sem assumir que a próxima massa
        # nucleie ali. A nova coluna sempre executa seu próprio bootstrap.
        return max(hits)

    try:
        for m8 in dict.fromkeys(plan.m8_GeV):
            section = {"m8_GeV": m8, "columns": [], "m6_stops": []}
            report["slices"].append(section)
            seed_j = scan_column(initial_i, initial_j, section)
            for direction, name in ((-1, "larger_m6"), (1, "smaller_m6")):
                start_j = seed_j if seed_j is not None else initial_j
                has_verified_seed = seed_j is not None
                empty_columns = []
                i = initial_i + direction
                reason = "bound"
                while 0 <= i < len(masses):
                    if plan.column_limit and len(section["columns"]) >= plan.column_limit:
                        reason = "column_limit"
                        break
                    hit_j = scan_column(i, start_j, section)
                    if hit_j is not None:
                        start_j, has_verified_seed = hit_j, True
                        empty_columns = []
                    elif has_verified_seed and section["columns"][-1]["state"] == "no_nucleation_on_grid":
                        empty_columns.append(masses[i])
                    else:
                        empty_columns = []
                    i += direction
                    if len(empty_columns) >= plan.patience:
                        reason = "consecutive_empty_columns"
                        break
                section["m6_stops"].append({
                    "direction": name, "reason": reason, "last_no_m6_GeV": empty_columns,
                    "first_skipped_m6_GeV": masses[i] if 0 <= i < len(masses) else None,
                })
            section["has_verified_seed"] = any("verified_seed_C" in c for c in section["columns"])
        report["state"] = "pilot_complete" if plan.column_limit or plan.C_steps_limit else "complete"
        if report["unresolved_points"]:
            report["state"] += "_with_unresolved"
    except PointBudgetReached:
        report["state"] = "paused_point_budget"
    except UnresolvedPoint as error:
        report.update(state="unresolved", message=str(error))
        raise
    except BaseException:
        report["state"] = "interrupted"
        raise
    return report


def export_adaptive(connection, output, plan, report):
    """Matrizes: células não calculadas ficam explicitamente não avaliadas."""
    from Articles.collect_data import canonical_json, json_safe

    temporary = output / "adaptive_state.json.tmp"
    temporary.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output / "adaptive_state.json")
    masses, cs = plan.axes()
    cells = {}
    for (record,) in connection.execute("SELECT record FROM points"):
        row = json.loads(record)
        cells[(str(row["m8_GeV"]), row["C"], str(row["m6_GeV"]))] = row.get("scan_decision", "unknown")
    for index, m8 in enumerate(dict.fromkeys(plan.m8_GeV)):
        temporary = output / f"matrix_m8_{index}.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["C / m6_GeV", *["inf" if math.isinf(m) else f"{m:g}" for m in masses]])
            for C in reversed(cs):
                writer.writerow([C, *[
                    cells.get((str(json_safe(m8)), C, str(json_safe(m))), "not_evaluated")
                    for m in masses
                ]])
        temporary.replace(output / f"matrix_m8_{index}.csv")
    # Mapeamento explícito: índices locais das matrizes não são índices globais.
    (output / "matrix_axes.json").write_text(canonical_json({
        "m8_GeV": list(dict.fromkeys(plan.m8_GeV)), "m6_GeV": masses, "C": cs,
        "cells": {"yes": "accepted_nucleation", "no": "no_nucleation_found", "unknown": "unresolved", "not_evaluated": "not_calculated"},
    }), encoding="utf-8")


def run_adaptive(output, plan, settings, manifest, max_points=None, retry_failed=False, retry_session=None):
    from Articles.collect_data import (
        evaluate_point,
        export_csv,
        open_database,
        point_id,
        save_point,
        single_writer,
    )

    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    report = {}
    retry_session = retry_session or uuid4().hex
    with single_writer(output):
        connection = open_database(output, manifest)
        cache = {key: (json.loads(record), path) for key, record, path in connection.execute(
            "SELECT point_id, record, details_path FROM points"
        )}
        # Cada falha é repetida no máximo uma vez por chamada, nunca em loop.
        retried = set()

        def evaluate(params):
            nonlocal completed
            key = point_id(params)
            old = cache.get(key)
            repair = old is None or not (output / old[1]).exists()
            retry = old and retry_failed and key not in retried and old[0].get("adaptive_retry_session") != retry_session and (
                old[0].get("scan_decision") == "unknown" or old[0]["status"] == "observables_unresolved"
            )
            if repair or retry:
                if max_points is not None and completed >= max_points:
                    raise PointBudgetReached
                result = evaluate_point(params, settings)
                result[0]["scan_decision"] = nucleation_decision(result[0], result[1])
                if retry:
                    # O runner reconstrói a sequência em cada lote. Um
                    # observável persistentemente inconclusivo não pode
                    # consumir o primeiro ponto de TODOS os lotes para sempre.
                    result[0]["adaptive_retry_session"] = retry_session
                save_point(connection, output, result)
                cache[key] = (result[0], result[0]["details_path"])
                retried.add(key)
                completed += 1
                row = result[0]
                print(f"[{completed}] m8={params.m8_GeV:g}, C={params.C:g}, m6={params.m6_GeV:g}: "
                      f"{row['status']} / {row['scan_decision']} ({row['runtime_seconds']:.1f}s)", flush=True)
            return cache[key][0]["scan_decision"]

        try:
            trace_scan(plan, evaluate, report)
        finally:
            try:
                export_csv(connection, output)
                export_adaptive(connection, output, plan, report)
            finally:
                connection.close()
    print(f"Scan: {report['state']}; {completed} pontos novos/reprocessados.", flush=True)
    return completed
