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
    # O piloto calcula exatamente duas linhas de C, sem fingir que isso
    # satisfaz o critério de três LINHAS vazias na direção de C.
    row_limit: int | None = None

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
        if self.row_limit is not None and self.row_limit < 1:
            raise ValueError("row-limit deve ser positivo.")
        a, b, h, seed = (Decimal(str(x)) for x in (lo, hi, step, self.seed_C))
        # O passo é ancorado na semente: 3.83, 3.81, ...; inclui 0 e 10,
        # mesmo quando o último intervalo é menor que h.
        below = [seed - i * h for i in range(1, int((seed - a) / h) + 1)]
        above = [seed + i * h for i in range(1, int((b - seed) / h) + 1)]
        cs = tuple(sorted({float(x) for x in [a, b, seed, *below, *above]}))
        return masses, cs

    def identity(self):
        return {"strategy": "adaptive_v1", **asdict(self), "Lambda_GeV": 1000.0}


def nucleation_decision(row, transitions):
    """Beta/GW não resolvidos não apagam uma nucleação já aceita."""
    if any(t.get("trantype") == 1 and t.get("nucleation_accepted") for t in transitions):
        return "yes"
    if row["status"] in ("no_first_order_found", "no_nucleation_found"):
        return "no"
    return "unknown"


def trace_scan(plan, evaluate, report=None):
    """Rastreia cada fatia; evaluate recebe ModelParameters e retorna decisão.

    Procura primeiro uma semente VERIFICADA no m6 declarado, descendo C.
    A regra das três linhas vazias só vale depois de encontrar essa semente.
    Ilhas desconectadas além de três falhas não são cobertas pela hipótese
    de continuação. Os trechos pulados são hipóteses, nunca resultados.
    """
    masses, cs = plan.axes()
    report = report if report is not None else {}
    report.update(state="running", slices=[], assumption="connected_band_at_declared_resolution")
    initial_i = masses.index(plan.seed_m6_GeV)
    initial_j = cs.index(plan.seed_C)

    def point(j, i, m8):
        return ModelParameters(m6_GeV=masses[i], m8_GeV=m8, C=cs[j])

    def ask(j, i, m8):
        decision = evaluate(point(j, i, m8))
        if decision not in ("yes", "no"):
            raise UnresolvedPoint(
                f"Ponto inconclusivo C={cs[j]:g}, m6={masses[i]:g}, m8={m8:g}. "
                "Nenhuma região foi podada a partir dele. Consulte details e "
                "retome com --retry-failed; para mudar código/configuração, use outra pasta."
            )
        return decision

    try:
        for m8 in dict.fromkeys(plan.m8_GeV):
            section = {"m8_GeV": m8, "rows": [], "C_stops": [], "bootstrap": []}
            report["slices"].append(section)
            # Uma semente que nucleia no modelo puro pode deixar de nuclear
            # ao ligar m8. Não encerramos a busca só por essa suposição falhar.
            seed_j = None
            for j in range(initial_j, -1, -1):
                d = ask(j, initial_i, m8)
                section["bootstrap"].append({"C": cs[j], "decision": d})
                if d == "yes":
                    seed_j = j
                    break
            if seed_j is None:
                raise UnresolvedPoint(
                    f"Nenhuma semente verificada em m6={plan.seed_m6_GeV:g}, m8={m8:g}. "
                    "A busca limitada terminou; escolha --seed-m6/--seed-C em outra campanha."
                )
            section["verified_seed_C"] = cs[seed_j]

            def scan_row(j, center, section=section, m8=m8):
                row = {"C": cs[j], "seed_m6_GeV": masses[center], "hits_m6_GeV": [], "stops": []}
                section["rows"].append(row)
                at_center = ask(j, center, m8)
                hits = [center] if at_center == "yes" else []
                for direction, name in ((-1, "larger_m6"), (1, "smaller_m6")):
                    trail = [center] if at_center == "no" else []
                    i = center + direction
                    while 0 <= i < len(masses) and len(trail) < plan.patience:
                        d = ask(j, i, m8)
                        if d == "yes":
                            hits.append(i)
                            trail = []
                        else:
                            trail.append(i)
                        i += direction
                    row["stops"].append({
                        "direction": name,
                        "reason": "consecutive_no_nucleation" if len(trail) >= plan.patience else "bound",
                        "last_no_m6_GeV": [masses[k] for k in trail],
                        "first_skipped_m6_GeV": masses[i] if 0 <= i < len(masses) else None,
                    })
                row["hits_m6_GeV"] = [masses[k] for k in sorted(set(hits))]
                row["state"] = "complete"
                # A mediana dos hits fica dentro da faixa e acompanha seu
                # deslocamento; uma semente presa à borda pode perder a faixa
                # inteira no próximo passo de C.
                unique_hits = sorted(set(hits))
                return unique_hits[len(unique_hits) // 2] if hits else None

            center = scan_row(seed_j, initial_i)
            assert center is not None  # nucleação do bootstrap foi verificada
            seed_center = center
            # No piloto: semente + próxima linha abaixo, duas linhas exatas.
            directions = ((-1, "decreasing_C"), (1, "increasing_C")) if plan.row_limit else (
                (1, "increasing_C"), (-1, "decreasing_C")
            )
            for direction, name in directions:
                center = seed_center
                empty_rows = []
                j = seed_j + direction
                while 0 <= j < len(cs):
                    if plan.row_limit and len(section["rows"]) >= plan.row_limit:
                        section["C_stops"].append({"direction": name, "reason": "row_limit"})
                        break
                    hit = scan_row(j, center)
                    if hit is None:
                        empty_rows.append(cs[j])
                    else:
                        center = hit
                        empty_rows = []
                    j += direction
                    if len(empty_rows) >= plan.patience:
                        break
                else:
                    section["C_stops"].append({"direction": name, "reason": "bound"})
                    continue
                if len(empty_rows) >= plan.patience:
                    section["C_stops"].append({
                        "direction": name, "reason": "consecutive_empty_rows",
                        "last_no_C": empty_rows,
                        "first_skipped_C": cs[j] if 0 <= j < len(cs) else None,
                    })
        report["state"] = "pilot_complete" if plan.row_limit else "complete"
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
