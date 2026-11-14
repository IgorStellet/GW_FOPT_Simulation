"""Segunda etapa: N pontos internos entre as bordas, mais suas duas testemunhas.

As coordenadas normalizadas u=(C-C_low)/(C_high-C_low) permitem amostrar a
faixa inteira em cada massa. A interpolação posterior continua limitada às
células com suporte calculado; o plano inclui pontos de validação adicionais.
"""

from __future__ import annotations

import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import pairwise
from pathlib import Path

from .boundaries import column_path, coordinate
from .execution import Paused
from .model import Parameters
from .storage import atomic_json, point_id, read_json


def sampling_masses(campaign, refine=False):
    """Refinamento insere colunas intermediárias já delimitadas, preservando a grade anterior."""
    original = campaign.observable_masses()
    if not refine:
        return original
    all_masses = campaign.masses()
    finite = [m for m in original if math.isfinite(m)]
    wanted = set(original)
    for a, b in pairwise(finite):
        left, right = all_masses.index(a), all_masses.index(b)
        if right - left > 1:
            wanted.add(all_masses[(left + right) // 2])
    return tuple(m for m in all_masses if m in wanted)


def plan(output, campaign, *, allow_partial=False, refine=False):
    """Lista coordenadas determinísticas; não avalia potencial nem altera resultados.

    Os pontos validation ficam fora da grade usada nos mapas, permitindo medir
    o erro da interpolação em C. refine dobra a resolução interna, reutilizando
    todos os pontos já salvos. Colunas com reentradas pedem tratamento próprio.
    """
    all_paths = [
        column_path(output, ci, mi)
        for ci in range(len(campaign.m8_values))
        for mi in range(len(campaign.masses()))
    ]
    missing = [p for p in all_paths if not p.exists()]
    if missing and not allow_partial:
        raise ValueError(
            f"Fronteiras ainda incompletas: faltam {len(missing)} colunas. "
            "Retome boundaries; --allow-partial é reservado ao piloto."
        )
    training_masses = sampling_masses(campaign, refine)
    wanted = set(training_masses)
    full_masses = campaign.masses()
    validation_masses = set()
    finite = [m for m in training_masses if math.isfinite(m)]
    for index, (a, b) in enumerate(pairwise(finite)):
        left, right = full_masses.index(a), full_masses.index(b)
        if index % 5 == 0 and right - left > 1:
            validation_masses.add(full_masses[(left + right) // 2])
    records, skipped, seen = [], [], set()
    divisions = (campaign.interior_points + 1) * (2 if refine else 1)
    for ci, case in enumerate(campaign.m8_values):
        ordinal = 0
        for mi, mass in enumerate(campaign.masses()):
            if mass not in wanted and mass not in validation_masses:
                continue
            path = column_path(output, ci, mi)
            if not path.exists():
                continue
            column = read_json(path)
            lo, hi = column["C_low_inside"], column["C_high_inside"]
            if lo is None or hi is None or column["state"] == "reentry_detected":
                skipped.append({"case_index": ci, "m6_GeV": mass, "state": column["state"]})
                continue
            fractions = (
                [(j / divisions, "grid") for j in range(divisions + 1)]
                if mass in wanted
                else [(0.25, "validation"), (0.5, "validation"), (0.75, "validation")]
            )
            if mass in wanted and ordinal % 5 == 0 and hi > lo:
                fractions += [
                    (0.5 / divisions, "validation"),
                    ((divisions - 0.5) / divisions, "validation"),
                ]
            for fraction, kind in fractions:
                C = coordinate(lo + fraction * (hi - lo))
                parameters = Parameters(mass, case, C)
                identifier = point_id(parameters)
                if identifier in seen:
                    continue
                seen.add(identifier)
                records.append(
                    {
                        "point_id": identifier,
                        "case_index": ci,
                        "mass_index": mi,
                        "m6_GeV": mass,
                        "m8_GeV": case,
                        "C": C,
                        "fraction": fraction,
                        "kind": kind,
                        "boundary_state": column["state"],
                        "C_low_inside": lo,
                        "C_high_inside": hi,
                    }
                )
            ordinal += mass in wanted
    return {
        "stage": "observables",
        "refined": refine,
        "divisions": divisions,
        "observable_masses": training_masses,
        "missing_boundary_columns": len(missing),
        "skipped_columns": skipped,
        "points": records,
    }


def run(output, planned, evaluator, *, shard_index=0, shard_count=1, workers=1):
    """Executa subprocessos limitados em paralelo; somente os coordenadores usam threads."""
    if shard_count < 1 or not 0 <= shard_index < shard_count or workers < 1:
        raise ValueError("Partes/workers positivos e índice válido.")
    requests = iter(
        row for i, row in enumerate(planned["points"]) if i % shard_count == shard_index
    )
    atomic_json(
        Path(output) / "plans" / f"observables-{'refined' if planned['refined'] else 'base'}.json",
        planned,
    )
    paused = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                row = next(requests, None)
                if row is None:
                    exhausted = True
                    break
                pending.add(
                    executor.submit(evaluator, Parameters(row["m6_GeV"], row["m8_GeV"], row["C"]))
                )
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Paused:
                    exhausted, paused = True, True
    if paused:
        raise Paused("Observáveis pausados; pontos terminados foram publicados.")
