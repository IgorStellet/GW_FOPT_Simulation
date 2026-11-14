"""Primeira etapa: encontrar testemunhas internas/externas das duas bordas.

O algoritmo assume uma faixa conectada em C depois de encontrar uma semente.
Faz continuação em m6 e bisseção, preserva o bracket, testa pontos além da
saída e sinaliza reentradas amostradas. Uma falha nunca é uma testemunha externa.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from .model import Parameters
from .storage import atomic_json, read_json, task_lock, timestamp


def coordinate(value):
    """Arredondamento apenas da coordenada de busca, para reutilizar pontos exatos."""
    return round(float(value), 10)


def search_axis(lo, hi, step):
    """Inclui ambos os extremos mesmo quando o intervalo não é múltiplo de step."""
    return tuple(dict.fromkeys((*[coordinate(x) for x in np.arange(lo, hi, step)], float(hi))))


def column_path(output, case_index, mass_index):
    """Um checkpoint independente por coluna de massa e cenário."""
    return Path(output) / "boundaries" / f"m8-{case_index}" / f"column-{mass_index:04d}.json"


def find_column(campaign, m6, m8, evaluate, hint=None):
    """Localiza uma faixa, retornando limites certificados ou incertezas explícitas.

    evaluate(Parameters) deve retornar um registro com summary.decision.
    seed_resolution limita a resolução da procura de faixas que ainda não têm
    semente; C_tolerance limita a largura final de cada bracket encontrado.
    """
    lo, hi = campaign.C_range
    samples = {}

    def ask(C):
        C = coordinate(min(hi, max(lo, C)))
        if C not in samples:
            result = evaluate(Parameters(m6, m8, C))
            summary = result["summary"]
            samples[C] = {
                "C": C,
                "decision": summary["decision"],
                "reason": summary.get("reason", summary["status"]),
                "status": summary["status"],
            }
        return samples[C]["decision"]

    def edge(seed, direction):
        bound = hi if direction > 0 else lo
        inside, outside, unknowns = seed, None, []
        previous = (hint or {}).get("upper" if direction > 0 else "lower") or {}
        guesses = [
            x
            for x in (previous.get("inside_C"), previous.get("outside_C"))
            if x is not None and direction * (x - seed) > 0 and lo <= x <= hi
        ]
        guesses.sort(reverse=direction < 0)
        step = campaign.walk_step
        while outside is None and inside != bound:
            candidate = (
                guesses.pop(0)
                if guesses
                else coordinate(min(hi, max(lo, inside + direction * step)))
            )
            decision = ask(candidate)
            if decision == "yes":
                inside = candidate
            elif decision == "no":
                outside = candidate
            else:
                unknowns.append(candidate)
                if len(unknowns) >= 3:
                    break
            step *= 1.7
            if candidate == bound:
                break
        if outside is None:
            state = "range_limit" if inside == bound else "unresolved"
            return {
                "state": state,
                "inside_C": inside,
                "outside_C": None,
                "width_C": None,
                "unknown_probes": unknowns,
                "confirmations": [],
            }
        blocked = False
        while abs(outside - inside) > campaign.C_tolerance:
            midpoint = coordinate((inside + outside) / 2)
            decision = ask(midpoint)
            if decision == "yes":
                inside = midpoint
            elif decision == "no":
                outside = midpoint
            else:
                unknowns.append(midpoint)
                # Duas sondas laterais podem contrair o bracket sem decidir
                # artificialmente se o ponto numericamente ruim pertence à faixa.
                changed = False
                for candidate in (
                    coordinate((inside + midpoint) / 2),
                    coordinate((outside + midpoint) / 2),
                ):
                    probe = ask(candidate)
                    if probe == "yes" and direction * (candidate - inside) > 0:
                        inside, changed = candidate, True
                    elif probe == "no" and direction * (outside - candidate) > 0:
                        outside, changed = candidate, True
                if direction * (outside - inside) <= 0:
                    return {
                        "state": "reentry_detected",
                        "inside_C": inside,
                        "outside_C": outside,
                        "width_C": abs(outside - inside),
                        "unknown_probes": unknowns,
                        "confirmations": [],
                    }
                if not changed:
                    blocked = True
                    break
        confirmations = []
        reentry = False
        for n in (1, 2, 3):
            candidate = coordinate(min(hi, max(lo, outside + direction * campaign.walk_step * n)))
            if candidate == outside:
                continue
            decision = ask(candidate)
            confirmations.append({"C": candidate, "decision": decision})
            reentry |= decision == "yes"
        return {
            "state": "reentry_detected" if reentry else ("unresolved" if blocked else "bracketed"),
            "inside_C": inside,
            "outside_C": outside,
            "width_C": abs(outside - inside),
            "unknown_probes": unknowns,
            "confirmations": confirmations,
            "outside_reason": samples[outside]["reason"],
        }

    # As curvas antigas orientam somente uma primeira tentativa. Toda inclusão
    # é novamente calculada com o potencial corrigido; nenhum dado antigo entra.
    guess = min(
        hi,
        max(
            lo,
            3.3
            - (0 if math.isinf(m6) else 0.5 * (1000 / m6) ** 2)
            - (0 if math.isinf(m8) else 0.05 * (1000 / m8) ** 4),
        ),
    )
    candidates = []
    if hint and hint.get("C_low_inside") is not None and hint.get("C_high_inside") is not None:
        candidates.append((hint["C_low_inside"] + hint["C_high_inside"]) / 2)
    candidates.append(guess)
    coarse = search_axis(lo, hi, campaign.seed_step)
    candidates.extend(sorted(coarse, key=lambda C: abs(C - guess)))
    seed = None
    for candidate in candidates:
        if ask(candidate) == "yes":
            seed = coordinate(candidate)
            break
    if seed is None:
        # Busca completa até a resolução declarada, inclusive C=0. Três
        # negativos consecutivos antes de uma semente nunca encerram a coluna.
        for candidate in search_axis(lo, hi, campaign.seed_resolution):
            if ask(candidate) == "yes":
                seed = coordinate(candidate)
                break
    report = {
        "m6_GeV": m6,
        "m8_GeV": m8,
        "completed_utc": timestamp(),
        "seed_C": seed,
        "lower": None,
        "upper": None,
        "C_low_inside": None,
        "C_high_inside": None,
    }
    if seed is None:
        report["state"] = (
            "unresolved_no_seed"
            if any(s["decision"] == "unknown" for s in samples.values())
            else "no_seed_on_grid"
        )
    else:
        report["lower"], report["upper"] = edge(seed, -1), edge(seed, 1)
        report["C_low_inside"] = report["lower"]["inside_C"]
        report["C_high_inside"] = report["upper"]["inside_C"]
        states = [report[side]["state"] for side in ("lower", "upper")]
        if "reentry_detected" in states:
            report["state"] = "reentry_detected"
        elif "unresolved" in states:
            report["state"] = "unresolved"
        elif report["upper"]["state"] == "range_limit":
            report["state"] = "upper_range_limit"
        elif report["lower"]["state"] == "range_limit" and lo > 0:
            report["state"] = "lower_range_limit"
        else:
            report["state"] = "bounded"
    report["samples"] = sorted(samples.values(), key=lambda s: s["C"])
    report["sample_counts"] = dict(Counter(s["decision"] for s in samples.values()))
    return report


def assigned_columns(campaign, index, count):
    """Distribui cenários/colunas sem sobreposição; count=4 dá um m8 por job."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("Exige shard-count>=1 e 0<=shard-index<shard-count.")
    for mass_index, mass in enumerate(campaign.masses()):
        for case_index, case in enumerate(campaign.m8_values):
            if (mass_index * len(campaign.m8_values) + case_index) % count == index:
                yield case_index, mass_index, mass, case


def run(output, campaign, evaluator, *, shard_index=0, shard_count=1, retry_failed=False):
    """Retoma colunas completas e publica cada nova coluna assim que termina."""
    hints = {}
    for ci, mi, mass, case in assigned_columns(campaign, shard_index, shard_count):
        path = column_path(output, ci, mi)
        with task_lock(output, f"column-{ci}-{mi}"):
            previous = read_json(path) if path.exists() else None
            incomplete = previous and previous["state"] in ("unresolved", "unresolved_no_seed")
            if previous and not (retry_failed and incomplete):
                hints[ci] = previous
                continue
            column = find_column(campaign, mass, case, evaluator, hints.get(ci))
            column.update(case_index=ci, mass_index=mi)
            atomic_json(path, column)
            hints[ci] = column
            print(
                f"Coluna m8={case:g}, m6={mass:g}: {column['state']}; "
                f"C=[{column['C_low_inside']}, {column['C_high_inside']}]",
                flush=True,
            )
