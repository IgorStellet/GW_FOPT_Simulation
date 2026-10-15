"""Localiza pares dentro/fora; não amostra densamente o interior das faixas.

A hipótese explícita é uma faixa conectada em C após uma semente verificada.
Falhas não são testemunhas externas e nunca certificam uma fronteira.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import pairwise


class BoundaryPaused(Exception):
    """Parada solicitada ou orçamento de pontos atingido; dados preservados."""


@dataclass(frozen=True)
class BoundaryPlan:
    m6_range_GeV: tuple = (500.0, 2000.0, 5.0)
    C_bounds: tuple = (0.0, 10.0)
    m8_GeV: float = math.inf
    seed_C: float = 3.83
    C_tolerance: float = 0.02
    walk_step: float = 0.1
    seed_step: float = 0.1
    seed_resolution: float = 0.02
    patience: int = 3
    unknown_limit: int = 3
    criterion: str = "strong"
    strong_threshold: float = 1.0
    column_limit: int | None = None

    def validate(self):
        from Articles.collect_data import inclusive_axis

        lo, hi = self.C_bounds
        if not (0 <= lo <= self.seed_C <= hi <= 10) or not all(math.isfinite(v) for v in (lo, hi, self.seed_C)):
            raise ValueError("Exige 0<=C_min<=seed-C<=C_max<=10.")
        for name in ("C_tolerance", "walk_step", "seed_step", "seed_resolution", "strong_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser finito e positivo.")
        if self.seed_resolution > self.seed_step:
            raise ValueError("seed-resolution deve ser <= seed-step.")
        if self.patience < 1 or self.unknown_limit < 1:
            raise ValueError("Contadores devem ser positivos.")
        if self.column_limit is not None and self.column_limit < 1:
            raise ValueError("column-limit deve ser positivo.")
        if self.criterion not in ("strong", "nucleation"):
            raise ValueError("Critério deve ser strong ou nucleation.")
        if math.isnan(self.m8_GeV) or self.m8_GeV <= 0:
            raise ValueError("m8 deve ser positivo ou inf.")
        if inclusive_axis(*self.m6_range_GeV)[0] <= 0:
            raise ValueError("m6 deve ser positivo.")

    def masses(self):
        from Articles.collect_data import inclusive_axis

        self.validate()
        return (math.inf, *reversed(inclusive_axis(*self.m6_range_GeV)))

    def identity(self):
        return {"strategy": "boundary_v1_bracketed", **asdict(self),
                "assumption": "connected_selected_band_after_verified_seed",
                "strong_definition": "abs(phi_low(Tn))/Tn > threshold"}


def _coordinate(value):
    return round(float(value), 12)


def _axis_around(start, lo, hi, step):
    """Busca para baixo, depois para cima, incluindo os bounds exatamente."""
    yield start
    for direction, bound in ((-1, lo), (1, hi)):
        count = int(abs(bound - start) / step)
        for i in range(1, count + 1):
            yield _coordinate(start + direction * i * step)
        yield bound


def trace_boundaries(plan, evaluate, report=None, on_progress=None, hint_samples=None, completed_columns=None):
    """evaluate(C,m6) retorna decisão ou {decision, reason}.

    Primeiro verifica uma semente na região selecionada, sem regra de três.
    Saltos crescentes delimitam a faixa; bisseção reduz o bracket a C_tolerance.
    Três amostras externas confirmam a saída (limitadas pelos bounds globais).
    Até unknown_limit falhas suspendem uma direção, sem chamar isso de exclusão.
    """
    masses = plan.masses()
    lo, hi = plan.C_bounds
    report = report if report is not None else {}
    report.update(state="running", plan=plan.identity(), columns=[], unknown_points=[],
                  evaluated_points=0, m6_stop=None, mass_connectivity_warnings=[])
    decisions = {}
    saved_columns = {float(c["m6_GeV"]): c for c in completed_columns or []
                     if c["state"] in ("bounded", "unresolved", "unresolved_no_seed", "no_selected_seed_on_grid")}

    def progress():
        if on_progress:
            on_progress(report)

    def ask(C, mass):
        C = _coordinate(min(hi, max(lo, C)))
        key = (C, mass)
        if key not in decisions:
            result = evaluate(C, mass)
            result = {"decision": result, "reason": result} if isinstance(result, str) else dict(result)
            if result["decision"] not in ("yes", "no", "unknown"):
                raise ValueError("Decisão de fronteira inválida.")
            decisions[key] = result
            report["evaluated_points"] += 1
            if result["decision"] == "unknown":
                report["unknown_points"].append({"C": C, "m6_GeV": mass, **result})
            progress()
        return decisions[key]["decision"]

    def seed_candidates(start, previous):
        cached = hint_samples(current_mass) if hint_samples else []
        positive = [p["C"] for p in cached if p["decision"] == "yes" and lo <= p["C"] <= hi]
        target = start
        if previous and previous.get("lower") and previous.get("upper"):
            target = (previous["lower"]["inside_C"] + previous["upper"]["inside_C"]) / 2
        if positive:
            yield min(positive, key=lambda C: abs(C - target))
        if previous:
            inside = [previous[s]["inside_C"] for s in ("lower", "upper") if previous.get(s)]
            if len(inside) == 2:
                yield _coordinate(sum(inside) / 2)
            yield previous["seed_C"]
        yield from _axis_around(start, lo, hi, plan.seed_step)
        for anchor in (1.0, 3.2):
            if lo <= anchor <= hi:
                yield anchor
        # Se o salto inicial não achou a faixa, completa a resolução original.
        # Três negativos ANTES da semente não encerram essa procura.
        yield from _axis_around(plan.seed_C, lo, hi, plan.seed_resolution)

    def find_boundary(seed, mass, direction, previous):
        bound = hi if direction == 1 else lo
        inside, outside = seed, None
        unknowns, seen = 0, set()
        label = "upper" if direction == 1 else "lower"

        def result(state, reason, confirmations=()):
            return {"state": state, "reason": reason, "inside_C": inside, "outside_C": outside,
                    "width_C": None if outside is None else abs(outside - inside),
                    "outside_kind": None if outside is None else decisions[(outside, mass)]["reason"],
                    "confirmation_C": list(confirmations), "unknown_probes": unknowns}

        guesses = []
        if previous and previous.get(label):
            old = previous[label]
            guesses = [c for c in (old.get("inside_C"), old.get("outside_C"))
                       if c is not None and direction * (c - seed) > 0 and lo <= c <= hi]
            guesses.sort(reverse=direction < 0)
        if hint_samples:
            cached = sorted([p for p in hint_samples(mass)
                             if lo <= p["C"] <= hi and direction * (p["C"] - seed) > 0],
                            key=lambda p: direction * (p["C"] - seed))
            # Usa o primeiro externo conhecido, sem saltar sobre ele com base
            # num ponto interno mais distante. Falhas nunca fornecem um bracket.
            external = next((p["C"] for p in cached if p["decision"] == "no"), None)
            internal = [p["C"] for p in cached if p["decision"] == "yes"
                        and (external is None or direction * (p["C"] - external) < 0)]
            known = ([max(internal, key=lambda C: direction * (C - seed))] if internal else [])
            if external is not None:
                known.append(external)
                if any(p["decision"] == "yes" and direction * (p["C"] - external) > 0 for p in cached):
                    inside = known[0] if internal else seed
                    outside = external
                    ask(inside, mass)
                    ask(outside, mass)
                    return result("unresolved", "known_nonconnected_samples")
            guesses = [*known, *[c for c in guesses if c not in known]]
        step = plan.walk_step
        while outside is None:
            if inside == bound:
                return result("bound_limited", "selected_at_search_bound")
            candidate = guesses.pop(0) if guesses else _coordinate(inside + direction * step)
            candidate = _coordinate(min(hi, max(lo, candidate)))
            if candidate in seen or direction * (candidate - inside) <= 0:
                guesses = []
                step *= 2
                candidate = _coordinate(min(hi, max(lo, inside + direction * step)))
            seen.add(candidate)
            d = ask(candidate, mass)
            if d == "yes":
                inside, unknowns = candidate, 0
            elif d == "no":
                outside = candidate
            else:
                unknowns += 1
                if unknowns >= plan.unknown_limit or candidate == bound:
                    return result("unresolved", "unknown_walk_limit")
            if not guesses:
                step *= 2

        tested = {inside, outside}
        unknowns = 0
        while abs(outside - inside) > plan.C_tolerance + 1e-11:
            # Após um midpoint inconclusivo, tenta o maior intervalo restante;
            # nenhum lado é descartado com base na falha.
            ordered = sorted({inside, outside, *[c for c in tested if min(inside, outside) < c < max(inside, outside)]})
            left, right = max(pairwise(ordered), key=lambda pair: pair[1] - pair[0])
            candidate = _coordinate((left + right) / 2)
            if candidate in tested:
                return result("unresolved", "coordinate_resolution")
            tested.add(candidate)
            d = ask(candidate, mass)
            if d == "yes":
                inside, unknowns = candidate, 0
            elif d == "no":
                outside, unknowns = candidate, 0
            else:
                unknowns += 1
                if unknowns >= plan.unknown_limit:
                    return result("unresolved", "unknown_refinement_limit")

        confirmations = [outside]
        for i in range(1, plan.patience):
            candidate = _coordinate(outside + direction * i * plan.C_tolerance)
            if not lo <= candidate <= hi:
                break
            d = ask(candidate, mass)
            if d == "yes":
                return result("unresolved", "nonconnected_samples", confirmations)
            if d == "unknown":
                return result("unresolved", "outside_confirmation_unresolved", confirmations)
            confirmations.append(candidate)
        return result("bracketed", "inside_outside_bracket", confirmations)

    previous, empty, has_seed, last_selected_mass = None, [], False, None
    def can_close_mass_axis(mass):
        if len(empty) < plan.patience:
            return False
        if hint_samples:
            remaining = masses[masses.index(mass) + 1:]
            later = [m for m in remaining if any(p["decision"] == "yes" for p in hint_samples(m))]
            if later:
                report["mass_connectivity_warnings"].append({"empty_m6_GeV": list(empty), "known_later_selected_m6_GeV": later})
                return False
        return True
    try:
        for mass in masses:
            current_mass = mass
            if plan.column_limit and len(report["columns"]) >= plan.column_limit:
                report["m6_stop"] = {"reason": "pilot_column_limit", "first_skipped_m6_GeV": mass}
                break
            if mass in saved_columns:
                # Mesma campanha/manifesto: uma coluna já terminada não refaz
                # a procura de fronteiras a cada alocação de 14 h. Pendências
                # só são reabertas explicitamente com --retry-failed.
                column = deepcopy(saved_columns[mass])
                report["columns"].append(column)
                if column["seed_C"] is not None:
                    previous, has_seed, last_selected_mass, empty = column, True, mass, []
                elif column["state"] == "no_selected_seed_on_grid" and has_seed:
                    empty.append(mass)
                else:
                    empty = []
                if can_close_mass_axis(mass):
                    report["m6_stop"] = {"reason": "consecutive_empty_columns_on_seed_grid", "last_m6_GeV": empty,
                                         "seed_resolution": plan.seed_resolution, "inside_m6_GeV": last_selected_mass,
                                         "outside_m6_GeV": empty[0],
                                         "width_m6_GeV": abs(last_selected_mass - empty[0]) if math.isfinite(last_selected_mass) else None}
                    break
                continue
            column = {"m6_GeV": mass, "state": "searching_seed", "seed_C": None,
                      "seed_probes": [], "lower": None, "upper": None}
            report["columns"].append(column)
            start = previous["seed_C"] if previous else plan.seed_C
            found, seed_unknown = None, False
            for candidate in seed_candidates(start, previous):
                candidate = _coordinate(candidate)
                d = ask(candidate, mass)
                if candidate not in column["seed_probes"]:
                    column["seed_probes"].append(candidate)
                seed_unknown |= d == "unknown"
                if d == "yes":
                    found = candidate
                    break
            if found is None:
                column["state"] = "unresolved_no_seed" if seed_unknown else "no_selected_seed_on_grid"
                column["seed_search_complete"] = True
                empty = [*empty, mass] if not seed_unknown and has_seed else []
                progress()
                if can_close_mass_axis(mass):
                    report["m6_stop"] = {"reason": "consecutive_empty_columns_on_seed_grid",
                                         "last_m6_GeV": empty, "seed_resolution": plan.seed_resolution,
                                         "inside_m6_GeV": last_selected_mass, "outside_m6_GeV": empty[0],
                                         "width_m6_GeV": abs(last_selected_mass - empty[0]) if math.isfinite(last_selected_mass) else None}
                    break
                continue
            column.update(seed_C=found, state="locating_boundaries")
            has_seed, empty = True, []
            last_selected_mass = mass
            column["upper"] = find_boundary(found, mass, 1, previous)
            column["lower"] = find_boundary(found, mass, -1, previous)
            column["state"] = "bounded" if all(column[s]["state"] != "unresolved" for s in ("lower", "upper")) else "unresolved"
            previous = column
            progress()
        else:
            report["m6_stop"] = {"reason": "mass_bound", "inside_m6_GeV": last_selected_mass, "outside_m6_GeV": None}
        report["state"] = "pilot_complete" if plan.column_limit else "complete"
        if report["unknown_points"] or any(c["state"].startswith("unresolved") for c in report["columns"]):
            report["state"] += "_with_unresolved"
    except BoundaryPaused as error:
        report.update(state="paused", message=str(error))
    except BaseException:
        report["state"] = "interrupted"
        raise
    finally:
        progress()
    return report


def interior_coordinates(report, step=0.02, samples_per_column=None, allow_unresolved=False):
    """Interior conservador entre duas testemunhas positivas.

    Pontos entre os brackets externos/internos ficam para refinamento posterior.
    Mesmo no interior, cada ponto voltará a verificar nucleação/observáveis.
    Colunas pendentes só entram com allow_unresolved explícito.
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Passo interior deve ser positivo e finito.")
    for column in report["columns"]:
        allowed = column["state"] == "bounded" or allow_unresolved and column["state"] == "unresolved"
        if not allowed or not column.get("lower") or not column.get("upper"):
            continue
        low, high = column["lower"]["inside_C"], column["upper"]["inside_C"]
        anchor = report["plan"]["seed_C"]
        first = math.ceil((low - anchor) / step - 1e-10)
        last = math.floor((high - anchor) / step + 1e-10)
        cs = sorted({_coordinate(anchor + i * step) for i in range(first, last + 1)}
                    | {low, high, column["seed_C"], 1.0, 3.2})
        cs = [c for c in cs if low - 1e-11 <= c <= high + 1e-11]
        if samples_per_column and len(cs) > samples_per_column:
            # O piloto usa pontos interiores, evitando encostar em Tc por acaso.
            indices = [min(len(cs) - 1, int((i + 1) * len(cs) / (samples_per_column + 1))) for i in range(samples_per_column)]
            cs = [cs[i] for i in dict.fromkeys(indices)]
        for C in cs:
            yield C, column["m6_GeV"]
