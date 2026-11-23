"""História térmica de um potencial radial com fase simétrica em phi=0.

O módulo recebe derivadas do potencial e limites de busca. O núcleo calcula
as fases quebradas, suas degenerescências e as ações dos bounces.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from scipy import optimize

from . import core


class ThermalSearchError(RuntimeError):
    """A história ou o cruzamento ficaram inconclusivos numericamente."""


def symmetric_phases(derivatives, settings):
    """Localiza os intervalos em que a origem é um mínimo local.

    Para um potencial par, phi=0 é estacionário em qualquer T. Rastrear
    uma spline para essa coordenada pode perder a fase ao cruzar o ponto
    de curvatura nula. Aqui a curvatura delimita a fase e phi fica fixo.
    """
    origin = np.zeros(1)

    def curvature(temperature):
        value = float(np.asarray(derivatives.hessV(origin, temperature)).item())
        if not math.isfinite(value):
            raise ThermalSearchError(f"Curvatura não finita na origem a T={temperature:g}.")
        return value

    temperatures = np.linspace(settings.T_min, settings.T_max, max(33, 2 * settings.n_T_seeds))
    stable = np.array([curvature(t) > 0 for t in temperatures])
    if not stable[-1]:
        raise ThermalSearchError("A origem não é um mínimo local na temperatura inicial.")
    changes = np.flatnonzero(stable[:-1] != stable[1:])
    edges = [
        float(settings.T_min),
        *(optimize.brentq(curvature, temperatures[i], temperatures[i + 1]) for i in changes),
        float(settings.T_max),
    ]
    return {
        f"symmetric-{index}": core.transitions.Phase(
            f"symmetric-{index}", np.zeros((2, 1)), np.array([low, high]), np.zeros((2, 1))
        )
        for index, (low, high) in enumerate(pairwise(edges))
        if low < high and bool(stable[0]) == (index % 2 == 0)
    }


def trace_phases(derivatives, settings, phi_max):
    """Traça as fases quebradas com o núcleo e combina com a origem exata."""
    tf = core.transitions
    temperature_refs = np.linspace(settings.T_min, settings.T_max, settings.n_T_seeds)
    # O núcleo descarta sementes exatamente no teto, pois o passo inicial é
    # positivo. O último palpite fica logo abaixo e pode ser traçado até T_max.
    temperature_refs[-1] -= min(settings.T_eps, (temperature_refs[-1] - temperature_refs[-2]) / 2)
    seeds = [
        (seed, float(temperature))
        for temperature in temperature_refs
        for seed in tf.findApproxLocalMin(
            derivatives.V,
            np.array([0.0]),
            np.array([phi_max]),
            args=(float(temperature),),
            n=settings.n_phi,
            edge=0.0,
        )
    ]

    def outside_domain(field):
        phi = float(np.asarray(field).item())
        return not math.isfinite(phi) or abs(phi) >= phi_max

    symmetric = symmetric_phases(derivatives, settings)
    with core.ordered_phase_inputs():
        traced = tf.traceMultiMin(
            derivatives.V,
            derivatives.dgradV_dT,
            derivatives.hessV,
            seeds,
            tLow=settings.T_min,
            tHigh=settings.T_max,
            deltaX_target=settings.phase_delta,
            dtstart=settings.phase_step,
            tjump=settings.phase_step,
            forbidCrit=outside_domain,
        )
        phases = {**traced, **symmetric}
        for key, phase in traced.items():
            broken = np.flatnonzero(np.abs(phase.X[:, 0]) > settings.x_eps)
            if len(broken) < 2:
                # Substitui cópias numéricas da origem e preserva os vínculos
                # com outras fases. A janela exata vem da curvatura.
                candidates = [
                    p for p in symmetric.values() if p.T[0] <= phase.T[-1] and p.T[-1] >= phase.T[0]
                ]
                if candidates and np.max(np.abs(phase.X)) <= settings.x_eps:
                    exact = max(
                        candidates,
                        key=lambda p: min(p.T[-1], phase.T[-1]) - max(p.T[0], phase.T[0]),
                    )
                    tf._removeRedundantPhase(phases, phase, exact)
                else:
                    del phases[key]
                    for neighbour in phases.values():
                        neighbour.low_trans.discard(key)
                        neighbour.high_trans.discard(key)
                continue
            # Uma continuação que alcançou a origem pode incluir nós do ramo
            # simétrico. Apenas a parte quebrada entra na busca de coexistência.
            if np.any(np.diff(broken) != 1):
                raise ThermalSearchError("Uma fase traçada contém trechos quebrados desconexos.")
            field = phase.X[broken]
            trimmed = tf.Phase(
                key, np.abs(field), phase.T[broken], np.sign(field) * phase.dXdT[broken]
            )
            trimmed.low_trans = phase.low_trans
            trimmed.high_trans = phase.high_trans
            phases[key] = trimmed
        tf.removeRedundantPhases(derivatives.V, phases)
    start = max(symmetric, key=lambda key: symmetric[key].T[-1])
    if start not in phases:
        raise ThermalSearchError("A fase simétrica inicial se confundiu com um ramo quebrado.")
    origin = phases[start].X[-1]
    origin_energy = derivatives.V(origin, settings.T_max)
    for phase in phases.values():
        if phase.T[-1] == settings.T_max:
            energy = derivatives.V(phase.X[-1], settings.T_max)
            if energy < origin_energy - 1e-5:
                raise ThermalSearchError(
                    "Um mínimo quebrado traçado é favorecido na temperatura inicial."
                )
    return phases, start


def highest_crossing(residual, lower, upper, *, tolerance, upper_is_critical):
    """Procura o cruzamento no resfriamento, inclusive se as pontas forem positivas.

    Sonda a coexistência de cima para baixo. Se necessário, minimiza a ação
    entre amostras vizinhas ao redor de mínimos locais. Um intervalo com
    falhas de integração é inconclusivo quando nenhum cruzamento foi resolvido.
    """
    if lower >= upper:
        return None
    values = {upper: math.inf} if upper_is_critical else {}
    errors = {}

    def probe(temperature):
        temperature = float(temperature)
        if temperature not in values and temperature not in errors:
            try:
                value = float(residual(temperature))
                if math.isnan(value) or value == -math.inf:
                    raise ValueError("Residual da ação não finito.")
                values[temperature] = value
            except (
                ArithmeticError,
                ValueError,
                RuntimeError,
                core.helpers.IntegrationError,
            ) as exc:
                errors[temperature] = exc
                print(f"Ação inconclusiva a T={temperature:.8g}: {type(exc).__name__}: {exc}")
        return values.get(temperature)

    def require(temperature):
        value = probe(temperature)
        if value is None:
            raise ThermalSearchError(f"Ação inconclusiva a T={temperature:.8g}.")
        return value

    def crossing_between(cold, hot):
        try:
            return float(optimize.brentq(require, cold, hot, xtol=tolerance, maxiter=100))
        except ThermalSearchError as error:
            # Um bounce que falha no passo de Brent não decide o sinal. Retoma
            # o bracket pelas ações válidas já calculadas e testa nós vizinhos.
            cold = max(t for t, value in values.items() if cold <= t < hot and value <= 0)
            hot = min(t for t, value in values.items() if cold < t <= hot and value > 0)
            while hot - cold > tolerance:
                for fraction in (0.5, 0.25, 0.75):
                    trial = cold + fraction * (hot - cold)
                    value = probe(trial)
                    if value is not None:
                        break
                else:
                    raise ThermalSearchError(
                        "Não foi possível fechar o bracket de S3/T."
                    ) from error
                if value == 0:
                    return trial
                if value < 0:
                    cold = trial
                else:
                    hot = trial
            raise ThermalSearchError("Bracket estreito sem ação dentro da tolerância.") from error

    hot = None
    nodes = np.linspace(upper, lower, 9)
    for temperature in nodes:
        value = probe(temperature)
        if value is None:
            continue
        if value == 0:
            return float(temperature)
        if value < 0:
            if hot is None:
                raise ThermalSearchError(
                    "A ação já está abaixo do limiar no início da coexistência."
                )
            return crossing_between(float(temperature), hot)
        hot = float(temperature)

    for hot, center, cold in zip(nodes[:-2], nodes[1:-1], nodes[2:], strict=True):
        shoulders = [values.get(float(t)) for t in (hot, center, cold)]
        if any(value is None for value in shoulders):
            continue
        if shoulders[1] <= min(shoulders[0], shoulders[2]) and shoulders[1] < max(shoulders):
            minimum = optimize.minimize_scalar(
                lambda t: value if (value := probe(t)) is not None else math.inf,
                bounds=(cold, hot),
                method="bounded",
                options={"xatol": max(tolerance, (hot - cold) / 512)},
            )
            if not minimum.success:
                raise ThermalSearchError("Não foi possível localizar o mínimo local de S3/T.")
            if minimum.fun == 0:
                return float(minimum.x)
            if minimum.fun < 0:
                return crossing_between(float(minimum.x), float(hot))
    if errors:
        raise ThermalSearchError(
            f"Ação inconclusiva em {len(errors)} temperaturas da coexistência."
        )
    return None


def find_transition(phases, critical, high_limit, derivatives, settings, criterion):
    """Resolve o cruzamento em um par de fases com coexistência conhecida."""
    high = phases[critical["high_phase"]]
    low = phases[critical["low_phase"]]
    lower = max(high.T[0], low.T[0])
    upper = min(high_limit, high.T[-1], low.T[-1], critical["Tcrit"])
    actions = {}

    def residual(temperature):
        return core.transitions._tunnelFromPhaseAtT(
            temperature,
            {low.key: low},
            high,
            derivatives.V,
            derivatives.gradV,
            settings.minima_tolerance,
            0.0,
            criterion,
            None,
            False,
            actions,
        )

    temperature = highest_crossing(
        residual,
        float(lower),
        float(upper),
        tolerance=settings.root_T_tolerance,
        upper_is_critical=upper == critical["Tcrit"],
    )
    if temperature is None:
        return None
    action = actions.get(temperature, {})
    if action.get("trantype") != 1:
        raise ThermalSearchError(f"Cruzamento sem bounce de primeira ordem a T={temperature:g}.")
    return {**action, "crit_trans": critical}


def build_history(derivatives, settings, criterion, phi_max):
    """Segue o resfriamento da fase inicial, procurando a transição mais quente."""
    phases, initial = trace_phases(derivatives, settings, phi_max)
    critical = core.transitions.findCriticalTemperatures(phases, derivatives.V, start_high=False)
    critical = [
        t
        for t in critical
        if t["trantype"] != 1 or np.linalg.norm(t["high_vev"] - t["low_vev"]) > settings.x_eps
    ]
    remaining = dict(phases)
    full = []
    history_error = None
    start, temperature = initial, settings.T_max
    while start in remaining:
        phase = remaining.pop(start)
        candidates = [
            t for t in critical if t["high_phase"] == start and t["low_phase"] in remaining
        ]
        transition = None
        try:
            for candidate in candidates:
                if candidate["trantype"] != 1:
                    continue
                if transition and candidate["Tcrit"] <= transition["Tnuc"]:
                    break
                found = find_transition(
                    phases, candidate, temperature, derivatives, settings, criterion
                )
                if found and (transition is None or found["Tnuc"] > transition["Tnuc"]):
                    transition = found
        except ThermalSearchError as error:
            if not full:
                raise
            # A história fica incompleta a partir daqui. As transições
            # resolvidas anteriormente continuam documentadas.
            history_error = {"phase": str(start), "T_start_GeV": temperature, "message": str(error)}
            break
        if transition is None:
            continuous = next(
                (t for t in candidates if t["trantype"] == 2 and t["Tcrit"] <= temperature), None
            )
            if continuous is None:
                break
            transition = core.transitions.secondOrderTrans(phase, phases[continuous["low_phase"]])
            transition["crit_trans"] = continuous
        full.append(transition)
        start, temperature = transition["low_phase"], transition["Tnuc"]
    return {
        "phases": phases,
        "start_phase_key": initial,
        "critical_transitions": critical,
        "full_transitions": full,
        "history_error": history_error,
    }
