"""Ponte pequena e explícita com o núcleo da main, sem editar src.

Uma correção de compatibilidade fica restrita à chamada de fases: a Phase da
main aplica duas vezes a ordenação de X. Entregando entradas já ordenadas,
preservamos a correspondência entre T e X sem mudar o arquivo original.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
# Importação deliberada do checkout: evita usar outro CosmoTransitions instalado.
_base = "src.CosmoTransitions"
finiteT = import_module(f"{_base}.finiteT")
generic = import_module(f"{_base}.generic_potential")
helpers = import_module(f"{_base}.helper_functions")
transitions = import_module(f"{_base}.transitionFinder")
gw = import_module(f"{_base}.gravitational_Waves")


@contextmanager
def ordered_phase_inputs():
    """Ordena juntos T, X e dX/dT antes do construtor original de Phase.

    O contexto é usado em um processo físico isolado, nunca por threads que
    estejam calculando dois potenciais. O método original é restaurado ao sair.
    """
    original = transitions.Phase.__init__

    def initialize(self, key, X, T, dXdT):
        order = np.argsort(np.asarray(T))
        original(self, key, np.asarray(X)[order], np.asarray(T)[order], np.asarray(dXdT)[order])

    transitions.Phase.__init__ = initialize
    try:
        yield
    finally:
        transitions.Phase.__init__ = original


def phase_from_record(key, record):
    """Reconstrói a spline de uma fase salva para enriquecer uma borda já calculada."""
    with ordered_phase_inputs():
        return transitions.Phase(key, record["X"], record["T"], record["dXdT"])


class RefinedPhase:
    """Refina os mínimos interpolados antes de pedir ações ao calculador do src.

    A classe delega os demais atributos à Phase original. Assim, o bounce
    continua sendo resolvido pelo GravitationalWaveCalculator da main.
    """

    def __init__(self, phase, potential, tolerance):
        self.phase, self.potential, self.tolerance = phase, potential, tolerance
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.phase, name)

    def valAt(self, temperature, deriv=0):
        """Retorna um mínimo refinado em T escalar; derivadas usam a spline original."""
        if deriv or np.ndim(temperature):
            return self.phase.valAt(temperature, deriv=deriv)
        T = float(temperature)
        if T not in self.cache:
            guess = np.asarray(self.phase.valAt(T), dtype=float)
            minimum = optimize.fmin(
                self.potential, guess, args=(T,), xtol=self.tolerance, ftol=np.inf, disp=False
            )
            if not np.all(np.isfinite(minimum)):
                raise ValueError("Mínimo não finito ao refinar a fase.")
            self.cache[T] = minimum
        return self.cache[T].copy()


def calculator(derivatives, phases, high_key, low_key, tolerance):
    """Usa o calculador original com mínimos refinados e cache por temperatura."""
    refined = {key: RefinedPhase(phase, derivatives.V, tolerance) for key, phase in phases.items()}
    return gw.GravitationalWaveCalculator(
        derivatives.V, derivatives.gradV, derivatives.dV_dT, refined, high_key, low_key
    )


def central_beta(calc, temperature, step):
    """Beta/H com dois bounces, erro de truncamento O(step²), sem mudar step.

    O src da main oferece ordem quatro. Aqui reutilizamos seu cálculo de S3 e
    os pesos de diferenças finitas do helper, escolhendo dois pontos centrais.
    """
    temperatures = np.array([temperature - step, temperature + step], dtype=float)
    if step <= 0 or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise ValueError("Stencil de beta exige temperaturas positivas e finitas.")
    values = np.array([calc._S3_at_T(float(T)) / T for T in temperatures])
    weights = helpers.fd_weights_1d(np.array([-1.0, 1.0]), 0.0, der=1) / step
    return float(temperature * np.dot(weights, values)), temperatures, values


def sound_spectrum(frequency_Hz, row):
    """h²Omega_sw: interface em Hz, convertendo para os mHz usados no src."""
    return gw.gw_omega_sw_h2(
        np.asarray(frequency_Hz) * 1e3,
        alpha=row["alpha_trace"],
        beta_over_H=row["beta_over_H"],
        T_star=row["Tn_GeV"],
        g_star=row["g_star"],
        v_w=row["v_w"],
    )


def beta_at_transition(calc, row, step, lower, upper):
    """Ordem dois com dois bounces novos; stencil central ou unilateral admissível.

    Perto de Tc, Tn+h pode estar fora da coexistência. Nesse caso usamos
    Tn-2h,Tn-h,Tn, aproveitando a ação de nucleação em Tn. O passo permanece
    fixo e a escolha fica salva. Se nenhum stencil cabe, beta fica pendente.
    """
    Tn = row["Tn_GeV"]
    choices = (
        ("central", (-1.0, 1.0)),
        ("backward", (-2.0, -1.0, 0.0)),
        ("forward", (0.0, 1.0, 2.0)),
    )
    for name, offsets in choices:
        offsets = np.asarray(offsets)
        temperatures = Tn + step * offsets
        if step <= 0 or np.any(temperatures <= max(0, lower)) or np.any(temperatures >= upper):
            continue
        values = np.array(
            [
                row["S3_over_T"] if offset == 0 else calc._S3_at_T(float(T)) / T
                for offset, T in zip(offsets, temperatures, strict=True)
            ]
        )
        weights = helpers.fd_weights_1d(offsets, 0.0, der=1) / step
        samples = [
            {
                "T_GeV": float(T),
                "S3_over_T": float(value),
                "cached_nucleation_action": bool(offset == 0),
            }
            for T, value, offset in zip(temperatures, values, offsets, strict=True)
        ]
        return float(Tn * np.dot(weights, values)), name, samples
    raise ValueError("Nenhum stencil de ordem dois cabe na coexistência; passo de beta mantido.")
