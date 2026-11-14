"""Fases, stencil de dois bounces, sinais de alpha e aceitação física."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from article_FOPT_combined import core, physics
from article_FOPT_combined.config import Numerics
from article_FOPT_combined.model import Parameters, Potential


def test_phase_constructor_keeps_temperature_field_pairs():
    original = core.transitions.Phase.__init__
    temperatures = np.array([4, 3, 2, 1.0])
    with core.ordered_phase_inputs():
        phase = core.transitions.Phase(
            "a", (2 * temperatures + 1)[:, None], temperatures, np.ones((4, 1)) * 2
        )
    np.testing.assert_allclose(phase.valAt([1, 2, 3, 4]).ravel(), [3, 5, 7, 9])
    assert core.transitions.Phase.__init__ is original


def test_beta_uses_exactly_two_action_evaluations():
    calls = []

    class Calculator:
        def _S3_at_T(self, T):
            calls.append(T)
            return T * (2 * T**2 + 3 * T + 4)

    beta, temperatures, values = core.central_beta(Calculator(), 100, 0.5)
    assert beta == pytest.approx(100 * 403)
    assert calls == [99.5, 100.5]


def test_action_criterion_retains_raw_residual():
    criterion = physics.NucleationCriterion(140, 0.5)
    assert criterion(140.4 * 100, 100) == 0
    assert criterion.samples[100]["residual"] == pytest.approx(0.4)
    assert criterion(141 * 100, 100) == pytest.approx(1)
    with pytest.raises(ValueError):
        criterion(-1, 100)


@pytest.mark.parametrize(
    "lower,upper,expected", [(1, 20, "central"), (1, 10.2, "backward"), (9.8, 20, "forward")]
)
def test_beta_near_phase_endpoints_keeps_step_and_two_new_bounces(lower, upper, expected):
    calls = []

    class Calculator:
        def _S3_at_T(self, T):
            calls.append(T)
            return T * (T**2 + 2 * T + 20)

    beta, stencil, samples = core.beta_at_transition(
        Calculator(), {"Tn_GeV": 10, "S3_over_T": 140}, 0.5, lower, upper
    )
    assert beta == pytest.approx(220)
    assert stencil == expected
    assert len(calls) == 2
    assert all(lower < T < upper for T in calls)


def test_signed_thermodynamic_definitions_and_partial_beta_failure(monkeypatch):
    class Phase:
        T = np.array([1, 20])

        def __init__(self, value):
            self.value = np.array([value])

        def valAt(self, T):
            return self.value

    calc = SimpleNamespace(high_phase=Phase(0), low_phase=Phase(1), _S3_cache={})
    monkeypatch.setattr(core, "calculator", lambda *args: calc)
    derivative = SimpleNamespace(
        V=lambda X, T: 2 if X[0] == 0 else 1, dV_dT=lambda X, T: -3 if X[0] == 0 else -1
    )
    row = {
        "Tn_GeV": 10,
        "Tc_GeV": 15,
        "high_phase": "h",
        "low_phase": "l",
        "nucleation_accepted": True,
        "selected_strong": True,
    }
    model = Potential(Parameters())
    result = physics.thermodynamic_record(row, {}, derivative, model, Numerics())
    assert result["delta_rho_GeV4"] == pytest.approx(21)
    assert result["delta_theta_GeV4"] == pytest.approx(6)
    assert result["alpha_energy"] == pytest.approx(21 / result["rho_rad_GeV4"])
    assert not result["observables_complete"]
    assert "alpha_trace" in result and "beta_over_H" not in result


def test_classification_never_relabels_bad_action_as_no_nucleation():
    assert physics.classify([{"trantype": 1, "nucleation_accepted": False}], 1)[0] == "unknown"
    assert (
        physics.classify([{"trantype": 1, "nucleation_accepted": True}], 1)[1] == "weak_nucleation"
    )
    assert physics.classify([{"trantype": 1, "selected_strong": True}], 1)[0] == "yes"


def test_unfavoured_zero_temperature_vacuum_skips_expensive_history(monkeypatch):
    monkeypatch.setattr(
        physics, "build_history", lambda *args: pytest.fail("Não deveria calcular fases.")
    )
    result = physics.evaluate(Parameters(500, 400, 0), Numerics())
    assert result["summary"]["status"] == "ew_vacuum_unfavoured"
    assert result["summary"]["decision"] == "no"


def test_initial_phase_check_exposes_large_field_high_T_minimum():
    model = Potential(Parameters(math.inf, 400, 0))
    assert physics.initial_phase_check(model, Numerics(T_max=200))["symmetric_on_grid"]
    assert not physics.initial_phase_check(model, Numerics(T_max=250))["symmetric_on_grid"]


def test_bad_initial_phase_is_unknown_and_skips_history(monkeypatch):
    monkeypatch.setattr(
        physics,
        "build_history",
        lambda *args: pytest.fail("A fase inicial deveria encerrar a avaliação."),
    )
    result = physics.evaluate(Parameters(math.inf, 400, 0), Numerics(T_max=250))
    assert result["summary"]["decision"] == "unknown"
    assert result["summary"]["reason"] == "high_temperature_phase_not_symmetric"


def test_narrow_phase_interval_uses_validated_adaptive_beta():
    calls = []

    class Calculator:
        def _S3_at_T(self, T):
            calls.append(T)
            return T * (T**2 + 2 * T + 20)

    row = {"Tn_GeV": 10, "S3_over_T": 140}
    result = physics.beta_record(Calculator(), row, Numerics(), 9.9, 10.2)
    assert result["beta_step_adapted"] and result["beta_stability_passed"]
    assert result["beta_step_GeV"] < 0.5
    assert result["beta_over_H"] == pytest.approx(220)
    assert len(calls) == 4
    assert all(9.9 < T < 10.2 for T in calls)


def test_unstable_adaptive_beta_remains_flagged():
    class Calculator:
        def _S3_at_T(self, T):
            return T * (140 + 1e8 * (T - 10) ** 3 + (T - 10))

    result = physics.beta_record(
        Calculator(), {"Tn_GeV": 10, "S3_over_T": 140}, Numerics(), 9.9, 10.2
    )
    assert not result["beta_stability_passed"]
