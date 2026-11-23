"""Regressões de coexistência e de cruzamentos da ação térmica."""

from types import SimpleNamespace

import numpy as np
import pytest
from article_FOPT_combined import core, thermal_history
from article_FOPT_combined.config import Numerics
from article_FOPT_combined.physics import NucleationCriterion


def test_nucleation_search_checks_interior_even_with_positive_endpoints():
    def residual(T):
        return (T - 6) ** 2 - 1

    hit = thermal_history.highest_crossing(residual, 1, 12, tolerance=1e-6, upper_is_critical=False)
    assert hit == pytest.approx(7, abs=1e-6)


def test_nucleation_search_refines_narrow_minimum_between_coarse_probes():
    hit = thermal_history.highest_crossing(
        lambda T: (T - 6.35) ** 2 - 0.05**2,
        1,
        12,
        tolerance=1e-6,
        upper_is_critical=False,
    )
    assert hit == pytest.approx(6.4, abs=1e-6)


def test_nucleation_search_retries_bracket_with_neighbouring_temperatures():
    criterion = NucleationCriterion(140, 0.01)
    failures = []

    def residual(T):
        if abs(T - 9.5) < 1e-8:
            failures.append(T)
            raise core.helpers.IntegrationError("bounce não convergiu")
        return criterion(T * (140 + T - 9.5), T)

    hit = thermal_history.highest_crossing(residual, 1, 17, tolerance=1e-6, upper_is_critical=False)
    assert failures
    assert hit == pytest.approx(9.5, abs=0.01)


def test_nucleation_search_keeps_failed_scan_inconclusive():
    def residual(T):
        if T < 3:
            raise core.helpers.IntegrationError("bounce não convergiu")
        return 3

    with pytest.raises(thermal_history.ThermalSearchError, match="inconclusiva"):
        thermal_history.highest_crossing(residual, 1, 12, tolerance=1e-6, upper_is_critical=False)


def test_symmetric_phase_has_exact_support_to_high_temperature(monkeypatch):
    T = np.array([1.0, 40.0, 60.0, 80.0])
    fields = np.array([[240.0], [100.0], [0.001], [0.0]])
    trace = core.transitions.Phase("broken", fields, T, np.zeros_like(fields))
    monkeypatch.setattr(
        core.transitions, "traceMultiMin", lambda *args, **kwargs: {"broken": trace}
    )
    monkeypatch.setattr(core.transitions, "findApproxLocalMin", lambda *args, **kwargs: [])
    model = SimpleNamespace(
        hessV=lambda x, t: np.array([[t - 50]]),
        dgradV_dT=lambda x, t: np.zeros_like(x),
        V=lambda x, t: np.sum(np.asarray(x) ** 2),
    )
    phases, first = thermal_history.trace_phases(model, Numerics(), 300)
    assert phases[first].T.tolist() == [50.0, 200.0]
    assert phases[first].valAt(200).item() == 0
    assert phases["broken"].T[-1] == 40


def test_late_failure_preserves_nucleation_already_resolved(monkeypatch):
    phases = dict.fromkeys(("symmetric", "broken", "cold"))
    critical = [
        {
            "trantype": 1,
            "Tcrit": T,
            "high_phase": high,
            "low_phase": low,
            "high_vev": np.array([0.0]),
            "low_vev": np.array([100.0]),
        }
        for T, high, low in ((110, "symmetric", "broken"), (90, "broken", "cold"))
    ]
    monkeypatch.setattr(thermal_history, "trace_phases", lambda *args: (phases, "symmetric"))
    monkeypatch.setattr(core.transitions, "findCriticalTemperatures", lambda *a, **k: critical)

    def find_transition(_phases, candidate, *args):
        if candidate["high_phase"] == "broken":
            raise thermal_history.ThermalSearchError("bounce posterior inconclusivo")
        return {**candidate, "Tnuc": 100.0}

    monkeypatch.setattr(thermal_history, "find_transition", find_transition)
    history = thermal_history.build_history(SimpleNamespace(V=None), Numerics(), None, 300)
    assert len(history["full_transitions"]) == 1
    assert history["full_transitions"][0]["Tnuc"] == 100
    assert history["history_error"]["phase"] == "broken"
