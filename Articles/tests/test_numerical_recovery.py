"""Recuperação limitada sem relaxar o critério físico nem ocultar falhas."""

from Articles import collect_data as collector
from Articles.combined_model import ModelParameters


def test_negative_action_is_recorded_and_rejected():
    import pytest
    criterion = collector.NucleationCriterion()
    with pytest.raises(ValueError, match="negative_nucleation_action"):
        criterion(-1, 100)
    assert criterion.samples[100]["S3_GeV"] == -1


def test_refinement_recovers_invalid_nucleation_and_preserves_initial_diagnostic(monkeypatch):
    calls = []
    def evaluate(p, settings):
        calls.append(settings)
        first = len(calls) == 1
        return ({"status": "observables_unresolved" if first else "nucleated", "runtime_seconds": 2},
                [{"trantype": 1, "nucleation_accepted": not first}], {"traceback": "original" if first else None})
    monkeypatch.setattr(collector, "_evaluate_point_once", evaluate)
    settings = collector.Settings(beta_check=True)
    row, _, details = collector.evaluate_point(ModelParameters(), settings)
    assert row["recovery_resolved"] and row["evaluation_attempts"] == 2
    assert row["runtime_seconds"] == 4
    assert calls[1].n_phi == 2 * settings.n_phi
    assert calls[1].phase_tjump_frac < settings.phase_tjump_frac
    for name in ("nucleation_target", "action_tolerance", "beta_step", "beta_check", "T_min", "T_max"):
        assert getattr(calls[1], name) == getattr(settings, name)
    assert details["numerical_recovery"]["initial_diagnostics"]["traceback"] == "original"


def test_persistent_failure_is_attempted_twice_and_remains_unknown(monkeypatch):
    calls = []
    def evaluate(*args):
        calls.append(args)
        return {"status": "numerical_failure", "runtime_seconds": 1}, [], {"traceback": "still fails"}
    monkeypatch.setattr(collector, "_evaluate_point_once", evaluate)
    row, trs, _ = collector.evaluate_point(ModelParameters(), collector.Settings())
    assert len(calls) == 2 and not row["recovery_resolved"]
    from Articles.adaptive_scan import nucleation_decision
    assert nucleation_decision(row, trs) == "unknown"


def test_accepted_nucleation_does_not_repeat_bounces_for_unresolved_beta(monkeypatch):
    calls = []
    def evaluate(*args):
        calls.append(args)
        return {"status": "observables_unresolved", "runtime_seconds": 1}, [{"trantype": 1, "nucleation_accepted": True}], {}
    monkeypatch.setattr(collector, "_evaluate_point_once", evaluate)
    row, _, _ = collector.evaluate_point(ModelParameters(), collector.Settings())
    assert len(calls) == row["evaluation_attempts"] == 1
