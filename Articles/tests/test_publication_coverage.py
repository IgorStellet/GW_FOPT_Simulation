"""Cobertura física, retomada e referências instrumentais da revisão v2."""

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_paper_figures import write_boundary_fixture

from Articles.collect_data import Settings, json_safe, point_id
from Articles.figures.data import load_dataset
from Articles.figures.points import write_plan
from Articles.figures.publication.coverage import (
    build_coverage_plan,
    complete_observables,
    complete_transition_set,
    coverage_targets,
    pilot_points,
)
from Articles.figures.publication.coverage_execution import run_shard
from Articles.figures.publication.coverage_worker import (
    evaluate_coverage,
    ladder_record,
)
from Articles.figures.publication.detectors import (
    YEAR_SECONDS,
    _plis_amplitudes,
    lisa_noise_h2,
    lisa_plis_h2,
)
from Articles.figures.publication.inputs import load_original

NUMERICS = {
    "policy": "central_step_ladder_v1",
    "maximum_steps_GeV": [0.5, 1.0, 0.25],
    "minimum_step_GeV": 0.005,
    "margin_fraction": 0.1,
    "phase_recovery": True,
}


def fake_physics(params, settings, **kwargs):
    row = {
        **asdict(params),
        "point_id": point_id(params),
        "status": "nucleated",
        "runtime_seconds": 0.0,
        "calculation_mode": "coverage_observables_v2",
    }
    transition = {
        **row,
        "transition_index": 0,
        "trantype": 1,
        "nucleation_accepted": True,
        "quality_flags": "",
        "Tn_GeV": 50.0,
        "Tc_GeV": 80.0,
        "phi_low_GeV": 200.0,
        "phi_high_GeV": 0.0,
        "phi_low_Tc_GeV": 160.0,
        "vev_over_Tc": 2.0,
        "S3_over_T": 140.0,
        "alpha_trace": 0.2,
        "beta_over_H": 100.0,
        "beta_over_H_half_step": 100.0,
        "beta_order": 2,
        "g_star": 106.75,
        "v_w": 1.0,
    }
    return row, [transition], {}


def runner_args(tmp_path, plan, **extra):
    return SimpleNamespace(
        plan=plan,
        output=tmp_path / "coverage",
        work_dir=tmp_path / "scratch",
        task_index=0,
        shards_per_scenario=1,
        point_timeout=900,
        max_points=None,
        retry_failed=False,
        reuse=[],
        dry_run=False,
        **extra,
    )


def hashes(root):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_plan_reuses_complete_points_and_targets_missing_interior(tmp_path):
    root = write_boundary_fixture(tmp_path / "original", with_gw=False)
    before = hashes(root)
    dataset, _ = load_original(root)
    targets = coverage_targets(dataset)
    assert any("interior_mesh" in p["roles"] for p in targets)
    assert any("repair_nucleated_point" in p["roles"] for p in targets)
    assert any("upper_reach_strip" in p["roles"] for p in targets)
    pilot = pilot_points(targets)
    assert len(pilot) == 12
    assert len({(p["m8_GeV"], p["m6_GeV"], p["C"]) for p in pilot}) == 12
    first = next(p for p in targets if "repair_nucleated_point" in p["roles"])
    existing = next(
        r
        for r in dataset.rows
        if float(r["m8_GeV"]) == first["m8_GeV"]
        and float(r["m6_GeV"]) == first["m6_GeV"]
        and math.isclose(r["C"], first["C"], abs_tol=1e-11)
    )
    existing.update(alpha_trace=0.3, beta_over_H=100.0, quality_flags="")
    assert complete_observables(existing)
    assert first not in coverage_targets(dataset)
    plan = build_coverage_plan(dataset, [root])
    assert plan["settings"]["action_tolerance"] == 0.5
    assert plan["settings"]["beta_check"]
    assert plan["numerics"]["maximum_steps_GeV"] == [0.5, 1.0, 0.25]
    assert hashes(root) == before


def test_bounded_campaign_resumes_and_preserves_sources(tmp_path, monkeypatch):
    import Articles.figures.publication.coverage_execution as execution

    root = write_boundary_fixture(tmp_path / "original", with_gw=False)
    before = hashes(root)
    dataset, _ = load_original(root)
    plan = build_coverage_plan(dataset, [root])
    plan["points"] = [p for p in plan["points"] if float(p["m8_GeV"]) == math.inf][:4]
    path = tmp_path / "plan.json"
    write_plan(path, plan)
    calls = []

    def evaluate(params, settings, **kwargs):
        calls.append(point_id(params))
        return fake_physics(params, settings)

    monkeypatch.setattr(execution, "evaluate_bounded", evaluate)
    args = runner_args(tmp_path, path)
    args.max_points = 1
    assert run_shard(args) == 0
    state = args.output / "m8-0-part-0/interior_state.json"
    assert json.loads(state.read_text())["state"] == "paused"
    args.work_dir = tmp_path / "scratch-resume"
    args.max_points = None
    assert run_shard(args) == 0
    assert len(calls) == 4 and len(set(calls)) == 4
    assert json.loads(state.read_text())["state"] == "complete"
    result = load_dataset(args.output)
    assert all(complete_observables(r) for r in result.rows)
    args.work_dir = tmp_path / "scratch-unused"
    args.dry_run = True
    assert run_shard(args) == 0 and not args.work_dir.exists()
    assert hashes(root) == before


def test_step_ladder_selects_first_stable_pair_without_detector_optimization(
    monkeypatch,
):
    import Articles.figures.publication.coverage_worker as worker

    tested = []

    def recorder(transition, phases, derivatives, settings, **kwargs):
        tested.append(settings.beta_step)
        row = {
            "nucleation_accepted": True,
            "alpha_trace": 0.2,
            "beta_over_H": 100,
            "quality_flags": "beta_step_sensitivity"
            if settings.beta_step == 0.5
            else "",
            "beta_step_GeV": settings.beta_step,
            "beta_metastable_margin_GeV": 20,
        }
        return row, {"original_diagnostics": True}

    monkeypatch.setattr(worker, "margin_record", recorder)
    row, details = ladder_record({}, {}, None, Settings(), numerics=NUMERICS)
    assert tested == [0.5, 1.0] and row["beta_step_GeV"] == 1.0
    assert row["beta_step_attempts"] == 2 and len(details["beta_step_ladder"]) == 2
    assert (
        details["beta_step_ladder"][0]["observables"]["quality_flags"]
        == "beta_step_sensitivity"
    )


def test_ladder_preserves_unresolved_and_skips_identical_margin_limited_steps(
    monkeypatch,
):
    import Articles.figures.publication.coverage_worker as worker

    calls = []

    def recorder(*args, **kwargs):
        calls.append(1)
        return {
            "nucleation_accepted": True,
            "alpha_trace": 0.2,
            "beta_over_H": -1,
            "quality_flags": "beta_step_sensitivity",
            "beta_step_GeV": 0.01,
            "beta_metastable_margin_GeV": 0.1,
        }, {}

    monkeypatch.setattr(worker, "margin_record", recorder)
    row, details = ladder_record({}, {}, None, Settings(), numerics=NUMERICS)
    assert len(calls) == 1 and row["quality_flags"] == "beta_step_sensitivity"
    assert len(details["beta_step_ladder"]) == 1


def test_worker_preserves_first_diagnostic_when_recovery_setup_fails(monkeypatch):
    import Articles.figures.publication.coverage_worker as worker
    from Articles.combined_model import ModelParameters

    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("synthetic phase recovery setup failure")
        return {"status": "numerical_failure"}, [], {"initial_trace": True}

    monkeypatch.setattr(worker, "_evaluate_point_once", evaluate)
    row, trs, details = evaluate_coverage(ModelParameters(), Settings(), NUMERICS)
    assert row["evaluation_attempts"] == 2 and not trs
    assert (
        details["initial_trace"]
        and details["phase_recovery_error"]["type"] == "ValueError"
    )


def test_lisa_full_spectrum_reference_has_declared_snr_and_time_scaling():
    from scipy.integrate import trapezoid

    f = np.geomspace(1e-5, 1, 6000)
    slopes, amps = _plis_amplitudes(4.0, 10.0, 4.0)
    for i in (0, 160, 320):
        shape = amps[i] * (f / 1e-3) ** slopes[i]
        snr = math.sqrt(
            4 * YEAR_SECONDS * trapezoid((shape / lisa_noise_h2(f)) ** 2, f)
        )
        assert snr == pytest.approx(10, rel=1e-10)
    small = np.geomspace(1e-4, 0.1, 200)
    assert np.allclose(lisa_plis_h2(small, years=4), lisa_plis_h2(small, years=1) / 2)
    assert np.all(lisa_noise_h2(small) >= lisa_noise_h2(small, foreground_years=None))
    with pytest.raises(ValueError):
        lisa_plis_h2([0])


def test_recovered_bounce_resolves_only_previous_search_non_detections(
    tmp_path, monkeypatch
):
    import Articles.figures.publication.coverage_execution as execution

    root = write_boundary_fixture(tmp_path / "original", cases=(math.inf,))
    dataset, _ = load_original(root)
    negative = next(
        r for r in dataset.rows if r["status"] == "no_nucleation_found" and r["C"] > 3
    )
    plan = build_coverage_plan(dataset, [root])
    plan["points"] = [{k: json_safe(negative[k]) for k in ("m6_GeV", "m8_GeV", "C")}]
    path = tmp_path / "plan.json"
    write_plan(path, plan)
    monkeypatch.setattr(execution, "evaluate_bounded", fake_physics)
    args = runner_args(tmp_path, path)
    assert run_shard(args) == 0
    strict = load_dataset(root, [args.output])
    assert strict.audit["conflicting_coordinates"]
    resolved, _ = load_original(root, [args.output])
    assert not resolved.audit["conflicting_coordinates"]
    assert len(resolved.audit["recovered_search_non_detections"]) == 1


def test_invalid_budget_or_nan_step_is_rejected_without_creating_campaign(tmp_path):
    root = write_boundary_fixture(tmp_path / "original", cases=(math.inf,))
    dataset, _ = load_original(root)
    plan = build_coverage_plan(dataset, [root])
    plan["numerics"]["minimum_step_GeV"] = float("nan")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = runner_args(tmp_path, path)
    with pytest.raises(ValueError, match="beta inválidos"):
        run_shard(args)
    assert not args.output.exists()


def test_negative_search_gets_refined_phase_resolution_before_being_reaccepted(
    monkeypatch,
):
    import Articles.figures.publication.coverage_worker as worker
    from Articles.combined_model import ModelParameters

    settings_seen = []

    def evaluate(params, settings, **kwargs):
        settings_seen.append(settings)
        if len(settings_seen) == 1:
            return {"status": "no_nucleation_found"}, [], {"phase_search": "initial"}
        return fake_physics(params, settings)

    monkeypatch.setattr(worker, "_evaluate_point_once", evaluate)
    row, trs, details = evaluate_coverage(ModelParameters(), Settings(), NUMERICS)
    assert row["evaluation_attempts"] == 2 and complete_transition_set(trs)
    assert settings_seen[1].n_phi == 2 * settings_seen[0].n_phi
    assert settings_seen[1].action_tolerance == settings_seen[0].action_tolerance
    assert details["phase_recovery"]["initial_point"]["status"] == "no_nucleation_found"


def test_later_complete_transition_does_not_hide_incomplete_primary():
    from Articles.combined_model import ModelParameters

    _, transitions, _ = fake_physics(ModelParameters(), Settings())
    first = {**transitions[0], "alpha_trace": None}
    second = {**transitions[0], "transition_index": 1}
    assert not complete_transition_set([second, first])


def test_new_sampling_does_not_resolve_disagreement_between_two_accepted_transitions(
    tmp_path, monkeypatch
):
    import Articles.figures.publication.coverage_execution as execution

    root = write_boundary_fixture(tmp_path / "original", cases=(math.inf,))
    dataset, _ = load_original(root)
    selected = next(r for r in dataset.rows if r["boundary_decision"] == "yes")
    # A origem agora tem uma transição aceita, porém fraca. Uma classificação
    # forte na nova avaliação deve permanecer como conflito físico explícito.
    with closing(sqlite3.connect(root / "m8-0/scan.sqlite")) as db, db:
        index, text = db.execute(
            "SELECT transition_index,record FROM transitions WHERE point_id=?",
            (selected["point_id"],),
        ).fetchone()
        tr = json.loads(text)
        tr["phi_low_GeV"] = 0.5 * tr["Tn_GeV"]
        db.execute(
            "UPDATE transitions SET record=? WHERE point_id=? AND transition_index=?",
            (json.dumps(tr), selected["point_id"], index),
        )
    dataset, _ = load_original(root)
    plan = build_coverage_plan(dataset, [root])
    plan["points"] = [{k: json_safe(selected[k]) for k in ("m6_GeV", "m8_GeV", "C")}]
    path = tmp_path / "plan.json"
    write_plan(path, plan)
    monkeypatch.setattr(execution, "evaluate_bounded", fake_physics)
    args = runner_args(tmp_path, path)
    assert run_shard(args) == 0
    combined, _ = load_original(root, [args.output])
    assert combined.audit["conflicting_coordinates"]
    assert not combined.audit["recovered_search_non_detections"]
