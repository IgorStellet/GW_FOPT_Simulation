"""Retomada, timeout, isolamento entre estágios e fronteiras com falhas."""

import math
from dataclasses import replace

import pytest
from article_FOPT_combined import boundaries, execution, observables, storage
from article_FOPT_combined.__main__ import main
from article_FOPT_combined.config import Campaign, Numerics, axis
from article_FOPT_combined.model import Parameters


def synthetic_evaluator(parameters):
    decision = "yes" if 1.2 < parameters.C < 3.3 else "no"
    return {"summary": {"decision": decision, "status": decision, "reason": decision}}


def test_brackets_use_opposite_calculated_decisions():
    campaign = Campaign()
    row = boundaries.find_column(campaign, 1000, math.inf, synthetic_evaluator)
    assert row["state"] == "bounded"
    for side, true_edge in (("lower", 1.2), ("upper", 3.3)):
        edge = row[side]
        assert edge["width_C"] <= campaign.C_tolerance
        assert (
            min(edge["inside_C"], edge["outside_C"])
            <= true_edge
            <= max(edge["inside_C"], edge["outside_C"])
        )


def test_unknown_is_never_an_external_witness():
    def evaluate(p):
        result = synthetic_evaluator(p)
        if 3.2 < p.C < 3.8:
            result["summary"].update(decision="unknown", status="numerical_failure")
        return result

    row = boundaries.find_column(Campaign(), 1000, math.inf, evaluate)
    assert row["state"] == "unresolved"
    assert row["upper"]["state"] == "unresolved"
    assert row["upper"]["inside_C"] < 3.3


def test_C_zero_is_a_physical_lower_bound():
    def evaluate(p):
        d = "yes" if p.C < 1 else "no"
        return {"summary": {"decision": d, "status": d}}

    row = boundaries.find_column(Campaign(), 800, math.inf, evaluate)
    assert row["state"] == "bounded"
    assert row["lower"]["inside_C"] == 0
    assert row["lower"]["state"] == "range_limit"


def test_shards_cover_all_columns_without_overlap():
    campaign = Campaign(m6_range=(500, 510, 5))
    seen = []
    for index in range(8):
        seen.extend(
            (ci, mi) for ci, mi, mass, case in boundaries.assigned_columns(campaign, index, 8)
        )
    assert len(seen) == len(set(seen)) == 4 * 4


def test_axis_does_not_accumulate_float_drift():
    assert axis(0, 0.3, 0.1) == (0, 0.1, 0.2, 0.3)
    with pytest.raises(ValueError):
        axis(0, 0.3, 0.2)


def test_manifest_compatibility_preserves_existing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "physics_identity", lambda: {"version": "test"})
    campaign = Campaign(m6_range=(1000, 1000, 5))
    storage.initialize(tmp_path, campaign)
    before = (tmp_path / "manifest.json").read_bytes()
    storage.initialize(tmp_path, campaign)
    with pytest.raises(ValueError, match="incompatível"):
        storage.initialize(tmp_path, replace(campaign, interior_points=7))
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_checkpoint_cache_and_retry_once(tmp_path, monkeypatch):
    calls = []

    def calculate(p, *args, **kwargs):
        calls.append(p)
        return execution.failed_record(p, "numerical_failure", "test")

    monkeypatch.setattr(execution, "bounded_evaluation", calculate)
    p = Parameters(1000, math.inf, 2)
    evaluator = execution.Evaluator(
        tmp_path, Numerics(), "nucleation", execution.Budget(max_points=1)
    )
    evaluator(p)
    evaluator(p)
    assert len(calls) == 1
    retry = execution.Evaluator(
        tmp_path, Numerics(), "nucleation", execution.Budget(max_points=1), retry_failed=True
    )
    retry(p)
    retry(p)
    assert len(calls) == 2
    assert (
        len(list((tmp_path / "attempts" / "nucleation" / storage.point_id(p)).glob("*.json"))) == 2
    )


def test_real_timeout_returns_unknown_and_reaps_child():
    result = execution.bounded_evaluation(Parameters(), Numerics(), "nucleation", timeout=0.05)
    assert result["summary"]["status"] == "point_timeout"
    assert result["summary"]["decision"] == "unknown"
    assert result["runtime_seconds"] < 10


def test_stages_have_separate_checkpoint_files(tmp_path):
    p = Parameters(1000, math.inf, 2)
    storage.save_point(tmp_path, "nucleation", p, execution.failed_record(p, "test", "n"))
    storage.save_point(tmp_path, "observables", p, execution.failed_record(p, "test", "o"))
    assert storage.point_path(tmp_path, "nucleation", p).is_file()
    assert storage.point_path(tmp_path, "observables", p).is_file()


def test_observables_use_N_interior_points_and_two_edges(tmp_path):
    campaign = Campaign(m6_range=(1000, 1000, 5), m8_values=(math.inf,), interior_points=3)
    for mi, mass in enumerate(campaign.masses()):
        row = boundaries.find_column(campaign, mass, math.inf, synthetic_evaluator)
        storage.atomic_json(boundaries.column_path(tmp_path, 0, mi), row)
    planned = observables.plan(tmp_path, campaign)
    grid = [p for p in planned["points"] if p["kind"] == "grid"]
    assert len(grid) == 2 * 5
    assert {p["fraction"] for p in grid} == {0, 0.25, 0.5, 0.75, 1}
    assert any(p["kind"] == "validation" for p in planned["points"])


def test_dry_run_creates_no_campaign(tmp_path):
    output = tmp_path / "dry"
    assert main(["init", "--preset", "full", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()


def test_archive_excludes_live_locks_and_private_temporaries(tmp_path):
    import zipfile

    output = tmp_path / "data"
    output.mkdir()
    storage.atomic_json(output / "manifest.json", {"test": True})
    storage.atomic_json(output / "points" / "observables" / "p.json", {"value": 1})
    storage.atomic_json(output / ".unfinished.json", {"value": 2})
    with storage.task_lock(output, "active"):
        archive = storage.archive_campaign(output, tmp_path / "complete.zip")
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {"manifest.json", "points/observables/p.json"}


def test_vacuum_exclusion_uses_same_physics_without_starting_worker(tmp_path, monkeypatch):
    from article_FOPT_combined.physics import evaluate

    p = Parameters(500, 400, 0)
    expected = evaluate(p, Numerics())["summary"]
    monkeypatch.setattr(
        execution,
        "bounded_evaluation",
        lambda *args, **kwargs: pytest.fail("Vácuo excluído deveria evitar o subprocesso."),
    )
    evaluator = execution.Evaluator(
        tmp_path, Numerics(), "nucleation", execution.Budget(max_points=1)
    )
    result = evaluator(p)
    assert result["summary"] == expected
    assert result["fast_vacuum_check"]
