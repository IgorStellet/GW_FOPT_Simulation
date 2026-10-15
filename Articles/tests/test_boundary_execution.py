"""Tempo real, retomada, publicação, física mínima e compatibilidade do reuso."""

import json
import math
import sys
import time
from contextlib import closing
from dataclasses import asdict

import pytest

from Articles.boundary_physics import boundary_record, classify
from Articles.bounded_evaluation import evaluate_bounded, run_with_budget
from Articles.collect_data import Settings, point_id
from Articles.combined_model import ModelParameters


def test_minimal_record_never_calls_gw_or_temperature_derivatives():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("A etapa de fronteiras não pode calcular observáveis de GW")
    tr = {"Tnuc": 100, "action": 14000, "trantype": 1, "high_vev": [0], "low_vev": [150]}
    row, details = boundary_record(tr, Forbidden(), Forbidden(), Settings())
    assert row["nucleation_accepted"] and row["vev_over_Tn"] == 1.5
    assert "beta_over_H" not in row and "alpha_trace" not in row
    assert not details["quality_flags"]


def test_weak_nucleation_is_not_converted_to_no_nucleation():
    tr = {"trantype": 1, "nucleation_accepted": True, "vev_over_Tn": 0.8}
    assert classify({"status": "nucleated"}, [tr]) == ("no", "weak_nucleation")
    assert classify({"status": "nucleated"}, [tr], "nucleation")[0] == "yes"


def test_secondary_weak_transition_cannot_hide_strong_transition():
    weak = {"trantype": 1, "nucleation_accepted": True, "vev_over_Tn": 0.2}
    strong = {**weak, "vev_over_Tn": 1.5}
    assert classify({"status": "observables_unresolved"}, [weak, strong])[0] == "yes"
    assert classify({"status": "observables_unresolved"}, [weak, {**strong, "nucleation_accepted": False}])[0] == "unknown"


def test_old_beta_warning_does_not_hide_valid_field_ratio():
    tr = {"trantype": 1, "nucleation_accepted": True, "phi_low_GeV": 150, "Tn_GeV": 100,
          "quality_flags": "beta_step_sensitivity"}
    assert classify({"status": "observables_unresolved"}, [tr])[0] == "yes"


def test_real_process_timeout_terminates_worker(tmp_path):
    started = time.monotonic()
    code, timeout, elapsed = run_with_budget([sys.executable, "-c", "import time; time.sleep(30)"],
                                             0.15, tmp_path / "log")
    assert timeout and code != 0
    assert elapsed < 4 and time.monotonic() - started < 4


def test_timeout_point_is_explicitly_unknown(tmp_path):
    row, transitions, _ = evaluate_bounded(ModelParameters(), Settings(recovery_attempts=0),
                                            timeout=0.01, scratch=tmp_path)
    assert row["status"] == "point_timeout"
    assert classify(row, transitions)[0] == "unknown"
    assert not list(tmp_path.iterdir())


def controlled_evaluator(calls, observable_failure=False):
    def evaluate(params, settings, **kwargs):
        calls.append(point_id(params))
        yes = 0.6 <= params.C <= 1.4
        row = {**asdict(params), "point_id": point_id(params), "runtime_seconds": 0,
               "status": "observables_unresolved" if yes and observable_failure else "nucleated" if yes else "no_nucleation_found"}
        records = [{**asdict(params), "point_id": point_id(params), "transition_index": 0,
                    "trantype": 1, "nucleation_accepted": True, "vev_over_Tn": 1.5,
                    "status": row["status"], "quality_flags": "beta_step_sensitivity" if observable_failure else ""}] if yes else []
        return row, records, {}
    return evaluate


def arguments(output):
    return ["--output", str(output), "--C-bounds", "0", "2", "--seed-C", "1",
            "--m6", "1000", "1000", "1"]


def test_resume_after_point_budget_does_not_repeat_committed_points(tmp_path, monkeypatch):
    from Articles import boundary_scan
    calls = []
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator(calls))
    assert boundary_scan.main([*arguments(tmp_path), "--max-points", "3"]) == 0
    assert json.loads((tmp_path / "boundary_state.json").read_text())["state"] == "paused"
    assert boundary_scan.main(arguments(tmp_path)) == 0
    assert len(calls) == len(set(calls))
    report = json.loads((tmp_path / "boundary_state.json").read_text())
    assert len(report["columns"]) == 2 and all(c["state"] == "bounded" for c in report["columns"])
    assert boundary_scan.main(arguments(tmp_path)) == 0
    assert len(calls) == len(set(calls))


def test_cluster_publishes_closed_checkpoints_and_resumes_new_scratch(tmp_path, monkeypatch):
    import sqlite3

    from Articles import boundary_scan
    calls = []
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator(calls))
    output = tmp_path / "persistent"
    boundary_scan.main([*arguments(output), "--work-dir", str(tmp_path / "scratch-a"), "--max-points", "3"])
    with closing(sqlite3.connect(output / "scan.sqlite")) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM points").fetchone()[0] == 3
    assert boundary_scan.main([*arguments(output), "--work-dir", str(tmp_path / "scratch-b")]) == 0
    assert len(calls) == len(set(calls))


def test_dry_run_creates_no_output(tmp_path):
    from Articles.boundary_scan import main
    output = tmp_path / "not-created"
    assert main([*arguments(output), "--dry-run"]) == 0
    assert not output.exists()


def test_changed_budget_does_not_silently_overwrite_campaign(tmp_path, monkeypatch):
    from Articles import boundary_scan
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator([]))
    boundary_scan.main([*arguments(tmp_path), "--max-points", "1"])
    with pytest.raises(SystemExit):
        boundary_scan.main([*arguments(tmp_path), "--point-timeout", "600", "--dry-run"])


def test_stop_file_pauses_before_new_bounce(tmp_path, monkeypatch):
    from Articles import boundary_scan
    (tmp_path / "STOP").touch()
    def forbidden(*args, **kwargs):
        raise AssertionError("STOP deve ser verificado antes do bounce")
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", forbidden)
    assert boundary_scan.main(arguments(tmp_path)) == 0
    assert json.loads((tmp_path / "boundary_state.json").read_text())["state"] == "paused"


def test_reuse_rejects_different_core_and_accepts_same_physics():
    from Articles.boundary_scan import reuse_compatible
    source = {"source_sha256": {"src/CosmoTransitions/core.py": "same", "Articles/combined_model.py": "same"},
              "versions": {"numpy": "test"}, "settings": asdict(Settings())}
    target = json.loads(json.dumps(source))
    target["settings"]["recovery_attempts"] = 0
    reuse_compatible(source, target, "boundaries")
    target["source_sha256"]["Articles/combined_model.py"] = "changed"
    with pytest.raises(ValueError, match="núcleo/modelo"):
        reuse_compatible(source, target, "boundaries")


def test_beta_failure_in_interior_does_not_report_clean_completion(tmp_path, monkeypatch):
    from Articles import boundary_scan
    boundaries = tmp_path / "boundaries"
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator([]))
    boundary_scan.main(arguments(boundaries))
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator([], observable_failure=True))
    interior = tmp_path / "interior"
    assert boundary_scan.main([*arguments(interior), "--stage", "interior", "--boundaries", str(boundaries),
                               "--interior-step", "0.4"]) == 3
    assert json.loads((interior / "interior_state.json").read_text())["state"] == "complete_with_unresolved"


def test_reuse_preserves_source_and_does_not_import_failures(tmp_path, monkeypatch):
    import hashlib

    from Articles import boundary_scan
    from Articles.collect_data import open_database, save_point

    source, output = tmp_path / "legacy", tmp_path / "new"
    source.mkdir()
    args = boundary_scan.parser().parse_args(arguments(output))
    _, _, manifest, _ = boundary_scan.prepare(args)
    origin = json.loads(json.dumps(manifest))
    origin["fingerprint"] = "legacy-fixture"
    origin["settings"]["recovery_attempts"] = 1
    origin["grid"].pop("stage")
    evaluator = controlled_evaluator([])
    with closing(open_database(source, origin)) as db:
        good = ModelParameters(C=1.0, m6_GeV=math.inf)
        save_point(db, source, evaluator(good, Settings()))
        bad = ModelParameters(C=1.1, m6_GeV=math.inf)
        save_point(db, source, ({**asdict(bad), "point_id": point_id(bad), "status": "numerical_failure"}, [], {}))
    before = hashlib.sha256((source / "scan.sqlite").read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator(calls))
    assert boundary_scan.main([*arguments(output), "--reuse", str(source), "--max-points", "1"]) == 0
    assert calls == [point_id(bad)]
    assert hashlib.sha256((source / "scan.sqlite").read_bytes()).hexdigest() == before
    from Articles.plot_results import load_campaign
    points = {p["point_id"]: p for p in load_campaign(output)["points"]}
    assert points[point_id(good)]["imported_fingerprint"] == "legacy-fixture"


def test_boundary_plots_and_status_only_use_saved_data(tmp_path, monkeypatch):
    from Articles import boundary_scan
    from Articles.boundary_status import main as status
    from Articles.plot_boundaries import main as plot

    monkeypatch.setattr(boundary_scan, "evaluate_bounded", controlled_evaluator([]))
    assert boundary_scan.main(arguments(tmp_path)) == 0
    def forbidden(*args, **kwargs):
        raise AssertionError("Gráficos/status não podem calcular física")
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", forbidden)
    assert plot(["--output", str(tmp_path)]) == status(["--output", str(tmp_path)]) == 0
    assert (tmp_path / "plots/boundary_map.png").stat().st_size > 10000
    assert (tmp_path / "plots/boundary_map.svg").exists()


def test_three_slice_launcher_preflights_all_and_preserves_review_exit(tmp_path, monkeypatch):
    from Articles import run_boundaries

    checked, started = [], []
    monkeypatch.setattr(run_boundaries.subprocess, "run", lambda command, **_: checked.append(command))
    class Child:
        def __init__(self, command, **kwargs):
            started.append(command)
            assert kwargs["cwd"] == run_boundaries.ROOT
            assert all((tmp_path / f"m8-{i}").is_dir() for i in range(3))
        def wait(self):
            return 3
    monkeypatch.setattr(run_boundaries.subprocess, "Popen", Child)
    assert run_boundaries.main(["--output", str(tmp_path), "--point-timeout", "90"]) == 3
    assert len(checked) == len(started) == 3
    assert all("--dry-run" in command for command in checked)
    assert [c[c.index("--scenario-index") + 1] for c in started] == ["0", "1", "2"]
    assert len({c[c.index("--output") + 1] for c in started}) == 3


def test_boundary_reuse_preserves_unknown_without_turning_it_into_no(tmp_path, monkeypatch):
    from Articles import boundary_scan
    from Articles.collect_data import open_database, save_point

    source, output = tmp_path / "previous", tmp_path / "next"
    source.mkdir()
    plan, settings, manifest, _ = boundary_scan.prepare(boundary_scan.parser().parse_args(arguments(output)))
    origin = json.loads(json.dumps(manifest))
    origin["fingerprint"] = "previous-boundary"
    params = ModelParameters(C=1.0, m6_GeV=math.inf)
    with closing(open_database(source, origin)) as db:
        save_point(db, source, ({**asdict(params), "point_id": point_id(params), "status": "point_timeout"}, [], {}))
    output.mkdir()
    store = boundary_scan.BoundaryStore(output, output, manifest, settings, plan, "boundaries", 300)
    assert store.import_campaign(source) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError("Pendência já salva não deve ser repetida automaticamente")
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", forbidden)
    assert store.evaluate(1.0, math.inf)["decision"] == "unknown"


def test_known_empty_mass_is_exclusion_not_pending_in_interior(tmp_path, monkeypatch):
    from Articles import boundary_scan

    base_evaluator = controlled_evaluator([])
    def evaluate(params, settings, **kwargs):
        row, transitions, details = base_evaluator(params, settings, **kwargs)
        if not math.isinf(params.m6_GeV):
            row["status"], transitions = "no_nucleation_found", []
        return row, transitions, details
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", evaluate)
    front = tmp_path / "front"
    assert boundary_scan.main([*arguments(front), "--C-bounds", "1", "1"]) == 0
    interior = tmp_path / "interior"
    assert boundary_scan.main([*arguments(interior), "--stage", "interior", "--boundaries", str(front)]) == 0
    report = json.loads((interior / "interior_state.json").read_text())
    assert report["excluded_columns"] == [1000.0] and not report["skipped_columns"]


def test_reviewed_partial_boundaries_can_use_positive_inner_interval(tmp_path, monkeypatch):
    from Articles import boundary_scan

    base_evaluator = controlled_evaluator([])
    def evaluate(params, settings, **kwargs):
        row, transitions, details = base_evaluator(params, settings, **kwargs)
        if params.C > 1.2:
            row["status"], transitions = "numerical_failure", []
        return row, transitions, details
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", evaluate)
    front = tmp_path / "front"
    assert boundary_scan.main(arguments(front)) == 3
    output = tmp_path / "interior"
    with pytest.raises(SystemExit):
        boundary_scan.main([*arguments(output), "--stage", "interior", "--boundaries", str(front)])
    monkeypatch.setattr(boundary_scan, "evaluate_bounded", base_evaluator)
    assert boundary_scan.main([*arguments(output), "--stage", "interior", "--boundaries", str(front),
                               "--allow-partial-boundaries", "--interior-step", "0.4"]) == 3
    report = json.loads((output / "interior_state.json").read_text())
    assert report["points_checked"] > 0 and len(report["partial_columns"]) == 2
