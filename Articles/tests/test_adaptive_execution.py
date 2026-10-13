"""Persistência, retomada e integração local/cluster da busca adaptativa."""

import csv
import json
import math
import sqlite3
from dataclasses import asdict

import pytest

from Articles.adaptive_scan import AdaptivePlan, UnresolvedPoint, run_adaptive
from Articles.cluster_runner import copy_campaign, restore_campaign
from Articles.collect_data import Settings, point_id


def test_batches_resume_identically_after_checkpoint_restore(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    plan = AdaptivePlan(m6_range_GeV=(500, 2000, 250), C_range=(0, 2, 0.1),
                        seed_C=1, m8_GeV=(math.inf,))
    manifest = {"fingerprint": "controlled-adaptive-test"}
    calls = []

    def evaluate(p, _):
        calls.append(point_id(p))
        yes = 0.8 <= p.C <= 1.2 and (math.isinf(p.m6_GeV) or p.m6_GeV >= 1500)
        row = {**asdict(p), "point_id": point_id(p), "status": "nucleated" if yes else "no_nucleation_found",
               "runtime_seconds": 0, "completed_utc": str(len(calls))}
        trs = [{"point_id": point_id(p), "transition_index": 0, "trantype": 1, "nucleation_accepted": True}] if yes else []
        return row, trs, {"controlled": True}

    monkeypatch.setattr(collector, "evaluate_point", evaluate)
    whole = tmp_path / "whole"
    run_adaptive(whole, plan, Settings(), manifest)
    expected_calls = calls.copy()
    calls.clear()
    local, persistent, restored = (tmp_path / name for name in ("local", "persistent", "restored"))
    assert run_adaptive(local, plan, Settings(), manifest, max_points=2) == 2
    assert json.loads((local / "adaptive_state.json").read_text())["state"] == "paused_point_budget"
    copy_campaign(local, persistent)
    restore_campaign(persistent, restored)
    while run_adaptive(restored, plan, Settings(), manifest, max_points=5):
        pass
    assert calls == expected_calls
    assert len(calls) == len(set(calls))
    assert (whole / "adaptive_state.json").read_bytes() == (restored / "adaptive_state.json").read_bytes()
    assert (whole / "matrix_m8_0.csv").read_bytes() == (restored / "matrix_m8_0.csv").read_bytes()
    with (whole / "matrix_m8_0.csv").open(newline="") as stream:
        cells = {cell for row in csv.reader(stream) for cell in row}
    assert {"yes", "no", "not_evaluated"} <= cells
    assert run_adaptive(restored, plan, Settings(), manifest) == 0


def test_unknown_point_is_committed_then_retry_repairs_it(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    plan = AdaptivePlan(m6_range_GeV=(1000, 1000, 1), C_range=(1, 1, 0.1), seed_C=1,
                        m8_GeV=(math.inf,))
    attempts = []

    def evaluate(p, _):
        attempts.append(p)
        failed = len(attempts) == 1
        row = {**asdict(p), "point_id": point_id(p), "status": "numerical_failure" if failed else "nucleated", "runtime_seconds": 0}
        tr = [] if failed else [{"point_id": point_id(p), "transition_index": 0, "trantype": 1, "nucleation_accepted": True}]
        return row, tr, {}

    monkeypatch.setattr(collector, "evaluate_point", evaluate)
    manifest = {"fingerprint": "retry-test"}
    with pytest.raises(UnresolvedPoint):
        run_adaptive(tmp_path, plan, Settings(), manifest)
    with sqlite3.connect(tmp_path / "scan.sqlite") as db:
        assert db.execute("SELECT status FROM points").fetchone()[0] == "numerical_failure"
    with pytest.raises(UnresolvedPoint):
        run_adaptive(tmp_path, plan, Settings(), manifest)
    assert len(attempts) == 1
    run_adaptive(tmp_path, plan, Settings(), manifest, retry_failed=True)
    assert len(attempts) == 3  # reparo da semente e a única massa finita
    assert json.loads((tmp_path / "adaptive_state.json").read_text())["state"] == "complete"


def test_local_launcher_preflights_then_starts_three_disjoint_slices(tmp_path, monkeypatch):
    from Articles import run_local

    checked, started = [], []
    monkeypatch.setattr(run_local.subprocess, "run", lambda command, **_: checked.append(command))

    class Child:
        def __init__(self, command):
            started.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(run_local.subprocess, "Popen", Child)
    assert run_local.main(["--smoke", "--output", str(tmp_path)]) == 0
    assert len(checked) == len(started) == 3
    assert all("--dry-run" in command for command in checked)
    assert [c[c.index("--scenario-index") + 1] for c in started] == ["0", "1", "2"]
    assert len({c[c.index("--output") + 1] for c in started}) == 3


def test_runner_publishes_a_saved_failure_before_exiting(tmp_path, monkeypatch):
    import Articles.cluster_runner as runner
    from Articles.collect_data import open_database, save_point

    def fail_after_commit(argv):
        folder = tmp_path / "scratch"
        db = open_database(folder, {"fingerprint": "saved-failure"})
        save_point(db, folder, ({"point_id": "failed", "status": "numerical_failure",
                                "completed_utc": "now"}, [], {"diagnostic": "kept"}))
        db.close()
        raise SystemExit(2)

    monkeypatch.setattr(runner, "collect_main", fail_after_commit)
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path / "persistent"),
                     "--work-dir", str(tmp_path / "scratch"), "--scenario-index", "0"])
    with sqlite3.connect(tmp_path / "persistent/scan.sqlite") as db:
        assert db.execute("SELECT status FROM points").fetchone()[0] == "numerical_failure"
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_persistent_observable_warning_is_retried_once_across_runner_batches(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    plan = AdaptivePlan(m6_range_GeV=(1000, 1000, 1), C_range=(1, 1, 0.1),
                        seed_C=1, m8_GeV=(math.inf,))
    manifest = {"fingerprint": "persistent-beta-warning"}
    calls = []

    def evaluate(p, _):
        calls.append(point_id(p))
        row = {**asdict(p), "point_id": point_id(p), "status": "observables_unresolved", "runtime_seconds": 0}
        tr = [{"point_id": point_id(p), "transition_index": 0, "trantype": 1, "nucleation_accepted": True}]
        return row, tr, {}

    monkeypatch.setattr(collector, "evaluate_point", evaluate)
    run_adaptive(tmp_path, plan, Settings(), manifest)
    calls.clear()
    for _ in range(3):
        run_adaptive(tmp_path, plan, Settings(), manifest, max_points=1,
                     retry_failed=True, retry_session="one-runner-invocation")
    assert len(calls) == len(set(calls)) == 2
    assert json.loads((tmp_path / "adaptive_state.json").read_text())["state"] == "complete"
