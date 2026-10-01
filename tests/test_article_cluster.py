"""Checkpoints mantêm dados consistentes e permitem retomada em outro nó."""

import gzip
import json
import sqlite3
from contextlib import closing

import pytest

from Articles.cluster_runner import copy_campaign, main, restore_campaign
from Articles.collect_data import open_database, save_point


def test_closed_checkpoint_restores_observables_and_ignores_incomplete_files(tmp_path):
    work, persistent, restored = (tmp_path / name for name in ("local", "shared", "new-node"))
    work.mkdir()
    database = open_database(work, {"fingerprint": "fixed-campaign"})
    row = {"point_id": "example", "status": "nucleated"}
    transition = {
        "point_id": "example", "transition_index": 0, "status": "nucleated",
        "alpha_trace": 0.3, "beta_over_H": 120.0, "omega_sw_peak_h2": 1e-11,
    }
    save_point(database, work, (row, [transition], {"observables": transition}))
    database.close()
    (work / "incomplete.json.tmp").write_text("broken")
    copy_campaign(work, persistent)
    assert not (persistent / "incomplete.json.tmp").exists()
    with closing(sqlite3.connect(persistent / "scan.sqlite")) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    restore_campaign(persistent, restored)
    with closing(open_database(restored, {"fingerprint": "fixed-campaign"})) as connection:
        saved = json.loads(connection.execute("SELECT record FROM transitions").fetchone()[0])
        assert saved == transition
    with gzip.open(restored / row["details_path"], "rt", encoding="utf-8") as stream:
        assert json.load(stream)["observables"] == transition


def test_live_wal_is_rejected_without_replacing_previous_checkpoint(tmp_path):
    work, persistent = tmp_path / "local", tmp_path / "shared"
    work.mkdir()
    persistent.mkdir()
    sentinel = persistent / "scan.sqlite"
    sentinel.write_bytes(b"previous confirmed checkpoint")
    database = open_database(work, {"fingerprint": "x"})
    try:
        with pytest.raises(RuntimeError, match="WAL"):
            copy_campaign(work, persistent)
        assert sentinel.read_bytes() == b"previous confirmed checkpoint"
    finally:
        database.close()


def test_runner_rejects_overlapping_output_and_work_paths(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path), "--work-dir", str(tmp_path / "scratch")])


def test_runner_continues_after_repair_that_does_not_increase_point_count(tmp_path, monkeypatch):
    import Articles.cluster_runner as runner

    calls = []

    def fake_collect(argv):
        calls.append(argv)
        folder = tmp_path / "local"
        with closing(open_database(folder, {"fingerprint": "repair-campaign"})) as database:
            if len(calls) <= 2:
                row = {
                    "point_id": "repaired-point", "status": "numerical_failure",
                    "completed_utc": f"2026-09-30T00:00:0{len(calls)}+00:00",
                }
                save_point(database, folder, (row, [], {"attempt": len(calls)}))
        return 0

    monkeypatch.setattr(runner, "collect_main", fake_collect)
    assert runner.main([
        "--output", str(tmp_path / "shared"), "--work-dir", str(tmp_path / "local"),
        "--smoke", "--batch-size", "1",
    ]) == 0
    assert len(calls) == 3  # O segundo lote atualizou uma linha; só o terceiro ficou vazio.
