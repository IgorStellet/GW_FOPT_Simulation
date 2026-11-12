"""Diagnóstico de campanha e saída explícita do runner com inconclusivos."""

import csv
import json

from Articles.status import main, summarize


def test_status_distinguishes_failure_before_any_finite_mass(tmp_path, capsys):
    (tmp_path / "adaptive_state.json").write_text(json.dumps({"state": "unresolved", "message": "bounce", "slices": []}))
    with (tmp_path / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["C", "m6_GeV", "scan_decision", "details_path"])
        writer.writeheader()
        writer.writerows([{"C": 3.83, "m6_GeV": "inf", "scan_decision": "yes"},
                          {"C": 3.19, "m6_GeV": "inf", "scan_decision": "unknown", "details_path": "details/failure.json.gz"}])
    report = summarize(tmp_path)
    assert report["points"] == 2 and report["finite_m6_columns"] == 0
    assert report["unresolved_points"][0]["C"] == "3.19"
    assert main(["--output", str(tmp_path)]) == 3
    assert "unresolved" in capsys.readouterr().out


def test_runner_publishes_review_state_before_returning_three(tmp_path, monkeypatch):
    from Articles import cluster_runner as runner
    from Articles.collect_data import open_database, save_point
    calls = []
    def collect(argv):
        calls.append(argv)
        work = tmp_path / "work"
        db = open_database(work, {"fingerprint": "review-test"})
        if len(calls) == 1:
            save_point(db, work, ({"point_id": "unknown", "status": "numerical_failure", "completed_utc": "now"}, [], {}))
        db.close()
        (work / "adaptive_state.json").write_text(json.dumps({"state": "complete_with_unresolved", "unresolved_points": [{}]}))
        return 0
    monkeypatch.setattr(runner, "collect_main", collect)
    assert runner.main(["--output", str(tmp_path / "output"), "--work-dir", str(tmp_path / "work")]) == 3
    assert (tmp_path / "output/scan.sqlite").exists()
    assert json.loads((tmp_path / "output/adaptive_state.json").read_text())["state"] == "complete_with_unresolved"
