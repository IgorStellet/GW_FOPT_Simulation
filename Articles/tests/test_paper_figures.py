"""Regressões científicas: lacunas, Tc/Tn, unidades, fontes e pontos dirigidos."""

import hashlib
import json
import math
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import asdict

import pytest

from Articles.boundary_scan import atomic_json
from Articles.boundary_search import BoundaryPlan
from Articles.collect_data import (
    M8_SCENARIOS_GEV,
    Settings,
    export_csv,
    open_database,
    point_id,
    provenance,
    save_point,
)
from Articles.combined_model import ModelParameters
from Articles.figures.__main__ import main
from Articles.figures.data import (
    critical_ratio,
    detector_metrics,
    load_dataset,
)
from Articles.figures.geometry import (
    adjacent_runs,
    detector_intervals,
    supported_triangulation,
)
from Articles.figures.points import build_plan, write_plan


def write_boundary_fixture(root, *, cases=M8_SCENARIOS_GEV, with_gw=True):
    """Dados analíticos só para testar/renderizar o estilo; não são resultados."""
    for index, case in enumerate(cases):
        folder = root / f"m8-{index}"
        folder.mkdir(parents=True)
        plan = BoundaryPlan(m8_GeV=case, m6_range_GeV=(990, 1010, 5))
        manifest = provenance(Settings(recovery_attempts=0, beta_check=True), {**plan.identity(), "stage": "boundaries", "point_timeout_seconds": 300})
        manifest["synthetic_data"] = True
        columns = []
        with closing(open_database(folder, manifest)) as db:
            for mass in (math.inf, 1010, 1005, 1000, 995, 990):
                shift = (0 if math.isinf(mass) else (1010-mass) / 150) + index * 0.08
                lower, upper = 2.0 - shift, 3.7 - shift
                for coupling, decision in ((0, "no"), (lower - 0.01, "no"), (lower, "yes"),
                                           ((lower + upper)/2, "yes"), (upper, "yes"), (upper + 0.01, "no")):
                    p = ModelParameters(m6_GeV=mass, m8_GeV=case, C=coupling)
                    row = {**asdict(p), "point_id": point_id(p), "status": "nucleated" if decision == "yes" else "no_nucleation_found",
                           "runtime_seconds": 0.01, "completed_utc": "synthetic", "V_at_v_GeV4": coupling - 4.14, "V_at_origin_GeV4": 0}
                    tr = {**asdict(p), "point_id": row["point_id"], "transition_index": 0, "trantype": 1,
                          "status": "nucleated", "nucleation_accepted": True, "quality_flags": "", "Tn_GeV": 50 + index,
                          "Tc_GeV": 80.0, "phi_low_GeV": 200.0, "phi_high_GeV": 0.0,
                          "S3_over_T": 140.1, "g_star": 106.75, "v_w": 1.0}
                    if with_gw:
                        tr.update(alpha_trace=0.2 + coupling/10 + index/10, beta_over_H=100 + coupling*20)
                    details = {"critical_transitions": [{"Tcrit": 80.0, "low_vev": [160.0], "high_vev": [0.0]}]}
                    save_point(db, folder, (row, [tr] if decision == "yes" else [], details))
                columns.append({"m6_GeV": "inf" if math.isinf(mass) else mass, "state": "bounded", "seed_C": (lower+upper)/2,
                                "lower": {"inside_C": lower, "outside_C": lower-0.01, "state": "bracketed", "outside_kind": "weak_nucleation"},
                                "upper": {"inside_C": upper, "outside_C": upper+0.01, "state": "bracketed", "outside_kind": "no_nucleation_found"}})
            export_csv(db, folder)
        atomic_json(folder / "boundary_state.json", {"state": "complete", "plan": plan.identity(), "columns": columns})
    return root


def test_Tc_ratio_uses_critical_field_and_rejects_ambiguous_match():
    tr = {"Tc_GeV": 80.0, "Tn_GeV": 40.0, "phi_low_GeV": 240.0}
    details = {"critical_transitions": [{"Tcrit": 80.0, "low_vev": [160.0]}]}
    assert critical_ratio(tr, details) == (2.0, "critical_transitions")
    details["critical_transitions"].append({"Tcrit": 80.0, "low_vev": [200.0]})
    assert critical_ratio(tr, details)[0] is None
    assert critical_ratio({**tr, "phi_low_Tc_GeV": 120.0}, details)[0] == 1.5
    assert critical_ratio({"Tn_GeV": 50}, details)[0] is None


def test_PIS_acoustic_SNR_has_year_scaling_and_Hz_conversion():
    tr = {"point_id": "a", "C": 3, "m6_GeV": 1000, "m8_GeV": math.inf, "transition_index": 0,
          "trantype": 1, "status": "nucleated", "quality_flags": "", "nucleation_accepted": True,
          "alpha_trace": 0.3, "beta_over_H": 100, "Tn_GeV": 40, "g_star": 106.75, "v_w": 1}
    one, four = detector_metrics(tr, {}, 1), detector_metrics(tr, {}, 4)
    assert 1e-5 < one["f_sw_peak_Hz"] < 0.1
    for detector in ("LISA", "DECIGO", "BBO"):
        assert four[f"{detector}_rho_sw"] == pytest.approx(2 * one[f"{detector}_rho_sw"])
    assert detector_metrics({**tr, "quality_flags": "beta_step_sensitivity"}, {}) == {}


def test_triangulation_masks_missing_values_and_large_unsampled_gaps():
    rows = [{"m6_GeV": m, "C": c, "field": v} for m, c, v in ((1000, 2, 1), (1005, 2, 2), (1000, 2.02, 3), (1005, 2.02, None), (1200, 3, 4))]
    tri, _ = supported_triangulation(rows, "field")
    for triangle, masked in zip(tri.triangles, tri.mask, strict=True):
        if 3 in triangle or 4 in triangle:
            assert masked
    assert len(adjacent_runs([{"m6_GeV": m} for m in (1000, 1005, 1020)])) == 2


def test_detector_intervals_do_not_bridge_unknowns_or_assume_monotonicity():
    def row(c, snr, valid=True):
        return {"C": c, "rho": snr, "gw_resolved": valid}
    rows = [row(1, 5), row(1.02, 15), row(1.04, 15), row(1.06, None, False), row(1.08, 20), row(1.10, 5)]
    intervals = detector_intervals(rows, "rho", 10)
    assert intervals[0][0] == pytest.approx(1.01)
    assert not any(lo < 1.06 < hi for lo, hi in intervals)
    assert len(intervals) == 3


def test_collinear_benchmarks_do_not_invent_a_two_dimensional_field():
    rows = [{"m6_GeV": 1000 + 5*i, "C": 2 + 0.02*i, "field": i} for i in range(4)]
    assert supported_triangulation(rows, "field") == (None, None)


def test_loader_and_plots_preserve_database_and_distinguish_Tc_Tn(tmp_path, monkeypatch):
    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,))
    folder = root / "m8-0"
    before = (folder / "scan.sqlite").read_bytes()
    import Articles.boundary_scan as numerical
    monkeypatch.setattr(numerical, "evaluate_bounded", lambda *_a, **_k: pytest.fail("Leitura não pode calcular pontos."))
    data = load_dataset(root)
    row = next(row for row in data.rows if row.get("gw_resolved"))
    assert row["vev_over_Tc"] == 2.0
    assert row["vev_over_Tn"] == 4.0
    assert not any(row["cooperation_verified"] for row in data.rows)
    assert (folder / "scan.sqlite").read_bytes() == before


def test_no_GW_fields_are_fabricated_from_boundary_only_data(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,), with_gw=False)
    data = load_dataset(root)
    assert data.audit["resolved_GW_points"] == 0
    assert all("alpha_trace" not in row and "LISA_rho_sw" not in row for row in data.rows)
    assert any(row.get("vev_over_Tc") == 2 for row in data.rows)
    with pytest.raises(SystemExit) as error:
        main(["render", "--boundaries", str(root), "--output", str(tmp_path / "fig"), "--require-gw"])
    assert error.value.code == 2


def test_publication_export_and_compact_bundle_are_auditable(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "figures"
    assert main(["render", "--boundaries", str(root), "--output", str(output), "--dpi", "90", "--bundle"]) == 0
    report = json.loads((output / "figure_report.json").read_text(encoding="utf-8"))
    assert report["synthetic_data"]
    assert report["assumptions"]["SNR_channel"] == "sound_waves_only_PIS"
    assert (output / "01_parameter_windows.pdf").read_bytes().startswith(b"%PDF")
    assert len(list(output.glob("*.png"))) == 14
    assert (output / "points_analysis.csv").exists()
    with zipfile.ZipFile(output / "review_bundle.zip") as archive:
        assert "figure_report.json" in archive.namelist()
        assert not any(name.endswith(("sqlite", ".gz")) for name in archive.namelist())
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_running_and_conflicting_sources_remain_explicit(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,))
    state_path = root / "m8-0/boundary_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "running"
    atomic_json(state_path, state)
    with pytest.raises(ValueError, match="execução"):
        load_dataset(root)
    state["state"] = "complete"
    atomic_json(state_path, state)
    other = write_boundary_fixture(tmp_path / "other", cases=(math.inf,))
    database = other / "m8-0/scan.sqlite"
    with closing(sqlite3.connect(database)) as db:
        record = json.loads(db.execute("SELECT record FROM points WHERE status='nucleated' LIMIT 1").fetchone()[0])
        record["status"] = "no_nucleation_found"
        with db:
            db.execute("UPDATE points SET record=?,status=? WHERE point_id=?", (json.dumps(record), record["status"], record["point_id"]))
            db.execute("DELETE FROM transitions WHERE point_id=?", (record["point_id"],))
    data = load_dataset(root, [other])
    assert len(data.audit["conflicting_coordinates"]) == 1
    conflict = next(row for row in data.rows if row["source_conflict"])
    assert not conflict["gw_resolved"] and conflict["boundary_decision"] == "unknown"


def test_plan_is_immutable_and_uses_exact_pure_measure_control(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary", with_gw=False)
    data = load_dataset(root)
    plan = build_plan(data, masses=2, samples=2, controls=True)
    assert plan["points"]
    assert all(point["m6_GeV"] == point["m8_GeV"] == "inf" for point in plan["points"] if point["purpose"] == "measure_control")
    output = tmp_path / "plan.json"
    write_plan(output, plan)
    with pytest.raises(ValueError, match="preservado"):
        write_plan(output, plan)


def test_selected_points_publish_and_resume_without_repeating_committed_points(tmp_path, monkeypatch):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,), with_gw=False)
    plan_path = tmp_path / "plan.json"
    write_plan(plan_path, build_plan(load_dataset(root), masses=1, samples=2))
    calls = []

    def evaluate(params, settings, **kwargs):
        calls.append(point_id(params))
        tr = {**asdict(params), "point_id": point_id(params), "transition_index": 0, "trantype": 1,
              "status": "nucleated", "nucleation_accepted": True, "quality_flags": "", "Tn_GeV": 50,
              "phi_low_GeV": 200, "alpha_trace": 0.3, "beta_over_H": 100, "g_star": 106.75, "v_w": 1}
        return ({**asdict(params), "point_id": point_id(params), "status": "nucleated", "runtime_seconds": 0, "completed_utc": str(len(calls))}, [tr], {})

    monkeypatch.setattr(numerical, "evaluate_bounded", evaluate)
    common = ["points", "--plan", str(plan_path), "--scenario-index", "0", "--output", str(tmp_path / "points")]
    assert main(common + ["--work-dir", str(tmp_path / "work1"), "--max-points", "1"]) == 0
    assert len(calls) == 1
    assert main(common + ["--work-dir", str(tmp_path / "work2")]) == 0
    assert len(calls) == len(set(calls)) == 2
    with closing(sqlite3.connect(tmp_path / "points/scan.sqlite")) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("SELECT count(*) FROM points").fetchone()[0] == 2


def test_explicit_point_replaces_automatic_plan_and_preserves_source(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,), with_gw=False)
    output = tmp_path / "explicit.json"
    assert main(["plan", "--boundaries", str(root), "--output", str(output),
                 "--point", "1000", "3.35", "668.740304976422"]) == 0
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["policy"] == "explicit_coordinates_only"
    assert plan["points"] == [{"m6_GeV": 1000, "m8_GeV": 668.740304976422, "C": 3.35, "purpose": "explicit_request"}]
    assert main(["points", "--plan", str(output), "--scenario-index", "2", "--dry-run",
                 "--output", str(tmp_path / "results"), "--work-dir", str(tmp_path / "local")]) == 0
    assert not (tmp_path / "results").exists()


def test_figure_subpackage_does_not_invalidate_v4_source_identity():
    sources = provenance(Settings(), {})["source_sha256"]
    assert not any(name.startswith("Articles/figures/") for name in sources)
    assert "Articles/collect_data.py" in sources
