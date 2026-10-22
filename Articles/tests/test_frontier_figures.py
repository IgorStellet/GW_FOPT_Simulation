"""Regressões das bordas: incerteza, observáveis parciais e alcance em C."""

import hashlib
import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_paper_figures import write_boundary_fixture

from Articles.collect_data import point_id
from Articles.figures.__main__ import write_csv
from Articles.figures.data import Dataset, coordinate
from Articles.figures.frontiers.__main__ import main
from Articles.figures.frontiers.data import (
    alpha_valid,
    beta_valid,
    edge_samples,
    load_exports,
    load_sources,
    read_table,
    strength_crossings,
    valid_runs,
)
from Articles.figures.frontiers.planning import (
    build_edge_plan,
    build_lisa_plan,
    partition,
)
from Articles.figures.frontiers.reach import observation, trace_cut
from Articles.figures.frontiers.render import (
    draw_edge_bands,
    interior_edge_violations,
    reach_profiles,
)
from Articles.figures.geometry import bracket_columns
from Articles.figures.points import write_plan
from Articles.figures.render import FigureWriter


def simple_dataset(rows, columns=None):
    return Dataset(rows, [], columns or {}, [], {})


def record(C, *, decision="yes", ratio=2.0, **kwargs):
    return {
        "C": C,
        "m6_GeV": 1000.0,
        "m8_GeV": math.inf,
        "point_id": str(C),
        "boundary_decision": decision,
        "nucleation_accepted": True,
        "vev_over_Tn": ratio,
        "status": "nucleated",
        "source_conflict": False,
        **kwargs,
    }


def fake_physics(params, settings, **kwargs):
    tr = {
        **asdict(params),
        "point_id": point_id(params),
        "transition_index": 0,
        "trantype": 1,
        "status": "nucleated",
        "nucleation_accepted": True,
        "quality_flags": "",
        "Tn_GeV": 50,
        "phi_low_GeV": 200,
        "phi_high_GeV": 0,
        "alpha_trace": 0.3,
        "beta_over_H": 100,
        "g_star": 106.75,
        "v_w": 1.0,
        "beta_over_H_half_step": 100.0,
        "beta_order": 2,
    }
    return (
        {
            **asdict(params),
            "point_id": point_id(params),
            "status": "nucleated",
            "runtime_seconds": 0.0,
            "completed_utc": "synthetic",
        },
        [tr],
        {},
    )


def test_edge_extremes_use_all_positive_samples_without_certifying_pending_columns():
    rows = [
        record(1.99, decision="no"),
        record(2),
        record(2.99, decision="no"),
        record(3),
        record(3.5),
        record(3.51, decision="no"),
    ]
    columns = {
        math.inf: [
            {
                "m6_GeV": 1000.0,
                "state": "unresolved",
                "lower": {
                    "inside_C": 3.0,
                    "outside_C": 2.99,
                    "state": "unresolved",
                    "reason": "known_nonconnected_samples",
                },
                "upper": {"inside_C": 3.5, "outside_C": 3.51, "state": "bracketed"},
            }
        ]
    }
    low, high = edge_samples(simple_dataset(rows, columns))
    assert low["C"] == 2 and low["edge_outside_C"] == 1.99 and not low["edge_certified"]
    assert high["C"] == 3.5 and high["edge_certified"]


def test_unit_strength_reference_has_multiple_branches_and_respects_holes():
    data = simple_dataset(
        [record(0, ratio=0.9), record(0.02, ratio=1.1), record(0.04, ratio=0.9)]
    )
    crosses = strength_crossings(data)
    assert [row["C"] for row in crosses] == pytest.approx([0.01, 0.03])
    data.rows.insert(1, record(0.01, ratio=None, decision="unknown"))
    assert [row["C"] for row in strength_crossings(data)] == pytest.approx([0.03])
    data.rows[-1]["high_phase"] = 7
    assert not strength_crossings(data)


def test_partial_alpha_survives_a_failed_beta_without_validating_beta():
    row = record(
        3, alpha_trace=0.2, beta_over_H=500, quality_flags="beta_step_sensitivity"
    )
    assert alpha_valid(row)
    assert not beta_valid(row)
    assert not alpha_valid({**row, "nucleation_accepted": False})


def test_valid_curve_does_not_skip_over_a_failed_close_sample():
    rows = [
        {"C": c, "valid": valid}
        for c, valid in ((1, True), (1.01, False), (1.02, True), (1.03, True))
    ]
    runs = valid_runs(rows, lambda row: row["valid"], 0.05)
    assert [[row["C"] for row in run] for run in runs] == [[1], [1.02, 1.03]]


def test_edges_are_not_assumed_to_bound_observable_extrema():
    rows = [
        record(2, alpha_trace=0.1),
        record(2.5, alpha_trace=0.5),
        record(3, alpha_trace=0.2),
    ]
    data = simple_dataset(rows)
    violations = interior_edge_violations(data, edge_samples(data))
    assert len(violations) == 1 and violations[0]["field"] == "alpha_trace"


def test_edge_plan_is_exact_and_partitioned_without_duplication(tmp_path):
    root = write_boundary_fixture(tmp_path / "boundary", with_gw=False)
    data = load_sources(root, details=False)
    plan = build_edge_plan(data, [root])
    expected = {coordinate(row) for row in edge_samples(data)}
    assert {coordinate(row) for row in plan["points"]} == expected
    all_parts = [partition(plan["points"], index, 4)[2] for index in range(12)]
    coordinates = [coordinate(row) for part in all_parts for row in part]
    assert len(coordinates) == len(set(coordinates)) == len(expected)
    assert plan["settings"]["beta_check"] and not plan["settings"]["recovery_attempts"]
    assert "not_alpha_beta_global_extrema" in plan["policy"]
    with pytest.raises(ValueError):
        partition(plan["points"], 12, 4)


def test_lisa_plan_contains_the_true_pure_measure_slice_and_declared_resolution(
    tmp_path,
):
    root = write_boundary_fixture(tmp_path / "boundary", with_gw=False)
    plan = build_lisa_plan(
        load_sources(root, details=False),
        [root],
        benchmark_masses=(1000.0,),
        C_step=0.1,
        C_tolerance=0.01,
    )
    pure = [
        row
        for row in plan["cuts"]
        if float(row["m6_GeV"]) == float(row["m8_GeV"]) == math.inf
    ]
    assert len(pure) == 1
    assert max(np.diff(pure[0]["grid_C"])) <= 0.1 + 1e-9
    assert plan["observation_years"] == 4 and plan["snr_threshold"] == 10
    with pytest.raises(ValueError):
        build_lisa_plan(
            load_sources(root, details=False), [root], C_step=0.01, C_tolerance=0.1
        )


def test_lisa_search_retains_nonmonotonic_intervals_and_refines_all_edges():
    cut = {"m6_GeV": 1000, "m8_GeV": math.inf, "grid_C": np.linspace(0, 1, 11).tolist()}
    calls = []

    def evaluate(C):
        calls.append(C)
        return {
            "decision": "yes" if 0.2 <= C <= 0.35 or 0.6 <= C <= 0.8 else "no",
            "rho_sw": 20,
        }

    result = trace_cut(cut, evaluate, 0.005)
    assert result["state"] == "complete"
    assert len(result["intervals"]) == 2
    assert len(result["brackets"]) == 4
    assert all(row["C_max"] - row["C_min"] <= 0.005 for row in result["brackets"])
    assert len(calls) == len(set(calls))


def test_lisa_unknown_sample_cannot_turn_into_a_threshold_crossing():
    result = trace_cut(
        {"grid_C": [0, 0.5, 1]},
        lambda C: {"decision": "unknown" if C == 0.5 else "yes" if C > 0.5 else "no"},
        0.01,
    )
    assert result["state"] == "complete_with_unresolved"
    assert not result["brackets"]
    assert not any(low < 0.5 < high for low, high in result["intervals"])


def test_missing_beta_is_unknown_LISA_not_zero_SNR():
    from Articles.boundary_search import BoundaryPlan
    from Articles.combined_model import ModelParameters

    p = ModelParameters(C=3, m6_GeV=1000)
    row, transitions, _ = fake_physics(p, None)
    transitions[0].update(
        status="observables_unresolved", quality_flags="beta_step_sensitivity"
    )
    row.update(status="observables_unresolved", boundary_decision="yes")
    store = SimpleNamespace(
        plan=BoundaryPlan(),
        cache={point_id(p): (row, transitions)},
        manifest={},
        evaluate=lambda *_: None,
    )
    result = observation(store, 1000, 3, 4, 10)
    assert result["decision"] == "unknown" and result["rho_sw"] is None


def test_edge_execution_publishes_and_resumes_without_repeating_points(
    tmp_path, monkeypatch
):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(
        tmp_path / "boundary", cases=(math.inf,), with_gw=False
    )
    plan = build_edge_plan(load_sources(root, details=False), [root])
    path = tmp_path / "edges.json"
    write_plan(path, plan)
    calls = []

    def evaluate(params, settings, **kwargs):
        calls.append(point_id(params))
        return fake_physics(params, settings, **kwargs)

    monkeypatch.setattr(numerical, "evaluate_bounded", evaluate)
    common = [
        "run-edges",
        "--plan",
        str(path),
        "--output",
        str(tmp_path / "new"),
        "--shards-per-scenario",
        "1",
    ]
    assert (
        main(common + ["--work-dir", str(tmp_path / "local1"), "--max-points", "1"])
        == 0
    )
    assert len(calls) == 1
    assert main(common + ["--work-dir", str(tmp_path / "local2")]) == 0
    assert len(calls) == len(set(calls)) == len(plan["points"])
    state = json.loads((tmp_path / "new/m8-0-part-0/interior_state.json").read_text())
    assert state["state"] == "complete"


def test_explicit_retry_includes_observables_pending_with_accepted_nucleation(
    tmp_path, monkeypatch
):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(
        tmp_path / "boundary", cases=(math.inf,), with_gw=False
    )
    plan = build_edge_plan(load_sources(root, details=False), [root])
    plan["points"] = plan["points"][:1]
    path = tmp_path / "edges.json"
    write_plan(path, plan)
    calls = []

    def evaluate(params, settings, **kwargs):
        result = fake_physics(params, settings, **kwargs)
        calls.append(params.C)
        if len(calls) == 1:
            result[0]["status"] = result[1][0]["status"] = "observables_unresolved"
            result[1][0]["quality_flags"] = "beta_step_sensitivity"
        return result

    monkeypatch.setattr(numerical, "evaluate_bounded", evaluate)
    common = [
        "run-edges",
        "--plan",
        str(path),
        "--output",
        str(tmp_path / "new"),
        "--shards-per-scenario",
        "1",
    ]
    assert main(common + ["--work-dir", str(tmp_path / "local1")]) == 3
    assert (
        main(common + ["--work-dir", str(tmp_path / "local2"), "--retry-failed"]) == 0
    )
    assert len(calls) == 2


def test_lisa_execution_resumes_its_mesh_then_publishes_a_complete_cut(
    tmp_path, monkeypatch
):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(
        tmp_path / "boundary", cases=(math.inf,), with_gw=False
    )
    plan = build_lisa_plan(
        load_sources(root, details=False),
        [root],
        benchmark_masses=(1000.0,),
        C_step=0.85,
        C_tolerance=0.45,
    )
    plan["cuts"] = plan["cuts"][:1]
    path = tmp_path / "lisa.json"
    write_plan(path, plan)
    calls = []

    def evaluate(params, settings, **kwargs):
        calls.append(point_id(params))
        return fake_physics(params, settings, **kwargs)

    monkeypatch.setattr(numerical, "evaluate_bounded", evaluate)
    common = [
        "run-lisa",
        "--plan",
        str(path),
        "--output",
        str(tmp_path / "new"),
        "--shards-per-scenario",
        "1",
    ]
    assert (
        main(common + ["--work-dir", str(tmp_path / "local1"), "--max-points", "1"])
        == 0
    )
    assert main(common + ["--work-dir", str(tmp_path / "local2")]) == 0
    assert len(calls) == len(set(calls))
    state = json.loads((tmp_path / "new/m8-0-part-0/interior_state.json").read_text())
    assert state["state"] == "complete" and state["cuts"][0]["intervals"]


def test_thin_edge_bands_have_no_vertical_connection_between_lower_and_upper(tmp_path):
    data = simple_dataset(
        [record(c, m6_GeV=mass) for mass in (1000, 1005) for c in (2, 3.5)]
    )
    edges = edge_samples(data)
    for row in edges:
        row["edge_outside_C"] = (
            row["C"] - 0.01 if row["edge"] == "lower" else row["C"] + 0.01
        )
    writer = FigureWriter(tmp_path)
    fig, ax = writer.plt.subplots()
    draw_edge_bands(ax, edges)
    assert all(np.ptp(line.get_ydata()) < 0.1 for line in ax.lines)
    assert all(
        np.ptp(path.vertices[:, 1]) < 0.1
        for collection in ax.collections
        for path in collection.get_paths()
    )
    writer.plt.close(fig)


def test_export_reader_preserves_false_and_missing_values(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        "point_id,gw_resolved,alpha_trace,C,m6_GeV\n001,False,,0,inf\n",
        encoding="utf-8",
    )
    row = read_table(path)[0]
    assert row["point_id"] == "001" and row["gw_resolved"] is False
    assert row["alpha_trace"] is None and row["C"] == 0 and math.isinf(row["m6_GeV"])


def test_margin_step_calls_original_recorder_and_preserves_settings(monkeypatch):
    from Articles.collect_data import Settings
    from Articles.figures.frontiers import worker

    calls = []

    def recorder(transition, phases, derivatives, settings):
        calls.append(settings.beta_step)
        return {"beta_step_GeV": settings.beta_step}, {}

    monkeypatch.setattr(worker, "transition_record", recorder)
    s = Settings()
    phases = {key: SimpleNamespace(T=np.array([1.0, 130.0])) for key in (0, 1)}
    transition = {
        "trantype": 1,
        "Tnuc": 121.2,
        "high_phase": 0,
        "low_phase": 1,
        "crit_trans": {"Tcrit": 121.7},
    }
    numerics = {"margin_fraction": 0.1, "minimum_step_GeV": 0.005}
    row, _ = worker.margin_record(transition, phases, None, s, numerics=numerics)
    assert calls == pytest.approx([0.05])
    assert row["beta_step_policy"] == "metastable_margin_v1" and s.beta_step == 0.5
    transition.update(Tnuc=50, crit_trans={"Tcrit": 100})
    worker.margin_record(transition, phases, None, s, numerics=numerics)
    assert calls[-1] == 0.5


@pytest.mark.parametrize("timed_out", [False, True])
def test_adaptive_worker_budget_preserves_failures_as_data(
    tmp_path, monkeypatch, timed_out
):
    from Articles.collect_data import Settings, json_safe
    from Articles.combined_model import ModelParameters
    from Articles.figures.frontiers import execution

    def budget(command, timeout, log):
        log.write_text("bounded worker", encoding="utf-8")
        assert "Articles.figures.frontiers.worker" in command and timeout == 3
        if not timed_out:
            result = command[command.index("--result") + 1]
            from pathlib import Path

            Path(result).write_text(
                json.dumps(json_safe(fake_physics(ModelParameters(C=3), Settings()))),
                encoding="utf-8",
            )
        return (1 if timed_out else 0), timed_out, 3.0

    monkeypatch.setattr(execution, "run_with_budget", budget)
    row, transitions, details = execution.evaluate_observables_bounded(
        ModelParameters(C=3),
        Settings(),
        numerics={"policy": "metastable_margin_v1"},
        timeout=3,
        scratch=tmp_path,
    )
    assert row["status"] == ("point_timeout" if timed_out else "nucleated")
    assert row["point_timeout_seconds"] == 3
    if timed_out:
        assert not transitions and "worker_log_tail" in details


def test_lisa_imports_only_precomputed_GW_coordinates_needed_by_this_cut(
    tmp_path, monkeypatch
):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(tmp_path / "boundary", cases=(math.inf,))
    data = load_sources(root, details=False)
    plan = build_lisa_plan(
        data, [root], benchmark_masses=(1000.0,), C_step=0.85, C_tolerance=0.45
    )
    plan["cuts"] = plan["cuts"][:1]
    path = tmp_path / "lisa.json"
    write_plan(path, plan)
    calls = []

    def evaluate(params, settings, **kwargs):
        calls.append(coordinate(asdict(params)))
        return fake_physics(params, settings, **kwargs)

    monkeypatch.setattr(numerical, "evaluate_bounded", evaluate)
    assert (
        main(
            [
                "run-lisa",
                "--plan",
                str(path),
                "--output",
                str(tmp_path / "new"),
                "--shards-per-scenario",
                "1",
                "--work-dir",
                str(tmp_path / "local"),
            ]
        )
        == 0
    )
    old_GW = {coordinate(row) for row in data.rows if row.get("gw_resolved")}
    assert calls and not set(calls) & old_GW


def test_reach_report_cannot_override_a_conflict_in_merged_sources(tmp_path):
    folder = tmp_path / "reach/m8-0-part-0"
    folder.mkdir(parents=True)
    cut = {
        "m6_GeV": 1000.0,
        "m8_GeV": "inf",
        "state": "complete",
        "samples": [
            {"C": 0.1, "decision": "yes", "rho_sw": 20},
            {"C": 0.2, "decision": "yes", "rho_sw": 30},
        ],
    }
    (folder / "interior_state.json").write_text(
        json.dumps(
            {
                "kind": "lisa_cuts",
                "cuts": [cut],
                "observation_years": 4,
                "snr_threshold": 10,
                "C_step": 0.1,
                "C_tolerance": 0.01,
            }
        ),
        encoding="utf-8",
    )
    data = simple_dataset([record(0.1, source_conflict=True), record(0.2)])
    args = SimpleNamespace(
        observation_years=4, snr_threshold=10, C_tolerance=0.01, maximum_C_gap=0.1
    )
    profile = reach_profiles(data, [tmp_path / "reach"], args)[math.inf, 1000.0]
    assert profile["state"] == "complete_with_unresolved"
    assert not any(low < 0.15 < high for low, high in profile["intervals"])


def test_seven_publication_figures_from_exports_leave_v1_inputs_untouched(
    tmp_path, monkeypatch
):
    import Articles.boundary_scan as numerical

    root = write_boundary_fixture(tmp_path / "boundary")
    data = load_sources(root)
    exports = tmp_path / "v1"
    exports.mkdir()
    write_csv(exports / "points_analysis.csv", data.rows)
    write_csv(exports / "all_saved_transitions.csv", data.transitions)
    write_csv(
        exports / "boundary_brackets.csv",
        [
            {"m8_GeV": case, **column}
            for case, columns in data.columns.items()
            for column in bracket_columns(columns)
        ],
    )
    (exports / "figure_report.json").write_text(
        json.dumps(data.audit), encoding="utf-8"
    )
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exports.iterdir()
    }
    loaded = load_exports(exports)
    assert len(loaded.rows) == len(data.rows)
    monkeypatch.setattr(
        numerical,
        "evaluate_bounded",
        lambda *_a, **_k: pytest.fail("Desenhar não pode calcular pontos."),
    )
    output = tmp_path / "v2"
    assert (
        main(
            [
                "render",
                "--exports",
                str(exports),
                "--output",
                str(output),
                "--benchmark-masses",
                "1000",
                "1010",
                "--dpi",
                "80",
                "--bundle",
            ]
        )
        == 0
    )
    report = json.loads((output / "frontier_report.json").read_text(encoding="utf-8"))
    assert (
        len(list(output.glob("*.pdf")))
        == len(list(output.glob("*.png")))
        == len(list(output.glob("*.svg")))
        == 7
    )
    assert not report["assumptions"]["edge_values_are_global_extrema"]
    assert not report["assumptions"]["LISA_threshold_uses_global_noise_minimum"]
    assert (output / "review_bundle_frontiers.zip").is_file()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exports.iterdir()
    }
