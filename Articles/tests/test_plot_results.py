"""Figuras usam dados salvos, com unidades corretas e sem novos bounces."""

import json
import math
from dataclasses import asdict

import numpy as np
from PIL import Image

from Articles.adaptive_scan import AdaptivePlan, export_adaptive
from Articles.collect_data import open_database, point_id, save_point
from Articles.combined_model import ModelParameters
from Articles.plot_results import (
    load_campaign,
    plot_campaign,
    plot_overview,
    reconstruct_spectrum,
    select_transitions,
)


def write_campaign(folder, m8=math.inf, only_no=False):
    folder.mkdir(parents=True)
    plan = AdaptivePlan(m6_range_GeV=(1000, 2000, 500), C_range=(3.75, 3.85, 0.02),
                        m8_GeV=(m8,), column_limit=2, C_steps_limit=3)
    manifest = {"fingerprint": "plot-fixture", "grid": plan.identity(),
                "spectrum_defaults": {"epsilon_turb": 0.05, "kappa_coll": 0.0}}
    points = [(math.inf, 3.83, "yes"), (math.inf, 3.81, "warning"), (math.inf, 3.85, "no"),
              (2000, 3.79, "yes"), (2000, 3.77, "yes"), (2000, 3.81, "no"),
              (2000, 3.83, "no"), (2000, 3.85, "no"), (2000, 3.75, "unknown")]
    with open_database(folder, manifest) as db:
        for m6, C, decision in points:
            if only_no:
                decision = "no"
            p = ModelParameters(m6_GeV=m6, m8_GeV=m8, C=C)
            key = point_id(p)
            status = {"yes": "nucleated", "warning": "observables_unresolved",
                      "no": "no_nucleation_found", "unknown": "numerical_failure"}[decision]
            row = {**asdict(p), "point_id": key, "status": status,
                   "scan_decision": "yes" if decision == "warning" else decision}
            transitions = []
            if decision in ("yes", "warning"):
                transitions = [{**asdict(p), "point_id": key, "transition_index": 0,
                                "trantype": 1, "nucleation_accepted": True, "status": status,
                                "quality_flags": "beta_step_sensitivity" if decision == "warning" else "",
                                "alpha_trace": 0.24, "beta_over_H": 205.0,
                                "Tn_GeV": 44.0, "g_star": 106.75, "v_w": 1.0}]
            save_point(db, folder, (row, transitions, {}))
        report = {"state": "unresolved", "slices": [{"m8_GeV": m8, "columns": [
            {"m6_GeV": math.inf, "verified_seed_C": 3.83, "C_stops": []},
            {"m6_GeV": 2000, "verified_seed_C": 3.79, "C_stops": [
                {"reason": "consecutive_no_nucleation", "last_no_C": [3.81, 3.83, 3.85]},
            ]},
        ]}]}
        export_adaptive(db, folder, plan, report)
    db.close()
    return folder


def test_spectrum_reconstruction_converts_Hz_to_mHz_and_keeps_efficiencies(monkeypatch):
    from Articles import plot_results

    captured = {}

    def core(f, **kwargs):
        captured.update(f=f, **kwargs)
        return {"total": np.asarray(f)}

    monkeypatch.setattr(plot_results, "gw_omega_total_h2", core)
    tr = {"alpha_trace": 0.3, "beta_over_H": 100, "Tn_GeV": 40, "g_star": 106.75, "v_w": 1}
    reconstruct_spectrum(tr, np.array([1e-3, 1]), {"epsilon_turb": 0.05, "kappa_coll": 0})
    np.testing.assert_array_equal(captured["f"], [1, 1000])
    assert captured["epsilon_turb"] == 0.05 and captured["kappa_coll"] == 0
    assert captured["alpha"] == tr["alpha_trace"] and captured["T_star"] == tr["Tn_GeV"]


def test_plots_preserve_database_and_skip_unresolved_spectra(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    folder = write_campaign(tmp_path / "campaign")
    before = (folder / "scan.sqlite").read_bytes()

    def unexpected(*_, **__):
        raise AssertionError("Gráficos não podem recalcular fases/bounces.")

    monkeypatch.setattr(collector, "evaluate_point", unexpected)
    selected = plot_campaign(folder)
    assert [t["C"] for t in selected["selected"]] == [3.77, 3.79, 3.83]
    assert any(t["reason"] == "beta_step_sensitivity" for t in selected["excluded"])
    assert (folder / "scan.sqlite").read_bytes() == before
    for name in ("nucleation_map", "gw_spectra"):
        with Image.open(folder / f"plots/{name}.png") as img:
            assert img.width > 1000 and img.height > 500
            assert np.asarray(img).std() > 10
        assert (folder / f"plots/{name}.svg").stat().st_size > 1000
    samples = (folder / "plots/spectrum_samples.csv").read_text()
    assert "f_Hz" in samples and "omega_total_h2" in samples
    assert len(samples.splitlines()) == 2401


def test_no_valid_spectrum_produces_explicit_empty_diagnostic(tmp_path):
    folder = write_campaign(tmp_path / "empty", only_no=True)
    selection = plot_campaign(folder)
    assert selection["selected"] == []
    assert len((folder / "plots/spectrum_samples.csv").read_text().splitlines()) == 1
    assert (folder / "plots/gw_spectra.png").exists()


def test_three_slices_can_be_compared_without_merging_their_databases(tmp_path):
    for index, m8 in enumerate((math.inf, 840.8964152537145, 668.740304976422)):
        write_campaign(tmp_path / f"m8-{index}", m8)
    plot_overview(tmp_path)
    with Image.open(tmp_path / "plots/nucleation_overview.png") as img:
        assert img.width > 2500
    assert not (tmp_path / "scan.sqlite").exists()


def test_selection_rejects_invalid_signed_alpha(tmp_path):
    campaign = load_campaign(write_campaign(tmp_path / "selection"))
    tr = next(t for t in campaign["transitions"] if t["status"] == "nucleated")
    selected, excluded = select_transitions([{**tr, "alpha_trace": -0.1}])
    assert not selected and excluded[0]["reason"] == "missing_or_invalid_spectrum_inputs"


def test_runner_publishes_figures_and_restores_without_new_calculations(tmp_path, monkeypatch):
    import Articles.collect_data as collector
    from Articles.cluster_runner import main

    calls = []

    def evaluate(p, _):
        calls.append(point_id(p))
        yes = p.C <= (3.83 if math.isinf(p.m6_GeV) else 3.69)
        row = {**asdict(p), "point_id": point_id(p), "status": "nucleated" if yes else "no_nucleation_found",
               "runtime_seconds": 0, "completed_utc": str(len(calls))}
        trs = [{**asdict(p), "point_id": point_id(p), "transition_index": 0, "trantype": 1,
                "nucleation_accepted": True, "status": "nucleated", "quality_flags": "",
                "alpha_trace": 0.24, "beta_over_H": 205., "Tn_GeV": 44., "g_star": 106.75, "v_w": 1.}] if yes else []
        return row, trs, {}

    monkeypatch.setattr(collector, "evaluate_point", evaluate)
    common = ["--smoke", "--scenario-index", "0", "--batch-size", "2", "--output", str(tmp_path / "persistent")]
    assert main([*common, "--work-dir", str(tmp_path / "scratch")]) == 0
    count = len(calls)
    assert count == len(set(calls))
    assert (tmp_path / "persistent/plots/nucleation_map.png").exists()
    assert (tmp_path / "persistent/plots/gw_spectra.png").exists()
    report = json.loads((tmp_path / "persistent/adaptive_state.json").read_text())
    assert report["slices"][0]["columns"][1]["verified_seed_C"] == 3.69
    assert main([*common, "--work-dir", str(tmp_path / "restored")]) == 0
    assert len(calls) == count
