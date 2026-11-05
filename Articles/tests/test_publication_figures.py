"""Regressões científicas do conjunto final: dados, limites e interpolação."""

import hashlib
import json
import math
import sqlite3
import zipfile
from contextlib import closing

import numpy as np
import pytest

from Articles.figures.__main__ import write_csv
from Articles.figures.data import Dataset
from Articles.figures.frontiers.data import read_table
from Articles.figures.publication.__main__ import main
from Articles.figures.publication.analysis import (
    detector_cuts,
    envelope_curves,
    field_mesh,
    fit_boundary_guides,
    match_spectra,
)
from Articles.figures.publication.inputs import (
    add_half_step_comparison,
    add_peak_metrics,
    load_received,
)
from Articles.figures.publication.supplement import build_supplement


def row(mass, coupling, case=math.inf, **extra):
    return {
        "m6_GeV": mass,
        "m8_GeV": case,
        "C": coupling,
        "point_id": f"{case}-{mass}-{coupling}",
        "nucleation_accepted": True,
        "boundary_decision": "yes",
        "source_conflict": False,
        "gw_resolved": True,
        "status": "nucleated",
        "quality_flags": "",
        "trantype": 1,
        "Tn_GeV": 50,
        "Tc_GeV": 100,
        "vev_over_Tc": 2,
        "alpha_trace": 0.2 + coupling / 100,
        "beta_over_H": 100 + coupling,
        "beta_over_H_half_step": 100 + coupling,
        "beta_order": 2,
        "beta_relative_change": 0,
        "g_star": 106.75,
        "v_w": 1,
        "source_folder": "synthetic_lisa_reach",
        "spectrum_defaults": {},
        "Lambda_GeV": 1000,
        **extra,
    }


def snapshots(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    cases = (math.inf, 840.8964152537145, 668.740304976422)
    rows = [
        row(mass, coupling, case)
        for case in cases
        for mass in (1000, 1005, 1010)
        for coupling in (2, 2.02, 2.04)
    ]
    write_csv(old / "points_analysis.csv", rows)
    write_csv(old / "all_saved_transitions.csv", rows)
    columns = [
        {
            "m8_GeV": case,
            "m6_GeV": mass,
            "state": "bounded",
            "lower_inside": 2,
            "lower_outside": 1.99,
            "lower_state": "bracketed",
            "upper_inside": 2.04,
            "upper_outside": 2.05,
            "upper_state": "bracketed",
        }
        for case in cases
        for mass in (1000, 1005, 1010)
    ]
    write_csv(old / "boundary_brackets.csv", columns)
    (old / "figure_report.json").write_text(
        json.dumps({"synthetic_data": True}), encoding="utf-8"
    )
    report = {
        "source_audit": {
            "points": len(rows),
            "synthetic_data": True,
            "sources": [],
            "conflicting_coordinates": [],
            "unverified_sources": [],
        },
        "assumptions": {"observation_years": 4, "snr_threshold": 10},
    }
    (new / "frontier_report.json").write_text(json.dumps(report), encoding="utf-8")
    write_csv(new / "edge_observables.csv", [r for r in rows if r["C"] in (2, 2.04)])
    write_csv(new / "lisa_endpoint_benchmarks.csv", [])
    (new / "lisa_profiles.json").write_text("[]", encoding="utf-8")
    full = tmp_path / "full.csv"
    write_csv(full, rows)
    return old, new, full


def test_received_zip_is_read_without_extracting_untrusted_members(tmp_path):
    old, new, full = snapshots(tmp_path)
    archive = tmp_path / "frontiers.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in new.iterdir():
            stream.write(path, path.name)
        stream.writestr("../escape.txt", "not extracted")
    dataset, profiles = load_received(old, archive, full)
    assert len(dataset.rows) == 27 and profiles == []
    assert not (tmp_path / "escape.txt").exists()
    with zipfile.ZipFile(archive) as stream:
        assert read_table(zipfile.Path(stream) / "edge_observables.csv")


def test_incomplete_or_different_full_table_is_rejected(tmp_path):
    old, new, full = snapshots(tmp_path)
    rows = read_table(full)
    write_csv(full, rows[:-1])
    with pytest.raises(ValueError, match="relatório v2"):
        load_received(old, new, full)
    rows[0]["alpha_trace"] *= 2
    write_csv(full, rows)
    with pytest.raises(ValueError, match="divergente"):
        load_received(old, new, full)


def test_old_profile_peak_never_overwrites_complete_thermodynamics(tmp_path):
    old, new, full = snapshots(tmp_path)
    first = read_table(full)[0]
    profile = [
        {
            "m8_GeV": "inf",
            "m6_GeV": 1000,
            "samples": [
                {
                    "point_id": first["point_id"],
                    "C": 2,
                    "reason": "resolved_acoustic_SNR",
                    "rho_sw": 123,
                    "decision": "yes",
                    "f_sw_peak_Hz": 0.2,
                    "omega_sw_peak_h2": 1e-6,
                }
            ],
        }
    ]
    (new / "lisa_profiles.json").write_text(json.dumps(profile), encoding="utf-8")
    dataset, profiles = load_received(old, new, full)
    result = dataset.rows[0]
    assert result["f_sw_peak_Hz"] != 0.2
    assert profiles[0]["samples"][0]["rho_sw"] == result["LISA_rho_sw"]


def test_one_sided_boundary_witness_does_not_create_a_false_dent():
    records = []
    for index in range(9):
        lower, upper = 2 + index * 0.01, 3.5 + index * 0.01
        records.append(
            {
                "m8_GeV": math.inf,
                "m6_GeV": 1000 + index * 5,
                "lower_sample_C": lower,
                "upper_sample_C": upper,
                "lower_bound_C": lower - 0.02,
                "upper_bound_C": upper + 0.02,
                "lower_C": lower,
                "upper_C": upper,
            }
        )
    records[4].update(upper_sample_C=2.8, upper_C=2.8, upper_bound_C=math.inf)
    fitted = fit_boundary_guides(records)
    assert fitted[4]["upper_C"] > 3.4
    for raw, guide in zip(records, fitted, strict=True):
        assert (
            raw["lower_bound_C"] - 1e-9
            <= guide["lower_C"]
            <= raw["lower_sample_C"] + 1e-9
        )
        assert (
            raw["upper_sample_C"] - 1e-9
            <= guide["upper_C"]
            <= raw["upper_bound_C"] + 1e-9
        )
    assert records[4]["upper_C"] == 2.8  # Fonte não alterada.


def test_envelope_never_connects_an_infinite_control_or_missing_column():
    records = [
        {"m6_GeV": m, "lower_C": 2, "upper_C": 3}
        for m in (1000, 1005, 1020, 1025, math.inf)
    ]
    curves = envelope_curves(records)
    assert len(curves) == 2
    assert curves[0][0].max() == 1005
    assert curves[1][0].min() == 1020


def test_boundary_only_alpha_does_not_paint_an_unsampled_interior():
    rows = [row(m, c) for m in (1000, 1005) for c in (2, 3)]
    tri, _, report = field_mesh(rows, "alpha_trace")
    assert tri is not None and np.all(tri.mask)
    assert report["supported_triangles"] == 0


def test_failed_beta_masks_the_mesh_but_keeps_alpha():
    rows = [row(m, c) for m in (1000, 1005) for c in (2, 2.02)]
    rows[1]["quality_flags"] = "beta_step_sensitivity"
    tri, _, _ = field_mesh(rows, "beta_over_H")
    assert all(
        masked
        for triangle, masked in zip(tri.triangles, tri.mask, strict=True)
        if 1 in triangle
    )
    _, _, alpha = field_mesh(rows, "alpha_trace")
    assert alpha["valid_nodes"] == 4


def test_detector_interpolation_respects_unknowns_and_log_SNR():
    rows = [
        row(1000, c, LISA_rho_sw=rho, DECIGO_rho_sw=rho, BBO_rho_sw=rho)
        for c, rho in ((2, 5), (2.02, 20), (2.04, None), (2.06, 20))
    ]
    profile = {
        "m6_GeV": 1000,
        "m8_GeV": "inf",
        "grid_C": [r["C"] for r in rows],
        "samples": [{"C": r["C"]} for r in rows],
    }
    intervals = detector_cuts(Dataset(rows, [], {}, [], {}), [profile])
    assert len(intervals) == 3
    assert intervals[0]["threshold_C"] == pytest.approx(2.01)
    assert not any(r["C_lower"] < 2.04 < r["C_upper"] for r in intervals)


def test_PIS_peak_metrics_preserve_units_and_time_scaling():
    first, second = {}, {}
    add_peak_metrics(first, 0.003, 1e-12, 1)
    add_peak_metrics(second, 0.003, 1e-12, 4)
    assert second["LISA_rho_sw"] == pytest.approx(2 * first["LISA_rho_sw"])
    assert first["f_sw_peak_Hz"] == 0.003


def test_accepted_beta_can_still_have_a_step_sensitive_detection_threshold():
    from Articles.figures.data import detector_metrics

    record = row(
        1000,
        3.312853459104,
        668.740304976422,
        alpha_trace=0.12751496195841275,
        Tn_GeV=52.02520450229633,
        beta_over_H=310.51526847855024,
        beta_over_H_half_step=364.0915356967048,
    )
    record.update(detector_metrics(record, {}, 4))
    dataset = Dataset([record], [], {}, [], {})
    add_half_step_comparison(dataset)
    assert record["LISA_rho_sw"] > 10 > record["LISA_rho_sw_half_step"]
    assert record["LISA_step_classification_agrees"] is False
    assert record["LISA_detectable_at_both_steps"] is False
    assert record["beta_over_H"] == 310.51526847855024


def test_spectrum_matching_does_not_claim_mass_variation_when_absent():
    rows = [
        row(1000, c, case)
        for case in (math.inf, 840.8964152537145, 668.740304976422)
        for c in (2, 2.02)
    ]
    for record in rows:
        add_peak_metrics(record, 0.003, 1e-12, 4)
        record["LISA_rho_sw_half_step"] = record["LISA_rho_sw"]
    benchmarks, _ = match_spectra(Dataset(rows, [], {}, [], {}))
    assert {r["group"] for r in benchmarks} == {"A"}


def test_render_requires_full_table_and_produces_six_figures_with_audit(tmp_path):
    old, new, full = snapshots(tmp_path)
    args = [
        "--exports",
        str(old),
        "--frontiers",
        str(new),
        "--output",
        str(tmp_path / "figures"),
    ]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert (
        main([*args, "--full-table", str(full), "--formats", "png", "--dpi", "72"]) == 0
    )
    output = tmp_path / "figures"
    assert len(list(output.glob("Fig*.png"))) == 6
    report = json.loads(
        (output / "publication_report.json").read_text(encoding="utf-8")
    )
    assert report["complete_table"] and report["source_audit"]["synthetic_data"]
    assert not report["assumptions"]["statistical_degeneracy_established"]
    assert (output / "Table01_benchmarks.tex").exists()
    with pytest.raises(SystemExit):
        main([*args, "--full-table", str(full)])  # Não sobrescreve outra edição.


def test_supplement_changes_only_the_declared_beta_maximum_step(tmp_path):
    from test_paper_figures import write_boundary_fixture

    from Articles.figures.publication.inputs import load_original

    root = write_boundary_fixture(tmp_path / "campaign")
    dataset, _ = load_original(root)
    selected = next(r for r in dataset.rows if r["boundary_decision"] == "yes")
    plan = build_supplement(dataset, [root], [selected], beta_step=1)
    assert len(plan["points"]) == 1
    assert plan["settings"]["beta_step"] == 1
    assert plan["settings"]["beta_check"] is True
    assert plan["numerics"]["policy"] == "metastable_margin_v1"
    old = plan["reference_manifest"]["settings"]
    for key, value in plan["settings"].items():
        if key not in {"beta_step", "beta_check", "recovery_attempts"}:
            assert old[key] == value
    with pytest.raises(ValueError, match="não corresponde"):
        build_supplement(dataset, [root], [{**selected, "point_id": "wrong"}])


def test_sqlite_mass_sentinels_render_all_six_figures_without_changing_inputs(tmp_path):
    """O CHE lê 'inf' do JSON/SQLite; CSV restaura float(inf) automaticamente."""
    from test_paper_figures import write_boundary_fixture

    from Articles.figures.publication.inputs import load_original
    from Articles.figures.publication.rendering import (
        PublicationWriter,
        plot_matched_spectra,
    )

    root = write_boundary_fixture(tmp_path / "campaign")
    # Dá ao fixture o mesmo diagnóstico de meio passo salvo na campanha real.
    for database in root.glob("m8-*/scan.sqlite"):
        with closing(sqlite3.connect(database)) as connection, connection:
            for point, index, payload in connection.execute(
                "SELECT point_id, transition_index, record FROM transitions"
            ).fetchall():
                record = json.loads(payload)
                record.update(
                    alpha_trace=0.3,
                    beta_over_H_half_step=record["beta_over_H"],
                    beta_order=2,
                    beta_relative_change=0,
                )
                connection.execute(
                    "UPDATE transitions SET record=? WHERE point_id=? AND transition_index=?",
                    (json.dumps(record), point, index),
                )
    hashes = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    with closing(sqlite3.connect(root / "m8-0/scan.sqlite")) as connection:
        assert any(
            json.loads(payload)["m8_GeV"] == "inf"
            for (payload,) in connection.execute("SELECT record FROM points")
        )
    dataset, _ = load_original(root)
    add_half_step_comparison(dataset)
    benchmarks, spectra = match_spectra(dataset)
    assert benchmarks  # Exercita os espectros, não apenas o painel vazio.
    args = type("Args", (), {"snr_threshold": 10, "observation_years": 4})()
    writer = PublicationWriter(tmp_path / "spectra", formats=("png",), dpi=72)
    plot_matched_spectra(benchmarks, spectra, dataset.cases, writer, args)
    assert all(
        isinstance(r["m8_GeV"], float) and isinstance(r["m6_GeV"], float)
        for r in dataset.rows
    )
    output = tmp_path / "figures"
    assert (
        main(
            [
                "--boundaries",
                str(root),
                "--output",
                str(output),
                "--formats",
                "png",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    assert len(list(output.glob("Fig*.png"))) == 6
    with zipfile.ZipFile(output / "article_figures_complete.zip") as archive:
        assert archive.testzip() is None
        assert len([name for name in archive.namelist() if name.endswith(".png")]) == 6
    assert hashes == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
