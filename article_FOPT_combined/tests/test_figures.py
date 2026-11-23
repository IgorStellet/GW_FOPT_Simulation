"""Geometria local, normalizações instrumentais e geração das seis figuras."""

import math
from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest
from article_FOPT_combined import boundaries, detectors, figures, storage
from article_FOPT_combined.__main__ import main, status
from article_FOPT_combined.config import Campaign
from article_FOPT_combined.model import Parameters
from article_FOPT_combined.observables import plan


def four_nodes(missing=False, gap=False):
    points, records = [], {}
    for mass in (700, 800):
        for fraction in (0, 1):
            identifier = f"{mass}-{fraction}"
            points.append(
                {
                    "kind": "grid",
                    "m6_GeV": mass,
                    "m8_GeV": math.inf,
                    "fraction": fraction,
                    "C": 1 + fraction,
                    "point_id": identifier,
                }
            )
            if not (missing and identifier == "800-1"):
                records[identifier] = {
                    "decision": "yes",
                    "nucleation_accepted": True,
                    "alpha_trace": 0.1,
                }
    return {"points": points, "observable_masses": [700, 750, 800] if gap else [700, 800]}, records


def test_mesh_uses_four_calculated_neighbours():
    planned, records = four_nodes()
    surface = figures.build_surface(planned, records, math.inf, "alpha_trace")
    assert len(surface.triangles) == 2
    np.testing.assert_allclose(surface.values, -1)


def test_mesh_preserves_valid_triangle_next_to_failed_point():
    planned, records = four_nodes(missing=True)
    surface = figures.build_surface(planned, records, math.inf, "alpha_trace")
    assert len(surface.triangles) == 1
    assert 3 not in surface.triangles


def test_mesh_does_not_bridge_absent_mass_column():
    planned, records = four_nodes(gap=True)
    surface = figures.build_surface(planned, records, math.inf, "alpha_trace")
    assert len(surface.triangles) == 0


def test_mesh_uses_extra_validated_vertex_on_sampled_column():
    planned, records = four_nodes()
    planned["points"].append(
        dict(planned["points"][0], kind="validation", C=1.5, fraction=0.5, point_id="check")
    )
    records["check"] = {
        **records["700-0"],
        "point_id": "check",
        "m8_GeV": math.inf,
        "m6_GeV": 700,
        "C": 1.5,
    }
    assert len(figures.build_surface(planned, records, math.inf, "alpha_trace").x) == 4
    drawn = figures.build_surface(
        planned, records, math.inf, "alpha_trace", include_validation=True
    )
    assert len(drawn.triangles) == 3
    assert 4 in drawn.triangles


def test_window_draws_upper_bracket_independently_of_lower_edge():
    from matplotlib.figure import Figure

    ax = Figure().subplots()
    columns = [
        {
            "m6_GeV": mass,
            "C_low_inside": 1.0,
            "C_high_inside": 3.0,
            "state": "unresolved",
            "lower": {"state": "unresolved"},
            "upper": {"state": "bracketed"},
        }
        for mass in (700, 800)
    ]
    figures.draw_boundary_guides(ax, columns, lambda row: row["m6_GeV"], "k")
    assert len(ax.collections) == 1
    assert any(line.get_linestyle() == "-" and np.all(line.get_ydata() == 3) for line in ax.lines)
    assert any(line.get_linestyle() == "--" and np.all(line.get_ydata() == 1) for line in ax.lines)
    columns[0]["samples"] = [{"decision": "no", "C": 2.0}]
    assert not figures.connected_witnesses(columns[0])


def test_lisa_requests_refine_both_sides_without_crossing_a_missing_point():
    planned, records = four_nodes()
    for identifier, row in records.items():
        row["LISA_rho_sw"] = 5 if identifier.endswith("-0") else 20
    records["800-0"]["LISA_rho_sw"] = 20
    records["800-1"]["LISA_rho_sw"] = 5
    requests = figures.lisa_refinement_requests(
        planned, records, (math.inf,), threshold=10, tolerance=0.1
    )
    assert [(p["m6_GeV"], p["C"]) for p in requests] == [(700, 1.5), (800, 1.5)]
    del records["700-0"]
    requests = figures.lisa_refinement_requests(
        planned, records, (math.inf,), threshold=10, tolerance=0.1
    )
    assert [(p["m6_GeV"], p["C"]) for p in requests] == [(800, 1.5)]


def test_pis_units_and_observation_time_scaling():
    f = np.array([1e-3, 1e-2])
    original = detectors.PIS["LISA"](f * 1e3)
    np.testing.assert_allclose(detectors.pis_h2(f, years=4, threshold=10), 5 * original)
    row = {"f_sw_peak_Hz": f[0], "omega_sw_peak_h2": original[0]}
    assert detectors.acoustic_snr(row, years=4) == pytest.approx(2)


def test_plis_is_finite_positive_and_scales_with_threshold():
    f = np.geomspace(1e-4, 0.1, 20)
    a = detectors.lisa_plis_h2(f, threshold=1)
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    np.testing.assert_allclose(detectors.lisa_plis_h2(f, threshold=10), 10 * a)


def test_six_figure_smoke_with_explicitly_synthetic_draft(tmp_path):
    # Estes dados são somente uma fixture de integração; nunca são uma validação física.
    output, destination = tmp_path / "campaign", tmp_path / "figures"
    campaign = Campaign(m6_range=(700, 900, 100), interior_stride=1, interior_points=1)
    storage.initialize(output, campaign)
    for ci, case in enumerate(campaign.m8_values):
        for mi, mass in enumerate(campaign.masses()):
            storage.atomic_json(
                boundaries.column_path(output, ci, mi),
                {
                    "state": "bounded",
                    "m6_GeV": mass,
                    "m8_GeV": case,
                    "C_low_inside": 1,
                    "C_high_inside": 3,
                },
            )
    planned = plan(output, campaign)
    for p in planned["points"]:
        parameters = Parameters(p["m6_GeV"], p["m8_GeV"], p["C"])
        summary = {
            "decision": "yes",
            "status": "nucleated",
            "nucleation_accepted": True,
            "observables_complete": True,
            "Tn_GeV": 80 + p["C"],
            "Tc_GeV": 100,
            "alpha_trace": 0.03 * 2 ** p["C"],
            "beta_over_H": 100 + 20 * p["C"],
            "vev_over_Tc": 0.8 + p["C"] / 3,
            "f_sw_peak_Hz": 0.003,
            "omega_sw_peak_h2": 1e-10,
            "g_star": 106.75,
            "v_w": 1,
        }
        storage.save_point(
            output,
            "observables",
            parameters,
            {
                "parameters": asdict(parameters),
                "summary": summary,
                "transitions": [],
                "data_origin": "synthetic_test_fixture",
            },
        )
    figures.render(
        Namespace(
            output=output,
            destination=destination,
            m8=None,
            years=4,
            snr=10,
            dpi=72,
            draft=True,
            replace=False,
        )
    )
    assert len(list(destination.glob("Fig*.pdf"))) == 6
    assert len(list(destination.glob("Fig*.png"))) == 6
    assert len(list(destination.glob("Supplement*.png"))) == 2
    assert (destination / "CAPTIONS.md").stat().st_size > 1000
    report = storage.read_json(destination / "figure_report.json")
    assert report["primary_figures"] == 6
    assert report["benchmark_groups"] == 2
    assert report["generation_complete"]
    assert main(["check-pilot", "--output", str(output)]) == 0
    assert all(
        case["observables_complete"] and case["observables_missing"] == 0
        for case in status(output)["cases"]
    )
