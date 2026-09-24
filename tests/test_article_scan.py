"""Checks of scientific conventions and crash-safe article collection."""

import gzip
import json
import math

import numpy as np
import pytest

from Articles.collect_data import (
    NucleationCriterion,
    Settings,
    canonical_json,
    evaluate_point,
    export_csv,
    inclusive_axis,
    open_database,
    pending_points,
    point_id,
    save_point,
    scan_points,
    single_writer,
)
from Articles.combined_model import CombinedPotential, ModelParameters
from CosmoTransitions.transitionFinder import Phase


def test_default_grid_contains_exact_endpoints_and_pure_limits():
    masses = inclusive_axis(500, 2000, 5)
    couplings = inclusive_axis(0, 10, 0.02)
    assert (len(masses), len(couplings)) == (301, 501)
    assert (masses[-1], couplings[-1]) == (2000, 10)
    points = list(scan_points((1000,), (3.2,), (668.740304976422,)))
    coordinates = {(p.m6_GeV, p.m8_GeV, p.C) for p in points}
    assert len(coordinates) == len(points) == 8
    assert (math.inf, math.inf, 3.2) in coordinates
    assert (1000, 668.740304976422, 0) in coordinates
    assert (math.inf, math.inf, 0) in coordinates
    with pytest.raises(ValueError, match="múltiplo"):
        inclusive_axis(0, 1, 0.3)


def test_nucleation_tolerance_is_on_action_not_temperature():
    criterion = NucleationCriterion(140, 0.5)
    for temperature in (10, 100):
        assert criterion(140.4 * temperature, temperature) == 0
        assert criterion(141 * temperature, temperature) == 1
        assert criterion(139 * temperature, temperature) == -1
    assert criterion.samples[100]["S3_over_T"] == 139


def test_mass_normalization_and_on_shell_tree_conditions():
    parameters = ModelParameters(m6_GeV=750, m8_GeV=840.8964152537145)
    model = CombinedPotential(parameters)
    v = parameters.v_GeV
    a, b = parameters.inverse_m6_squared, parameters.inverse_m8_fourth
    derivative = (
        -parameters.mu2_tree_GeV2 * v
        + parameters.lambda_tree * v**3
        + 0.75 * a * v**5
        + 0.5 * b * v**7
    )
    assert derivative == pytest.approx(0, abs=1e-8)
    assert model.field_masses_squared(v)["h"] == pytest.approx(parameters.mh_GeV**2)
    off = ModelParameters(m6_GeV=math.inf, m8_GeV=math.inf)
    assert off.lambda_tree == off.mh_GeV**2 / (2 * off.v_GeV**2)


def test_domain_and_vector_temperature_broadcasting():
    model = CombinedPotential(ModelParameters(C=3.2))
    assert np.isnan(model(model.domain_limit, 100))
    assert np.isnan(model(-model.domain_limit, 100))
    X, temperatures = np.array([[0.0], [100.0], [246.0]]), np.array([0, 50, 100])
    vector = model.Vtot(X, temperatures)
    scalar = np.array([model(x, t) for x, t in zip(X[:, 0], temperatures, strict=True)])
    np.testing.assert_allclose(vector, scalar)


def test_phase_interpolation_preserves_pairs_when_temperature_is_unsorted():
    temperatures = np.array([4, 1, 3, 2], dtype=float)
    phase = Phase(
        0, (2 * temperatures + 1)[:, None], temperatures, np.full((4, 1), 2.0)
    )
    np.testing.assert_allclose(phase.valAt(temperatures).ravel(), 2 * temperatures + 1)


def test_numerical_failure_is_saved_as_failure(monkeypatch):
    import Articles.collect_data as collector

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(collector, "_build_phases_and_transitions", unavailable)
    row, transitions, details = evaluate_point(ModelParameters(), Settings(n_phi=30))
    assert row["status"] == "numerical_failure"
    assert not transitions
    assert "synthetic solver failure" in details["traceback"]


def test_resume_retry_atomic_details_and_csv(tmp_path):
    manifest = {"fingerprint": "a"}
    db = open_database(tmp_path, manifest)
    params = ModelParameters()
    key = point_id(params)
    row = {"point_id": key, "status": "numerical_failure"}
    save_point(db, tmp_path, (row, [], {"attempt": 1}))
    first_path = row["details_path"]
    assert not list(pending_points(db, tmp_path, [params]))
    assert list(pending_points(db, tmp_path, [params], retry_failed=True)) == [params]
    row["status"] = "nucleated"
    save_point(
        db,
        tmp_path,
        (
            row,
            [
                {
                    "transition_index": 0,
                    "point_id": key,
                    "status": "nucleated",
                    "Tn_GeV": 100,
                }
            ],
            {"attempt": 2},
        ),
    )
    assert row["details_path"] != first_path
    with gzip.open(tmp_path / first_path, "rt", encoding="utf-8") as stream:
        assert json.load(stream)["attempt"] == 1
    export_csv(db, tmp_path)
    assert "Tn_GeV" in (tmp_path / "transitions.csv").read_text(encoding="utf-8")
    assert db.execute("SELECT count(*) FROM points").fetchone()[0] == 1
    db.close()
    with pytest.raises(ValueError, match="mudaram"):
        open_database(tmp_path, {"fingerprint": "b"})
    assert json.loads(canonical_json({"mass": math.inf, "missing": math.nan})) == {
        "mass": "inf",
        "missing": None,
    }


def test_output_lock_rejects_another_writer(tmp_path):
    with (
        single_writer(tmp_path),
        pytest.raises(RuntimeError, match="Já existe"),
        single_writer(tmp_path),
    ):
        pass
