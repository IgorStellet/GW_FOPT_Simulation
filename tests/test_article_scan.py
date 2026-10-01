"""Checks of scientific conventions and crash-safe article collection."""

import gzip
import hashlib
import json
import math
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from Articles.collect_data import (
    IncompatibleCampaignError,
    NucleationCriterion,
    Settings,
    canonical_json,
    evaluate_point,
    export_csv,
    inclusive_axis,
    open_database,
    pending_points,
    point_id,
    provenance,
    save_point,
    scan_points,
    shard_points,
    single_writer,
    validate_campaign,
)
from Articles.combined_model import CombinedPotential, ModelParameters
from CosmoTransitions.transitionFinder import Phase


def test_provenance_hashes_imported_core_even_without_repository_layout(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    # Simula arquivos instalados em site-packages, sem uma pasta ROOT/src.
    core = tmp_path / "site-packages" / "CosmoTransitions"
    core.mkdir(parents=True)
    (core / "__init__.py").write_bytes(b"installed core")
    (core / "Jb_spline_v1.npz").write_bytes(b"installed thermal table")
    monkeypatch.setattr(collector.CosmoTransitions, "__file__", str(core / "__init__.py"))
    monkeypatch.setattr(collector, "ROOT", tmp_path / "absent-repository")
    manifest = provenance(Settings(), {})
    sources = manifest["source_sha256"]
    assert sources["src/CosmoTransitions/__init__.py"] == hashlib.sha256(b"installed core").hexdigest()
    assert sources["src/CosmoTransitions/Jb_spline_v1.npz"] == hashlib.sha256(b"installed thermal table").hexdigest()
    assert "Articles/combined_model.py" in sources
    assert manifest["git_commit"] is None


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


def test_campaign_mismatch_explains_changes_without_modifying_checkpoint(tmp_path):
    old = {
        "fingerprint": "old", "settings": {"beta_step": 0.5},
        "grid": {"C_range": [0, 1, 0.1]}, "versions": {"numpy": "old"},
        "python": "3.11.7", "source_sha256": {"Articles/collect_data.py": "old"},
    }
    new = {
        **old, "fingerprint": "new", "settings": {"beta_step": 1.0},
        "grid": {"C_range": [0, 2, 0.1]}, "versions": {"numpy": "new"},
        "python": "3.12.3", "source_sha256": {"Articles/collect_data.py": "new"},
    }
    with closing(open_database(tmp_path, old)) as db:
        save_point(db, tmp_path, ({"point_id": "kept", "status": "nucleated"}, [], {}))
        db.execute("PRAGMA journal_mode=DELETE")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(IncompatibleCampaignError) as error:
        open_database(tmp_path, new)
    message = str(error.value)
    for expected in ("beta_step", "C_range", "numpy", "Python", "Articles/collect_data.py", "--output"):
        assert expected in message
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    with closing(sqlite3.connect(tmp_path / "scan.sqlite")) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert db.execute("SELECT point_id FROM points").fetchone()[0] == "kept"


def test_dry_run_checks_compatibility_without_creating_or_running_campaign(tmp_path, monkeypatch, capsys):
    import Articles.collect_data as collector

    output = tmp_path / "new-campaign"
    manifest = {"fingerprint": "current"}
    monkeypatch.setattr(collector, "provenance", lambda *_: manifest)

    def unexpected_scan(*args, **kwargs):
        pytest.fail("--dry-run não deve iniciar a coleta.")

    monkeypatch.setattr(collector, "run_scan", unexpected_scan)
    assert collector.main(["--dry-run", "--output", str(output)]) == 0
    assert not output.exists()
    output.mkdir()
    with closing(open_database(output, {"fingerprint": "earlier"})):
        pass
    with pytest.raises(SystemExit) as error:
        collector.main(["--dry-run", "--output", str(output)])
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "Campanha incompatível" in diagnostic
    assert "usage:" not in diagnostic


def test_explicit_default_axes_have_same_campaign_identity(tmp_path, monkeypatch):
    import Articles.collect_data as collector

    captured = []
    real_provenance = collector.provenance

    def capture(settings, grid):
        manifest = real_provenance(settings, grid)
        captured.append(manifest["fingerprint"])
        return manifest

    monkeypatch.setattr(collector, "provenance", capture)
    common = ["--dry-run", "--output", str(tmp_path / "unused")]
    assert collector.main(common) == 0
    assert collector.main(common + ["--m6", "500", "2000", "5", "--C", "0", "10", "0.02"]) == 0
    assert captured[0] == captured[1]
    assert validate_campaign(tmp_path / "unused", {}) is None


def test_shards_cover_each_point_exactly_once_and_preserve_coordinates():
    points = list(scan_points((500, 1000), (0, 3.35), (math.inf, 668.740304976422)))
    parts = [list(shard_points(points, index, 7)) for index in range(7)]
    identities = [point_id(p) for part in parts for p in part]
    assert len(identities) == len(set(identities)) == len(points)
    assert set(identities) == {point_id(p) for p in points}
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    assert list(shard_points(points)) == points
    with pytest.raises(ValueError, match="shard-count"):
        list(shard_points(points, 2, 2))
