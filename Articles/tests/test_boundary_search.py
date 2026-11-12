"""Fronteiras, saltos abruptos, incerteza e hipóteses de continuidade."""

import math

import pytest

from Articles.boundary_search import (
    BoundaryPlan,
    interior_coordinates,
    trace_boundaries,
)


def plan(**changes):
    return BoundaryPlan(**{"m6_range_GeV": (1000, 1000, 1), "C_bounds": (0, 4), "seed_C": 1.5,
                           "C_tolerance": 0.02, "column_limit": 1, **changes})


def test_brackets_boundaries_without_dense_interior():
    calls = []
    def evaluate(C, mass):
        calls.append(C)
        return "yes" if 1.13 <= C <= 2.36 else "no"
    report = trace_boundaries(plan(), evaluate)
    column = report["columns"][0]
    assert column["state"] == "bounded"
    assert len(calls) < 35  # A grade de 0.02 exigiria >60 pontos só no interior.
    for name, value in (("lower", 1.13), ("upper", 2.36)):
        edge = column[name]
        assert edge["state"] == "bracketed"
        assert edge["width_C"] <= 0.02 + 1e-10
        assert min(edge["inside_C"], edge["outside_C"]) <= value <= max(edge["inside_C"], edge["outside_C"])
        assert len(edge["confirmation_C"]) == 3


def test_abrupt_mass_change_searches_new_seed_beyond_three_negatives():
    def evaluate(C, mass):
        low, high = (3.6, 3.8) if math.isinf(mass) else (0.8, 1.1)
        return "yes" if low <= C <= high else "no"
    report = trace_boundaries(plan(seed_C=3.83, column_limit=None), evaluate)
    columns = report["columns"]
    assert len(columns) == 2
    assert columns[1]["state"] == "bounded"
    assert 0.8 <= columns[1]["seed_C"] <= 1.1
    assert len(columns[1]["seed_probes"]) > 3


def test_first_seed_falls_back_to_original_resolution_for_narrow_band():
    report = trace_boundaries(plan(C_bounds=(0, 1), seed_C=0.83),
                              lambda C, m: "yes" if 0.47 <= C <= 0.49 else "no")
    assert report["columns"][0]["seed_C"] == pytest.approx(0.49)
    assert report["columns"][0]["state"] == "bounded"


def test_unknown_midpoint_is_not_an_outside_witness():
    def evaluate(C, mass):
        return "unknown" if C == 2 else "yes" if 1 <= C < 2 else "no"
    report = trace_boundaries(plan(), evaluate)
    upper = report["columns"][0]["upper"]
    assert upper["state"] == "bracketed" and upper["outside_C"] != 2
    assert 2 not in upper["confirmation_C"]
    assert report["state"] == "pilot_complete_with_unresolved"


def test_persistent_unknown_suspends_direction_and_advances_mass():
    report = trace_boundaries(plan(column_limit=None),
                              lambda C, m: "unknown" if C >= 1.6 else "yes" if C >= 1 else "no")
    assert len(report["columns"]) == 2
    assert all(c["upper"]["state"] == "unresolved" for c in report["columns"])
    assert all(c["upper"]["outside_C"] is None for c in report["columns"])
    assert report["state"] == "complete_with_unresolved"
    assert not list(interior_coordinates(report))


def test_unknown_seed_columns_cannot_close_mass_axis():
    report = trace_boundaries(plan(C_bounds=(1.5, 1.5), m6_range_GeV=(1000, 1100, 25), column_limit=None),
                              lambda C, m: "unknown")
    assert len(report["columns"]) == 6
    assert report["m6_stop"]["reason"] == "mass_bound"
    assert all(c["state"] == "unresolved_no_seed" for c in report["columns"])


def test_three_fully_searched_empty_columns_close_mass_axis():
    report = trace_boundaries(plan(C_bounds=(1.5, 1.5), m6_range_GeV=(1000, 1100, 25), column_limit=None),
                              lambda C, m: "yes" if math.isinf(m) else "no")
    assert len(report["columns"]) == 4
    assert report["m6_stop"]["reason"] == "consecutive_empty_columns_on_seed_grid"


def test_search_bound_is_not_reported_as_physical_boundary():
    report = trace_boundaries(plan(), lambda C, m: "yes")
    column = report["columns"][0]
    assert column["lower"]["state"] == column["upper"]["state"] == "bound_limited"
    assert column["lower"]["outside_C"] is column["upper"]["outside_C"] is None
    assert column["lower"]["inside_C"] == 0 and column["upper"]["inside_C"] == 4


def test_reappearing_band_during_confirmation_requires_review():
    report = trace_boundaries(plan(), lambda C, m: "yes" if 1 <= C < 2 or 2.015 <= C <= 2.08 else "no")
    assert report["columns"][0]["upper"]["reason"] == "nonconnected_samples"
    assert report["columns"][0]["state"] == "unresolved"


def test_interior_uses_conservative_bracket_and_can_make_small_pilot():
    report = trace_boundaries(plan(), lambda C, m: "yes" if 1.13 <= C <= 2.36 else "no")
    coordinates = list(interior_coordinates(report))
    assert all(1.13 <= C <= 2.36 for C, _ in coordinates)
    assert len(list(interior_coordinates(report, samples_per_column=2))) == 2


def test_saved_brackets_avoid_new_jumps_and_unknown_cannot_supply_outside():
    known = [{"C": C, "decision": d} for C, d in ((1.0, "no"), (1.12, "no"), (1.14, "yes"),
                                                 (1.5, "yes"), (2.34, "yes"), (2.36, "unknown"), (2.35, "no"))]
    calls = []
    def evaluate(C, mass):
        calls.append(C)
        if C == 2.36:
            return "unknown"
        return "yes" if 1.13 <= C < 2.35 else "no"
    report = trace_boundaries(plan(), evaluate, hint_samples=lambda m: known)
    assert report["columns"][0]["state"] == "bounded"
    assert report["columns"][0]["upper"]["outside_C"] == 2.35
    assert 2.2 not in calls and 2.7 not in calls


def test_completed_unresolved_columns_are_not_reexplored_on_resume():
    p = plan(column_limit=None)
    first = trace_boundaries(p, lambda C, m: "unknown" if C > 1.6 else "yes" if C >= 1 else "no")
    def forbidden(*args):
        raise AssertionError("Coluna concluída deve ser retomada sem novos pontos")
    resumed = trace_boundaries(p, forbidden, completed_columns=first["columns"])
    assert resumed["columns"] == first["columns"]
    assert resumed["state"] == "complete_with_unresolved"
    assert all(1 <= C <= 1.6 for C, _ in interior_coordinates(resumed, allow_unresolved=True))
    assert list(interior_coordinates(resumed, allow_unresolved=True))


def test_known_distant_positive_beyond_negative_requires_review():
    known = [{"C": 1.5, "decision": "yes"}, {"C": 2.3, "decision": "no"}, {"C": 2.8, "decision": "yes"}]
    report = trace_boundaries(plan(), lambda C, m: "yes" if 1 <= C < 2.2 or 2.7 <= C <= 3 else "no",
                              hint_samples=lambda m: known)
    assert report["columns"][0]["upper"]["reason"] == "known_nonconnected_samples"


def test_known_later_mass_seed_is_not_pruned_after_three_empty_columns():
    p = plan(C_bounds=(1.5, 1.5), m6_range_GeV=(500, 2000, 500), column_limit=None)
    report = trace_boundaries(p, lambda C, m: "yes" if math.isinf(m) or m == 500 else "no",
                              hint_samples=lambda m: [{"C": 1.5, "decision": "yes"}] if m == 500 else [])
    assert report["columns"][-1]["m6_GeV"] == 500
    assert report["columns"][-1]["state"] == "bounded"
    assert report["mass_connectivity_warnings"]


@pytest.mark.parametrize("changes", [{"C_tolerance": 0}, {"walk_step": float("nan")},
                                     {"seed_resolution": 0.2}, {"unknown_limit": 0}, {"strong_threshold": -1}])
def test_invalid_plans_rejected(changes):
    with pytest.raises(ValueError):
        plan(**changes).validate()
