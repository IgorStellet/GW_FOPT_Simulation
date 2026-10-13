"""Teste 1: fronteiras controladas, sem o custo dos bounces reais."""

import math
from dataclasses import replace

import pytest

from Articles.adaptive_scan import (
    AdaptivePlan,
    PointBudgetReached,
    UnresolvedPoint,
    nucleation_decision,
    trace_scan,
)


def plan(**changes):
    return replace(AdaptivePlan(m6_range_GeV=(500, 1500, 100), C_range=(0, 2, 0.1),
                                seed_C=1, seed_m6_GeV=1000, m8_GeV=(math.inf,)), **changes)


def test_three_consecutive_points_in_both_mass_directions_and_three_empty_C_rows():
    calls = []

    def evaluate(p):
        calls.append(p)
        return "yes" if 0.8 <= p.C <= 1.2 and 900 <= p.m6_GeV <= 1100 else "no"

    report = trace_scan(plan(), evaluate)
    assert report["state"] == "complete"
    section = report["slices"][0]
    row = next(r for r in section["rows"] if r["C"] == 1)
    assert row["hits_m6_GeV"] == [1100, 1000, 900]
    assert row["stops"][0]["last_no_m6_GeV"] == [1200, 1300, 1400]
    assert row["stops"][1]["last_no_m6_GeV"] == [800, 700, 600]
    assert row["stops"][0]["first_skipped_m6_GeV"] == 1500
    assert row["stops"][1]["first_skipped_m6_GeV"] == 500
    stops = {s["direction"]: s for s in section["C_stops"]}
    assert stops["increasing_C"]["last_no_C"] == [1.3, 1.4, 1.5]
    assert stops["decreasing_C"]["last_no_C"] == [0.7, 0.6, 0.5]
    assert not any(p.C < 0.5 or p.C > 1.5 for p in calls)


def test_two_misses_then_a_hit_reset_the_counter():
    calls = []

    def evaluate(p):
        calls.append(p.m6_GeV)
        return "yes" if p.m6_GeV in (math.inf, 1500) else "no"

    p = plan(m6_range_GeV=(500, 2000, 250), seed_m6_GeV=math.inf, row_limit=1)
    report = trace_scan(p, evaluate)
    row = report["slices"][0]["rows"][0]
    assert row["stops"][1]["last_no_m6_GeV"] == [1250, 1000, 750]
    assert 500 not in calls
    assert report["state"] == "pilot_complete"


def test_a_nucleated_C_row_resets_the_outer_counter():
    p = plan(m6_range_GeV=(1000, 1000, 1))
    report = trace_scan(p, lambda x: "yes" if x.C in (1, 1.3) else "no")
    stop = report["slices"][0]["C_stops"][0]
    assert stop["last_no_C"] == [1.4, 1.5, 1.6]


def test_bounds_and_exact_seed_are_preserved():
    masses, cs = AdaptivePlan().axes()
    assert (cs[0], cs[-1]) == (0, 10)
    assert 3.83 in cs and 3.81 in cs and 3.85 in cs
    assert masses[0] == math.inf and masses[1] == 2000 and masses[-1] == 500
    report = trace_scan(plan(C_range=(1, 1, 0.1)), lambda _: "yes")
    assert all(s["reason"] == "bound" for s in report["slices"][0]["rows"][0]["stops"])
    assert all(s["reason"] == "bound" for s in report["slices"][0]["C_stops"])


def test_bootstrap_does_not_apply_three_C_misses_before_verifying_a_seed():
    report = trace_scan(plan(row_limit=2), lambda x: "yes" if x.C == 0.5 else "no")
    section = report["slices"][0]
    assert section["verified_seed_C"] == 0.5
    assert len(section["bootstrap"]) == 6
    assert len(section["rows"]) == 2


def test_failure_aborts_without_closing_a_boundary():
    report = {}
    with pytest.raises(UnresolvedPoint, match="Nenhuma região"):
        trace_scan(plan(), lambda x: "unknown" if x.m6_GeV == 1200 else "yes", report)
    assert report["state"] == "unresolved"
    assert not report["slices"][0]["rows"][0]["stops"]


def test_accepted_nucleation_with_unresolved_observables_is_still_a_hit():
    row = {"status": "observables_unresolved"}
    assert nucleation_decision(row, [{"trantype": 1, "nucleation_accepted": True}]) == "yes"
    assert nucleation_decision(row, [{"trantype": 1, "nucleation_accepted": False}]) == "unknown"
    assert nucleation_decision({"status": "numerical_failure"}, []) == "unknown"
    assert nucleation_decision({"status": "no_nucleation_found"}, []) == "no"


def test_missing_seed_is_explicit_and_finite():
    report = {}
    with pytest.raises(UnresolvedPoint, match="Nenhuma semente"):
        trace_scan(plan(), lambda _: "no", report)
    assert len(report["slices"][0]["bootstrap"]) == 11
    assert report["state"] == "unresolved"


def test_partial_budget_is_not_a_physical_stop():
    def evaluate(_):
        raise PointBudgetReached

    report = trace_scan(plan(), evaluate)
    assert report["state"] == "paused_point_budget"
    assert not report["slices"][0]["C_stops"]


@pytest.mark.parametrize("changes", [
    {"patience": 0}, {"C_range": (-1, 10, 0.02)}, {"C_range": (0, 11, 0.02)},
    {"C_range": (0, 10, 0)}, {"seed_m6_GeV": 1234}, {"m8_GeV": (0,)},
    {"m6_range_GeV": (0, 1000, 5)},
])
def test_invalid_bounds_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(AdaptivePlan(), **changes).axes()
