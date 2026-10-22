"""Busca de sementes por massa antes de qualquer poda em C."""

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


def test_three_C_misses_only_after_verified_seed_and_three_exhausted_mass_columns():
    calls = []

    def evaluate(p):
        calls.append(p)
        return "yes" if 0.8 <= p.C <= 1.2 and 900 <= p.m6_GeV <= 1100 else "no"

    report = trace_scan(plan(), evaluate)
    assert report["state"] == "complete"
    section = report["slices"][0]
    col = next(c for c in section["columns"] if c["m6_GeV"] == 1000)
    assert col["verified_seed_C"] == 1
    assert col["C_stops"][0]["last_no_C"] == [1.3, 1.4, 1.5]
    assert col["C_stops"][1]["last_no_C"] == [0.7, 0.6, 0.5]
    stops = {s["direction"]: s for s in section["m6_stops"]}
    assert stops["larger_m6"]["last_no_m6_GeV"] == [1200, 1300, 1400]
    assert stops["smaller_m6"]["last_no_m6_GeV"] == [800, 700, 600]
    assert stops["larger_m6"]["first_skipped_m6_GeV"] == 1500
    assert stops["smaller_m6"]["first_skipped_m6_GeV"] == 500
    for c in section["columns"]:
        if c["state"] == "no_nucleation_on_grid":
            assert c["C_search_complete"]
            assert len(c["bootstrap"]) == len(plan().axes()[1])
    assert len(calls) == len({(p.C, p.m6_GeV) for p in calls})


def test_first_finite_mass_can_require_more_than_three_C_misses():
    p = plan(m6_range_GeV=(1500, 2000, 500), seed_m6_GeV=math.inf, column_limit=2)
    calls = []

    def evaluate(x):
        calls.append(x)
        return "yes" if x.C == (1 if math.isinf(x.m6_GeV) else 0.3) else "no"

    report = trace_scan(p, evaluate)
    finite = report["slices"][0]["columns"][1]
    assert finite["m6_GeV"] == 2000
    assert finite["verified_seed_C"] == 0.3
    assert [a["decision"] for a in finite["bootstrap"]] == ["no"] * 7 + ["yes"]
    # Os C até 1.0 já foram calculados no bootstrap, apesar de a expansão
    # depois da semente encerrar após 0.4, 0.5, 0.6 sem nucleação.
    assert finite["C_stops"][0]["first_after_walk_C"] == 0.7
    assert finite["C_stops"][0]["first_skipped_C"] == 1.1
    assert not any(x.m6_GeV == 1500 for x in calls)


def test_abrupt_shift_between_two_finite_masses_is_reseeded():
    p = plan(m6_range_GeV=(1500, 2000, 500), seed_m6_GeV=2000)
    report = trace_scan(p, lambda x: "yes" if x.C == (0.2 if x.m6_GeV == 1500 else 1) else "no")
    col = next(c for c in report["slices"][0]["columns"] if c["m6_GeV"] == 1500)
    assert col["verified_seed_C"] == 0.2
    assert len(col["bootstrap"]) == 9


def test_upward_fallback_does_not_assume_monotonicity_between_masses():
    p = plan(m6_range_GeV=(1500, 2000, 500), seed_m6_GeV=math.inf, column_limit=2)
    report = trace_scan(p, lambda x: "yes" if x.C == (1 if math.isinf(x.m6_GeV) else 1.8) else "no")
    col = report["slices"][0]["columns"][1]
    assert col["verified_seed_C"] == 1.8
    assert col["bootstrap"][-1]["direction"] == "increasing_C_fallback"
    assert sum(s["direction"] == "decreasing_C" for s in col["bootstrap"]) == 11


def test_each_m8_verifies_its_own_seed_after_any_number_of_misses():
    p = plan(seed_m6_GeV=math.inf, m8_GeV=(math.inf, 840.8964152537145), column_limit=1)
    report = trace_scan(p, lambda x: "yes" if x.C == (1 if math.isinf(x.m8_GeV) else 0.3) else "no")
    assert [s["columns"][0]["verified_seed_C"] for s in report["slices"]] == [1, 0.3]


def test_two_C_misses_then_a_hit_reset_the_counter():
    report = trace_scan(plan(column_limit=1), lambda x: "yes" if x.C in (1, 1.3) else "no")
    stop = report["slices"][0]["columns"][0]["C_stops"][0]
    assert stop["last_no_C"] == [1.4, 1.5, 1.6]
    assert stop["first_skipped_C"] == 1.7


def test_a_nucleated_mass_column_resets_the_outer_counter():
    report = trace_scan(plan(), lambda x: "yes" if x.m6_GeV in (1000, 1300) and x.C == 1 else "no")
    stop = report["slices"][0]["m6_stops"][0]
    assert stop["last_no_m6_GeV"] == [1400, 1500, math.inf]
    assert stop["reason"] == "consecutive_empty_columns"


def test_initial_empty_columns_do_not_trigger_pruning_before_any_seed():
    p = plan(m6_range_GeV=(1000, 2000, 250), seed_m6_GeV=math.inf)
    report = trace_scan(p, lambda x: "yes" if x.m6_GeV == 1000 and x.C == 0.5 else "no")
    assert report["slices"][0]["columns"][-1]["verified_seed_C"] == 0.5
    assert len(report["slices"][0]["columns"]) == 6


def test_bounds_and_exact_seed_are_preserved():
    masses, cs = AdaptivePlan().axes()
    assert (cs[0], cs[-1]) == (0, 10)
    assert 3.83 in cs and 3.81 in cs and 3.85 in cs
    assert 1.0 in cs and 3.2 in cs  # Cortes exatos de referência da dissertação.
    assert masses[0] == math.inf and masses[1] == 2000 and masses[-1] == 500
    report = trace_scan(plan(C_range=(1, 1, 0.1)), lambda _: "yes")
    assert all(s["reason"] == "bound" for s in report["slices"][0]["columns"][0]["C_stops"])
    assert all(s["reason"] == "bound" for s in report["slices"][0]["m6_stops"])


def test_pilot_has_two_columns_and_does_not_truncate_seed_search():
    p = plan(m6_range_GeV=(1500, 2000, 500), seed_m6_GeV=math.inf, column_limit=2, C_steps_limit=3)
    report = trace_scan(p, lambda x: "yes" if x.C <= (1 if math.isinf(x.m6_GeV) else 0.3) else "no")
    section = report["slices"][0]
    assert len(section["columns"]) == 2
    assert section["columns"][1]["verified_seed_C"] == 0.3
    assert len(section["columns"][1]["bootstrap"]) == 8
    assert section["columns"][0]["C_stops"][1]["reason"] == "pilot_step_limit"
    assert report["state"] == "pilot_complete"


def test_failure_aborts_without_closing_a_boundary():
    report = {}
    with pytest.raises(UnresolvedPoint, match="Nenhuma região"):
        trace_scan(plan(on_unresolved="stop"), lambda x: "unknown" if x.C == 1.2 else "yes", report)
    assert report["state"] == "unresolved"
    assert not report["slices"][0]["columns"][0]["C_stops"]


def test_unknown_does_not_stop_later_C_or_finite_masses():
    p = plan(m6_range_GeV=(500, 2000, 500), seed_m6_GeV=math.inf)
    def evaluate(x):
        if math.isinf(x.m6_GeV):
            return "unknown" if x.C == 0.7 else "yes" if x.C == 1 else "no"
        return "yes" if x.C == 0.2 else "no"
    report = trace_scan(p, evaluate)
    assert report["state"] == "complete_with_unresolved"
    assert report["unresolved_points"] == [{"C": 0.7, "m6_GeV": math.inf, "m8_GeV": math.inf}]
    cols = report["slices"][0]["columns"]
    assert any(c["m6_GeV"] == 500 and c["verified_seed_C"] == 0.2 for c in cols)
    assert cols[0]["C_stops"][1]["last_no_C"] == [0.6, 0.5, 0.4]


def test_unknown_seed_candidates_are_skipped_and_not_classified_as_empty_columns():
    p = plan(m6_range_GeV=(500, 2000, 500), seed_m6_GeV=math.inf)
    def evaluate(x):
        if x.m6_GeV == 500 or math.isinf(x.m6_GeV):
            return "yes" if x.C == 1 else "no"
        return "unknown" if x.C == 1 else "no"
    report = trace_scan(p, evaluate)
    cols = report["slices"][0]["columns"]
    assert cols[-1]["m6_GeV"] == 500
    assert all(c["state"] == "unresolved_no_seed" for c in cols[1:-1])
    assert all(s["reason"] != "consecutive_empty_columns" for s in report["slices"][0]["m6_stops"])


def test_all_unknowns_are_bounded_and_never_become_a_physical_exclusion():
    p = plan(m6_range_GeV=(1000, 1000, 1), C_range=(0, 1, 0.1), seed_m6_GeV=math.inf)
    report = trace_scan(p, lambda _: "unknown")
    assert report["state"] == "complete_with_unresolved"
    assert len(report["unresolved_points"]) == 22
    assert all(c["state"] == "unresolved_no_seed" for c in report["slices"][0]["columns"])


def test_unknown_before_seed_does_not_prevent_verified_nucleation():
    report = trace_scan(plan(column_limit=1), lambda x: "yes" if x.C == 0.5 else "unknown")
    assert report["slices"][0]["columns"][0]["verified_seed_C"] == 0.5


def test_accepted_nucleation_with_unresolved_observables_is_still_a_hit():
    row = {"status": "observables_unresolved"}
    assert nucleation_decision(row, [{"trantype": 1, "nucleation_accepted": True}]) == "yes"
    assert nucleation_decision(row, [{"trantype": 1, "nucleation_accepted": False}]) == "unknown"
    assert nucleation_decision({"status": "numerical_failure"}, []) == "unknown"
    assert nucleation_decision({"status": "no_nucleation_found"}, []) == "no"


def test_missing_seed_is_explicit_and_search_is_bounded():
    p = plan(m6_range_GeV=(1000, 1000, 1), C_range=(0, 1, 0.1), seed_m6_GeV=math.inf)
    report = trace_scan(p, lambda _: "no")
    assert report["state"] == "complete"
    assert not report["slices"][0]["has_verified_seed"]
    assert sum(len(c["bootstrap"]) for c in report["slices"][0]["columns"]) == 22


def test_partial_budget_is_not_a_physical_stop():
    def evaluate(_):
        raise PointBudgetReached

    report = trace_scan(plan(), evaluate)
    assert report["state"] == "paused_point_budget"
    assert not report["slices"][0]["m6_stops"]


@pytest.mark.parametrize("changes", [
    {"patience": 0}, {"C_range": (-1, 10, 0.02)}, {"C_range": (0, 11, 0.02)},
    {"C_range": (0, 10, 0)}, {"seed_m6_GeV": 1234}, {"m8_GeV": (0,)},
    {"m6_range_GeV": (0, 1000, 5)}, {"column_limit": 0}, {"C_steps_limit": 0},
])
def test_invalid_bounds_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(AdaptivePlan(), **changes).axes()
