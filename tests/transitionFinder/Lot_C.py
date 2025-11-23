"""
Block C – Transition history: tests & tutorial-style examples for
secondOrderTrans, findAllTransitions, findCriticalTemperatures and
addCritTempsForFullTransitions, using the same 1D Landau–Ginzburg toy
model as in Block A/B.

This file is deliberately verbose: it prints intermediate results and
creates plots so that you can *see* how the transition-history routines
turn:

- Phase structure (Block A) and
- Bounce/nucleation information (Block B)

into a coherent thermal history: a sequence of first- and second-order
transitions as the Universe cools.
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions.transitionFinder import (
    Phase,
    traceMultiMin,
    getStartPhase,
    secondOrderTrans,
    findAllTransitions,
    findCriticalTemperatures,
    addCritTempsForFullTransitions,
)

# ---------------------------------------------------------------------------
# Model parameters: same Landau–Ginzburg finite-T potential as Block A/B
# ---------------------------------------------------------------------------

D: float = 0.1
E: float = 0.02
lambda_: float = 0.1
T0: float = 100.0


# ---------------------------------------------------------------------------
# Potential and gradient (1D model, but written with array interface)
# ---------------------------------------------------------------------------

def V(phi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Landau–Ginzburg finite-temperature potential.

    Parameters
    ----------
    phi :
        Field value(s). Either scalar-like, shape (1,), or shape (n, 1).
    T :
        Temperature.

    Returns
    -------
    float or ndarray
        Potential evaluated at the given field(s) and temperature.
    """
    phi_arr = np.asarray(phi, dtype=float)

    if phi_arr.ndim == 0:
        phi_val = phi_arr
    elif phi_arr.ndim == 1:
        if phi_arr.size != 1:
            raise ValueError("This test potential is strictly 1D in field space.")
        phi_val = phi_arr[0]
    elif phi_arr.ndim == 2:
        if phi_arr.shape[1] != 1:
            raise ValueError("For batched evaluation, use shape (n_samples, 1).")
        phi_val = phi_arr[:, 0]
    else:
        raise ValueError(f"Unsupported phi shape {phi_arr.shape} for this test potential.")

    V_val = (
        D * (T**2 - T0**2) * phi_val**2
        - E * T * phi_val**3
        + 0.25 * lambda_ * phi_val**4
    )
    return V_val


def dV_dphi(phi: np.ndarray | float, T: float) -> np.ndarray:
    """
    First derivative dV/dphi for the Landau–Ginzburg potential.

    Parameters
    ----------
    phi :
        Field value(s). Only the first component is used.
    T :
        Temperature.

    Returns
    -------
    ndarray, shape (1,)
        Gradient with respect to phi.
    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    phi0 = float(phi_arr[0])
    dV_val = (
        2.0 * D * (T**2 - T0**2) * phi0
        - 3.0 * E * T * phi0**2
        + lambda_ * phi0**3
    )
    return np.array([dV_val], dtype=float)


def d2V_dphi2(phi: np.ndarray | float, T: float) -> np.ndarray:
    """
    Second derivative d^2 V / d phi^2 (Hessian for the 1D toy model).

    Parameters
    ----------
    phi :
        Field value(s). Only the first component is used.
    T :
        Temperature.

    Returns
    -------
    ndarray, shape (1, 1)
        Hessian with respect to phi.
    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    phi0 = float(phi_arr[0])
    d2V_val = (
        2.0 * D * (T**2 - T0**2)
        - 6.0 * E * T * phi0
        + 3.0 * lambda_ * phi0**2
    )
    return np.array([[d2V_val]], dtype=float)


def d2V_dphidT(phi: np.ndarray | float, T: float) -> np.ndarray:
    """
    Mixed derivative ∂/∂T (∂V/∂phi), required for traceMinimum.

    We start from
        ∂V/∂phi = 2D(T^2 - T0^2) phi - 3E T phi^2 + λ phi^3

    and take ∂/∂T:
        ∂/∂T (∂V/∂phi)
        = 4D T phi - 3E phi^2.
    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    phi0 = float(phi_arr[0])
    val = 4.0 * D * T * phi0 - 3.0 * E * phi0**2
    return np.array([val], dtype=float)




# ---------------------------------------------------------------------------
# Small utilities reused across tests
# ---------------------------------------------------------------------------

def _find_broken_minimum(T: float, phi_guess: float = 1.0) -> float:
    """
    Find the broken-phase minimum at a given T by a simple 1D search.

    For Block C we only need an approximate seed for traceMultiMin.
    """
    # Sample V on a coarse grid and pick the minimum; good enough for seeding.
    phi_grid = np.linspace(-1.0, 8.0, 400)
    V_vals = V(phi_grid.reshape(-1, 1), T)
    idx = int(np.argmin(V_vals))
    return float(phi_grid[idx])


def _build_phases() -> Dict[Hashable, Phase]:
    """
    Build the phase structure with traceMultiMin in [T_low, T_high].

    This reproduces the symmetric and broken branches used in Blocks A/B.
    """
    T_low = 50.0
    T_high = 200.0

    phi_b_50 = _find_broken_minimum(T=T_low, phi_guess=1.0)
    points = [
        (np.array([0.0], dtype=float), T_high),        # symmetric seed
        (np.array([phi_b_50], dtype=float), T_low),    # broken seed
    ]

    phases = traceMultiMin(
        f=V,
        d2f_dxdt=d2V_dphidT,
        d2f_dx2=d2V_dphi2,
        points=points,
        tLow=T_low,
        tHigh=T_high,
        deltaX_target=0.02,
    )
    return phases


def _classify_phase(phase: Phase) -> str:
    """
    Classify a Phase as 'symmetric' or 'broken' by its φ at the highest T.

    This is only for diagnostics and plotting.
    """
    T_eval = float(phase.T[-1])
    phi_eval = np.asarray(phase.valAt(T_eval), dtype=float)
    phi0 = float(phi_eval[0])
    return "symmetric" if abs(phi0) < 0.1 else "broken"


def _plot_phase_structure_with_markers(
    phases: Dict[Hashable, Phase],
    transitions: List[dict] | None = None,
    crit_transitions: List[dict] | None = None,
    title: str = "Phase structure and transition temperatures",
) -> None:
    """
    Utility for Block C tests: plot φ_min(T) for each Phase, and optionally
    overlay vertical lines and markers at Tcrit/Tnuc.
    """
    fig, ax = plt.subplots()
    ax.set_title(title)

    # Plot phase branches
    for key, phase in phases.items():
        T_dense = np.linspace(float(phase.T[0]), float(phase.T[-1]), 200)
        phi_dense = np.asarray(phase.valAt(T_dense), dtype=float)
        ax.plot(
            T_dense,
            phi_dense,
            label=f"Phase {key} ({_classify_phase(phase)})",
        )

    # Overlay critical temperatures
    if crit_transitions:
        for tcdict in crit_transitions:
            Tcrit = float(tcdict["Tcrit"])
            ax.axvline(Tcrit, linestyle="--", alpha=0.5, label=None)

    # Overlay nucleation temperatures and vevs
    if transitions:
        for tdict in transitions:
            if "Tnuc" not in tdict:
                continue
            Tnuc = float(tdict["Tnuc"])
            high_vev = np.asarray(tdict["high_vev"], dtype=float)[0]
            low_vev = np.asarray(tdict["low_vev"], dtype=float)[0]

            # vertical line at Tnuc
            ax.axvline(Tnuc, color="k", linewidth=1.0, alpha=0.6, label=None)
            # markers at high/low vev
            ax.plot(Tnuc, high_vev, "o", label=None)
            ax.plot(Tnuc, low_vev, "s", label=None)

    ax.set_xlabel("T")
    ax.set_ylabel(r"$\phi_{\min}(T)$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 0 – basic structure of secondOrderTrans
# ---------------------------------------------------------------------------

def test_blockC_0_secondOrderTrans_basic_structure():
    """
    Test 0 – sanity check for secondOrderTrans:

    - Build the phase structure and pick two phases.
    - Construct a 'second-order' transition dictionary between them.
    - Check that the dictionary has the expected fields and types.
    """
    print("\n[Block C / Test 0] secondOrderTrans: basic structure and fields")

    phases = _build_phases()
    keys = list(phases.keys())
    assert len(keys) >= 2, "We expect at least symmetric + broken phases."

    high_phase = phases[keys[0]]
    low_phase = phases[keys[1]]

    tdict = secondOrderTrans(high_phase=high_phase, low_phase=low_phase, Tstr="Tnuc")

    print("  Constructed second-order transition dictionary:")
    for k, v in tdict.items():
        print(f"    {k:>10s} : {v}")

    assert "Tnuc" in tdict
    assert tdict["trantype"] == 2
    assert tdict["action"] == 0.0
    assert tdict["instanton"] is None

    # Tnuc should be the average of high_phase.T[0] and low_phase.T[-1]
    T_ref = 0.5 * (float(high_phase.T[0]) + float(low_phase.T[-1]))
    assert abs(tdict["Tnuc"] - T_ref) < 1e-8

    # high_vev and low_vev should coincide with high_phase.X[0]
    phi0 = np.asarray(high_phase.X[0], dtype=float)
    assert np.allclose(np.asarray(tdict["high_vev"], dtype=float), phi0)
    assert np.allclose(np.asarray(tdict["low_vev"], dtype=float), phi0)


# ---------------------------------------------------------------------------
# Test 1 – findAllTransitions: full thermal history from high T to low T
# ---------------------------------------------------------------------------

def test_blockC_1_findAllTransitions_full_history():
    """
    Test 1 – build the thermal history with findAllTransitions:

    - Reconstruct phases with traceMultiMin.
    - Identify the high-T start phase.
    - Call findAllTransitions to obtain the ordered list of transitions.
    - Print a table summarizing Tnuc, trantype, phases and action.
    - Plot φ_min(T) for all phases and mark nucleation points.
    """
    print("\n[Block C / Test 1] findAllTransitions – full thermal history")

    phases = _build_phases()
    print(f"  Number of phases found: {len(phases)}")

    start_key = getStartPhase(phases, V)
    print(f"  getStartPhase selected: {start_key}")

    # Arguments for tunnelFromPhase inside findAllTransitions
    tunnel_args = dict(
        Ttol=1e-2,
        maxiter=200,
        phitol=1e-8,
        overlapAngle=45.0,
        verbose=False,   # keep quiet; we'll print our own summary
    )

    transitions = findAllTransitions(phases, V, dV_dphi, tunnelFromPhase_args=tunnel_args)

    print(f"  Number of transitions in the thermal history: {len(transitions)}")
    assert len(transitions) >= 1, "We expect at least one transition in this model."

    # Basic consistency checks: temperatures should be decreasing, phases should exist.
    Tnucs = []
    print("\n  Summary of transitions (ordered from high T to low T):")
    print("    idx |  Tnuc    | type | high_phase -> low_phase |  S(Tnuc)")
    print("    ----+----------+------+-------------------------+---------")
    for i, tdict in enumerate(transitions):
        Tnuc = float(tdict["Tnuc"])
        Tnucs.append(Tnuc)
        trantype = int(tdict["trantype"])
        high_phase = tdict["high_phase"]
        low_phase = tdict["low_phase"]
        action = float(tdict["action"])

        print(
            f"    {i:3d} | {Tnuc:8.3f} |  {trantype:d}   | "
            f"{high_phase} -> {low_phase:>8} | {action:7.3f}"
        )

        # Phases must be part of the original phase dictionary
        assert high_phase in phases
        assert low_phase in phases
        assert trantype in (1, 2)

    # Temperatures should not increase along the history
    if len(Tnucs) > 1:
        diffs = np.diff(np.array(Tnucs, dtype=float))
        assert np.all(diffs <= 1e-6), "Tnuc should be non-increasing along the history."

    # Plot φ_min(T) with markers at Tnuc
    _plot_phase_structure_with_markers(
        phases,
        transitions=transitions,
        crit_transitions=None,
        title="Block C / Test 1 – φ_min(T) and nucleation temperatures",
    )


# ---------------------------------------------------------------------------
# Test 2 – findCriticalTemperatures: phase degeneracies Tcrit
# ---------------------------------------------------------------------------

def test_blockC_2_findCriticalTemperatures_degeneracies():
    """
    Test 2 – scan for critical temperatures with findCriticalTemperatures:

    - Rebuild the phases.
    - Call findCriticalTemperatures to find all Tcrit where phases are degenerate.
    - Print a list of Tcrit and associated (high_phase -> low_phase).
    - For first-order entries, check that V(high_vev, Tcrit) ≈ V(low_vev, Tcrit).
    """
    print("\n[Block C / Test 2] findCriticalTemperatures – phase degeneracies")

    phases = _build_phases()
    crit_trans = findCriticalTemperatures(phases, V, start_high=False)

    print(f"  Number of critical-temperature transitions found: {len(crit_trans)}")
    assert len(crit_trans) >= 1, "We expect at least one critical temperature in this model."

    print("\n  Summary of critical transitions (sorted by decreasing Tcrit):")
    print("    idx |  Tcrit   | type | high_phase -> low_phase | ΔV(high-low)")
    print("    ----+----------+------+-------------------------+--------------")

    for i, tcdict in enumerate(crit_trans):
        Tcrit = float(tcdict["Tcrit"])
        trantype = int(tcdict["trantype"])
        high_phase = tcdict["high_phase"]
        low_phase = tcdict["low_phase"]
        high_vev = np.asarray(tcdict["high_vev"], dtype=float)
        low_vev = np.asarray(tcdict["low_vev"], dtype=float)

        V_high = float(V(high_vev, Tcrit))
        V_low = float(V(low_vev, Tcrit))
        dV = V_high - V_low

        print(
            f"    {i:3d} | {Tcrit:8.3f} |  {trantype:d}   | "
            f"{high_phase} -> {low_phase:>8} | {dV:12.3e}"
        )

        # For first-order degeneracies we expect nearly equal free energies
        if trantype == 1:
            assert abs(dV) < 1e-3, (
                "At Tcrit the free energies of the two phases should be nearly equal."
            )

    # Plot φ_min(T) and mark all Tcrit with vertical dashed lines
    _plot_phase_structure_with_markers(
        phases,
        transitions=None,
        crit_transitions=crit_trans,
        title="Block C / Test 2 – φ_min(T) and critical temperatures",
    )


# ---------------------------------------------------------------------------
# Test 3 – addCritTempsForFullTransitions: matching Tcrit and Tnuc
# ---------------------------------------------------------------------------

def test_blockC_3_addCritTemps_match_Tcrit_and_Tnuc():
    """
    Test 3 – combine full transitions and critical temperatures:

    - Build phases and compute:
        * full_trans: thermal history from findAllTransitions.
        * crit_trans: degeneracies from findCriticalTemperatures.
    - Call addCritTempsForFullTransitions to attach each Tcrit to the
      corresponding Tnuc entry (when possible).
    - Print a table for each full transition, showing:
        * Tcrit (if found), Tnuc, ΔT = Tcrit - Tnuc, trantype.
    - Check that for first-order transitions with a matched Tcrit we have
      Tcrit >= Tnuc (supercooling or no supercooling, but not the reverse).
    """
    print("\n[Block C / Test 3] addCritTempsForFullTransitions – matching Tcrit and Tnuc")

    phases = _build_phases()

    # Build the full thermal history
    tunnel_args = dict(
        Ttol=1e-2,
        maxiter=200,
        phitol=1e-8,
        overlapAngle=45.0,
        verbose=False,
    )
    full_trans = findAllTransitions(phases, V, dV_dphi, tunnelFromPhase_args=tunnel_args)

    # Find all critical temperatures
    crit_trans = findCriticalTemperatures(phases, V, start_high=False)

    # Attach crit_trans to full_trans
    addCritTempsForFullTransitions(phases, crit_trans, full_trans)

    print(f"  Number of full transitions: {len(full_trans)}")
    print(f"  Number of critical transitions: {len(crit_trans)}")

    print("\n  Matching Tcrit and Tnuc for each full transition:")
    print("    idx |  type |  Tcrit   |  Tnuc    |  ΔT=Tcrit-Tnuc | high_phase -> low_phase")
    print("    ----+-------+----------+----------+----------------+------------------------")

    for i, tdict in enumerate(full_trans):
        trantype = int(tdict["trantype"])
        Tnuc = float(tdict["Tnuc"])
        high_phase = tdict["high_phase"]
        low_phase = tdict["low_phase"]

        crit = tdict.get("crit_trans", None)
        if crit is not None:
            Tcrit = float(crit["Tcrit"])
            dT = Tcrit - Tnuc
            print(
                f"    {i:3d} |  {trantype:d}    | {Tcrit:8.3f} | {Tnuc:8.3f} | "
                f"{dT:14.3f} | {high_phase} -> {low_phase}"
            )
            if trantype == 1:
                # For first-order transitions with a matched Tcrit, require Tcrit >= Tnuc
                assert Tcrit >= Tnuc - 1e-6, (
                    "For a first-order transition, nucleation should not occur above Tcrit."
                )
        else:
            print(
                f"    {i:3d} |  {trantype:d}    |   (none) | {Tnuc:8.3f} | "
                f"      (n/a)      | {high_phase} -> {low_phase}"
            )

    # Plot φ_min(T) with both Tcrit and Tnuc markers to visualize supercooling
    _plot_phase_structure_with_markers(
        phases,
        transitions=full_trans,
        crit_transitions=crit_trans,
        title="Block C / Test 3 – φ_min(T), Tcrit and Tnuc (supercooling)",
    )


# ---------------------------------------------------------------------------
# Optional: manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Running this file directly will execute all Block C tests sequentially.
    # This is handy if you want to see the printed diagnostics and plots
    # without going through pytest.
    test_blockC_0_secondOrderTrans_basic_structure()
    test_blockC_1_findAllTransitions_full_history()
    test_blockC_2_findCriticalTemperatures_degeneracies()
    test_blockC_3_addCritTemps_match_Tcrit_and_Tnuc()
    print("\n[Block C] All example tests executed.")
