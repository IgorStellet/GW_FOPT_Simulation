"""
Block D – Observables:
tutorial-style tests for thermalObservablesForTransition and
addObservablesToTransitions, using the same 1D Landau–Ginzburg
finite-temperature potential as Blocks A–C.

This file is deliberately verbose: it prints intermediate results and
creates plots (closed immediately in CI), so that you can *see* what
the observables look like in a concrete model.

Main goals
----------
- Build a phase structure with traceMultiMin (Block A).
- Construct a full transition history with findAllTransitions and
  addCritTempsForFullTransitions (Block C).
- Attach thermodynamic observables with addObservablesToTransitions
  (Block D).
- Visualize:
    * α vs supercooling ΔT/T_crit,
    * β_eff/H_* vs α,
    * energy budget (ρ_high, ρ_low, Δρ, ρ_rad),
    * a simple “table-like” summary of all transitions.
"""

from __future__ import annotations

from typing import Dict, Hashable, Tuple, List, Any

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions.transitionFinder import (
    Phase,
    traceMultiMin,
    removeRedundantPhases,
    findAllTransitions,
    findCriticalTemperatures,
    addCritTempsForFullTransitions,
    addObservablesToTransitions,
)

# ---------------------------------------------------------------------------
# Model parameters: same Landau–Ginzburg finite-T potential as Block A
# ---------------------------------------------------------------------------

D: float = 0.1
E: float = 0.02
lambda_: float = 0.1
T0: float = 100.0


# ---------------------------------------------------------------------------
# Potential and analytic derivatives (1D in field space)
# ---------------------------------------------------------------------------

def V(phi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Landau–Ginzburg finite-temperature potential.

        V(phi, T) = D (T^2 - T0^2) phi^2 - E T phi^3 + (lambda_/4) phi^4

    This implementation is 1D in field space but supports both scalar and
    batched inputs, compatible with traceMultiMin:

    - For traceMultiMin / Phase, we use phi.shape == (1,).
    - For vectorized calls, we use phi.shape == (n, 1).
    """
    phi_arr = np.asarray(phi, dtype=float)

    if phi_arr.ndim == 0:
        # Single scalar phi
        phi_val = phi_arr
    elif phi_arr.ndim == 1:
        # Treat as a single 1D field vector; this toy model is strictly 1D
        if phi_arr.size != 1:
            raise ValueError("This test potential is strictly 1D in field space.")
        phi_val = phi_arr[0]
    elif phi_arr.ndim == 2:
        # Batched samples: shape (n_samples, 1)
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

    Returns an array of shape (1,), consistent with the main code.
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

    Returns an array of shape (1, 1).
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
    Mixed derivative ∂/∂T (∂V/∂phi), required by traceMinimum / traceMultiMin.

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


def dV_dT(phi: np.ndarray | float, T: float) -> float:
    """
    Analytic partial derivative ∂V/∂T at fixed phi.

    From
        V(phi, T) = D (T^2 - T0^2) phi^2 - E T phi^3 + (lambda_/4) phi^4

    we have
        ∂V/∂T = 2D T phi^2 - E phi^3.

    This function is passed to thermalObservablesForTransition as dVdT.
    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    # In this 1D model we only use the first component
    phi0 = float(phi_arr.flat[0])
    return float(2.0 * D * T * phi0**2 - E * phi0**3)


# ---------------------------------------------------------------------------
# Small utilities: build phases and full transition history with observables
# ---------------------------------------------------------------------------

def _find_broken_minimum(T: float, phi_guess: float = 1.0) -> float:
    """
    1D minimization to find the broken minimum at a given temperature T.
    """

    def V_scalar(phi: float) -> float:
        return float(V(np.array([phi], dtype=float), T))

    # Simple 1D scan around the guess
    grid = np.linspace(phi_guess - 2.0, phi_guess + 6.0, 400)
    vals = [V_scalar(p) for p in grid]
    idx = int(np.argmin(vals))
    return float(grid[idx])


def _build_phases() -> Dict[Hashable, Phase]:
    """
    Build the phase structure in [T_low, T_high] with traceMultiMin.

    This mirrors Block A, so that Block D is exercised on the same
    phase structure used in previous blocks.
    """
    T_low = 50.0
    T_high = 200.0

    phi_b_50 = _find_broken_minimum(T=T_low, phi_guess=1.0)
    points = [
        (np.array([0.0], dtype=float), T_high),   # symmetric seed
        (np.array([phi_b_50], dtype=float), T_low),  # broken seed
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

    removeRedundantPhases(V, phases, xeps=1e-6, diftol=1e-2)
    return phases


def _build_transition_history_with_observables() -> Tuple[List[Dict[str, Any]], Dict[Hashable, Phase]]:
    """
    High-level helper for all tests in this file:

    - Build phases with traceMultiMin.
    - Construct full transition history with findAllTransitions.
    - Compute critical temperatures with findCriticalTemperatures.
    - Attach critical transitions to the full history.
    - Attach observables (α, β_eff/H_*, etc.) to each transition.

    Returns
    -------
    full_trans : list of dict
        Full transitions, each including an 'obs' sub-dictionary.
    phases : dict
        Phase structure used to build the history.
    """
    phases = _build_phases()

    # Tunneling solver parameters: mild tolerances, no debug prints.
    tunnel_args: Dict[str, Any] = dict(
        Ttol=5e-2,
        maxiter=100,
        phitol=1e-6,
        overlapAngle=0.0,
        verbose=False,
    )

    full_trans = findAllTransitions(
        phases=phases,
        V=V,
        dV=dV_dphi,
        tunnelFromPhase_args=tunnel_args,
    )

    if not full_trans:
        print("  [WARN] findAllTransitions returned no transitions.")
        return full_trans, phases

    # Critical temperatures between phases (where V_high = V_low)
    crit_trans = findCriticalTemperatures(phases, V, start_high=False)

    # Attach critical transitions to the full (possibly supercooled) transitions
    addCritTempsForFullTransitions(phases, crit_trans, full_trans)

    # Add thermal observables at Tnuc (default)
    addObservablesToTransitions(
        transitions=full_trans,
        V=V,
        dVdT=dV_dT,
        T_key="Tnuc",
        g_star=106.75,
        beta_from_geometry=True,
        beta_geom_method="rscale",
    )

    return full_trans, phases


# ---------------------------------------------------------------------------
# Test 1: basic sanity and printed summary of observables
# ---------------------------------------------------------------------------

def test_blockD_1_observables_summary() -> None:
    """
    Test 1 – build the full transition history and attach observables.

    - Calls _build_transition_history_with_observables.
    - Prints a compact summary per transition:
        * T_star, Tcrit, ΔT,
        * S/T, α, β_eff/H_*,
        * phase labels.
    - Asserts basic sanity on a first-order transition, if present.
    """
    print("\n[Block D / Test 1] Summary of observables for each transition")

    full_trans, phases = _build_transition_history_with_observables()
    assert phases, "Phase structure should not be empty."
    assert full_trans, "We expect at least one transition in this toy model."

    for idx, tdict in enumerate(full_trans):
        obs = tdict.get("obs", {})
        print(f"\n  Transition #{idx}: "
              f"{tdict.get('high_phase', '?')} → {tdict.get('low_phase', '?')}")
        print(f"    trantype     = {tdict.get('trantype', '?')}")
        print(f"    T_star (Tnuc)= {obs.get('T_star', np.nan):.3f}")
        print(f"    Tcrit        = {obs.get('Tcrit', np.nan):.3f}")
        print(f"    DeltaT       = {obs.get('DeltaT', np.nan):.3f}")
        print(f"    S/T          = {obs.get('S_over_T', np.nan):.3f}")
        print(f"    alpha        = {obs.get('alpha_strength', np.nan):.3e}")
        print(f"    beta_eff/H_* = {obs.get('beta_over_H_eff', np.nan):.3e} "
              f" (method={obs.get('beta_method', 'none')})")

        # Basic sanity checks for first-order transitions
        if int(tdict.get("trantype", 0)) == 1:
            alpha = float(obs["alpha_strength"])
            S_over_T = float(obs["S_over_T"])
            rho_rad = float(obs["rho_rad"])

            assert rho_rad > 0.0, "Radiation energy density must be positive."
            assert alpha > 0.0, "We expect alpha > 0 for a first-order transition."
            assert 0.0 < S_over_T < 1e3, "S/T should be finite and not absurdly large."

    print("\n[Block D / Test 1] Summary printed successfully.")


# ---------------------------------------------------------------------------
# Test 2: α vs supercooling ΔT/T_crit
# ---------------------------------------------------------------------------

def test_blockD_2_strength_vs_supercooling_plot() -> None:
    """
    Test 2 – α vs supercooling ΔT/T_crit.

    - Uses the same transition history as Test 1.
    - Builds arrays:
        x = DeltaT / Tcrit
        y = alpha_strength
      for transitions where Tcrit is available and finite.
    - Produces a scatter/line plot for visual inspection.
    """
    print("\n[Block D / Test 2] Strength α vs supercooling ΔT/Tcrit")

    full_trans, _ = _build_transition_history_with_observables()
    if not full_trans:
        print("  No transitions found; skipping plot.")
        return

    x_vals: List[float] = []
    y_vals: List[float] = []

    for tdict in full_trans:
        obs = tdict.get("obs", {})
        Tcrit = float(obs.get("Tcrit", np.nan))
        DeltaT = float(obs.get("DeltaT", np.nan))
        alpha = float(obs.get("alpha_strength", np.nan))

        if not np.isfinite(Tcrit) or Tcrit <= 0.0:
            continue
        if not np.isfinite(alpha) or alpha <= 0.0:
            continue

        x_vals.append(DeltaT / Tcrit)
        y_vals.append(alpha)

    if not x_vals:
        print("  No transitions with finite Tcrit and alpha; skipping plot.")
        return

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)

    print("  Points (ΔT/Tcrit, α):")
    for xv, yv in zip(x_arr, y_arr):
        print(f"    ({xv:.3f}, {yv:.3e})")

    fig, ax = plt.subplots()
    ax.set_title(r"Block D: strength $\alpha$ vs supercooling $\Delta T/T_{\rm crit}$")
    ax.plot(x_arr, y_arr, marker="o", linestyle="-")
    ax.set_xlabel(r"$\Delta T / T_{\rm crit}$")
    ax.set_ylabel(r"$\alpha$")
    fig.tight_layout()
    plt.show()  # closed automatically in most CI setups

    # Very mild sanity: α should not be negative and not absurdly huge
    assert np.all(y_arr > 0.0)
    assert np.all(y_arr < 1e3)


# ---------------------------------------------------------------------------
# Test 3: β_eff/H_* vs α
# ---------------------------------------------------------------------------

def test_blockD_3_beta_over_H_vs_alpha_plot() -> None:
    """
    Test 3 – β_eff/H_* vs strength α.

    - For each transition, reads:
        alpha_strength
        beta_over_H_eff
      from tdict["obs"].
    - Filters transitions with finite beta_over_H_eff and alpha > 0.
    - Produces a scatter/line plot β_eff/H_* vs α.

    Notes
    -----
    If the geometric β proxy cannot be computed (e.g. pathDeformation backend
    without a betaEff method), beta_over_H_eff may be NaN, in which case those
    transitions are skipped in this plot.
    """
    print("\n[Block D / Test 3] β_eff/H_* vs strength α")

    full_trans, _ = _build_transition_history_with_observables()
    if not full_trans:
        print("  No transitions found; skipping plot.")
        return

    alpha_list: List[float] = []
    betaH_list: List[float] = []

    for tdict in full_trans:
        obs = tdict.get("obs", {})
        alpha = float(obs.get("alpha_strength", np.nan))
        beta_over_H = float(obs.get("beta_over_H_eff", np.nan))

        if not np.isfinite(alpha) or alpha <= 0.0:
            continue
        if not np.isfinite(beta_over_H) or beta_over_H <= 0.0:
            continue

        alpha_list.append(alpha)
        betaH_list.append(beta_over_H)

    if not alpha_list:
        print("  No transitions with finite β_eff/H_*; skipping plot.")
        return

    alpha_arr = np.asarray(alpha_list, dtype=float)
    betaH_arr = np.asarray(betaH_list, dtype=float)

    print("  Points (α, β_eff/H_*):")
    for a, b in zip(alpha_arr, betaH_arr):
        print(f"    ({a:.3e}, {b:.3e})")

    fig, ax = plt.subplots()
    ax.set_title(r"Block D: $\beta_{\rm eff}/H_*$ vs $\alpha$")
    ax.plot(alpha_arr, betaH_arr, marker="o", linestyle="-")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta_{\rm eff}/H_*$")
    fig.tight_layout()
    plt.show()

    # Mild sanity: β/H should be positive and not insanely small.
    assert np.all(betaH_arr > 0.0)


# ---------------------------------------------------------------------------
# Test 4: Energy budget for a single transition
# ---------------------------------------------------------------------------

def test_blockD_4_energy_budget_plot() -> None:
    """
    Test 4 – visualize the energy budget (ρ_high, ρ_low, Δρ, ρ_rad)
    for a representative transition.

    - Selects the first transition in the history.
    - Extracts from obs:
        rho_high, rho_low, delta_rho, rho_rad.
    - Prints the numbers.
    - Produces a simple bar plot to visualise where the energy lives.
    """
    print("\n[Block D / Test 4] Energy budget for the first transition")

    full_trans, _ = _build_transition_history_with_observables()
    if not full_trans:
        print("  No transitions found; skipping energy budget plot.")
        return

    tdict = full_trans[0]
    obs = tdict.get("obs", {})

    rho_high = float(obs.get("rho_high", np.nan))
    rho_low = float(obs.get("rho_low", np.nan))
    delta_rho = float(obs.get("delta_rho", np.nan))
    rho_rad = float(obs.get("rho_rad", np.nan))

    print(f"  Transition: {tdict.get('high_phase', '?')} → {tdict.get('low_phase', '?')}")
    print(f"    rho_high = {rho_high:.6e}")
    print(f"    rho_low  = {rho_low:.6e}")
    print(f"    delta_rho= {delta_rho:.6e}")
    print(f"    rho_rad  = {rho_rad:.6e}")

    # Basic sanity: radiation > 0, and for a proper first-order transition,
    # we expect rho_high > rho_low and delta_rho > 0.
    assert rho_rad > 0.0
    if int(tdict.get("trantype", 0)) == 1:
        assert rho_high > rho_low
        assert delta_rho > 0.0

    labels = [r"$\rho_{\rm high}$", r"$\rho_{\rm low}$",
              r"$\Delta\rho$", r"$\rho_{\rm rad}$"]
    values = [rho_high, rho_low, delta_rho, rho_rad]

    fig, ax = plt.subplots()
    ax.set_title("Block D: energy budget for one transition")
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy density (arb. units)")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 5: Table-like printed summary for all transitions
# ---------------------------------------------------------------------------

def test_blockD_5_summary_table_print() -> None:
    """
    Test 5 – print a compact, table-like summary of all transitions.

    Columns:
    - index
    - high_phase → low_phase
    - T_star (Tnuc)
    - Tcrit
    - DeltaT
    - alpha_strength
    - beta_over_H_eff
    """
    print("\n[Block D / Test 5] Table-like summary of transitions and observables")

    full_trans, _ = _build_transition_history_with_observables()
    if not full_trans:
        print("  No transitions found; nothing to summarize.")
        return

    header = (
        "  idx  high→low    "
        "T_star    Tcrit    DeltaT    alpha        beta/H_eff"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for idx, tdict in enumerate(full_trans):
        obs = tdict.get("obs", {})
        T_star = float(obs.get("T_star", np.nan))
        Tcrit = float(obs.get("Tcrit", np.nan))
        DeltaT = float(obs.get("DeltaT", np.nan))
        alpha = float(obs.get("alpha_strength", np.nan))
        betaH = float(obs.get("beta_over_H_eff", np.nan))

        label = f"{tdict.get('high_phase', '?')}→{tdict.get('low_phase', '?')}"
        print(
            f"  {idx:3d}  {label:8s}  "
            f"{T_star:7.3f}  {Tcrit:7.3f}  {DeltaT:7.3f}  "
            f"{alpha:9.3e}  {betaH:11.3e}"
        )

        # If Tcrit is known, T_star should not exceed it by a large margin
        if np.isfinite(Tcrit) and Tcrit > 0.0 and np.isfinite(T_star):
            assert T_star <= Tcrit + 5.0, "T_star should not be far above Tcrit."

    print("\n[Block D / Test 5] Summary table printed.")


# ---------------------------------------------------------------------------
# Optional: manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Running this file directly executes all tests sequentially.
    # This is useful if you want to see the printed diagnostics and plots
    # without going through pytest.
    test_blockD_1_observables_summary()
    test_blockD_2_strength_vs_supercooling_plot()
    test_blockD_3_beta_over_H_vs_alpha_plot()
    test_blockD_4_energy_budget_plot()
    test_blockD_5_summary_table_print()
    print("\n[Block D] All example tests executed.")
