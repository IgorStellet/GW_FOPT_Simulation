from CosmoTransitions.transitionFinder import (
    Phase,
    traceMultiMin,
    removeRedundantPhases,
    getStartPhase,
    tunnelFromPhase,
    _potentialDiffForPhase,
    _maxTCritForPhase,
    _solve_bounce,
)

from typing import Dict, Hashable, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# ---------------------------------------------------------------------------
# Model parameters: simple Landau–Ginzburg finite-T potential
# ---------------------------------------------------------------------------

D: float = 0.1
E: float = 0.02
lambda_: float = 0.1
T0: float = 100.0


# ---------------------------------------------------------------------------
# Potential and analytic derivatives
# ---------------------------------------------------------------------------

def V(phi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Landau–Ginzburg finite-temperature potential.

    This implementation is 1D in field space but supports both scalar and
    batched inputs for use with findApproxLocalMin:

    - For traceMinimum / Phase, we use phi.shape == (1,).
    - For vectorized calls, we use phi.shape == (n, 1).

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
        # single scalar phi
        phi_val = phi_arr
    elif phi_arr.ndim == 1:
        # treat as a single 1D field vector; this toy model is 1D
        if phi_arr.size != 1:
            raise ValueError("This test potential is strictly 1D in field space.")
        phi_val = phi_arr[0]
    elif phi_arr.ndim == 2:
        # batched samples: shape (n_samples, 1)
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
# Small utilities for this test module
# ---------------------------------------------------------------------------

def _broken_minimum_analytic(T: float) -> float:
    """
    Analytic location of the broken minimum for the 1D Landau–Ginzburg model.

    For the stationary points with phi ≠ 0 we solve

        λ φ^2 - 3 E T φ + 2 D (T^2 - T0^2) = 0,

    and pick the root that:
    - is not too close to phi = 0, and
    - has positive curvature (d^2 V / d phi^2 > 0),
    - and minimizes V among the candidates.

    This is only used in the Block B tests to construct a clean
    metastable configuration (false vacuum at phi=0, true vacuum at
    phi > 0) for the bounce solver.
    """
    a = lambda_
    b = -3.0 * E * T
    c = 2.0 * D * (T**2 - T0**2)

    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        raise RuntimeError(f"No non-trivial broken minimum for T={T} (discriminant <= 0).")

    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]

    candidates: list[float] = []
    for phi in roots:
        if abs(phi) < 1e-6:
            continue
        m2 = d2V_dphi2(np.array([phi], dtype=float), T)[0, 0]
        if m2 > 0.0:
            candidates.append(float(phi))

    if not candidates:
        raise RuntimeError(f"No stable broken minimum for T={T} among analytic roots.")

    # Choose the one with the lowest potential
    phi_best = min(
        candidates,
        key=lambda phi_val: float(V(np.array([phi_val], dtype=float), T)),
    )
    return float(phi_best)


def _find_broken_minimum(T: float, phi_guess: float = 1.0) -> float:
    """
    Find the broken-phase minimum at a given T by 1D minimization.

    Uses scipy.optimize.fmin on the scalar potential V(phi, T).
    """

    def V_scalar(phi: float) -> float:
        return float(V(np.array([phi], dtype=float), T))

    result = optimize.fmin(V_scalar, x0=phi_guess, disp=False)
    phi_min = float(result[0])
    return phi_min


def _build_phases() -> Dict[Hashable, Phase]:
    """
    Build the phase structure with traceMultiMin in [T_low, T_high].

    Returns
    -------
    dict
        Mapping key -> Phase instance, after removing redundancies.
    """
    T_low = 50.0
    T_high = 200.0

    phi_b_50 = _find_broken_minimum(T=T_low, phi_guess=1.0)
    points = [
        (np.array([0.0], dtype=float), T_high),  # symmetric seed
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
        # keep other kwargs as defaults; they are already tuned in the module
    )

    # Clean up any duplicated phases
    removeRedundantPhases(V, phases, xeps=1e-6, diftol=1e-2)

    return phases


# ---------------------------------------------------------------------------
# Block B – tunneling core: tests & tutorial-style examples
# ---------------------------------------------------------------------------

def test_blockB_1_potential_diff_and_Tcrit():
    """
    Block B / Test 1 – _potentialDiffForPhase and _maxTCritForPhase

    Physics picture:
    ----------------
    For a given 'start_phase' (here the symmetric phase at high T) we
    look at the free-energy difference

        ΔV(T) = V(other, T) - V(start_phase, T),

    minimized over all other phases. Then:

    - ΔV(T) > 0 → start_phase is energetically preferred.
    - ΔV(T) < 0 → some other phase is preferred.
    - ΔV(Tcrit) = 0 → degeneracy: definition of a critical temperature.

    This test:
    ----------
    - builds the phase structure;
    - identifies the symmetric high-T phase as start_phase;
    - uses _maxTCritForPhase to find Tcrit;
    - checks that ΔV(Tcrit) is numerically close to zero;
    - plots ΔV(T) vs T and marks Tcrit.
    """
    print("\n[Block B / Test 1] Critical temperature and free-energy differences")

    phases = _build_phases()
    start_key = getStartPhase(phases, V)
    start_phase = phases[start_key]
    other_phases = [phase for key, phase in phases.items() if key != start_key]

    assert other_phases, "We need at least one 'other' phase to define a critical temperature."

    # Compute Tcrit using the helper we are testing
    Tcrit = _maxTCritForPhase(phases, start_phase, V, Ttol=1e-3)
    DV_Tcrit = _potentialDiffForPhase(Tcrit, start_phase, other_phases, V)

    print(f"  start_phase key           = {start_key}")
    print(f"  Critical temperature T_c  ≈ {Tcrit:.4f}")
    print(f"  ΔV(T_c) = V(other) - V(start) ≈ {DV_Tcrit:.4e} (should be ~ 0)")

    # At Tcrit the free-energy difference should be very small
    assert abs(DV_Tcrit) < 1

    # For intuition, scan ΔV(T) across the overlapping T-range
    Tmin = max(start_phase.T[0], min(phase.T[0] for phase in other_phases))
    Tmax = min(start_phase.T[-1], max(phase.T[-1] for phase in other_phases))
    Tmin = float(Tmin)
    Tmax = float(Tmax)

    T_grid = np.linspace(Tmin, Tmax, 200)
    DV_grid = np.array(
        [_potentialDiffForPhase(Tg, start_phase, other_phases, V) for Tg in T_grid],
        dtype=float,
    )

    fig, ax = plt.subplots()
    ax.set_title("Block B / Test 1: free-energy difference between phases")
    ax.plot(T_grid, DV_grid, label=r"$\Delta V(T)$")
    ax.axhline(0.0, linestyle="--", label=r"$\Delta V = 0$")
    ax.axvline(Tcrit, linestyle="--", label=r"$T_{\mathrm{crit}}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\Delta V(T) = V_{\text{other}} - V_{\text{start}}$")
    ax.legend()
    fig.tight_layout()
    plt.show()


def test_blockB_2_solve_bounce_single_field_example():
    """
    Block B / Test 2 – direct check of _solve_bounce in the 1D toy model.

    Idea:
    -----
    We want a clean 1D configuration where:
    - phi = 0 is a *metastable* minimum (false vacuum);
    - there is a deeper broken minimum at phi = phi_true > 0 (true vacuum);
    - the two minima are separated by a barrier.

    This happens in our Landau–Ginzburg model for T slightly above T0:
    the cubic term creates a second, deeper minimum at large phi while
    the origin is still locally stable.

    Steps:
    ------
    - scan a small range T in (T0, T0 + ΔT) until we find such a
      configuration;
    - call _solve_bounce with x_high = 0 and x_low = phi_true;
    - check that a non-trivial bounce is found (trantype = 1) with a
      finite positive action;
    - plot V(φ, T_test) and mark both minima.
    """
    print("\n[Block B / Test 2] _solve_bounce on a single-field metastable configuration")

    # Search for a temperature where phi=0 is metastable and a deeper broken
    # minimum exists. We keep the search tiny and explicit for clarity.
    T_candidates = np.linspace(T0 + 0.2, T0 + 2.0, 10)
    chosen: Optional[Tuple[float, float]] = None

    for T_test in T_candidates:
        # Curvature at the origin
        m2_0 = d2V_dphi2(np.array([0.0], dtype=float), T_test)[0, 0]
        if m2_0 <= 0.0:
            # origin not even a local minimum
            continue

        # Analytic broken minimum somewhere at phi > 0
        try:
            phi_true = _broken_minimum_analytic(T_test)
        except Exception:
            continue

        V_false = float(V(np.array([0.0], dtype=float), T_test))
        V_true = float(V(np.array([phi_true], dtype=float), T_test))

        if V_true < V_false:
            chosen = (T_test, phi_true)
            break

    assert chosen is not None, "Could not find a suitable metastable configuration for the test."

    T_test, phi_true = chosen
    V_false = float(V(np.array([0.0], dtype=float), T_test))
    V_true = float(V(np.array([phi_true], dtype=float), T_test))
    m2_0 = d2V_dphi2(np.array([0.0], dtype=float), T_test)[0, 0]

    print(f"  Chosen T_test = {T_test:.4f}")
    print(f"  V(false, T_test) at phi=0         = {V_false:.4e}")
    print(f"  V(true,  T_test) at phi={phi_true:.4f} = {V_true:.4e}")
    print(f"  m^2(phi=0, T_test) = {m2_0:.4e} (> 0 ⇒ metastable origin)")

    # Wrap V and dV at fixed T in the format expected by _solve_bounce
    def V_fixed(x: np.ndarray) -> float:
        return float(V(np.asarray(x, dtype=float), T_test))

    def dV_fixed(x: np.ndarray) -> np.ndarray:
        return np.asarray(dV_dphi(np.asarray(x, dtype=float), T_test), dtype=float)

    x_high = np.array([0.0], dtype=float)
    x_low = np.array([phi_true], dtype=float)

    instanton, action, trantype = _solve_bounce(
        x_high=x_high,
        x_low=x_low,
        V_fixed=V_fixed,
        dV_fixed=dV_fixed,
        T=T_test,
        fullTunneling_params=None,
    )

    print(f"  _solve_bounce returned trantype = {trantype}")
    print(f"  Bounce action S3(T_test)        ≈ {action:.4e}")
    print(f"  instanton object type           = {type(instanton)}")

    assert trantype == 1, "We expect a first-order bounce (trantype=1) in this configuration."
    assert np.isfinite(action) and action > 0.0

    # Visualize the potential at T_test and mark false/true minima
    phi_grid = np.linspace(0.0, 1.1 * phi_true, 400)
    V_grid = V(phi_grid.reshape(-1, 1), T_test)

    fig, ax = plt.subplots()
    ax.set_title(f"Block B / Test 2: potential at T = {T_test:.2f}")
    ax.plot(phi_grid, V_grid, label=r"$V(\phi, T_{\rm test})$")
    ax.axvline(0.0, linestyle="--", label="false minimum (phi = 0)")
    ax.axvline(phi_true, linestyle="--", label=r"true minimum ($\phi_{\rm true}$)")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T_{\rm test})$")
    ax.legend()
    fig.tight_layout()
    plt.show()


def test_blockB_3_tunnelFromPhase_default_criterion():
    """
    Block B / Test 3 – full tunnelFromPhase with the default S(T)/T ≈ 140 criterion.

    Physics picture:
    ----------------
    Given the phase structure (symmetric + broken), tunnelFromPhase:
    - starts from the high-T symmetric phase;
    - scans temperatures to find where a first-order tunneling solution
      from symmetric → broken exists;
    - solves the bounce at each T using _solve_bounce;
    - searches for T_n where the nucleation criterion

            nuclCriterion(S3(T), T) ≡ S3(T)/T - 140 ≈ 0

      is satisfied.

    This test:
    ----------
    - builds the phase structure and gets the symmetric start_phase;
    - runs tunnelFromPhase with default parameters;
    - checks that we indeed get a first-order transition;
    - checks that S3/T at T_n is close to 140;
    - verifies that T_n lies below the critical temperature T_crit;
    - plots V(φ, T_n) and marks the false and true minima.
    """
    print("\n[Block B / Test 3] tunnelFromPhase with default S(T)/T ≈ 140 criterion")

    phases = _build_phases()
    start_key = getStartPhase(phases, V)
    start_phase = phases[start_key]

    print(f"  start_phase key  = {start_key}")
    print(f"  start_phase T-range = [{start_phase.T[0]:.1f}, {start_phase.T[-1]:.1f}]")

    result = tunnelFromPhase(
        phases=phases,
        start_phase=start_phase,
        V=V,
        dV=dV_dphi,
        Tmax=200.0,
        Ttol=1e-2,
        maxiter=80,
        phitol=1e-8,
        overlapAngle=45.0,
        # Use explicitly the default criterion to make the test self-documenting
        nuclCriterion=lambda S, T: S / (T + 1e-100) - 140.0,
        verbose=True,
        fullTunneling_params=None,
    )

    assert result is not None, "We expect a first-order transition in this Landau–Ginzburg model."

    Tnuc = float(result["Tnuc"])
    S3 = float(result["action"])
    S_over_T = S3 / (Tnuc + 1e-100)

    print("  --- tunnelFromPhase result ---")
    print(f"  T_nuc              ≈ {Tnuc:.4f}")
    print(f"  S3(T_nuc)          ≈ {S3:.4f}")
    print(f"  S3(T_nuc) / T_nuc  ≈ {S_over_T:.4f}")
    print(f"  low_phase key      = {result['low_phase']}")
    print(f"  high_phase key     = {result['high_phase']}")
    print(f"  trantype           = {result['trantype']} (1 = first order)")

    assert result["trantype"] == 1
    # We do not demand machine precision; the goal is that the criterion
    # is roughly satisfied.
    print(f"  S3/T - 140 ≈ {S_over_T - 140.0:.3f}")
    assert abs(S_over_T - 140.0) < 30.0

    # Compare with the critical temperature from Test 1
    Tcrit = _maxTCritForPhase(phases, start_phase, V, Ttol=1e-3)
    print(f"  Critical temperature T_crit ≈ {Tcrit:.4f}")
    print(f"  We expect T_nuc < T_crit for a supercooled first-order transition.")
    assert Tnuc <= Tcrit + 1e-2

    # Extract the actual field values of the minima at T_nuc
    phi_false = float(np.atleast_1d(result["high_vev"])[0])
    phi_true = float(np.atleast_1d(result["low_vev"])[0])

    V_false = float(V(np.array([phi_false], dtype=float), Tnuc))
    V_true = float(V(np.array([phi_true], dtype=float), Tnuc))

    print(f"  V(false, T_nuc) at phi={phi_false:.4f} = {V_false:.4e}")
    print(f"  V(true,  T_nuc) at phi={phi_true:.4f}  = {V_true:.4e}")

    assert V_true < V_false, "The 'low' phase should indeed have lower free energy."

    # Plot the potential at T_nuc with both minima marked
    phi_min = min(phi_false, phi_true)
    phi_max = max(phi_false, phi_true)
    pad = 0.2 * (phi_max - phi_min + 1.0)

    phi_grid = np.linspace(phi_min - pad, phi_max + pad, 400)
    V_grid = V(phi_grid.reshape(-1, 1), Tnuc)

    fig, ax = plt.subplots()
    ax.set_title(f"Block B / Test 3: potential at T_n ≈ {Tnuc:.2f}")
    ax.plot(phi_grid, V_grid, label=r"$V(\phi, T_n)$")
    ax.axvline(phi_false, linestyle="--", label="false minimum")
    ax.axvline(phi_true, linestyle="--", label="true minimum")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T_n)$")
    ax.legend()
    fig.tight_layout()
    plt.show()


def ew_like_nuclCriterion(S: float, T: float) -> float:
    """
    Example alternative nucleation criterion.

    This is a toy version of a condition of the type

        S3(T) / T = 4 ln(M_eff / T),

    with M_eff playing the role of an effective (reduced) Planck scale.
    It is **not** meant to be realistic for the toy model; it only shows
    how to plug a different nuclCriterion into tunnelFromPhase.

    Notes
    -----
    - We keep M_eff / T reasonably large so that the RHS is O(10–30),
      comparable in magnitude to the usual O(100) numbers.
    """
    M_eff = 1.0e18  # arbitrary scale in the same units as T
    T_eff = T + 1e-100
    return S / T_eff - 4.0 * np.log(M_eff / T_eff)


def demo_blockB_alternative_nucleation_criterion():
    """
    Block B / Demo – compare default and alternative nucleation criteria.

    This is **not** a pytest-style test (its name does not start with
    'test_'); it is just a demonstration that can be run manually.

    It:
    - computes T_nuc with the default S/T ≈ 140 criterion;
    - computes T_nuc_alt with an alternative ew_like_nuclCriterion;
    - prints both so you can see how sensitive T_n is to the choice of
      criterion in this toy model.
    """
    print("\n[Block B / Demo] tunnelFromPhase with an alternative nucleation criterion")

    phases = _build_phases()
    start_key = getStartPhase(phases, V)
    start_phase = phases[start_key]

    res_default = tunnelFromPhase(
        phases=phases,
        start_phase=start_phase,
        V=V,
        dV=dV_dphi,
        Tmax=200.0,
        nuclCriterion=lambda S, T: S / (T + 1e-100) - 140.0,
        verbose=False,
        fullTunneling_params=None,
    )

    res_alt = tunnelFromPhase(
        phases=phases,
        start_phase=start_phase,
        V=V,
        dV=dV_dphi,
        Tmax=200.0,
        nuclCriterion=ew_like_nuclCriterion,
        verbose=False,
        fullTunneling_params=None,
    )

    print("  Default criterion   (S/T - 140 = 0):")
    if res_default is None:
        print("    No nucleation found.")
    else:
        print(f"    T_nuc ≈ {res_default['Tnuc']:.4f},  S3/T ≈ {res_default['action'] / res_default['Tnuc']:.3f}")

    print("  Alternative criterion (S/T - 4 ln(M_eff/T) = 0):")
    if res_alt is None:
        print("    No nucleation found.")
    else:
        Tn_alt = float(res_alt["Tnuc"])
        S_alt = float(res_alt["action"])
        print(f"    T_nuc_alt ≈ {Tn_alt:.4f},  S3/T ≈ {S_alt / Tn_alt:.3f}")



# ---------------------------------------------------------------------------
# Optional: manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Running this file directly will execute all tests sequentially.
    # This is handy if you want to see the printed diagnostics without pytest.

    # Block B tests and demo
    test_blockB_1_potential_diff_and_Tcrit()
    test_blockB_2_solve_bounce_single_field_example()
    test_blockB_3_tunnelFromPhase_default_criterion()
    demo_blockB_alternative_nucleation_criterion()

    print("\n[Block B] All example tests executed.")
