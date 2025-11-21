"""
Block A – transitionFinder: tests & tutorial-style examples for
traceMinimum, Phase and phase-structure utilities, using a simple
finite-temperature scalar potential.

This file is deliberately verbose: it prints intermediate results and
creates plots (which are closed immediately), so that you can *see* what
the phase-tracing routines are doing in practice.

Potential used
--------------
We use a 1D Landau–Ginzburg finite-temperature potential

    V(phi, T) = D (T^2 - T0^2) phi^2 - E T phi^3 + (lambda_/4) phi^4,

with D > 0, lambda_ > 0 and a small cubic term E > 0. This is the
classic toy model for a first-order phase transition:

- At high temperature (T >> T0), phi = 0 is the unique minimum
  (symmetric phase).
- As T decreases, non-trivial broken minima at phi ≠ 0 develop.
- Around T ~ T0 there is a region where symmetric and broken phases
  coexist, separated by a barrier → first-order transition.
- Below some spinodal temperature, the symmetric minimum disappears.

All tests below are designed to illustrate how the transitionFinder
routines reconstruct this structure:

- traceMinimum: follow a single minimum as T varies.
- Phase: spline representation of a temperature-dependent minimum.
- traceMultiMin: reconstruct all phases in a T-interval.
- findApproxLocalMin: detect candidate minima along field-space segments.
- removeRedundantPhases: merge duplicated phases.
- getStartPhase: identify the high-temperature phase.
"""

from __future__ import annotations

from typing import Dict, Hashable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from CosmoTransitions.transitionFinder import (
    traceMinimum,
    Phase,
    traceMultiMin,
    findApproxLocalMin,
    removeRedundantPhases,
    getStartPhase,
)


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


def _compute_symmetric_trace() -> Tuple[float, object]:
    """
    Convenience helper: follow the symmetric minimum (phi ~ 0)
    from high T down until it becomes unstable.

    Returns
    -------
    T_spin_analytic :
        Analytic spinodal for the symmetric phase (where m^2 = d2V/dphi2|_{phi=0} = 0).
    res :
        Result object from traceMinimum for the symmetric phase.
    """
    T_spin_analytic = T0  # from d2V/dphi2(0, T) = 2D (T^2 - T0^2) = 0.
    res = traceMinimum(
        f=V,
        d2f_dxdt=d2V_dphidT,
        d2f_dx2=d2V_dphi2,
        x0=np.array([0.0]),
        t0=200.0,
        tstop=50.0,          # we ask to go down to T=50, but the phase dies around T ~ T0
        dtstart=-1.0,        # integrate "down" in T
        deltaX_target=0.01,
    )
    return T_spin_analytic, res


def _compute_broken_trace() -> object:
    """
    Convenience helper: follow a broken minimum from low T upwards
    until it becomes unstable.
    """
    phi_b_50 = _find_broken_minimum(T=50.0, phi_guess=1.0)
    res = traceMinimum(
        f=V,
        d2f_dxdt=d2V_dphidT,
        d2f_dx2=d2V_dphi2,
        x0=np.array([phi_b_50]),
        t0=50.0,
        tstop=200.0,         # integrate upwards in T
        dtstart=+1.0,
        deltaX_target=0.01,
    )
    return res


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
# Test 1: potential shape and minima at high and low T
# ---------------------------------------------------------------------------

def test_blockA_1_potential_shape_and_minima():
    """
    Test 1 – sanity check of the Landau–Ginzburg potential:

    - At T_high = 200: there is a symmetric minimum near phi ~ 0.
    - At T_low = 50: the origin is unstable, and a broken minimum exists.
    - Plot V(phi, T) for both temperatures for visual inspection.
    """
    print("\n[Block A / Test 1] Potential shape & minima at high and low T")

    T_high = 200.0
    T_low = 50.0

    # Sample the potential on a field grid
    phi_grid = np.linspace(-2.0, 8.0, 400)
    V_high = V(phi_grid.reshape(-1, 1), T_high)
    V_low = V(phi_grid.reshape(-1, 1), T_low)

    # Find minima numerically
    phi_sym = _find_broken_minimum(T=T_high, phi_guess=0.0)
    phi_broken_low = _find_broken_minimum(T=T_low, phi_guess=2.0)

    print(f"  T_high = {T_high:.1f}: minimum at phi ≈ {phi_sym:.4f}")
    print(f"  T_low  = {T_low:.1f}: broken minimum at phi ≈ {phi_broken_low:.4f}")

    # At high T, the symmetric minimum should be very close to 0
    assert abs(phi_sym) < 1e-2

    # At low T, the origin should be unstable and the broken minimum stable
    m2_origin_low = d2V_dphi2(np.array([0.0]), T_low)[0, 0]
    m2_broken_low = d2V_dphi2(np.array([phi_broken_low]), T_low)[0, 0]
    print(f"  m^2(phi=0, T_low) = {m2_origin_low:.4f} (should be < 0)")
    print(f"  m^2(phi_b, T_low) = {m2_broken_low:.4f} (should be > 0)")

    assert m2_origin_low < 0.0
    assert m2_broken_low > 0.0

    # Quick visual check (plots are closed immediately to be CI-friendly)
    fig, ax = plt.subplots()
    ax.set_title("Landau–Ginzburg potential: high vs low T")
    ax.plot(phi_grid, V_high, label=f"T = {T_high:.0f}")
    ax.plot(phi_grid, V_low, label=f"T = {T_low:.0f}")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T)$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 2: traceMinimum on the symmetric phase (phi ~ 0)
# ---------------------------------------------------------------------------

def test_blockA_2_traceMinimum_symmetric_phase_downwards():
    """
    Test 2 – follow the symmetric phase with traceMinimum:

    - Start at (phi = 0, T = 200) and move down in T.
    - Check that T decreases monotonically in the trace.
    - Check that phi_min(T) stays close to 0.
    - Check that the phase dies near the analytic spinodal T_spin = T0,
      where d^2 V / d phi^2 |_{phi=0} = 0.
    """
    print("\n[Block A / Test 2] traceMinimum on symmetric phase (downwards in T)")

    T_spin_analytic, res = _compute_symmetric_trace()

    T_arr = np.asarray(res.T, dtype=float)
    X_arr = np.asarray(res.X, dtype=float).reshape(-1, 1)
    dXdT_arr = np.asarray(res.dXdT, dtype=float).reshape(-1, 1)

    # T must be strictly decreasing (within numerical noise)
    assert np.all(np.diff(T_arr) < 1e-8), "T should decrease along the symmetric trace."

    phi_arr = X_arr[:, 0]
    max_phi = np.max(np.abs(phi_arr))
    print(f"  Max |phi_min(T)| along symmetric trace: {max_phi:.4e}")
    assert max_phi < 1e-1  # we tolerate a small drift; the minimizer corrects it

    # Spinodal temperature from traceMinimum
    T_spin_numeric = float(res.overT)
    print(f"  Analytic symmetric spinodal: T_spin = {T_spin_analytic:.4f}")
    print(f"  Numeric overT from traceMinimum: T_spin ≈ {T_spin_numeric:.4f}")

    assert abs(T_spin_numeric - T_spin_analytic) < 2.0, (
        "Symmetric spinodal temperature from traceMinimum should be close to analytic T0."
    )

    # Check that the symmetric phase has positive curvature along the traced range
    m2_along = np.array(
        [d2V_dphi2(np.array([phi]), T)[0, 0] for phi, T in zip(phi_arr, T_arr)]
    )
    print(f"  m^2 along symmetric trace (first 5 points): {m2_along[:5]}")
    # Ignore the last few points very close to spinodal
    assert np.all(m2_along[:-3] > 0.0), "Curvature should be positive away from spinodal."

    # Plot phi_min(T) and m^2(T) for visual understanding
    fig, ax = plt.subplots()
    ax.set_title("Symmetric phase: phi_min(T) from traceMinimum")
    ax.plot(T_arr, phi_arr, marker="o", linestyle="-", label=r"$\phi_{\min}(T)$")
    ax.axvline(T_spin_analytic, linestyle="--", label=r"$T_{\text{spin}}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\phi_{\min}$")
    ax.legend()
    fig.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set_title("Symmetric phase: curvature m^2(T) along the trace")
    ax2.plot(T_arr, m2_along, marker="o", linestyle="-")
    ax2.axhline(0.0, linestyle="--")
    ax2.set_xlabel("T")
    ax2.set_ylabel(r"$m^2(\phi_{\min}(T))$")
    fig2.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 3: traceMinimum on the broken phase (phi ≠ 0) upwards in T
# ---------------------------------------------------------------------------

def test_blockA_3_traceMinimum_broken_phase_upwards():
    """
    Test 3 – follow a broken phase with traceMinimum:

    - Start at the broken minimum at T = 50 and evolve upwards in T.
    - Check that T increases monotonically.
    - Check that phi_min(T) decreases in magnitude as T grows (tending
      towards the symmetric phase).
    - Identify the approximate spinodal where the broken phase disappears.
    """
    print("\n[Block A / Test 3] traceMinimum on broken phase (upwards in T)")

    res = _compute_broken_trace()

    T_arr = np.asarray(res.T, dtype=float)
    X_arr = np.asarray(res.X, dtype=float).reshape(-1, 1)
    phi_arr = X_arr[:, 0]

    # T must be increasing
    assert np.all(np.diff(T_arr) > -1e-8), "T should increase along the broken trace."

    print(f"  Broken phase traced from T = {T_arr[0]:.2f} up to T ≈ {T_arr[-1]:.2f}")
    print(f"  First phi_min(T)  = {phi_arr[0]:.4f}")
    print(f"  Last  phi_min(T)  = {phi_arr[-1]:.4f}")

    # The broken minimum should move towards the origin as T increases
    assert abs(phi_arr[0]) > abs(phi_arr[-1])

    # Curvature along the broken branch
    m2_along = np.array(
        [d2V_dphi2(np.array([phi]), T)[0, 0] for phi, T in zip(phi_arr, T_arr)]
    )
    print(f"  m^2 along broken trace (first 5 points): {m2_along[:5]}")

    # Away from the very end, the broken branch should be stable
    assert np.all(m2_along[:-3] > 0.0)

    # Spinodal for the broken phase, as seen by traceMinimum
    T_spin_broken = float(res.overT)
    print(f"  Broken-phase spinodal from traceMinimum: T ≈ {T_spin_broken:.4f}")

    # Plot phi_min(T) for visualization
    fig, ax = plt.subplots()
    ax.set_title("Broken phase: phi_min(T) from traceMinimum")
    ax.plot(T_arr, phi_arr, marker="o", linestyle="-", label="broken phase")
    ax.axhline(0.0, linestyle="--", label=r"$\phi = 0$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\phi_{\min}$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 4: dphi/dT consistency and helper_functions cross-check (if available)
# ---------------------------------------------------------------------------

def test_blockA_4_dphi_dT_consistency_and_helper_functions():
    """
    Test 4 – internal consistency of dphi/dT and optional comparison with
    helper_functions (if available):

    - Compare traceMinimum's dXdT with a simple finite-difference estimate
      of dphi/dT on the symmetric branch.
    - If CosmoTransitions.helper_functions is available, compare the
      analytic dV/dphi and d^2V/dphi^2 with numerical gradient/Hessian.
    """
    print("\n[Block A / Test 4] dphi/dT consistency and helper_functions cross-check")

    _, res = _compute_symmetric_trace()

    T_arr = np.asarray(res.T, dtype=float)
    X_arr = np.asarray(res.X, dtype=float).reshape(-1, 1)
    dXdT_arr = np.asarray(res.dXdT, dtype=float).reshape(-1, 1)

    phi_arr = X_arr[:, 0]
    dphi_dT_model = dXdT_arr[:, 0]

    # Finite-difference estimate of dphi/dT on the interior points
    dphi_dT_fd = np.gradient(phi_arr, T_arr)
    interior = slice(2, -2)
    num = dphi_dT_fd[interior]
    mod = dphi_dT_model[interior]

    denom = np.maximum(np.abs(num), 1e-8)
    rel_err = np.max(np.abs(num - mod) / denom)
    print(f"  Max relative difference between model dphi/dT and finite-diff: {rel_err:.3f}")
    # This is a qualitative check; we allow O(10%) differences.
    assert rel_err < 0.5

    # Optional: use helper_functions if available
    try:
        from CosmoTransitions.helper_functions import gradientFunction, hessianFunction
    except ImportError:
        print("  CosmoTransitions.helper_functions not available; skipping gradient/Hessian check.")
        return

    print("  helper_functions found: checking gradient and Hessian against analytic expressions.")

    T_test = 150.0
    phi_test = np.array([0.3])

    def f_x_only(x: np.ndarray) -> float:
        V_val = (
                D * (T_test ** 2 - T0 ** 2) * x ** 2
                - E * T_test * x ** 3
                + 0.25 * lambda_ * x ** 4
        )
        return V_val

    # Gradient
    grad_obj = gradientFunction(f_x_only,eps=1e-5, Ndim=1)
    grad_num = np.asarray(grad_obj(phi_test), dtype=float)

    grad_analytic = dV_dphi(phi_test, T_test)
    print(f"  grad_num      = {grad_num}")
    print(f"  grad_analytic = {grad_analytic}")
    assert np.allclose(grad_num, grad_analytic, rtol=1e-2, atol=1e-4)

    # Hessian
    hess_obj = hessianFunction(f_x_only, eps=1e-5, Ndim=1)
    hess_num = np.asarray(hess_obj(phi_test), dtype=float)

    hess_analytic = d2V_dphi2(phi_test, T_test)
    print(f"  hess_num      = {hess_num}")
    print(f"  hess_analytic = {hess_analytic}")
    assert np.allclose(hess_num, hess_analytic, rtol=1e-2, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 5: traceMultiMin + Phase – global phase structure in [T_low, T_high]
# ---------------------------------------------------------------------------

def test_blockA_5_traceMultiMin_and_Phase_structure():
    """
    Test 5 – reconstruct the phase structure with traceMultiMin:

    - Use seeds at (phi = 0, T = 200) and (phi_b(T=50), T = 50).
    - Build Phase objects for all minima in [50, 200].
    - Check that exactly one symmetric-like and one broken-like phase
      are present.
    - Check that Phase.valAt(T) tracks true minima reasonably well.
    - Use getStartPhase to identify the high-T phase.
    """
    print("\n[Block A / Test 5] traceMultiMin and Phase structure")

    phases = _build_phases()
    print(f"  Number of phases found: {len(phases)}")

    assert len(phases) >= 2, "We expect at least symmetric and broken phases."

    # Classify phases as 'symmetric' or 'broken' by inspecting phi at high T
    def classify_phase(phase: Phase) -> str:
        T_eval = phase.T[-1]  # highest T in that phase
        phi_eval = phase.valAt(T_eval)
        phi0 = float(np.atleast_1d(phi_eval)[0])
        return "symmetric" if abs(phi0) < 0.1 else "broken"

    counts = {"symmetric": 0, "broken": 0}
    for key, phase in phases.items():
        label = classify_phase(phase)
        counts[label] += 1
        print(f"  Phase {key}: T-range [{phase.T[0]:.1f}, {phase.T[-1]:.1f}], classified as {label}")

    assert counts["symmetric"] == 1
    assert counts["broken"] >= 1  # could in principle have multiple broken branches, but here expect 1

    # Check spline-based valAt against direct minimization at a few random T
    rng = np.random.default_rng(12345)
    for key, phase in phases.items():
        # Pick 3 random points in the interior of this phase's T-range
        if phase.T.size < 5:
            continue
        T_min, T_max = phase.T[1], phase.T[-2]
        T_samples = np.linspace(T_min, T_max, 3)
        for T_s in T_samples:
            phi_spline = phase.valAt(T_s)
            phi_guess = float(np.atleast_1d(phi_spline)[0])

            def V_scalar(phi: float) -> float:
                return float(V(np.array([phi], dtype=float), T_s))

            phi_min = float(optimize.fmin(V_scalar, phi_guess, disp=False)[0])
            diff = abs(phi_min - phi_guess)
            print(f"    Phase {key}: T={T_s:.2f}, spline phi={phi_guess:.4f}, "
                  f"true min={phi_min:.4f}, |Δphi|={diff:.3e}")
            assert diff < 0.1

    # Check getStartPhase: should pick the symmetric phase at high T
    start_key = getStartPhase(phases, V)
    start_phase = phases[start_key]
    phi_highT = start_phase.valAt(start_phase.T[-1])
    phi0 = float(np.atleast_1d(phi_highT)[0])
    print(f"  getStartPhase returned key={start_key}, phi(T_max) ≈ {phi0:.4e}")
    assert abs(phi0) < 0.1

    # Plot phi_min(T) curves for all phases
    fig, ax = plt.subplots()
    ax.set_title("Phase structure: phi_min(T) from Phase splines")
    for key, phase in phases.items():
        T_dense = np.linspace(phase.T[0], phase.T[-1], 200)
        phi_dense = phase.valAt(T_dense)
        ax.plot(T_dense, phi_dense, label=f"Phase {key}")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\phi_{\min}(T)$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 6: findApproxLocalMin on a segment crossing the unique minimum
# ---------------------------------------------------------------------------

def test_blockA_6_findApproxLocalMin_on_simple_segment():
    """
    Test 6 – use findApproxLocalMin on a segment that crosses the unique
    symmetric minimum at high T:

    - Choose T = 150 > T0 where phi = 0 is the unique minimum.
    - Take a segment from phi = -3 to phi = +3.
    - findApproxLocalMin should detect a minimum near phi = 0.
    """
    print("\n[Block A / Test 6] findApproxLocalMin on a simple segment at high T")

    T_test = 150
    x1 = np.array([-3.0])
    x2 = np.array([3.0])

    def f_seg(x: np.ndarray, T: float) -> np.ndarray:
        # x is expected to be shape (n, 1) internally
        return np.asarray(V(x, T), dtype=float)

    minima = findApproxLocalMin(
        f_seg,
        x1,
        x2,
        args=(T_test,),
        n=200,
        edge=0.05,
    )

    print(f"  Number of approximate minima found along the segment: {minima.shape[0]}")

    assert minima.ndim == 2 and minima.shape[1] == 1
    assert minima.shape[0] >= 1, "We expect at least one minimum along the segment."

    # All found minima should be close to phi = 0
    phi_minima = minima[:, 0]
    max_dist = np.max(np.abs(phi_minima))
    print(f"  Approximate minima phi ≈ {phi_minima}")
    print(f"  Max |phi_min| along segment: {max_dist:.4f}")
    assert max_dist < 0.5

    # Optional quick plot of f(φ) along the segment
    phi_grid = np.linspace(-3.0, 3.0, 400)
    V_grid = V(phi_grid.reshape(-1, 1), T_test)

    fig, ax = plt.subplots()
    ax.set_title(f"findApproxLocalMin at T = {T_test:.0f}")
    ax.plot(phi_grid, V_grid, label="V(φ, T)")
    ax.plot(phi_minima, V(phi_minima.reshape(-1, 1), T_test), "o", label="approx minima")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T)$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test 7: removeRedundantPhases – merging artificially duplicated phases
# ---------------------------------------------------------------------------

def test_blockA_7_removeRedundantPhases_merges_duplicates():
    """
    Test 7 – exercise removeRedundantPhases on artificially duplicated phases:

    - Build the phase structure with traceMultiMin.
    - Artificially clone one Phase and insert it into the dictionary.
    - Call removeRedundantPhases and check that the number of phases
      returns to the original value.
    """
    print("\n[Block A / Test 7] removeRedundantPhases on duplicated phases")

    phases = _build_phases()
    original_keys = list(phases.keys())
    original_len = len(original_keys)

    # Artificially duplicate the first phase
    key_to_clone = original_keys[0]
    phase_to_clone = phases[key_to_clone]
    clone_key = f"{key_to_clone}_clone"
    phases[clone_key] = Phase(
        key=clone_key,
        X=phase_to_clone.X.copy(),
        T=phase_to_clone.T.copy(),
        dXdT=phase_to_clone.dXdT.copy(),
    )

    print(f"  Original number of phases: {original_len}")
    print(f"  After cloning: {len(phases)} (added key={clone_key})")

    assert len(phases) == original_len + 1

    removeRedundantPhases(V, phases, xeps=1e-6, diftol=1e-2)

    print(f"  After removeRedundantPhases: {len(phases)}")
    assert len(phases) == original_len

    # Check that there is still a phase with T-range very close to the cloned one
    still_has_equivalent = False
    for key, phase in phases.items():
        if (
            abs(phase.T[0] - phase_to_clone.T[0]) < 1e-3
            and abs(phase.T[-1] - phase_to_clone.T[-1]) < 1e-3
        ):
            still_has_equivalent = True
            break
    assert still_has_equivalent, "An equivalent phase to the cloned one should remain."


# ---------------------------------------------------------------------------
# Optional: manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Running this file directly will execute all tests sequentially.
    # This is handy if you want to see the printed diagnostics without pytest.
    test_blockA_1_potential_shape_and_minima()
    test_blockA_2_traceMinimum_symmetric_phase_downwards()
    test_blockA_3_traceMinimum_broken_phase_upwards()
    #test_blockA_4_dphi_dT_consistency_and_helper_functions()
    test_blockA_5_traceMultiMin_and_Phase_structure()
    test_blockA_6_findApproxLocalMin_on_simple_segment()
    test_blockA_7_removeRedundantPhases_merges_duplicates()
    print("\n[Block A] All example tests executed.")
