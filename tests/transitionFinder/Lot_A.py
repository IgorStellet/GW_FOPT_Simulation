"""
Block A – transitionFinder: tests & tutorial-style examples for
traceMinimum and Phase using a simple finite-temperature scalar potential.

This file is deliberately verbose: it prints intermediate results and
produces plots, so that you can *see* what the phase-tracing routines
are doing in practice.

Potential used
--------------
We use a 1D Landau-Ginzburg finite-temperature potential

    V(phi, T) = D * (T^2 - T0^2) * phi^2 + (lambda_/4) * phi^4

with
    D       = 0.5
    lambda_ = 1.0
    T0      = 1.0

This gives:
  - For T > T0: single, symmetric minimum at phi = 0.
  - For T < T0: two degenerate minima at phi = ±phi_min(T).
  - At T = T0: second-order transition (mass^2 -> 0 at the origin).

These examples focus on:
  - traceMinimum: following a local minimum as T changes.
  - Phase: smoothing the traced minimum with a spline and interpolating it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions import transitionFinder

# ---------------------------------------------------------------------------
# Model parameters: simple Landau-Ginzburg finite-T potential
# ---------------------------------------------------------------------------

D_LANDAU: float = 0.5
LAMBDA_LANDAU: float = 1.0
T0_CRIT: float = 1.0


def V_finite_T(x: np.ndarray, T: float) -> float:
    """
    Finite-temperature Landau-Ginzburg potential V(phi, T) for a single scalar.

    Parameters
    ----------
    x : array_like, shape (1,)
        Field value(s). For this test we use a single scalar phi.
    T : float
        Temperature.

    Returns
    -------
    float
        Potential V(phi, T).
    """
    x_arr = np.atleast_1d(x)
    phi = float(x_arr[0])
    return D_LANDAU * (T**2 - T0_CRIT**2) * phi**2 + 0.25 * LAMBDA_LANDAU * phi**4


def d2V_dphidT(x: np.ndarray, T: float) -> np.ndarray:
    """
    Mixed derivative d/dT (dV/dphi) for the test potential.

    Starting from
        V(phi, T) = D (T^2 - T0^2) phi^2 + (lambda/4) phi^4
    we have
        dV/dphi = 2 D (T^2 - T0^2) phi + lambda phi^3
    so
        d/dT (dV/dphi) = 4 D T phi

    Parameters
    ----------
    x : array_like, shape (1,)
        Field value(s).
    T : float
        Temperature.

    Returns
    -------
    ndarray, shape (1,)
        d/dT (gradient V) evaluated at (phi, T).
    """
    x_arr = np.atleast_1d(x)
    phi = float(x_arr[0])
    return np.array([4.0 * D_LANDAU * T * phi], dtype=float)


def d2V_dphi2(x: np.ndarray, T: float) -> np.ndarray:
    """
    Hessian d^2V/dphi^2 for the test potential.

    From
        dV/dphi = 2 D (T^2 - T0^2) phi + lambda phi^3
    we obtain
        d^2V/dphi^2 = 2 D (T^2 - T0^2) + 3 lambda phi^2

    Parameters
    ----------
    x : array_like, shape (1,)
        Field value(s).
    T : float
        Temperature.

    Returns
    -------
    ndarray, shape (1, 1)
        Hessian matrix evaluated at (phi, T).
    """
    x_arr = np.atleast_1d(x)
    phi = float(x_arr[0])
    second = 2.0 * D_LANDAU * (T**2 - T0_CRIT**2) + 3.0 * LAMBDA_LANDAU * phi**2
    return np.array([[second]], dtype=float)


def analytic_minima(T: float) -> np.ndarray:
    """
    Analytic positions of the minima for the Landau potential.

    For T < T0:
        phi_min^2 = 2 D (T0^2 - T^2) / lambda
    For T >= T0:
        only minimum is at phi = 0.

    Parameters
    ----------
    T : float
        Temperature.

    Returns
    -------
    ndarray
        Array with the minima positions. It has length 1 (phi=0) or 2 (±phi_min).
    """
    if T <= T0_CRIT:
        phi2 = 2.0 * D_LANDAU * (T0_CRIT**2 - T**2) / LAMBDA_LANDAU
        phi_min = np.sqrt(max(phi2, 0.0))
        return np.array([-phi_min, +phi_min], dtype=float)
    return np.array([0.0], dtype=float)


# ---------------------------------------------------------------------------
# Helper: thin wrapper around traceMinimum for our 1D potential
# ---------------------------------------------------------------------------

def run_trace_minimum(
    x0: float,
    t0: float,
    tstop: float,
    dtstart: float,
    deltaX_target: float = 1e-3,
) -> transitionFinder._traceMinimum_rval:
    """
    Convenience wrapper: run traceMinimum on the 1D Landau potential.

    Parameters
    ----------
    x0 : float
        Initial guess for the minimum (phi) at temperature t0.
    t0 : float
        Initial temperature where tracing starts.
    tstop : float
        Final temperature where tracing stops (may be above or below t0).
    dtstart : float
        Initial step size in T (sign determines direction).
    deltaX_target : float, optional
        Target step size in field space for the trace.

    Returns
    -------
    _traceMinimum_rval
        Namedtuple with (X, T, dXdT, overX, overT).
    """
    result = transitionFinder.traceMinimum(
        f=V_finite_T,
        d2f_dxdt=d2V_dphidT,
        d2f_dx2=d2V_dphi2,
        x0=np.array([x0], dtype=float),
        t0=float(t0),
        tstop=float(tstop),
        dtstart=float(dtstart),
        deltaX_target=float(deltaX_target),
        # A bit conservative: smallish dt so the trace is smooth.
        dtabsMax=20.0,
        dtfracMax=0.25,
        dtmin=1e-3,
        deltaX_tol=1.2,
        minratio=1e-2,
    )
    return result


# ---------------------------------------------------------------------------
# Test / Example A1: shape of V(phi, T) at low and high T
# ---------------------------------------------------------------------------

def test_A1_potential_shapes():
    """
    Test A1 – Visualize the Landau potential at T = 0, 1, 2.

    This is a sanity check:
      - For T = 0, we expect minima near phi = ±1.
      - For T = 1 = T0, we expect a single flat-ish minimum at phi = 0.
      - For T = 2, we expect a single sharp minimum at phi = 0.
    """
    print("\n[Block A – Test A1] Landau finite-T potential: shapes at T=0, 1, 2")

    phi_grid = np.linspace(-2.5, 2.5, 400)
    T_values = [0.0, 1.0, 2.0]

    fig, ax = plt.subplots()
    for T in T_values:
        V_vals = np.array([V_finite_T(np.array([phi]), T) for phi in phi_grid])
        label = f"T = {T:.1f}"
        ax.plot(phi_grid, V_vals, label=label)

        mins = analytic_minima(T)
        print(f"  T = {T:.1f} → analytic minima phi ≈ {mins}")
        for phi_min in mins:
            ax.axvline(phi_min, linestyle="--", alpha=0.5)

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T)$")
    ax.set_title("Landau finite-temperature potential (Block A – Test A1)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    #fig.savefig("transitionFinder_blockA_A1_potential_shapes.png")

    # Numerical sanity checks
    mins_T0 = analytic_minima(0.0)
    assert np.allclose(mins_T0, np.array([-1.0, 1.0]), atol=1e-6)
    mins_T2 = analytic_minima(2.0)
    assert np.allclose(mins_T2, np.array([0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Test / Example A2: trace the symmetric phase (phi ~ 0) from high T downwards
# ---------------------------------------------------------------------------

def test_A2_trace_symmetric_phase():
    """
    Test A2 – Trace the symmetric-phase minimum from T = 2 down to T = 0.

    We start at T = 2 with phi ≈ 0 (symmetric minimum) and trace downwards.
    The algorithm should follow the local minimum at phi ≈ 0 until it
    ceases to be a minimum near T ≈ T0.

    We then:
      - print a summary of the trace,
      - plot phi(T) for the traced branch,
      - check that phi ≈ 0 for T >> T0.
    """
    print("\n[Block A – Test A2] Tracing symmetric phase from T=2 → T=0")

    T_start = 2.0
    T_stop = 0.0
    dtstart = -0.05  # negative sign: tracing downwards in T
    x0 = 0.0

    res = run_trace_minimum(x0=x0, t0=T_start, tstop=T_stop,
                            dtstart=dtstart, deltaX_target=1e-3)

    T_array = res.T
    phi_array = res.X[:, 0]
    overT = res.overT
    overX = res.overX

    print(f"  Number of traced points: {len(T_array)}")
    print(f"  T range (trace): {T_array[0]:.4f} → {T_array[-1]:.4f}")
    print(f"  overT (where phase disappears or step too small): {overT:.4f}")
    print(f"  overX (field there): {overX}")

    # Plot phi(T)
    fig, ax = plt.subplots()
    ax.plot(T_array, phi_array, "-o", markersize=3, label="symmetric phase (traceMinimum)")
    ax.axvline(T0_CRIT, linestyle="--", label=r"$T_0$ (critical)")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\phi_{\min}(T)$")
    ax.set_title("Trace of symmetric phase (Block A – Test A2)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    #fig.savefig("transitionFinder_blockA_A2_trace_symmetric_phase.png")

    # Sanity: for sufficiently high T, phi should be close to zero
    mask_highT = T_array > 1.5
    if np.any(mask_highT):
        assert np.allclose(phi_array[mask_highT], 0.0, atol=1e-2)


# ---------------------------------------------------------------------------
# Test / Example A3: trace the broken phase (phi > 0) from low T upwards
# ---------------------------------------------------------------------------

def test_A3_trace_broken_phase():
    """
    Test A3 – Trace the broken-phase minimum from T = 0 up to T = 1.5.

    We start at T = 0 with phi ≈ +1 (one of the broken minima) and trace
    upwards in T. The algorithm should follow the broken minimum until it
    merges back into the symmetric minimum near T ≈ T0.

    We then:
      - print a summary of the trace,
      - plot phi(T) for the traced branch,
      - compare with the analytic expectation for T < T0.
    """
    print("\n[Block A – Test A3] Tracing broken phase from T=0 → T=1.5")

    T_start = 0.0
    T_stop = 1.5
    dtstart = +0.05  # upwards in T
    x0 = analytic_minima(0.0)[1]  # +1.0

    res = run_trace_minimum(x0=x0, t0=T_start, tstop=T_stop,
                            dtstart=dtstart, deltaX_target=1e-3)

    T_array = res.T
    phi_array = res.X[:, 0]
    overT = res.overT
    overX = res.overX

    print(f"  Number of traced points: {len(T_array)}")
    print(f"  T range (trace): {T_array[0]:.4f} → {T_array[-1]:.4f}")
    print(f"  overT (where phase disappears or step too small): {overT:.4f}")
    print(f"  overX (field there): {overX}")

    # Plot phi(T) and compare with analytic broken minimum (for T < T0)
    fig, ax = plt.subplots()
    ax.plot(T_array, phi_array, "-o", markersize=3, label="broken phase (traceMinimum)")
    ax.axvline(T0_CRIT, linestyle="--", label=r"$T_0$ (critical)")

    T_analytic = np.linspace(0.0, T0_CRIT, 100)
    phi_analytic = np.array([analytic_minima(T)[1] for T in T_analytic])
    ax.plot(T_analytic, phi_analytic, "-", alpha=0.5,
            label="broken phase (analytic)")

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\phi_{\min}(T)$")
    ax.set_title("Trace of broken phase (Block A – Test A3)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    #fig.savefig("transitionFinder_blockA_A3_trace_broken_phase.png")

    # Sanity: for T well below T0, traced phi should agree with analytic curve
    mask_lowT = T_array < 0.6 * T0_CRIT
    if np.any(mask_lowT):
        phi_interp = np.interp(T_array[mask_lowT], T_analytic, phi_analytic)
        assert np.allclose(phi_array[mask_lowT], phi_interp, atol=1e-2)


# ---------------------------------------------------------------------------
# Test / Example A4: Phase class & spline interpolation over a traced minimum
# ---------------------------------------------------------------------------
def test_A4_phase_object_and_spline():
    """
    Test A4 – Build a Phase object from a traced broken minimum and
    test its spline interpolation.

    Steps:
      1. Re-use the broken-phase trace from Test A3.
      2. Construct Phase(key=0, X, T, dXdT).
      3. Check that phase.valAt(T) reproduces X at the original sample points.
      4. Evaluate phase.valAt(T_dense) on a dense grid and compare visually
         with the original sampled points.
      5. (Diagnostic only) Compute a numerical derivative from the spline.
    """
    print("\n[Block A – Test A4] Phase object and spline interpolation")

    # 1. Trace the broken phase again (small overhead, cleaner test)
    T_start = 0.0
    T_stop = 1.5
    dtstart = +0.05
    x0 = analytic_minima(0.0)[1]

    res = run_trace_minimum(
        x0=x0, t0=T_start, tstop=T_stop,
        dtstart=dtstart, deltaX_target=1e-3
    )

    X = res.X          # shape (N, 1)
    T = res.T          # shape (N,)
    dXdT = res.dXdT    # shape (N, 1)

    # 2. Build the Phase object
    phase = transitionFinder.Phase(key=0, X=X, T=T, dXdT=dXdT)
    print("  Phase representation:")
    print("   ", repr(phase))

    # 3. Checar que valAt(T) recupera os pontos originais
    phi_on_grid = phase.valAt(T)   # deve ter mesma forma que X
    assert phi_on_grid.shape == X.shape
    assert np.allclose(phi_on_grid, X, atol=1e-8)

    # 4. Avaliar em um grid denso para visualização
    T_dense = np.linspace(T.min(), T.max(), 200)
    phi_dense = phase.valAt(T_dense)  # shape (N_dense, 1)

    fig, ax = plt.subplots()
    ax.plot(T, X[:, 0], "o", label="traceMinimum samples")
    ax.plot(T_dense, phi_dense[:, 0], "-", label="Phase spline")
    ax.axvline(T0_CRIT, linestyle="--", label=r"$T_0$ (critical)")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\phi_{\min}(T)$")
    ax.set_title("Phase spline interpolation (Block A – Test A4)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    #fig.savefig("transitionFinder_blockA_A4_phase_spline.png")

    # 5. Derivada aproximada (diagnóstico, sem assert forte por enquanto)
    #    Usamos diferenças finitas no spline em T_dense.
    dphi_dT_dense = np.gradient(phi_dense[:, 0], T_dense)
    print("  Sample of dphi/dT from spline (T, dphi/dT):")
    for T_i, dphi_i in zip(T_dense[::40], dphi_dT_dense[::40]):
        print(f"    T = {T_i:6.3f}, dphi/dT ≈ {dphi_i: .3e}")

    # Se quiser um teste bem suave de consistência de sinal / ordem de grandeza,
    # podemos, por exemplo, comparar o sinal da derivada em um intervalo onde
    # sabemos que phi(T) está diminuindo:
    mask_mid = (T_dense > 0.2 * T0_CRIT) & (T_dense < 0.9 * T0_CRIT)
    if np.any(mask_mid):
        # phi decresce com T nesse intervalo → derivada deve ser negativa em média
        mean_dphi = dphi_dT_dense[mask_mid].mean()
        print(f"  Mean dphi/dT on 0.2–0.9 T0 ≈ {mean_dphi:.3e}")
        assert mean_dphi < 0.0


# ---------------------------------------------------------------------------
# Run all examples when executing this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # These calls make it easy to run the examples by hand:
    #   python test_transitionFinder_blockA.py
    test_A1_potential_shapes()
    test_A2_trace_symmetric_phase()
    test_A3_trace_broken_phase()
    test_A4_phase_object_and_spline()
    print("\n[Block A] All example tests finished.")

# ------------------------------------------------------------------
# Additional utilities for Block A tests: thermal 1D potential
# ------------------------------------------------------------------

from typing import Dict, Hashable

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions.transitionFinder import (
    traceMultiMin,
    findApproxLocalMin,
    removeRedundantPhases,
    getStartPhase,
    Phase,
)

# Electroweak-like single-field thermal potential:
#   V(φ, T) = D (T^2 - T0^2) φ^2 - E T φ^3 + λ/4 φ^4
_D_TM = 0.4
_E_TM = 0.1
_LAMBDA_TM = 0.4
_T0_TM = 1.0


def V_thermal_1d(x: np.ndarray, T: float) -> float:
    """
    Scalar potential used in the Block A multi-phase tests.

    Parameters
    ----------
    x : array_like, shape (1,)
        Field value φ packed in a 1-element array.
    T : float
        Temperature.

    Returns
    -------
    float
        Potential V(φ, T).
    """
    x = np.asarray(x, dtype=float).ravel()
    phi = x[0]

    return (
        _D_TM * (T**2 - _T0_TM**2) * phi**2
        - _E_TM * T * phi**3
        + 0.25 * _LAMBDA_TM * phi**4
    )


def d2V_thermal_dxdt(x: np.ndarray, T: float) -> np.ndarray:
    """
    ∂/∂T of the gradient ∂V/∂φ, returned as a length-1 array.
    """
    x = np.asarray(x, dtype=float).ravel()
    phi = x[0]
    # d/dT (∂V/∂φ) = d/dT [ 2 D (T^2 - T0^2) φ - 3 E T φ^2 + λ φ^3 ]
    d2 = 4.0 * _D_TM * T * phi - 3.0 * _E_TM * phi**2
    return np.array([d2], dtype=float)


def d2V_thermal_dx2(x: np.ndarray, T: float) -> np.ndarray:
    """
    Hessian ∂^2 V / ∂φ^2, returned as a (1, 1) array.
    """
    x = np.asarray(x, dtype=float).ravel()
    phi = x[0]
    # ∂^2V/∂φ^2 = 2 D (T^2 - T0^2) - 6 E T φ + 3 λ φ^2
    d2 = 2.0 * _D_TM * (T**2 - _T0_TM**2) - 6.0 * _E_TM * T * phi + 3.0 * _LAMBDA_TM * phi**2
    return np.array([[d2]], dtype=float)


def _grid_minimum_for_T(T: float, phi_min: float = -0.5, phi_max: float = 3.0, n: int = 400) -> np.ndarray:
    """
    Crude 1D grid search to find a minimum of V_thermal_1d at fixed T.
    This is only used to build good seeds for traceMultiMin tests.
    """
    phis = np.linspace(phi_min, phi_max, n)
    vals = np.array([V_thermal_1d(np.array([phi]), T) for phi in phis])
    idx_min = np.argmin(vals)
    return np.array([phis[idx_min]], dtype=float)


def build_thermal_phases_example(
    t_low: float = 0.6,
    t_high: float = 1.2,
    deltaX_target: float = 1e-2,
) -> Dict[Hashable, Phase]:
    """
    Helper: reconstruct phase structure for the thermal 1D potential
    using traceMultiMin, to be reused by multiple tests.
    """
    # Seed 1: symmetric phase at high T (φ = 0 is exactly a stationary point).
    x_sym = np.array([0.0], dtype=float)
    T_sym = 1.2

    # Seed 2: broken-like minimum at low T (approximate via grid search).
    T_broken = 0.7
    x_broken = _grid_minimum_for_T(T_broken)

    points = [
        (x_sym, T_sym),
        (x_broken, T_broken),
    ]

    phases = traceMultiMin(
        f=V_thermal_1d,
        d2f_dxdt=d2V_thermal_dxdt,
        d2f_dx2=d2V_thermal_dx2,
        points=points,
        tLow=t_low,
        tHigh=t_high,
        deltaX_target=deltaX_target,
        dtstart=5e-3,
        tjump=5e-3,
        forbidCrit=None,
        single_trace_args={"dtabsMax": 10.0, "dtfracMax": 0.25},
        local_min_args={"n": 80, "edge": 0.05},
    )

    return phases


# ------------------------------------------------------------------
# Test 5 – findApproxLocalMin on a simple 1D triple-well potential
# ------------------------------------------------------------------

def test_5_findApproxLocalMin_triple_well():
    """
    Test 5:
    Use a purely 1D toy potential with three minima and show that
    findApproxLocalMin detects the *intermediate* minimum along the line
    between the leftmost and rightmost minima.

    Potential:
        V(φ) = (φ + 2)^2 (φ - 0.5)^2 (φ - 2)^2
    Minima at φ ≈ -2, 0.5, 2.
    """

    def V_triple_line(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        phi = x[..., 0]  # works for shape (N,1)
        return (phi + 2.0) ** 2 * (phi - 0.5) ** 2 * (phi - 2.0) ** 2

    x1 = np.array([-2.0], dtype=float)
    x2 = np.array([2.0], dtype=float)

    approx_minima = findApproxLocalMin(
        V_triple_line,
        x1=x1,
        x2=x2,
        n=200,
        edge=0.05,
    )

    print("\n=== Test 5: findApproxLocalMin on triple-well ===")
    print(f"Number of approximate minima found between x1={x1[0]} and x2={x2[0]}: {len(approx_minima)}")
    if len(approx_minima) > 0:
        phis_mid = approx_minima[:, 0]
        print("Approximate interior minima positions:", phis_mid)
        # We expect at least one minimum near φ ≈ 0.5
        assert np.any(np.abs(phis_mid - 0.5) < 0.2), (
            "findApproxLocalMin did not detect a minimum near φ ≈ 0.5 "
            f"(found {phis_mid})"
        )
    else:
        raise AssertionError("findApproxLocalMin returned no interior minima for the triple-well.")

    # Plot to visualize what the function is doing
    fig, ax = plt.subplots(figsize=(6, 4))
    phi_grid = np.linspace(-2.5, 2.5, 400)
    V_vals = V_triple_line(phi_grid.reshape(-1, 1))

    ax.plot(phi_grid, V_vals, label="V_triple(φ)")
    ax.axvline(x1[0], linestyle="--", alpha=0.5, label="x1")
    ax.axvline(x2[0], linestyle="--", alpha=0.5, label="x2")

    if len(approx_minima) > 0:
        ax.scatter(
            approx_minima[:, 0],
            V_triple_line(approx_minima),
            marker="o",
            zorder=5,
            label="approx. interior minima",
        )

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi)$")
    ax.set_title("Test 5: findApproxLocalMin on 1D triple-well")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Test 6 – traceMultiMin on the thermal 1D potential
# ------------------------------------------------------------------

def test_6_traceMultiMin_thermal_1d():
    """
    Test 6:
    Use traceMultiMin on the EW-like thermal potential to reconstruct the
    phase structure. Plot φ_min(T) for all phases and illustrate the
    symmetric and broken minima.

    We check that:
      - at high T (~1.15), we find a phase with φ ≈ 0 (symmetric);
      - at low T (~0.65), we find at least one phase with |φ| > 0 (broken).
    """
    t_low = 0.6
    t_high = 1.2
    phases = build_thermal_phases_example(t_low=t_low, t_high=t_high, deltaX_target=1e-2)

    print("\n=== Test 6: traceMultiMin on thermal 1D potential ===")
    print(f"Number of phases found: {len(phases)}")
    for key, phase in phases.items():
        print(
            f"  Phase {key}: T in [{phase.T[0]:.3f}, {phase.T[-1]:.3f}], "
            f"N_points = {len(phase.T)}"
        )

    # Check for at least one symmetric-like phase at high T
    T_high_probe = 1.15
    symmetric_phases = []
    for phase in phases.values():
        if phase.T[0] <= T_high_probe <= phase.T[-1]:
            phi_val = np.asarray(phase.valAt(T_high_probe)).ravel()[0]
            if abs(phi_val) < 5e-2:
                symmetric_phases.append((phase, phi_val))
    print(f"Phases symmetric-like at T={T_high_probe:.2f}: {[(p.key, phi) for p, phi in symmetric_phases]}")

    assert len(symmetric_phases) >= 1, (
        "No phase with φ ≈ 0 found at high T, "
        "traceMultiMin may have missed the symmetric phase."
    )

    # Check for at least one broken-like phase at low T
    T_low_probe = 0.65
    broken_phases = []
    for phase in phases.values():
        if phase.T[0] <= T_low_probe <= phase.T[-1]:
            phi_val = np.asarray(phase.valAt(T_low_probe)).ravel()[0]
            if abs(phi_val) > 0.1:
                broken_phases.append((phase, phi_val))
    print(f"Phases broken-like at T={T_low_probe:.2f}: {[(p.key, phi) for p, phi in broken_phases]}")

    assert len(broken_phases) >= 1, (
        "No phase with |φ| > 0 found at low T, "
        "traceMultiMin may have failed to capture the broken minimum."
    )

    # Plot φ_min(T) for all phases
    fig, (ax_phi, ax_V) = plt.subplots(1, 2, figsize=(11, 4))

    for key, phase in phases.items():
        phi_vals = phase.X[:, 0]
        ax_phi.plot(phase.T, phi_vals, marker=".", label=f"phase {key}")

    ax_phi.set_xlabel(r"Temperature $T$")
    ax_phi.set_ylabel(r"Minimum $\phi_{\rm min}(T)$")
    ax_phi.set_title("Test 6: traced minima φ(T)")
    ax_phi.legend()

    # Plot V(φ, T) slices and mark minima from phases
    phi_grid = np.linspace(-0.5, 3.0, 400)
    T_slices = [0.7, 0.9, 1.1]
    for T in T_slices:
        V_vals = [V_thermal_1d(np.array([phi]), T) for phi in phi_grid]
        ax_V.plot(phi_grid, V_vals, label=f"T = {T:.2f}")

        # mark minima from each phase at this T (if in range)
        for key, phase in phases.items():
            if phase.T[0] <= T <= phase.T[-1]:
                phi_min = float(np.asarray(phase.valAt(T)).ravel()[0])
                V_min = V_thermal_1d(np.array([phi_min]), T)
                ax_V.scatter(phi_min, V_min, marker="o")

    ax_V.set_xlabel(r"$\phi$")
    ax_V.set_ylabel(r"$V(\phi, T)$")
    ax_V.set_title("V(φ, T) slices with traced minima")
    ax_V.legend()
    fig.tight_layout()
    plt.show()

    return phases  # convenient if you want to reuse in a notebook


# ------------------------------------------------------------------
# Test 7 – removeRedundantPhases: synthetic duplicated phase
# ------------------------------------------------------------------

def test_7_removeRedundantPhases_synthetic():
    """
    Test 7:
    Construct two *identical* Phase objects for the same thermal minimum
    and verify that removeRedundantPhases merges them into a single phase.

    This isolates the redundancy logic from traceMultiMin.
    """
    # Build a broken-like branch by hand using grid search
    Ts = np.linspace(0.65, 1.0, 6)
    X_vals = np.zeros((len(Ts), 1))
    for i, T in enumerate(Ts):
        X_vals[i, :] = _grid_minimum_for_T(T)

    dXdT_vals = np.zeros_like(X_vals)

    phase_a = Phase(key=0, X=X_vals, T=Ts, dXdT=dXdT_vals)
    phase_b = Phase(key=1, X=X_vals.copy(), T=Ts.copy(), dXdT=dXdT_vals.copy())

    phases: Dict[Hashable, Phase] = {0: phase_a, 1: phase_b}

    print("\n=== Test 7: removeRedundantPhases (synthetic) ===")
    print(f"Initial number of phases: {len(phases)}")
    for key, phase in phases.items():
        print(f"  Phase {key}: T in [{phase.T[0]:.3f}, {phase.T[-1]:.3f}]")

    removeRedundantPhases(
        f=V_thermal_1d,
        phases=phases,
        xeps=1e-5,
        diftol=1e-3,
    )

    print(f"Number of phases after redundancy removal: {len(phases)}")
    assert len(phases) == 1, (
        "removeRedundantPhases did not merge two identical branches as expected."
    )

    remaining_key = list(phases.keys())[0]
    remaining_phase = phases[remaining_key]
    print(
        f"Remaining phase key: {remaining_key}, "
        f"T in [{remaining_phase.T[0]:.3f}, {remaining_phase.T[-1]:.3f}]"
    )


# ------------------------------------------------------------------
# Test 8 – getStartPhase: identify the high-temperature phase
# ------------------------------------------------------------------

def test_8_getStartPhase_highT():
    """
    Test 8:
    Use getStartPhase on the thermal 1D phase structure and verify that
    it selects a phase whose high-T endpoint is symmetric (φ ≈ 0).
    """
    t_low = 0.6
    t_high = 1.2
    phases = build_thermal_phases_example(t_low=t_low, t_high=t_high, deltaX_target=1e-2)

    start_key = getStartPhase(phases, V_thermal_1d)
    start_phase = phases[start_key]

    print("\n=== Test 8: getStartPhase on thermal 1D phases ===")
    print(f"Start phase key (high-T phase): {start_key}")
    print(f"  T-range: [{start_phase.T[0]:.3f}, {start_phase.T[-1]:.3f}]")

    phi_high = float(np.asarray(start_phase.X[-1]).ravel()[0])
    print(f"  φ at highest T: φ(T_max) = {phi_high:.4f}")

    # For our potential and seeds, the high-T phase should be symmetric:
    assert abs(phi_high) < 5e-2, (
        "getStartPhase did not return a phase with φ ≈ 0 at highest T. "
        "Check the phase structure and seeds."
    )


# ------------------------------------------------------------------
# Manual runner (extend your existing __main__ block)
# ------------------------------------------------------------------

if __name__ == "__main__":
    test_5_findApproxLocalMin_triple_well()
    test_6_traceMultiMin_thermal_1d()
    test_7_removeRedundantPhases_synthetic()
    test_8_getStartPhase_highT()
