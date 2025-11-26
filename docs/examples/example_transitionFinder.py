"""
example_transitionFinder.py

Pedagogical, end-to-end showcase for the CosmoTransitions.transitionFinder
module, using a simple 1D Landau–Ginzburg finite-temperature potential.

The script is structured in four examples (A..D):

  * Example A – Potential snapshots at very high and very low temperature.
  * Example B – Critical, spinodal and nucleation temperatures (Tc, Tspin, Tn)
                 plus V(φ, T) at these special points.
  * Example C – How the minima evolve with T: φ_min(T) and the curvature
                 m²(T) = ∂²V/∂φ² evaluated on each phase.
  * Example D – Suggestions for further numerical/physical experiments.

Along the way we:
  - Build phases with `traceMultiMin`.
  - Clean them with `removeRedundantPhases`.
  - Extract critical temperatures with `findCriticalTemperatures`.
  - Find the nucleation temperature with `findAllTransitions`
    (which internally uses `tunnelFromPhase` and the bounce solvers).
  - Attach Tc information to the supercooled transitions with
    `addCritTempsForFullTransitions`.

The goal is to illustrate *how a cosmologist would actually use*
transitionFinder: starting from V(φ, T), reconstruct a coherent thermal
history and identify the relevant temperatures.

Everything is kept 1D for clarity, but the logic is directly generalizable
to multi-field models.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Hashable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions import transitionFinder as TF


# -----------------------------------------------------------------------------
# Global configuration – model and numerics
# -----------------------------------------------------------------------------

# Model parameters: simple Landau–Ginzburg finite-T potential
D: float = 0.1
E: float = 0.02
lambda_: float = 0.1
T0: float = 100.0

MODEL_LABEL: str = "LG1D"

# Temperature range used for phase tracing
T_LOW: float = 0.0
T_HIGH: float = 200.0

# Field range for plotting / crude scans
PHI_MIN: float = -5.0
PHI_MAX: float = 5.0
N_PHI: int = 801

# traceMultiMin / traceMinimum controls
DELTA_X_TARGET: float = 0.05     # Target step in field space
DTSTART_FRAC: float = 1e-3       # dtstart as a fraction of (T_HIGH - T_LOW)
TJUMP_FRAC: float = 1e-3         # temperature jump between successive traces

# Nucleation criterion: classic S(T) / T ≈ 140
TARGET_S_OVER_T: float = 140.0


# -----------------------------------------------------------------------------
# Potential and analytic derivatives
# -----------------------------------------------------------------------------

def V(phi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Landau–Ginzburg finite-temperature potential in one scalar field.

    V(φ, T) = D (T² - T₀²) φ² - E T φ³ + (λ / 4) φ⁴

    This implementation is 1D in field space but supports both scalar and
    batched inputs:

    - For traceMinimum / Phase, we use phi.shape == (1,).
    - For vectorized calls (e.g. findApproxLocalMin), we use phi.shape == (n, 1)
      and return an array of shape (n,).

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
        # Single scalar φ
        phi_val = phi_arr                      # scalar
    elif phi_arr.ndim == 1:
        # Vector of φ values (for scans/plots or single-field x[0])
        phi_val = phi_arr                      # shape (n,)
    elif phi_arr.ndim == 2:
        # Batched samples: shape (n_samples, 1)
        if phi_arr.shape[1] != 1:
            raise ValueError("For batched evaluation, use shape (n_samples, 1).")
        phi_val = phi_arr[:, 0]                # shape (n,)
    else:
        raise ValueError(f"Unsupported phi shape {phi_arr.shape} for this test potential.")

    V_val = (
        D * (T**2 - T0**2) * phi_val**2
        - E * T * phi_val**3
        + 0.25 * lambda_ * phi_val**4
    )
    return V_val


def dV(phi: np.ndarray | float, T: float) -> np.ndarray:
    """
    Gradient ∂V/∂φ for the Landau–Ginzburg potential.

    Returns an array with the same "batching convention" as V:

    - For phi.ndim <= 1 (scalar or (1,)), we return shape (1,).
    - For phi.ndim == 2 and shape (n, 1), we return shape (n, 1).
    """
    phi_arr = np.asarray(phi, dtype=float)

    if phi_arr.ndim == 0:
        phi_val = phi_arr
        dV_val = (
            2.0 * D * (T**2 - T0**2) * phi_val
            - 3.0 * E * T * phi_val**2
            + lambda_ * phi_val**3
        )
        return np.array([dV_val], dtype=float)

    if phi_arr.ndim == 1:
        if phi_arr.size != 1:
            raise ValueError("This showcase gradient assumes a single scalar field.")
        phi_val = phi_arr[0]
        dV_val = (
            2.0 * D * (T**2 - T0**2) * phi_val
            - 3.0 * E * T * phi_val**2
            + lambda_ * phi_val**3
        )
        return np.array([dV_val], dtype=float)

    if phi_arr.ndim == 2:
        if phi_arr.shape[1] != 1:
            raise ValueError("For batched evaluation, use shape (n_samples, 1).")
        phi_val = phi_arr[:, 0]
        dV_val = (
            2.0 * D * (T**2 - T0**2) * phi_val
            - 3.0 * E * T * phi_val**2
            + lambda_ * phi_val**3
        )
        return dV_val.reshape(-1, 1)

    raise ValueError(f"Unsupported phi shape {phi_arr.shape} in dV.")


def d2V_dphi2(phi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Second derivative ∂²V/∂φ² evaluated at (φ, T).

    For the 1D Landau–Ginzburg potential,

        ∂²V/∂φ² = 2D (T² - T₀²) - 6E T φ + 3λ φ².
    """
    phi_arr = np.asarray(phi, dtype=float)
    m2 = 2.0 * D * (T**2 - T0**2) - 6.0 * E * T * phi_arr + 3.0 * lambda_ * phi_arr**2
    return m2


# --- Wrappers in the shape expected by transitionFinder ----------------------

def free_energy(x: np.ndarray, T: float) -> float:
    """
    Free-energy density f(x, T) used by traceMinimum / traceMultiMin.

    In this 1D example, x is a 1-element array with x[0] = φ.
    """
    return float(V(x, T))


def d2f_dxdt(x: np.ndarray, T: float) -> np.ndarray:
    """
    Mixed derivative ∂/∂T (∂f/∂x) evaluated at (x, T).

    For our 1D potential, with x[0] = φ,

        ∂V/∂φ = 2D (T² - T₀²) φ - 3E T φ² + λ φ³

    so that

        ∂/∂T (∂V/∂φ) = 4D T φ - 3E φ².
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    if x_arr.size != 1:
        raise ValueError("d2f_dxdt assumes a single scalar field.")
    phi = x_arr[0]
    val = 4.0 * D * T * phi - 3.0 * E * phi**2
    return np.array([val], dtype=float)


def d2f_dx2(x: np.ndarray, T: float) -> np.ndarray:
    """
    Hessian ∂²f/∂x² evaluated at (x, T).

    In 1D this is just a 1×1 matrix containing ∂²V/∂φ².
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    if x_arr.size != 1:
        raise ValueError("d2f_dx2 assumes a single scalar field.")
    phi = x_arr[0]
    m2 = d2V_dphi2(phi, T)
    return np.array([[float(m2)]], dtype=float)


# -----------------------------------------------------------------------------
# Nucleation criterion and utility helpers
# -----------------------------------------------------------------------------

def nuclCriterion(S: float, T: float, target: float = TARGET_S_OVER_T) -> float:
    """
    Default nucleation condition: S(T) / T - TARGET_S_OVER_T.

    The transition is "efficient" when this function crosses zero from
    positive to negative values.
    """
    return S / (T + 1e-100) - target


def scan_minimum_1D(T: float) -> float:
    """
    Crude 1D scan of V(φ, T) to locate the deepest minimum in a fixed range.

    This is only used to build intuitive seed points for traceMultiMin.
    """
    phi_grid = np.linspace(PHI_MIN, PHI_MAX, N_PHI)
    V_vals = np.asarray(V(phi_grid, T), dtype=float)
    idx_min = int(np.argmin(V_vals))
    return float(phi_grid[idx_min])


def build_seed_points() -> List[Tuple[np.ndarray, float]]:
    """
    Construct initial seeds (x, T) for traceMultiMin.

    For this symmetric 1D model we pick:
      - one seed in the high-T symmetric minimum (around φ ≈ 0),
      - one seed in a low-T broken minimum (|φ| > 0).

    From these, traceMultiMin and findApproxLocalMin will automatically
    discover additional branches if they exist.
    """
    # High-T symmetric-phase seed
    T_seed_high = 180.0
    phi_high = scan_minimum_1D(T_seed_high)

    # Low-T broken-phase seed
    T_seed_low = 20.0
    phi_low = scan_minimum_1D(T_seed_low)

    seeds: List[Tuple[np.ndarray, float]] = [
        (np.array([phi_high], dtype=float), T_seed_high),
        (np.array([phi_low], dtype=float), T_seed_low),
    ]

    print("\n[Seeds] Initial seed points for traceMultiMin:")
    for (x, T) in seeds:
        print(f"  (T = {T:7.3f}, phi = {x[0]:8.4f})  V = {V(x[0], T):10.4f}")

    return seeds


def build_phases_and_transitions() -> Tuple[Dict[Hashable, TF.Phase],
                                            List[Dict[str, Any]],
                                            List[Dict[str, Any]]]:
    """
    Block A + B core: trace phases, find critical and nucleation temperatures.

    Returns
    -------
    phases : dict
        Mapping from phase keys to Phase objects.
    crit_transitions : list of dict
        Output of findCriticalTemperatures.
    full_transitions : list of dict
        Output of findAllTransitions, with Tc information attached
        via addCritTempsForFullTransitions.
    """
    print("\n" + "=" * 79)
    print("[Block A] Tracing temperature-dependent minima with traceMultiMin")
    print("=" * 79)

    seeds = build_seed_points()

    dt_scale = T_HIGH - T_LOW
    dtstart_abs = DTSTART_FRAC * dt_scale
    tjump_abs = TJUMP_FRAC * dt_scale

    print(f"\n[traceMultiMin configuration]")
    print(f"  T-range        : [{T_LOW:.3f}, {T_HIGH:.3f}]")
    print(f"  deltaX_target  : {DELTA_X_TARGET:.4f}")
    print(f"  dtstart (frac) : {DTSTART_FRAC:.3e}  "
          f"→ |ΔT| ≈ {dtstart_abs:.4g}")
    print(f"  tjump  (frac)  : {TJUMP_FRAC:.3e}  "
          f"→ |ΔT| ≈ {tjump_abs:.4g}")

    phases: Dict[Hashable, TF.Phase] = TF.traceMultiMin(
        f=free_energy,
        d2f_dxdt=d2f_dxdt,
        d2f_dx2=d2f_dx2,
        points=seeds,
        tLow=T_LOW,
        tHigh=T_HIGH,
        deltaX_target=DELTA_X_TARGET,
        # IMPORTANT: pass *fractions* here; traceMultiMin rescales internally
        dtstart=DTSTART_FRAC,
        tjump=TJUMP_FRAC,
        single_trace_args={
            "dtabsMax": 20.0,
            "dtfracMax": 0.25,
            "dtmin": 1e-4,
            "minratio": 1e-2,
        },
        local_min_args={"n": 200, "edge": 0.05},
    )

    print(f"\n[Block A] Raw phases traced: {len(phases)}")

    # Clean up any accidental duplicates
    phases_mut: MutableMapping[Hashable, TF.Phase] = dict(phases)
    TF.removeRedundantPhases(f=free_energy, phases=phases_mut)
    phases_clean: Dict[Hashable, TF.Phase] = dict(phases_mut)

    print(f"[Block A] Phases after removeRedundantPhases: {len(phases_clean)}")

    for key, phase in phases_clean.items():
        T_min = float(phase.T[0])
        T_max = float(phase.T[-1])
        phi_min = float(np.min(phase.X[:, 0]))
        phi_max = float(np.max(phase.X[:, 0]))
        print(
            f"  Phase {key!r}: T in [{T_min:7.3f}, {T_max:7.3f}]  "
            f"phi in [{phi_min:+8.4f}, {phi_max:+8.4f}]  "
            f"({len(phase.T)} support points)"
        )

    print("\n" + "=" * 79)
    print("[Block B] Critical temperatures and nucleation")
    print("=" * 79)

    # Critical temperatures between all phases
    crit_transitions: List[Dict[str, Any]] = TF.findCriticalTemperatures(
        phases_clean,
        V,
        start_high=False,
    )

    print(f"\n[Critical temperatures] Found {len(crit_transitions)} degeneracy points.")
    for i, tdict in enumerate(sorted(crit_transitions, key=lambda d: d["Tcrit"], reverse=True), 1):
        print(
            f"  #{i:2d}: Tcrit = {tdict['Tcrit']:7.3f}  "
            f"high_phase = {tdict['high_phase']!r}, "
            f"low_phase = {tdict['low_phase']!r}"
        )

    # Full supercooled transitions with nucleation temperatures
    tunnel_args: Dict[str, Any] = {
        "Ttol": 1e-2,
        "maxiter": 80,
        "phitol": 1e-6,
        "overlapAngle": 45.0,
        "nuclCriterion": nuclCriterion,
        "verbose": False,
        "fullTunneling_params": {},
    }

    full_transitions: List[Dict[str, Any]] = TF.findAllTransitions(
        phases_clean,
        V,
        dV,
        tunnelFromPhase_args=tunnel_args,
    )

    if not full_transitions:
        print("\n[Warning] No first-order transitions with a bounce were found "
              "in the specified T-range.")
    else:
        print(f"\n[Full transitions] Found {len(full_transitions)} candidate transitions.")
        for i, tdict in enumerate(full_transitions, 1):
            Tn = float(tdict["Tnuc"])
            S = float(tdict["action"])
            print(
                f"  #{i:2d}: high_phase -> low_phase = "
                f"{tdict['high_phase']!r} → {tdict['low_phase']!r}\n"
                f"        Tnuc = {Tn:7.3f},  S(Tn) = {S:9.3f},  "
                f"S/T = {S / (Tn + 1e-100):7.2f}"
            )

    # Attach critical-temperature info to supercooled transitions
    TF.addCritTempsForFullTransitions(phases_clean, crit_transitions, full_transitions)

    # Return cleaned phases and both transition lists
    return phases_clean, crit_transitions, full_transitions


# -----------------------------------------------------------------------------
# Plot helpers and configuration printing
# -----------------------------------------------------------------------------

def _output_dir() -> str:
    """
    Directory where figures are stored.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    outdir = os.path.join(here, "assets_transitionFinder")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _save_figure(fig: plt.Figure, label: str, case: str) -> None:
    """
    Save figure with a simple, consistent naming scheme: figX_<case>.png
    where X is 'A', 'B', 'C', ...
    """
    fname = f"fig{label}_{case}.png"
    path = os.path.join(_output_dir(), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> saved figure to {path}")
    plt.close(fig)


def print_configuration(case_label: str) -> None:
    """
    Print a compact summary of all physical and numerical parameters that
    a user might want to tweak.
    """
    print("=" * 79)
    print(f"Configuration for example_transitionFinder (case = {case_label!r})")
    print("=" * 79)
    print("\n[Model parameters]")
    print(f"  D       = {D:7.4f}")
    print(f"  E       = {E:7.4f}")
    print(f"  lambda  = {lambda_:7.4f}")
    print(f"  T0      = {T0:7.3f}")

    print("\n[Temperature range]")
    print(f"  T_LOW   = {T_LOW:7.3f}")
    print(f"  T_HIGH  = {T_HIGH:7.3f}")

    print("\n[Tracing parameters (traceMultiMin / traceMinimum)]")
    print(f"  DELTA_X_TARGET = {DELTA_X_TARGET:7.4f}")
    print(f"  DTSTART_FRAC   = {DTSTART_FRAC:7.2e}")
    print(f"  TJUMP_FRAC     = {TJUMP_FRAC:7.2e}")

    print("\n[Nucleation criterion]")
    print(f"  TARGET_S_OVER_T = {TARGET_S_OVER_T:7.2f}")
    print("  nuclCriterion(S, T) = S / T - TARGET_S_OVER_T\n")


# -----------------------------------------------------------------------------
# Example A – Potential snapshots at very high and very low T
# -----------------------------------------------------------------------------

def example_A_potential_snapshots(case: str) -> None:
    """
    Example A:
      - Plot V(φ, T) at a very low temperature and at a very high temperature.
      - Mark the minima obtained from a simple scan.
    """
    print("\n" + "-" * 79)
    print("Example A – Potential snapshots at very low and very high T")
    print("-" * 79)

    T_lo = 0.0
    T_hi = 200.0

    phi_grid = np.linspace(PHI_MIN, PHI_MAX, N_PHI)
    V_lo = np.asarray(V(phi_grid, T_lo), dtype=float)
    V_hi = np.asarray(V(phi_grid, T_hi), dtype=float)

    phi_min_lo = scan_minimum_1D(T_lo)
    phi_min_hi = scan_minimum_1D(T_hi)

    print(f"  At T = {T_hi:7.3f}, the symmetric minimum is near φ ≈ {phi_min_hi:+8.4f}")
    print(f"  At T = {T_lo:7.3f}, the broken minimum is near   φ ≈ {phi_min_lo:+8.4f}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(phi_grid, V_hi, label=f"T = {T_hi:.1f}")
    ax.plot(phi_grid, V_lo, label=f"T = {T_lo:.1f}")
    ax.scatter([phi_min_hi], [V(phi_min_hi, T_hi)], marker="o")
    ax.scatter([phi_min_lo], [V(phi_min_lo, T_lo)], marker="o")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T)$")
    ax.set_title("Landau–Ginzburg potential at low and high T")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.show()
    _save_figure(fig, "A", case)


# -----------------------------------------------------------------------------
# Example B – Tc, Tspin and Tn with V(φ, T) at special temperatures
# -----------------------------------------------------------------------------

def _extract_main_transition(
    phases: Mapping[Hashable, TF.Phase],
    full_transitions: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any] | None, TF.Phase | None, TF.Phase | None]:
    """
    For this simple 1D example we expect a single dominant first-order
    transition. This helper picks the one with the highest nucleation
    temperature.
    """
    if not full_transitions:
        return None, None, None

    tdict = max(full_transitions, key=lambda d: float(d["Tnuc"]))
    high_phase = phases[tdict["high_phase"]]
    low_phase = phases[tdict["low_phase"]]
    return tdict, high_phase, low_phase


def example_B_transition_temperatures(
    phases: Mapping[Hashable, TF.Phase],
    crit_transitions: Sequence[Dict[str, Any]],
    full_transitions: Sequence[Dict[str, Any]],
    case: str,
) -> None:
    """
    Example B:
      - Extract spinodal, critical and nucleation temperatures.
      - Print them in a small table.
      - Plot V(φ, T) at those special temperatures.
    """
    del crit_transitions  # only used indirectly via main_trans["crit_trans"]

    print("\n" + "-" * 79)
    print("Example B – Tc, Tspin and Tn with V(φ, T) at special temperatures")
    print("-" * 79)

    main_trans, high_phase, low_phase = _extract_main_transition(phases, full_transitions)
    if main_trans is None or high_phase is None or low_phase is None:
        print("  [Example B] No first-order transition found; skipping Tc/Tn analysis.")
        return

    Tn = float(main_trans["Tnuc"])
    S = float(main_trans["action"])
    S_over_T = S / (Tn + 1e-100)

    # Critical temperature attached by addCritTempsForFullTransitions, if any
    Tc: float | None = None
    if main_trans.get("crit_trans") is not None:
        Tc = float(main_trans["crit_trans"]["Tcrit"])

    # Spinodal temperatures: where each phase branch "starts" or "ends"
    T_spin_high = float(high_phase.T[0])   # lowest T where high-T phase still exists
    T_spin_low = float(low_phase.T[-1])    # highest T where low-T phase first appears

    print("\n[Characteristic temperatures for the main transition]")
    print("  (Interpretation: Universe cools from high to low T.)\n")

    header = (
        "+----------------------+-----------+-------------------------------+\n"
        "| quantity             |  value    | comment                       |\n"
        "+----------------------+-----------+-------------------------------+"
    )
    rows = [
        ("T_spin(high phase)", T_spin_high, "symmetric minimum disappears"),
        ("T_crit",             Tc,          "phases degenerate (if available)"),
        ("T_nuc",              Tn,          "S(T)/T ≈ TARGET_S_OVER_T"),
        ("T_spin(low phase)",  T_spin_low,  "broken minimum ceases to exist"),
    ]

    print(header)
    for name, val, comment in rows:
        if val is None:
            val_str = "   n/a  "
        else:
            val_str = f"{val:9.3f}"
        print(f"| {name:20s} | {val_str} | {comment:29s} |")
    print("+----------------------+-----------+-------------------------------+\n")

    print(f"  Action at nucleation: S(Tn) = {S:9.3f}")
    print(f"  S(Tn) / Tn           : {S_over_T:9.3f}  (target = {TARGET_S_OVER_T:.1f})")

    # Collect temperatures at which to plot V(φ, T)
    T_plot_list: List[float] = [T_spin_high, Tn, T_spin_low]
    T_labels: List[str] = [r"$T_{\mathrm{spin}}^{\mathrm{high}}$",
                           r"$T_{\mathrm{nuc}}$",
                           r"$T_{\mathrm{spin}}^{\mathrm{low}}$"]
    if Tc is not None:
        # Insert Tc between spinodal and Tn if it lies in between
        T_plot_list.insert(1, Tc)
        T_labels.insert(1, r"$T_{\mathrm{crit}}$")

    phi_grid = np.linspace(PHI_MIN, PHI_MAX, N_PHI)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for Tval, label in zip(T_plot_list, T_labels):
        V_vals = np.asarray(V(phi_grid, Tval), dtype=float)
        ax.plot(phi_grid, V_vals, label=f"{label} = {Tval:.3f}")

    # Mark the high- and low-phase minima at Tn
    phi_high_Tn = float(high_phase.valAt(Tn)[0])
    phi_low_Tn = float(low_phase.valAt(Tn)[0])
    ax.scatter([phi_high_Tn], [V(phi_high_Tn, Tn)], marker="o", label="high-phase minimum at Tn")
    ax.scatter([phi_low_Tn], [V(phi_low_Tn, Tn)], marker="s", label="low-phase minimum at Tn")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi, T)$")
    ax.set_title("Potential at spinodal, critical and nucleation temperatures")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.show()
    _save_figure(fig, "B", case)


# -----------------------------------------------------------------------------
# Example C – Evolution of minima and curvature with T
# -----------------------------------------------------------------------------

def example_C_minima_evolution(
    phases: Mapping[Hashable, TF.Phase],
    full_transitions: Sequence[Dict[str, Any]],
    case: str,
) -> None:
    """
    Example C:
      - Show φ_min(T) for each Phase.
      - Show the curvature m²(T) = ∂²V/∂φ² evaluated at the minima.
      - Highlight Tspin, Tcrit and Tn when available.
    """
    print("\n" + "-" * 79)
    print("Example C – Evolution of minima and curvature with temperature")
    print("-" * 79)

    main_trans, high_phase, low_phase = _extract_main_transition(phases, full_transitions)

    # Gather landmark temperatures if possible
    Tn = None
    Tc = None
    T_spin_high = None
    T_spin_low = None
    if main_trans is not None and high_phase is not None and low_phase is not None:
        Tn = float(main_trans["Tnuc"])
        T_spin_high = float(high_phase.T[0])
        T_spin_low = float(low_phase.T[-1])
        if main_trans.get("crit_trans") is not None:
            Tc = float(main_trans["crit_trans"]["Tcrit"])

    # --- φ_min(T) for all phases --------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))

    for key, phase in phases.items():
        T_vals = np.asarray(phase.T, dtype=float)
        phi_vals = np.asarray(phase.X[:, 0], dtype=float)
        ax1.plot(T_vals, phi_vals, marker=".", linestyle="-", label=f"Phase {key!r}")

    if T_spin_high is not None:
        ax1.axvline(T_spin_high, linestyle="--", alpha=0.4, label="T_spin (high)")
    if Tc is not None:
        ax1.axvline(Tc, linestyle=":", alpha=0.6, label="T_crit")
    if Tn is not None:
        ax1.axvline(Tn, linestyle="-.", alpha=0.6, label="T_nuc")
    if T_spin_low is not None:
        ax1.axvline(T_spin_low, linestyle="--", alpha=0.4, label="T_spin (low)")

    ax1.set_xlabel(r"Temperature $T$")
    ax1.set_ylabel(r"$\phi_{\mathrm{min}}(T)$")
    ax1.set_title("Evolution of the minima with temperature")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    plt.show()
    _save_figure(fig1, "C_phi", case)

    # --- Curvature m²(T) = ∂²V/∂φ² at each minimum --------------------------
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

    for key, phase in phases.items():
        T_vals = np.asarray(phase.T, dtype=float)
        phi_vals = np.asarray(phase.X[:, 0], dtype=float)
        m2_vals = np.array([d2V_dphi2(phi, T) for phi, T in zip(phi_vals, T_vals)], dtype=float)
        ax2.plot(T_vals, m2_vals, marker=".", linestyle="-", label=f"Phase {key!r}")

    ax2.axhline(0.0, color="k", linewidth=1.0)
    if T_spin_high is not None:
        ax2.axvline(T_spin_high, linestyle="--", alpha=0.4, label="T_spin (high)")
    if Tc is not None:
        ax2.axvline(Tc, linestyle=":", alpha=0.6, label="T_crit")
    if Tn is not None:
        ax2.axvline(Tn, linestyle="-.", alpha=0.6, label="T_nuc")
    if T_spin_low is not None:
        ax2.axvline(T_spin_low, linestyle="--", alpha=0.4, label="T_spin (low)")

    ax2.set_xlabel(r"Temperature $T$")
    ax2.set_ylabel(r"$m^2(T) = \partial^2 V / \partial \phi^2$")
    ax2.set_title("Curvature at the minima (spinodal points where m² → 0)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    plt.show()
    _save_figure(fig2, "C_m2", case)


# -----------------------------------------------------------------------------
# Example D – Suggestions for further experiments
# -----------------------------------------------------------------------------

def example_D_further_ideas() -> None:
    """
    Example D:
      Pedagogical suggestions for how to extend this showcase with
      additional physically interesting experiments.
    """
    print("\n" + "-" * 79)
    print("Example D – Further numerical and physical experiments")
    print("-" * 79)

    print(
        "Here are some ideas for extensions you can implement directly in this\n"
        "example file (or in your own scripts) using the same transitionFinder\n"
        "infrastructure:\n"
    )
    print("  1. Change the nucleation criterion:")
    print("       - Replace TARGET_S_OVER_T = 140 by another threshold.")
    print("       - Or implement a more sophisticated nuclCriterion(S, T) that")
    print("         accounts for percolation or a finite Hubble rate.\n")
    print("  2. Scan over model parameters (D, E, lambda, T0):")
    print("       - For each point in parameter space, recompute Tc, Tn, and")
    print("         T_spin and build contour plots in the (D, E) plane.")
    print("       - Identify regions with strong first-order transitions.\n")
    print("  3. Compare with a second-order benchmark:")
    print("       - Set E = 0 so that the cubic term vanishes.")
    print("       - Repeat the analysis and observe how the first-order")
    print("         characteristics (bounce, Tn) disappear, leaving only")
    print("         a continuous second-order transition at T ≈ T0.\n")
    print("  4. Add a second scalar field:")
    print("       - Promote phi to a 2-component vector and generalize V.")
    print("       - The same transitionFinder interface works, but tunneling")
    print("         will use pathDeformation instead of the 1D backend.\n")
    print("  5. Connect to gravitational-wave predictions:")
    print("       - Use the extracted Tn and latent heat to estimate GW spectra.")
    print("       - This provides a full pipeline from V(φ, T) to a potentially")
    print("         observable stochastic gravitational-wave background.\n")


# -----------------------------------------------------------------------------
# Summary helpers and entry point
# -----------------------------------------------------------------------------

def collect_summary(
    phases: Mapping[Hashable, TF.Phase],
    full_transitions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a compact summary dictionary with the main temperatures and
    bounce diagnostics, useful for notebooks or further processing.
    """
    summary: Dict[str, Any] = {
        "model": {
            "D": D,
            "E": E,
            "lambda": lambda_,
            "T0": T0,
        },
        "tracing": {
            "T_LOW": T_LOW,
            "T_HIGH": T_HIGH,
            "DELTA_X_TARGET": DELTA_X_TARGET,
            "DTSTART_FRAC": DTSTART_FRAC,
            "TJUMP_FRAC": TJUMP_FRAC,
        },
        "nucleation": {
            "TARGET_S_OVER_T": TARGET_S_OVER_T,
        },
        "phases": {},
        "transitions": [],
    }

    # Phase summary
    for key, phase in phases.items():
        T_min = float(phase.T[0])
        T_max = float(phase.T[-1])
        phi_min = float(np.min(phase.X[:, 0]))
        phi_max = float(np.max(phase.X[:, 0]))
        summary["phases"][str(key)] = {
            "T_min": T_min,
            "T_max": T_max,
            "phi_min": phi_min,
            "phi_max": phi_max,
            "n_points": int(len(phase.T)),
        }

    # Transition summary (if any)
    for tdict in full_transitions:
        Tn = float(tdict["Tnuc"])
        S = float(tdict["action"])
        Tc = None
        if tdict.get("crit_trans") is not None:
            Tc = float(tdict["crit_trans"]["Tcrit"])
        summary["transitions"].append(
            {
                "high_phase": str(tdict["high_phase"]),
                "low_phase": str(tdict["low_phase"]),
                "Tnuc": Tn,
                "S_Tn": S,
                "S_over_Tn": S / (Tn + 1e-100),
                "Tcrit": Tc,
            }
        )

    return summary


def print_final_summary(summary: Dict[str, Any]) -> None:
    """
    Nicely print the content of the summary dictionary at the end of the run.
    """
    print("\n" + "=" * 79)
    print("Final compact summary (temperatures and actions)")
    print("=" * 79)

    if not summary["transitions"]:
        print("  No first-order transitions were recorded.")
        return

    for i, tdict in enumerate(summary["transitions"], 1):
        print(f"\n  Transition #{i}: "
              f"{tdict['high_phase']} → {tdict['low_phase']}")
        print(f"    Tnuc        = {tdict['Tnuc']:9.4f}")
        if tdict["Tcrit"] is not None:
            print(f"    Tcrit       = {tdict["Tcrit"]:9.4f}")
        else:
            print("    Tcrit       =       n/a")
        print(f"    S(Tn)       = {tdict['S_Tn']:9.4f}")
        print(f"    S(Tn)/Tn    = {tdict['S_over_Tn']:9.4f} "
              f"(target = {TARGET_S_OVER_T:.1f})")

    print("\n  (All parameters controlling the behaviour of the example are listed")
    print("   at the beginning of the run in print_configuration().)\n")


def run_all(case_label: str = MODEL_LABEL) -> Dict[str, Any]:
    """
    Run all examples A..D in a physically chronological order:

      1. Print configuration and model parameters.
      2. Build phases and transitions (Blocks A and B of transitionFinder).
      3. Example A: basic potential geometry.
      4. Example B: characteristic temperatures (Tc, Tspin, Tn).
      5. Example C: full evolution of minima and curvatures.
      6. Example D: suggestions for further work.
      7. Final compact summary with all key numbers.

    Returns
    -------
    summary : dict
        A dictionary with the main temperatures and actions, convenient
        for programmatic use in notebooks.
    """
    print_configuration(case_label)

    phases, crit_transitions, full_transitions = build_phases_and_transitions()

    example_A_potential_snapshots(case_label)
    example_B_transition_temperatures(phases, crit_transitions, full_transitions, case_label)
    example_C_minima_evolution(phases, full_transitions, case_label)
    example_D_further_ideas()

    summary = collect_summary(phases, full_transitions)
    print_final_summary(summary)
    return summary


if __name__ == "__main__":
    run_all()
