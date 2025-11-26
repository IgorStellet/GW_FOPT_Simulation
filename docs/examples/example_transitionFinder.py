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
from typing import Any, Dict, Hashable, List, Mapping, MutableMapping, Sequence, Tuple, Optional

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
PHI_MIN: float = -5
PHI_MAX: float = 5
N_PHI: int = 1000

# traceMultiMin / traceMinimum controls
DELTA_X_TARGET: float = 0.05     # Target step in field space
DTSTART_FRAC: float = 1e-4       # dtstart as a fraction of (T_HIGH - T_LOW)
TJUMP_FRAC: float = 1e-4         # temperature jump between successive traces

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
      - Plot V(φ, T) at those special temperatures (4 panels: spinodal-high,
        critical, nucleation, spinodal-low), with an adaptive φ-range that
        covers the relevant minima for each T.
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

    # ------------------------------------------------------------------
    # Four panels: one for each characteristic temperature
    # ------------------------------------------------------------------
    T_items: List[Tuple[Optional[float], str, str]] = [
        (T_spin_high, r"$T_{\mathrm{spin}}^{\mathrm{high}}$", "High-phase spinodal"),
        (Tc,          r"$T_{\mathrm{crit}}$",                "Critical (degenerate)"),
        (Tn,          r"$T_{\mathrm{nuc}}$",                 "Nucleation"),
        (T_spin_low,  r"$T_{\mathrm{spin}}^{\mathrm{low}}$", "Low-phase spinodal"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    for ax, (Tval, label_tex, desc) in zip(axes_flat, T_items):
        if Tval is None:
            # No critical temperature: keep the panel but show a message
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{desc}\n(no {label_tex} for this transition)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # --------------------------------------------------------------
        # Determine which phases exist at this T and where their minima are
        # --------------------------------------------------------------
        phis_for_range: List[float] = []

        T_hp_min = float(high_phase.T[0])
        T_hp_max = float(high_phase.T[-1])
        phi_high: float | None = None
        V_high: float | None = None
        if T_hp_min <= Tval <= T_hp_max:
            phi_high = float(high_phase.valAt(Tval)[0])
            V_high = float(V(phi_high, Tval))
            phis_for_range.append(phi_high)

        T_lp_min = float(low_phase.T[0])
        T_lp_max = float(low_phase.T[-1])
        phi_low: float | None = None
        V_low: float | None = None
        if T_lp_min <= Tval <= T_lp_max:
            phi_low = float(low_phase.valAt(Tval)[0])
            V_low = float(V(phi_low, Tval))
            phis_for_range.append(phi_low)

        # --------------------------------------------------------------
        # Build an adaptive φ-range that covers all relevant minima
        # --------------------------------------------------------------
        if phis_for_range:
            phi_min = min(phis_for_range)
            phi_max = max(phis_for_range)
            span = max(1e-6, phi_max - phi_min)

            # Margin: 30% of the span, but at least 0.5 in absolute units
            margin = max(0.5, 0.3 * span)

            # If both minima coincide (e.g. second-order or at the appearance
            # of a new phase), still give a symmetric window around them.
            if span < 1e-6:
                center = phi_min
                margin = max(margin, 0.5 * max(1.0, abs(center)))
                phi_left = center - margin
                phi_right = center + margin
            else:
                phi_left = phi_min - margin
                phi_right = phi_max + margin
        else:
            # Fallback: no minima from these two phases at this T
            # (very unlikely in this example, but be robust).
            phi_left = PHI_MIN
            phi_right = PHI_MAX

        phi_grid = np.linspace(phi_left, phi_right, N_PHI)
        V_vals = np.asarray(V(phi_grid, Tval), dtype=float)

        # --------------------------------------------------------------
        # Plot potential and mark minima (when defined)
        # --------------------------------------------------------------
        ax.plot(phi_grid, V_vals)

        if phi_high is not None and V_high is not None:
            ax.scatter([phi_high], [V_high], marker="o", s=30, label="high-phase min")

        if phi_low is not None and V_low is not None:
            ax.scatter([phi_low], [V_low], marker="s", s=30, label="low-phase min")

        ax.set_title(f"{desc}\n{label_tex} = {Tval:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="best")

        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$V(\phi, T)$")

    fig.suptitle(
        "Example B – Potential at spinodal, critical and nucleation temperatures",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

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
      - Highlight T_spin, T_crit and T_n when available.
      - First figure: φ_min(T) and m²(T) stacked in a single panel.
      - Second figure: m²(T) only, zoomed around the spinodal region.
    """
    print("\n" + "-" * 79)
    print("Example C – Evolution of minima and curvature with temperature")
    print("-" * 79)

    main_trans, high_phase, low_phase = _extract_main_transition(phases, full_transitions)

    # ------------------------------------------------------------------
    # Landmark temperatures (if a main first-order transition exists)
    # ------------------------------------------------------------------
    Tn: float | None = None
    Tc: float | None = None
    T_spin_high: float | None = None
    T_spin_low: float | None = None

    if main_trans is not None and high_phase is not None and low_phase is not None:
        Tn = float(main_trans["Tnuc"])
        T_spin_high = float(high_phase.T[0])   # lowest T where high-T phase still exists
        T_spin_low = float(low_phase.T[-1])    # highest T where low-T phase first appears
        if main_trans.get("crit_trans") is not None:
            Tc = float(main_trans["crit_trans"]["Tcrit"])

    # ------------------------------------------------------------------
    # Precompute φ_min(T) and m²(T) for all phases once
    # ------------------------------------------------------------------
    phase_data: Dict[Hashable, Dict[str, np.ndarray]] = {}

    for key, phase in phases.items():
        T_vals = np.asarray(phase.T, dtype=float)
        phi_vals = np.asarray(phase.X[:, 0], dtype=float)
        m2_vals = np.array(
            [d2V_dphi2(phi, T) for phi, T in zip(phi_vals, T_vals)],
            dtype=float,
        )
        phase_data[key] = {"T": T_vals, "phi": phi_vals, "m2": m2_vals}

    # ==================================================================
    # Figure 1: φ_min(T) and m²(T) in a single stacked figure
    # ==================================================================
    fig1, (ax_phi, ax_m2) = plt.subplots(
        2, 1, figsize=(6.5, 7.0), sharex=True
    )

    # --- Top: φ_min(T) -------------------------------------------------
    for key, pdata in phase_data.items():
        ax_phi.plot(
            pdata["T"],
            pdata["phi"],
            marker=".",
            linestyle="-",
            label=f"Phase {key!r}",
        )

    if T_spin_high is not None:
        ax_phi.axvline(
            T_spin_high,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{high}}$",
        )
    if Tc is not None:
        ax_phi.axvline(
            Tc,
            linestyle=":",
            alpha=0.6,
            label=r"$T_{\mathrm{crit}}$",
        )
    if Tn is not None:
        ax_phi.axvline(
            Tn,
            linestyle="-.",
            alpha=0.6,
            label=r"$T_{\mathrm{nuc}}$",
        )
    if T_spin_low is not None:
        ax_phi.axvline(
            T_spin_low,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{low}}$",
        )

    ax_phi.set_ylabel(r"$\phi_{\mathrm{min}}(T)$")
    ax_phi.set_title("Evolution of the minima with temperature")
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(loc="best", fontsize=8)

    # --- Bottom: m²(T) -------------------------------------------------
    for key, pdata in phase_data.items():
        ax_m2.plot(
            pdata["T"],
            pdata["m2"],
            marker=".",
            linestyle="-",
            label=f"Phase {key!r}",
        )

    ax_m2.axhline(0.0, color="k", linewidth=1.0)

    if T_spin_high is not None:
        ax_m2.axvline(
            T_spin_high,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{high}}$",
        )
    if Tc is not None:
        ax_m2.axvline(
            Tc,
            linestyle=":",
            alpha=0.6,
            label=r"$T_{\mathrm{crit}}$",
        )
    if Tn is not None:
        ax_m2.axvline(
            Tn,
            linestyle="-.",
            alpha=0.6,
            label=r"$T_{\mathrm{nuc}}$",
        )
    if T_spin_low is not None:
        ax_m2.axvline(
            T_spin_low,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{low}}$",
        )

    ax_m2.set_xlabel(r"Temperature $T$")
    ax_m2.set_ylabel(r"$m^2(T) = \partial^2 V / \partial \phi^2$")
    ax_m2.set_title("Curvature at the minima (spinodal points where $m^2 \to 0$)")
    ax_m2.grid(True, alpha=0.3)
    ax_m2.legend(loc="best", fontsize=8)

    fig1.tight_layout()
    plt.show()
    _save_figure(fig1, "C_phi", case)

    # ==================================================================
    # Figure 2: m²(T) zoomed around the spinodal region
    # ==================================================================
    fig2, ax_zoom = plt.subplots(figsize=(6.5, 4.5))

    for key, pdata in phase_data.items():
        ax_zoom.plot(
            pdata["T"],
            pdata["m2"],
            marker=".",
            linestyle="-",
            label=f"Phase {key!r}",
        )

    ax_zoom.axhline(0.0, color="k", linewidth=1.0)

    # Vertical markers again (they help to orient in the zoomed window)
    if T_spin_high is not None:
        ax_zoom.axvline(
            T_spin_high,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{high}}$",
        )
    if Tc is not None:
        ax_zoom.axvline(
            Tc,
            linestyle=":",
            alpha=0.6,
            label=r"$T_{\mathrm{crit}}$",
        )
    if Tn is not None:
        ax_zoom.axvline(
            Tn,
            linestyle="-.",
            alpha=0.6,
            label=r"$T_{\mathrm{nuc}}$",
        )
    if T_spin_low is not None:
        ax_zoom.axvline(
            T_spin_low,
            linestyle="--",
            alpha=0.4,
            label=r"$T_{\mathrm{spin}}^{\mathrm{low}}$",
        )

    # Decide zoom window: from T_spin_high to T_spin_low, with a bit of margin
    if (T_spin_high is not None) and (T_spin_low is not None):
        T_low = min(T_spin_high, T_spin_low)
        T_high = max(T_spin_high, T_spin_low)
        span = max(1e-6, T_high - T_low)
        margin = 0.1 * span  # "a bit more" than the spinodal interval
        ax_zoom.set_xlim(T_low - margin, T_high + margin)

    ax_zoom.set_xlabel(r"Temperature $T$")
    ax_zoom.set_ylabel(r"$m^2(T) = \partial^2 V / \partial \phi^2$")
    ax_zoom.set_title("Curvature at the minima – zoom on the spinodal region")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="best", fontsize=8)

    fig2.tight_layout()
    plt.show()
    _save_figure(fig2, "C_m2", case)


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
            Tcrit = "Tcrit"
            print(f"    Tcrit       = {tdict[Tcrit]:9.4f}")
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

    summary = collect_summary(phases, full_transitions)
    print_final_summary(summary)
    return summary


if __name__ == "__main__":
    run_all()
