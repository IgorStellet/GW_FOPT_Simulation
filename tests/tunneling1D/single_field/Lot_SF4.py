# -----------------------------------------------------------------------------
# Lot SF-4 — ODE core (EOM + RKQS driver + sampler)
#
# What this file demonstrates
# --------------------------
# 1) EOM sanity checks: evaluate equationOfMotion at a couple of (phi, dphi, r).
# 2) Event detection via integrateProfile:
#    - "undershoot" case: initial field too close to the true minimum (low energy),
#      the trajectory turns around (dphi -> 0 before hitting phi_metaMin).
#    - "overshoot" case: initial field too close to the false minimum (high energy),
#      the trajectory crosses phi_metaMin within the step.
#    For both cases we:
#       (a) print the event classification and the (r_event, phi_event, dphi_event),
#       (b) plot phi(r) up to the event, and annotate the event point and phi_metaMin.
# 3) integrateAndSaveProfile: sample the profile on a user grid R using cubic
#    Hermite interpolation between accepted RK steps.
# 4) Error demo: show IntegrationError when rmax is chosen too small.
#
# Notes
# -----
# - We reuse the same polynomial potentials from previous lots:
#     thin-wall:  V = 0.25*phi^4 - 0.49*phi^3 + 0.235*phi^2
#     thick-wall: V = 0.25*phi^4 - 0.40*phi^3 + 0.100*phi^2
# - The initial conditions (r0, phi(r0), dphi(r0)) are computed with
#   SingleFieldInstanton.initialConditions(delta_phi0, rmin, delta_phi_cutoff),
#   where delta_phi0 = exp(-x) * (phi_metaMin - phi_absMin). Large x -> undershoot-ish,
#   small x -> overshoot-ish.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Import the modernized class
from src.CosmoTransitions import SingleFieldInstanton, PotentialError
from src.CosmoTransitions.helper_functions import IntegrationError

np.set_printoptions(precision=6, suppress=True)

# -----------------------------------------------------------------------------
# Potentials (same as in Lot SF-1/SF-2)
# -----------------------------------------------------------------------------
def V_thin(phi: float) -> float:
    return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2

def dV_thin(phi: float) -> float:
    return phi*(phi - 0.47)*(phi - 1.0)

def V_thick(phi: float) -> float:
    return 0.25*phi**4 - 0.40*phi**3 + 0.100*phi**2

def dV_thick(phi: float) -> float:
    return phi*(phi - 0.20)*(phi - 1.0)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def make_inst(V, dV, label: str, alpha: int = 2) -> SingleFieldInstanton:
    """
    Construct a SingleFieldInstanton with standard vacua used in the examples.

    We keep phi_absMin = 1.0 (true/stable) and phi_metaMin = 0.0 (false/metastable),
    as in the legacy demos.
    """
    phi_absMin = 1.0
    phi_metaMin = 0.0
    return SingleFieldInstanton(
        phi_absMin=phi_absMin,
        phi_metaMin=phi_metaMin,
        V=V,
        dV=dV,          # we pass dV; d2V is obtained via finite-difference unless provided
        alpha=alpha,
        phi_eps=1e-3,   # relative finite-difference scale (rescaled internally)
    )

def build_tolerances(inst: SingleFieldInstanton, phitol: float = 1e-6):
    """
    Construct relative/absolute tolerances for [phi, dphi] that play nicely
    with the internal normalization in integrateProfile/integrateAndSaveProfile.
    """
    delta_phi = abs(inst.phi_metaMin - inst.phi_absMin)
    rscale = inst.rscale
    epsfrac = np.array([phitol, phitol], dtype=float)
    # absolute tolerances: scale phi by delta_phi and dphi by delta_phi/rscale
    epsabs = np.array([phitol * delta_phi, phitol * delta_phi / max(rscale, 1e-14)], dtype=float)
    return epsfrac, epsabs

def initial_conditions_from_x(inst: SingleFieldInstanton, x: float,
                              thinCutoff: float = 0.01):
    """
    Convert the shooting parameter x into (r0, phi0, dphi0) using the helper:
       delta_phi0 = exp(-x) * (phi_metaMin - phi_absMin)
    and a thin-wall-ish cutoff for the starting surface.
    """
    delta_phi = inst.phi_metaMin - inst.phi_absMin
    delta_phi0 = np.exp(-x) * delta_phi
    rmin = 1e-3 * inst.rscale
    delta_phi_cutoff = thinCutoff * abs(delta_phi)
    r0, phi0, dphi0 = inst.initialConditions(delta_phi0, rmin, delta_phi_cutoff)
    return r0, np.array([phi0, dphi0], dtype=float)

def sample_up_to_event(inst: SingleFieldInstanton, r0: float, y0: np.ndarray,
                       r_evt: float, epsfrac, epsabs, drmin: float):
    """
    Build a dense R grid from r0 to r_evt and sample the profile with integrateAndSaveProfile.
    """
    R = np.linspace(r0, r_evt, 200)
    dr0 = R[1] - R[0]
    prof = inst.integrateAndSaveProfile(R, y0, dr0, epsfrac, epsabs, drmin)
    return prof

# -----------------------------------------------------------------------------
# Test A — EOM sanity checks
# -----------------------------------------------------------------------------
def test_A_eom_sanity():
    print("\n=== Test A: EOM sanity checks ===")
    for label, (V, dV) in {
        "thin-wall":  (V_thin,  dV_thin),
        "thick-wall": (V_thick, dV_thick),
    }.items():
        inst = make_inst(V, dV, label)
        # Two points: moderate r and very small r (friction term check)
        y_mid = np.array([0.5, 0.0])      # phi ~ midway between vacua, dphi=0
        y_slope = np.array([0.3, 0.2])    # some slope
        rhs1 = inst.equationOfMotion(y_mid, r=1.0)
        rhs2 = inst.equationOfMotion(y_slope, r=1e-9)
        print(f"[{label}] EOM at r=1.0, y={y_mid}:  RHS = {rhs1}")
        print(f"[{label}] EOM at r=1e-9, y={y_slope}: RHS = {rhs2} (friction term large)")

# -----------------------------------------------------------------------------
# Test B — Event detection: undershoot & overshoot (thin-wall)
# -----------------------------------------------------------------------------
def test_B_events_thin():
    print("\n=== Test B: Event detection on thin-wall potential ===")
    inst = make_inst(V_thin, dV_thin, "thin-wall")
    epsfrac, epsabs = build_tolerances(inst, phitol=1e-4)
    drmin = 0.01 * 1e-3 * inst.rscale    # consistent with rmin used in ICs
    rmax  = 100.0 * inst.rscale

    # Choose x values to provoke undershoot and overshoot
    x_under = 6.0   # too close to absMin => low energy => turning point before metaMin
    x_over  = 0.2 # too close to metaMin => high energy => crosses metaMin (?)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax_i, (mode, x) in zip(ax, [("undershoot", x_under), ("undershoot", x_over)]):
        # Initial conditions
        r0, y0 = initial_conditions_from_x(inst, x, thinCutoff=0.1)
        # Integrate until event
        dy, dr0_try = None, 1e-3 * inst.rscale
        try:
            r_evt, y_evt, ctype = inst.integrateProfile(
                r0=r0, y0=y0, dr0=dr0_try,
                epsfrac=epsfrac, epsabs=epsabs,
                drmin=drmin, rmax=rmax
            )
        except IntegrationError as err:
            print(f"[thin-wall :: x={x:.2f}] IntegrationError: {err}")
            continue

        print(f"[thin-wall :: x={x:.2f}] event = {ctype:>10s} at r={r_evt:.6e} "
              f"(phi={y_evt[0]:.6e}, dphi={y_evt[1]:.6e})")

        # Sample up to the event for a smooth curve
        prof = sample_up_to_event(inst, r0, y0, r_evt, epsfrac, epsabs, drmin)
        ax_i.plot(prof.R, prof.Phi, lw=2, label=r"$\phi(r)$")
        ax_i.axhline(inst.phi_metaMin, ls="--", lw=1.0, label=r"$\phi_{\rm metaMin}$")
        ax_i.axvline(r_evt, ls=":", lw=1.0, label="event $r$")
        ax_i.plot([r_evt], [y_evt[0]], "o", ms=5)
        ax_i.set_title(f"thin-wall — expected {mode}")
        ax_i.set_xlabel("r")
        ax_i.grid(True, alpha=0.3)

    ax[0].set_ylabel(r"$\phi(r)$")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Test C — Event detection: undershoot & overshoot (thick-wall)
# -----------------------------------------------------------------------------
def test_C_events_thick():
    print("\n=== Test C: Event detection on thick-wall potential ===")
    inst = make_inst(V_thick, dV_thick, "thick-wall")
    epsfrac, epsabs = build_tolerances(inst, phitol=1e-5)
    drmin = 0.01 * 1e-3 * inst.rscale
    rmax  = 100.0 * inst.rscale

    x_under = 0.2
    x_over  = 6

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax_i, (mode, x) in zip(ax, [("undershoot", x_under), ("overshoot", x_over)]):
        r0, y0 = initial_conditions_from_x(inst, x, thinCutoff=0.01)
        try:
            r_evt, y_evt, ctype = inst.integrateProfile(
                r0=r0, y0=y0, dr0=1e-3 * inst.rscale,
                epsfrac=epsfrac, epsabs=epsabs,
                drmin=drmin, rmax=rmax
            )
        except IntegrationError as err:
            print(f"[thick-wall :: x={x:.2f}] IntegrationError: {err}")
            continue

        print(f"[thick-wall :: x={x:.2f}] event = {ctype:>10s} at r={r_evt:.6e} "
              f"(phi={y_evt[0]:.6e}, dphi={y_evt[1]:.6e})")

        prof = sample_up_to_event(inst, r0, y0, r_evt, epsfrac, epsabs, drmin)
        ax_i.plot(prof.R, prof.Phi, lw=2, label=r"$\phi(r)$")
        ax_i.axhline(inst.phi_metaMin, ls="--", lw=1.0, label=r"$\phi_{\rm metaMin}$")
        ax_i.axvline(r_evt, ls=":", lw=1.0, label="event $r$")
        ax_i.plot([r_evt], [y_evt[0]], "o", ms=5)
        ax_i.set_title(f"thick-wall — expected {mode}")
        ax_i.set_xlabel("r")
        ax_i.grid(True, alpha=0.3)

    ax[0].set_ylabel(r"$\phi(r)$")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Test D — integrateAndSaveProfile on a user R-grid
# -----------------------------------------------------------------------------
def test_D_sampler():
    print("\n=== Test D: integrateAndSaveProfile on a user R-grid ===")
    # We'll pick the thin-wall potential and a moderate x (may or may not converge;
    # here the goal is to demonstrate sampling on a fixed R grid).
    inst = make_inst(V_thin, dV_thin, "thin-wall")
    epsfrac, epsabs = build_tolerances(inst, phitol=2e-6)
    drmin = 0.01 * 1e-3 * inst.rscale

    x_demo = 15
    r0, y0 = initial_conditions_from_x(inst, x_demo, thinCutoff=0.01)
    # Build a user grid (monotone) and sample
    R = np.linspace(r0, r0 + 10.0 * inst.rscale, 300)
    dr0 = (R[1] - R[0])

    prof = inst.integrateAndSaveProfile(R, y0, dr0, epsfrac, epsabs, drmin)
    print(f"Profile sampled at {len(prof.R)} points.")
    print(f"Rerr (first clamped step radius) = {prof.Rerr}")

    plt.figure(figsize=(7.6, 4.2))
    plt.plot(prof.R, prof.Phi, lw=2, label=r"$\phi(r)$")
    plt.axhline(inst.phi_metaMin, ls="--", lw=1.0, label=r"$\phi_{\rm metaMin}$")
    plt.xlabel("r"); plt.ylabel(r"$\phi(r)$")
    plt.title("Sampling with integrateAndSaveProfile (thin-wall, demo)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# Test E — Error demo (small rmax)
# -----------------------------------------------------------------------------
def test_E_error_guard():
    print("\n=== Test E: Error guards (rmax too small) ===")
    inst = make_inst(V_thick, dV_thick, "thick-wall")
    epsfrac, epsabs = build_tolerances(inst, phitol=2e-6)
    drmin = 0.01 * 1e-3 * inst.rscale

    # Pick a value that tends to overshoot and set a tiny rmax to force an error.
    x_over = 0.2
    r0, y0 = initial_conditions_from_x(inst, x_over, thinCutoff=0.01)
    try:
        inst.integrateProfile(
            r0=r0, y0=y0, dr0=1e-3 * inst.rscale,
            epsfrac=epsfrac, epsabs=epsabs,
            drmin=drmin, rmax=1e-4 * inst.rscale  # ridiculously small
        )
    except IntegrationError as err:
        print(f"[expected] IntegrationError: {err}")

# -----------------------------------------------------------------------------
# Test F — Explict Initial Conditions (thin-wall)
# -----------------------------------------------------------------------------
def test_F_explicit_initial_Condition_thin():
    print("\n=== Test F: Explict Initial Conditions on thin-wall potential ===")
    inst = make_inst(V_thin, dV_thin, "thin-wall")
    epsfrac, epsabs = build_tolerances(inst, phitol=1e-4)
    drmin = 0.01 * 1e-3 * inst.rscale    # consistent with rmin used in ICs
    rmax  = 100.0 * inst.rscale

    # Choose x values to provoke undershoot and overshoot
    x_under = [0.001, [0.9, -0.1]] # r_0, phi_0 , dphi_0
    x_over  = [1, [0.3, -2]]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax_i, (mode, x) in zip(ax, [("undershoot", x_under), ("overshoot", x_over)]):
        # Initial conditions
        r0, y0 = x
        # Integrate until event
        dy, dr0_try = None, 1e-3 * inst.rscale
        try:
            r_evt, y_evt, ctype = inst.integrateProfile(
                r0=r0, y0=y0, dr0=dr0_try,
                epsfrac=epsfrac, epsabs=epsabs,
                drmin=drmin, rmax=rmax
            )
        except IntegrationError as err:
            print(f"[thin-wall :: x={x}] IntegrationError: {err}")
            continue

        print(f"[thin-wall :: x={x}] event = {ctype:>10s} at r={r_evt:.6e} "
              f"(phi={y_evt[0]:.6e}, dphi={y_evt[1]:.6e})")

        # Sample up to the event for a smooth curve
        prof = sample_up_to_event(inst, r0, y0, r_evt, epsfrac, epsabs, drmin)
        ax_i.plot(prof.R, prof.Phi, lw=2, label=r"$\phi(r)$")
        ax_i.axhline(inst.phi_metaMin, ls="--", lw=1.0, label=r"$\phi_{\rm metaMin}$")
        ax_i.axvline(r_evt, ls=":", lw=1.0, label="event $r$")
        ax_i.plot([r_evt], [y_evt[0]], "o", ms=5)
        ax_i.set_title(f"thin-wall — expected {mode}")
        ax_i.set_xlabel("r")
        ax_i.grid(True, alpha=0.3)

    ax[0].set_ylabel(r"$\phi(r)$")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("---------- Lot SF-4: ODE core (EOM + RKQS driver + sampler) ----------")

    # A) EOM evaluation sanity
    test_A_eom_sanity()

    # B) Events on thin-wall potential
    test_B_events_thin()

    # C) Events on thick-wall potential
    test_C_events_thick()

    # D) Sampler demo
    test_D_sampler()

    # E) Error guard demo
    test_E_error_guard()

    # F) Explicit Initial Conditions
    test_F_explicit_initial_Condition_thin()

    print("---------- END: Lot SF-4 examples ----------")
