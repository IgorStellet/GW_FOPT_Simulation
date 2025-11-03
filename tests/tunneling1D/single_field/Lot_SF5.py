# -----------------------------------------------------------------------------
# Lot SF-5 — Profile search (overshoot/undershoot) using findProfile
#
# What this file demonstrates
# --------------------------
# 1) Legacy demo (thin & thick): call findProfile and plot φ(r).
# 2) Sensitivity to thinCutoff and interior fill (on/off).
# 3) Convergence tuning via phitol.
# 4) Error guard demo (rmax too small -> IntegrationError).
#
# Notes
# -----
# - Potentials are the same quartics used across previous lots:
#     thin-wall:  V = 0.25*φ^4 - 0.49*φ^3 + 0.235*φ^2
#     thick-wall: V = 0.25*φ^4 - 0.40*φ^3 + 0.100*φ^2
# - We report simple diagnostics (start/end radii, residual vs φ_meta, action).
# - Plots add horizontal guides at φ_abs and φ_meta for physical context.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions import SingleFieldInstanton, PotentialError
from CosmoTransitions.helper_functions import IntegrationError

np.set_printoptions(precision=6, suppress=True)


# -----------------------------------------------------------------------------
# Potentials (same as in earlier lots)
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
    Construct a SingleFieldInstanton with the standard vacua used in examples.
    """
    phi_absMin = 1.0   # true (stable) vacuum
    phi_metaMin = 0.0  # false (metastable) vacuum
    return SingleFieldInstanton(
        phi_absMin=phi_absMin,
        phi_metaMin=phi_metaMin,
        V=V,
        dV=dV,          # pass analytic dV; d2V is FD unless provided
        alpha=alpha,
        phi_eps=1e-3,   # relative FD scale (rescaled internally)
    )


def print_profile_diagnostics(inst: SingleFieldInstanton, label: str, profile):
    """
    Minimal, useful diagnostics for a returned Profile1D.
    """
    r0, rf = float(profile.R[0]), float(profile.R[-1])
    phi0, phif = float(profile.Phi[0]), float(profile.Phi[-1])
    dphif = float(profile.dPhi[-1])
    residual = abs(phif - inst.phi_metaMin)

    print(f"[{label}] profile: R[0]={r0:.6e}, R[-1]={rf:.6e}")
    print(f"         φ(r0)={phi0:.6e}, φ(rf)={phif:.6e}, |φ(rf)-φ_meta|={residual:.3e}, φ'(rf)={dphif:.3e}")
    print(f"         Rerr={profile.Rerr}\n")


def add_phi_guides(ax, inst: SingleFieldInstanton):
    """
    Add horizontal guides at φ_absMin and φ_metaMin to φ(r) plots.
    """
    ax.axhline(inst.phi_metaMin, ls="--", lw=1.0, alpha=0.9, label=r"$\phi_{\rm metaMin}$")
    ax.axhline(inst.phi_absMin,  ls=":",  lw=1.0, alpha=0.7, label=r"$\phi_{\rm absMin}$")


# -----------------------------------------------------------------------------
# Test 1 — Legacy demo (thin & thick) with improved plot
# -----------------------------------------------------------------------------
def test_1_legacy_demo():
    print("\n=== Test 1: Legacy demo — thin & thick walls ===")
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))

    # Thin-wall
    inst_thin = make_inst(V_thin, dV_thin, "thin")
    profile_thin = inst_thin.findProfile()  # all defaults
    ax.plot(profile_thin.R, profile_thin.Phi, lw=2, label="thin-wall φ(r)")
    print_profile_diagnostics(inst_thin, "thin-wall", profile_thin)

    # Thick-wall
    inst_thick = make_inst(V_thick, dV_thick, "thick")
    profile_thick = inst_thick.findProfile()  # all defaults
    ax.plot(profile_thick.R, profile_thick.Phi, lw=2, label="thick-wall φ(r)")
    print_profile_diagnostics(inst_thick, "thick-wall", profile_thick)

    # Guides and cosmetics
    add_phi_guides(ax, inst_thin)  # same φ_abs/φ_meta values (1.0 / 0.0)
    ax.set_xlabel("r"); ax.set_ylabel(r"$\phi(r)$")
    ax.set_title("findProfile — Legacy thin/thick demos")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
# Test 2 — Sensitivity: thinCutoff and interior fill
# -----------------------------------------------------------------------------
def test_2_thinCutoff_and_interior_fill():
    print("\n=== Test 2: Sensitivity to thinCutoff and interior fill (thin wall) ===")
    inst = make_inst(V_thin, dV_thin, "thin")

    # Case A: default thinCutoff, interior ON (default max_interior_pts)
    prof_A = inst.findProfile(
        phitol=1e-5, thinCutoff=0.01, npoints=600, rmin=1e-4, rmax=1e4, max_interior_pts=None
    )
    print_profile_diagnostics(inst, "thin (A: thinCutoff=0.01, interior ON)", prof_A)

    # Case B: looser thinCutoff, interior OFF (start strictly at r0>0)
    prof_B = inst.findProfile(
        phitol=1e-5, thinCutoff=0.15, npoints=600, rmin=1e-4, rmax=1e4, max_interior_pts=0
    )
    print_profile_diagnostics(inst, "thin (B: thinCutoff=0.15, interior OFF)", prof_B)

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
    ax.plot(prof_A.R, prof_A.Phi, lw=2, label=r"A: thinCutoff=0.01 (interior ON)")
    ax.plot(prof_B.R, prof_B.Phi, lw=2, label=r"B: thinCutoff=0.15 (interior OFF)")
    add_phi_guides(ax, inst)
    ax.set_xlabel("r"); ax.set_ylabel(r"$\phi(r)$")
    ax.set_title("Effect of thinCutoff & interior fill (thin-wall)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
# Test 3 — Convergence tuning via phitol (thick wall)
# -----------------------------------------------------------------------------
def test_3_convergence_tuning():
    print("\n=== Test 3: Convergence tuning via phitol (thick wall) ===")
    inst = make_inst(V_thick, dV_thick, "thick")

    # Tight tolerances
    prof_tight = inst.findProfile(phitol=2e-6, npoints=700)
    print_profile_diagnostics(inst, "thick (tight phitol=2e-6)", prof_tight)

    # Looser tolerances
    prof_loose = inst.findProfile(phitol=5e-5, npoints=700)
    print_profile_diagnostics(inst, "thick (loose phitol=5e-5)", prof_loose)

    # Visual comparison
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
    ax.plot(prof_tight.R, prof_tight.Phi, lw=2, label="tight phitol=2e-6")
    ax.plot(prof_loose.R, prof_loose.Phi, lw=2, ls="--", label="loose phitol=5e-5")
    add_phi_guides(ax, inst)
    ax.set_xlabel("r"); ax.set_ylabel(r"$\phi(r)$")
    ax.set_title("Convergence tuning (thick-wall)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
# Test 4 — Error guard (tiny rmax)
# -----------------------------------------------------------------------------
def test_4_error_guard():
    print("\n=== Test 4: Error guard (rmax too small) ===")
    inst = make_inst(V_thick, dV_thick, "thick")
    try:
        # Force failure: rmax so small the integration cannot proceed
        _ = inst.findProfile(phitol=1e-5, rmax=1e-6)  # scaled by rscale internally
    except IntegrationError as err:
        print(f"[expected] IntegrationError: {err}")
    except PotentialError as err:
        print(f"[unexpected] PotentialError: {err}")
    else:
        print("Warning: expected an IntegrationError but the call succeeded.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("---------- Lot SF-5: Profile search (findProfile) ----------")

    # 1) Legacy thin/thick, with improved diagnostics
    test_1_legacy_demo()

    # 2) Sensitivity to thinCutoff and interior fill (thin-wall)
    test_2_thinCutoff_and_interior_fill()

    # 3) Convergence tuning (thick-wall)
    test_3_convergence_tuning()

    # 4) Error guard demo
    test_4_error_guard()

    print("---------- END: Lot SF-5 examples ----------")
