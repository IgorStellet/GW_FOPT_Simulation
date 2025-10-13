# -----------------------------------------------------------------------------
# Lot SF-6 — Action & post-processing
#
# What this file demonstrates
# --------------------------
# 1) Compute a bounce profile for thin- and thick-wall benchmark potentials.
# 2) Compute the Euclidean action S via `findAction`, and a detailed split via
#    `actionBreakdown` (kinetic, potential, interior-bulk; plus per-r density).
# 3) Resample (phi, dphi) on a uniform phi-grid using `evenlySpacedPhi`.
# 4) Extract wall geometry via `wallDiagnostics` (r_peak, thickness, etc.).
# 5) Print/plot proxies for β_eff via `betaEff` with methods: rscale/curvature/wall.
#
# Expected physics highlights
# ---------------------------
# - Thin-wall: sharper barrier → thinner wall (smaller thickness), typically
#   larger |phi'| peak, so β_eff(wall) tends to be larger than in thick-wall.
# - Thick-wall: broader transition in r; action density spreads more.
# - Action density typically localizes around the wall region.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Import the modernized class
from src.CosmoTransitions import SingleFieldInstanton, PotentialError

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
    Construct a SingleFieldInstanton with the standard vacua used in the examples:
    phi_absMin = 1.0 (true), phi_metaMin = 0.0 (false/metastable).
    """
    return SingleFieldInstanton(
        phi_absMin=1.0,
        phi_metaMin=0.0,
        V=V,
        dV=dV,
        alpha=alpha,
        phi_eps=1e-3,  # finite-diff scale (relative; rescaled internally)
    )

def compute_profile(inst: SingleFieldInstanton, phitol: float = 2e-6):
    """
    Wrapper to produce a profile with a moderately strict tolerance.
    """
    profile = inst.findProfile(
        xguess=None,     # default heuristic from phi_bar
        xtol=1e-4,
        phitol=phitol,   # relative error tolerance used internally
        thinCutoff=0.01,
        npoints=600,
        rmin=1e-4,
        rmax=3e3,        # large enough for both demo potentials
        max_interior_pts=None,
    )
    return profile

# -----------------------------------------------------------------------------
# Test A — Actions: total S and breakdown (densities)
# -----------------------------------------------------------------------------
def test_A_actions_and_densities():
    print("\n=== Test A: Actions and density breakdown ===")

    cases = {
        "thin-wall":  (V_thin,  dV_thin),
        "thick-wall": (V_thick, dV_thick),
    }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for i, (label, (V, dV)) in enumerate(cases.items()):
        inst = make_inst(V, dV, label)
        profile = compute_profile(inst, phitol=2e-6)

        # Total action
        S = inst.findAction(profile)

        # Detailed breakdown and densities
        br = inst.actionBreakdown(profile)

        print(f"[{label}]")
        print(f"  S_total    = {br.S_total:.6e}")
        print(f"   S_kin     = {br.S_kin:.6e}")
        print(f"   S_pot     = {br.S_pot:.6e}")
        print(f"   S_interior= {br.S_interior:.6e}")
        print(f"  (check) S_kin + S_pot + S_interior = {br.S_kin + br.S_pot + br.S_interior:.6e}")

        # Plot density vs r
        ax[i].plot(br.r, br.density["kin"], lw=1.8, label="kinetic density")
        ax[i].plot(br.r, br.density["pot"], lw=1.8, label="potential density", ls="--")
        ax[i].plot(br.r, br.density["tot"], lw=2.2, label="total (line)", alpha=0.8)
        ax[i].axhline(0.0, color="k", lw=0.7)
        ax[i].set_title(f"{label}: action densities vs r")
        ax[i].set_xlabel("r")
        ax[i].grid(True, alpha=0.3)

    ax[0].set_ylabel("action density  (units of integrand)")
    ax[1].legend(loc="best")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Test B — Resample on a uniform phi-grid (evenlySpacedPhi)
# -----------------------------------------------------------------------------
def test_B_evenly_spaced_phi():
    print("\n=== Test B: Resampling on a uniform phi-grid ===")
    for label, (V, dV) in {
        "thin-wall":  (V_thin,  dV_thin),
        "thick-wall": (V_thick, dV_thick),
    }.items():
        inst = make_inst(V, dV, label)
        profile = compute_profile(inst, phitol=2e-6)

        # Uniform phi-grid sampling
        phi_u, dphi_u = inst.evenlySpacedPhi(profile.Phi, profile.dPhi, npoints=200, k=1, fixAbs=True)

        print(f"[{label}] uniform-phi samples: {len(phi_u)} points; "
              f"phi range = [{phi_u[0]:.3f}, {phi_u[-1]:.3f}]")

        # Plot dphi vs phi (phi-space view of the wall)
        plt.figure(figsize=(7.0, 4.2))
        plt.plot(phi_u, dphi_u, lw=2)
        plt.title(f"{label}: φ-space sampling (dφ vs φ)")
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$d\phi/d r$ (resampled)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------
# Test C — Wall diagnostics (location & thickness)
# -----------------------------------------------------------------------------
def test_C_wall_diagnostics():
    print("\n=== Test C: Wall diagnostics (location & thickness) ===")
    for label, (V, dV) in {
        "thin-wall":  (V_thin,  dV_thin),
        "thick-wall": (V_thick, dV_thick),
    }.items():
        inst = make_inst(V, dV, label)
        profile = compute_profile(inst, phitol=2e-6)

        ws = inst.wallDiagnostics(profile, frac=(0.1, 0.9))
        print(f"[{label}] r_peak={ws.r_peak:.6e}, r_mid={ws.r_mid:.6e}, "
              f"r_lo={ws.r_lo:.6e}, r_hi={ws.r_hi:.6e}, thickness={ws.thickness:.6e} "
              f"(phi_lo={ws.phi_lo:.3f}, phi_hi={ws.phi_hi:.3f})")

        # Plot φ(r) with the diagnostic radii as vertical markers
        plt.figure(figsize=(7.8, 4.2))
        plt.plot(profile.R, profile.Phi, lw=2, label=r"$\phi(r)$")
        plt.axhline(inst.phi_metaMin, ls="--", lw=1.0, label=r"$\phi_{\rm metaMin}$", alpha=0.8)
        for x, name, ls in [(ws.r_lo, "r_lo", ":"), (ws.r_hi, "r_hi", ":"), (ws.r_mid, "r_mid", "-."), (ws.r_peak, "r_peak", "--")]:
            plt.axvline(x, ls=ls, lw=1.1, alpha=0.9, label=name)
        plt.title(f"{label}: wall location & thickness")
        plt.xlabel("r"); plt.ylabel(r"$\phi(r)$")
        plt.legend(ncol=2, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------
# Test D — β_eff proxies: rscale vs curvature vs wall
# -----------------------------------------------------------------------------
def test_D_beta_proxies():
    print("\n=== Test D: β_eff proxies (rscale, curvature, wall) ===")
    labels = []
    beta_r = []
    beta_c = []
    beta_w = []

    for label, (V, dV) in {
        "thin-wall":  (V_thin,  dV_thin),
        "thick-wall": (V_thick, dV_thick),
    }.items():
        inst = make_inst(V, dV, label)
        profile = compute_profile(inst, phitol=2e-6)

        b_r = inst.betaEff(profile, method="rscale")
        b_c = inst.betaEff(profile, method="curvature")  # requires barrier top
        b_w = inst.betaEff(profile, method="wall")

        labels.append(label)
        beta_r.append(b_r)
        beta_c.append(b_c)
        beta_w.append(b_w)

        print(f"[{label}] β_rscale={b_r:.6e}, β_curv={b_c:.6e}, β_wall={b_w:.6e}")

    # Simple grouped bar chart
    x = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(8.8, 4.2))
    plt.bar(x - width, beta_r, width, label=r"$\beta_{\rm eff}$ (1/rscale)")
    plt.bar(x,          beta_c, width, label=r"$\beta_{\rm eff}$ (curvature)")
    plt.bar(x + width,  beta_w, width, label=r"$\beta_{\rm eff}$ (1/thickness)")
    plt.xticks(x, labels)
    plt.ylabel(r"$\beta_{\rm eff}$  (inverse length units)")
    plt.title(r"β proxies: rscale vs curvature vs wall thickness")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("---------- Lot SF-6: Action & post-processing ----------")

    # A) Actions and densities
    test_A_actions_and_densities()

    # B) Uniform φ-grid resampling
    test_B_evenly_spaced_phi()

    # C) Wall geometry
    test_C_wall_diagnostics()

    # D) β_eff proxies comparison
    test_D_beta_proxies()

    print("---------- END: Lot SF-6 examples ----------")
