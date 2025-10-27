# -----------------------------------------------------------------------------
# Complete, end-to-end demo for SingleFieldInstanton
# -----------------------------------------------------------------------------
# What this script provides
# ------------------------
# Six cohesive examples (A..F) that take you from the potential geometry, to
# initial conditions, to the final bounce profile and physically meaningful
# diagnostics/visualizations. This is intended for users who prefer a “single
# narrative” over per-function unit tests.
#
# You can switch between the "thin-wall" and "thick-wall" toy potentials, and
# optionally save every figure to disk by passing save_dir="some/folder" | Or choose your on potential V(phi).
#
# Usage
# -----
#   python -m tests.tunneling1D.single_field.Complete_Showcase
#
# or import and call `run_all(case="thin", save_dir=None)` from your notebook.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import math
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, interpolate

# Import the modernized class
from src.CosmoTransitions import SingleFieldInstanton

np.set_printoptions(precision=6, suppress=True)

# -----------------------------------------------------------------------------
# Potentials used across the examples
# -----------------------------------------------------------------------------
def V_thin(phi: float) -> float:
    return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2

def V_thick(phi: float) -> float:
    return 0.25*phi**4 - 0.40*phi**3 + 0.100*phi**2

def V_mine(phi: float) ->float:
    return 0.25 * phi ** 4 - 0.49 * phi ** 3 + 0.235 * phi ** 2

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def make_inst(case: str = "thin", alpha: int = 2) -> Tuple[SingleFieldInstanton, str]:
    """
    Construct a SingleFieldInstanton with the standard vacua:
      phi_absMin = 1.0 (true/stable), phi_metaMin = 0.0 (false/metastable).
    """
    case = case.lower().strip()
    if case == "thin":
        V, label = V_thin, "thin-wall"
    elif case == "thick":
        V, label = V_thick, "thick-wall"
    elif case == "mine":
        V, label = V_mine, "my potential V"
    else:
        raise ValueError("Unknown case (use 'thin' 'thick' or mine).")
    inst = SingleFieldInstanton(
        phi_absMin=1.0,
        phi_metaMin=0.0,
        V=V,
        alpha=alpha,
        phi_eps=1e-3,
    )
    return inst, label

def ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path

def build_phi_grid(inst: SingleFieldInstanton, margin: float = 0.1, n: int = 800):
    lo = min(inst.phi_metaMin, inst.phi_absMin)
    hi = max(inst.phi_metaMin, inst.phi_absMin)
    span = hi - lo
    return np.linspace(lo - margin*span, hi + margin*span, n)

def savefig(fig: plt.Figure, save_dir: Optional[str], name: str):
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=160, bbox_inches="tight")

# -----------------------------------------------------------------------------
# Example A
# -----------------------------------------------------------------------------
def example_A_potential_geometry(inst: SingleFieldInstanton,
                                 profile,
                                 save_dir: Optional[str] = None):
    """
    A) Plot V(φ) with markers at phi_meta, phi_abs(true), phi_bar, phi_top.
       Also plot the *inverted* potential -V(φ) with vertical markers and a
       single red dot at φ0 = profile.Phi[0].
       Print rscale (cubic & curvature) and ΔV diagnostics.
    """
    phi0 = float(profile.Phi[0])
    r0   = float(profile.R[0])

    sinfo = getattr(inst, "_scale_info", {}) or {}
    rscale_cubic = sinfo.get("rscale_cubic", np.nan)
    rscale_curv  = sinfo.get("rscale_curv", np.inf)
    phi_top = sinfo.get("phi_top", None)

    # Values & deltas
    V_meta = inst.V(inst.phi_metaMin)
    V_abs  = inst.V(inst.phi_absMin)
    V_top  = inst.V(phi_top)
    dV_true_meta = V_abs - V_meta
    dV_top = sinfo.get("V_top_minus_Vmeta", None)

    # Console summary
    print("\n[A] Potential geometry & scales")
    print(f"  phi_meta = {inst.phi_metaMin:.9f}, V(phi_meta) = {V_meta:.9e}")
    print(f"  phi_abs  = {inst.phi_absMin:.9f}, V(phi_abs)  = {V_abs :.9e}")
    print(f"  phi_bar  = {inst.phi_bar:.9f}, V(phi_bar)  = {inst.V(inst.phi_bar):.9e}")
    print(f"  phi_top  = {phi_top:.9f}, V(phi_top)  = {V_top :.9e}")
    print(f"  phi_0    = {phi0:.9f}, V(phi_0)    = {inst.V(phi0):.9e}, r0 = {r0:.6e}")
    print(f"  ΔV_true-meta = {dV_true_meta:.9e}")
    print(f"  ΔV_top -meta = {dV_top:.9e}")
    print(f"  rscale_cubic = {rscale_cubic:.9e}")
    if math.isfinite(rscale_curv):
        print(f"  rscale_curv  = {rscale_curv :.9e}   (from |V''(phi_top)|)")
    else:
        print( "  rscale_curv  = ∞ (flat top)")

    # Colors
    c_meta, c_abs, c_bar, c_top, c_0 = "#d62728", "#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"

    # rosa: #e377c2
    # laranja: #ff7f0e

    # Left: V(φ) with markers
    phi_grid = build_phi_grid(inst, margin=0.10, n=900)
    V_grid = inst.V(phi_grid)

    fig1, ax1 = plt.subplots(figsize=(7.8, 4.5))
    ax1.plot(phi_grid, V_grid, lw=2.2, color="#444444", label="V(φ)")
    ax1.scatter([inst.phi_metaMin], [V_meta], color=c_meta, s=40, label="φ_meta")
    ax1.scatter([inst.phi_absMin ], [V_abs ], color=c_abs , s=40, label="φ_true")
    ax1.scatter([inst.phi_bar    ], [inst.V(inst.phi_bar)], color=c_bar, s=40, label="φ_bar")
    ax1.scatter([phi_top         ], [V_top], color=c_top, s=40, label="φ_top")

    # Horizontal line at V(phi_meta); vertical lines at the markers
    ax1.axhline(V_meta, lw=1.0, ls="--", color=c_meta, alpha=0.8)
    for x, col in [(inst.phi_metaMin, c_meta), (inst.phi_absMin, c_abs),
                   (inst.phi_bar, c_bar), (phi_top, c_top)]:
        ax1.axvline(x, lw=1.0, ls=":", color=col, alpha=0.9)

    ax1.set_xlabel("φ"); ax1.set_ylabel("V(φ)")
    ax1.set_title("Potential with barrier markers")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", ncol=2)
    plt.tight_layout()
    savefig(fig1, save_dir, "A1_V_with_markers")

    # Right: -V(φ) with vertical lines and a red dot at φ0
    fig2, ax2 = plt.subplots(figsize=(7.8, 4.5))
    ax2.plot(phi_grid, -V_grid, lw=2.2, color="#444444", label="-V(φ)")
    for x, col in [(inst.phi_metaMin, c_meta), (inst.phi_absMin, c_abs),
                   (inst.phi_bar, c_bar), (phi_top, c_top)]:
        ax2.axvline(x, lw=1.0, ls=":", color=col, alpha=0.9)
    ax2.scatter([phi0], [-inst.V(phi0)], color="#e377c2", s=45, zorder=5, label="φ0")
    ax2.scatter([inst.phi_metaMin], [-V_meta], color=c_meta, s=40, label="φ_meta")
    ax2.scatter([inst.phi_absMin ], [-V_abs ], color=c_abs , s=40, label="φ_true")
    ax2.scatter([inst.phi_bar    ], [-inst.V(inst.phi_bar)], color=c_bar, s=40, label="φ_bar")
    ax2.scatter([phi_top         ], [-V_top], color=c_top, s=40, label="φ_top")
    ax2.set_xlabel("φ"); ax2.set_ylabel("-V(φ)")
    ax2.set_title("Inverted potential with φ0")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    savefig(fig2, save_dir, "A2_inverted_with_phi0")

# -----------------------------------------------------------------------------
# Example B
# -----------------------------------------------------------------------------
def example_B_local_quadratic_at_phi0(inst: SingleFieldInstanton,
                                      profile,
                                      save_dir: Optional[str] = None):
    """
    B) Print initial choice (r0, φ0, φ'(r0), V(φ0)) and overlay the potential
       with its quadratic Taylor expansion around φ0.
    """
    r0   = float(profile.R[0])
    phi0 = float(profile.Phi[0])
    dphi0= float(profile.dPhi[0])
    V0   = inst.V(phi0)
    dV0  = inst.dV(phi0)
    d2V0 = inst.d2V(phi0)

    print("\n[B] Initial local data at r0")
    print(f"  r0      = {r0:.9e}")
    print(f"  φ(r0)   = {phi0:.9f}")
    print(f"  φ'(r0)  = {dphi0:.9e}  (should be ~ 0 by regularity)")
    print(f"  V(φ0)   = {V0:.9e}")
    print(f"  dV(φ0)  = {dV0:.9e}")
    print(f"  d2V(φ0) = {d2V0:.9e}")

    # Plot V and its quadratic approximation around φ0
    span = abs(inst.phi_absMin - inst.phi_metaMin)
    phi_loc = np.linspace(phi0 - 0.1*span, phi0 + 0.1*span, 600)
    V_taylor = V0 + dV0*(phi_loc - phi0) + 0.5*d2V0*(phi_loc - phi0)**2

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(phi_loc, inst.V(phi_loc), lw=2.2, color="#444444", label="V(φ)")
    ax.plot(phi_loc, V_taylor, lw=2.0, ls="--", color="#d45500", label="quadratic @ φ0")
    ax.scatter([phi0], [V0], color="#888888", s=40, label="(φ0, V(φ0))")
    ax.set_xlabel("φ"); ax.set_ylabel("V(φ)")
    ax.set_title("Local quadratic approximation at φ0")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    savefig(fig, save_dir, "B_local_quadratic_at_phi0")

# -----------------------------------------------------------------------------
# Example C
# -----------------------------------------------------------------------------
def example_C_inverted_path(inst: SingleFieldInstanton,
                            profile,
                            save_dir: Optional[str] = None):
    """
    C) Left: inverted potential -V(φ) with a red line indicating the path from
       φ0 to φ_meta. Right: V(φ) with dots at φ0 (gray), φ_bar, φ_true, φ_meta (black).
    """
    phi0 = float(profile.Phi[0])

    phi_grid = build_phi_grid(inst, margin=0.12, n=900)
    V_grid = inst.V(phi_grid)

    # Values
    sinfo = getattr(inst, "_scale_info", {}) or {}
    phi_top = sinfo.get("phi_top", None)
    V_meta = inst.V(inst.phi_metaMin)
    V_abs  = inst.V(inst.phi_absMin)
    V_top  = inst.V(phi_top)

    # LEFT: -V with a red segment from φ0 to φ_meta
    fig1, ax1 = plt.subplots(figsize=(7.8, 4.5))
    ax1.plot(phi_grid, -V_grid, lw=2.2, color="#1f5fb4", label="-V(φ)")
    # Red path (straight in φ-coordinate): from φ0 to φ_meta on the same curve
    ph_a, ph_b = sorted([phi0, inst.phi_metaMin])
    mask = (phi_grid >= ph_a) & (phi_grid <= ph_b)
    ax1.plot(phi_grid[mask], -V_grid[mask], lw=3.0, color="#e377c2", label="path φ0→φ_meta")
    ax1.scatter([phi0], [-inst.V(phi0)], color="red", s=50, label="φ0")
    ax1.scatter([inst.phi_metaMin], [-V_meta], color="black", s=40, label="φ_meta")
    ax1.scatter([inst.phi_absMin ], [-V_abs ], color="black" , s=40, label="φ_true")
    ax1.scatter([inst.phi_bar    ], [-inst.V(inst.phi_bar)], color="black", s=40, label="φ_bar")
    ax1.scatter([phi_top         ], [-V_top], color="black", s=40, label="φ_top")

    ax1.set_xlabel("φ"); ax1.set_ylabel("-V(φ)")
    ax1.set_title("Inverted potential and the descent path")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    plt.tight_layout()
    plt.show()
    savefig(fig1, save_dir, "C_inverted_path")


# -----------------------------------------------------------------------------
# Example D
# -----------------------------------------------------------------------------
def example_D_phi_of_r(inst: SingleFieldInstanton,
                       profile,
                       save_dir: Optional[str] = None):
    """
    D) Plot φ(r) highlighting the starting point (r0, φ0); shade the interior
       region r ∈ [0, r0] to distinguish bubble interior vs exterior.
    """
    r = np.asarray(profile.R); phi = np.asarray(profile.Phi)
    r0, phi0 = float(r[0]), float(phi[0])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(r, phi, lw=2.2, color="#444444", label="φ(r)")
    ax.scatter([r0], [phi0], color="#e377c2", s=50, zorder=5, label="(r0, φ0)")
    # Shade the interior (0..r0)
    ax.axvspan(0.0, r0, color="#2ca02c", alpha=0.10, label="interior (shaded)")
    ax.axhline(inst.phi_metaMin, ls="--", lw=1.0, color="#d62728", label="φ_meta")
    ax.axhline(inst.phi_absMin , ls="--", lw=1.0, color="#2ca02c", label="φ_true")
    ax.set_xlabel("r"); ax.set_ylabel("φ(r)")
    ax.set_title("Bounce profile in radius")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    savefig(fig, save_dir, "D_phi_of_r")

# -----------------------------------------------------------------------------
# Example E
# -----------------------------------------------------------------------------
def example_E_spherical_maps(inst: SingleFieldInstanton,
                             profile,
                             save_dir: Optional[str] = None):
    """
    E) Visualize the spherical profile φ(r) in 2D:
       - Cartesian slice (x,y) colored by φ(√(x^2+y^2)).
       - Polar pcolormesh (r, θ) colored by φ(r).
       Color-bar is limited to [φ_meta, φ_true].
    """
    r = np.asarray(profile.R); phi = np.asarray(profile.Phi)
    r0 = float(r[0])

    # Interpolant φ(r) with sensible fill outside the tabulated range
    fr = interpolate.interp1d(
        r, phi, kind="linear", bounds_error=False,
        fill_value=(phi[0], inst.phi_metaMin)
    )

    # Cartesian slice
    Rmax = float(r[-1])
    pad  = 0.10 * Rmax
    Nx = Ny = 400
    x = np.linspace(-Rmax-pad, Rmax+pad, Nx)
    y = np.linspace(-Rmax-pad, Rmax+pad, Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    RAD = np.hypot(X, Y)
    PHI = fr(RAD)
    # Clip for colorbar range
    vmin, vmax = min(inst.phi_metaMin, inst.phi_absMin), max(inst.phi_metaMin, inst.phi_absMin)

    fig1, ax1 = plt.subplots(figsize=(6.2, 5.6))
    im = ax1.imshow(PHI, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_title("Cartesian slice: φ(√(x²+y²)) | Bounce profile at t=0")
    cb = plt.colorbar(im, ax=ax1, pad=0.02); cb.set_label("φ")
    plt.tight_layout()
    plt.show()
    savefig(fig1, save_dir, "E1_cartesian_slice")


# -----------------------------------------------------------------------------
# Example F
# -----------------------------------------------------------------------------
def example_F_ode_terms(inst: SingleFieldInstanton,
                        profile,
                        save_dir: Optional[str] = None):
    """
    F) Plot the contributions to the ODE:
         φ''(r)           (acceleration)
         (α/r) φ'(r)      (friction)
         V'(φ(r))         (force)
       and also overlay V(φ(r)) for reference (secondary axis).
    """
    r = np.asarray(profile.R); phi = np.asarray(profile.Phi); dphi = np.asarray(profile.dPhi)

    # Second derivative φ'' via centered finite difference (with end corrections)
    d2phi = np.zeros_like(phi)
    dr = np.diff(r)
    # Internal points (2..N-1): use three-point stencil
    for i in range(1, len(r)-1):
        dr_f = r[i+1] - r[i]
        dr_b = r[i]   - r[i-1]
        # Non-uniform second-derivative approximation
        d2phi[i] = 2.0 * (
            (phi[i+1] - phi[i]) / (dr_f*(dr_f + dr_b)) -
            (phi[i]   - phi[i-1]) / (dr_b*(dr_f + dr_b))
        )
    # One-sided for endpoints (order 2)
    d2phi[0]  = (phi[2] - 2*phi[1] + phi[0]) / ((r[1]-r[0])*(r[2]-r[0]))
    d2phi[-1] = (phi[-1] - 2*phi[-2] + phi[-3]) / ((r[-1]-r[-2])*(r[-1]-r[-3]))

    friction = inst.alpha * dphi / np.maximum(r, 1e-30)  # (α/r) φ'
    force = inst.dV(phi)                                  # V'(φ)

    # Plot ODE terms vs r
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(r, d2phi, lw=2.0, label="φ''(r)")
    ax.plot(r, friction, lw=2.0, label="(α/r) φ'(r)")
    ax.plot(r, force, lw=2.0, label="V'(φ(r))")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)

    # Second y-axis for V(φ(r))
    ax2 = ax.twinx()
    ax2.plot(r, inst.V(phi), lw=1.8, ls="--", color="#444444", label="V(φ(r))")
    ax.set_xlabel("r")
    ax.set_ylabel("ODE terms")
    ax2.set_ylabel("V(φ(r))")
    ax.set_title("ODE term decomposition along the profile")
    ax.grid(True, alpha=0.3)

    # Build a unified legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    savefig(fig, save_dir, "F_ode_terms_and_potential")

# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def compute_profile(inst: SingleFieldInstanton,
                    xguess: Optional[float] = None,
                    phitol: float = 1e-5,
                    thinCutoff: float = 0.01,
                    npoints: int = 600,
                    max_interior_pts: int = 0,
                    _MAX_ITERS=200
                    ) -> object:
    """
    Run the overshoot/undershoot solver to obtain the profile.
    Returns the profile namedtuple (R, Phi, dPhi, Rerr).
    """
    profile = inst.findProfile(
        xguess=xguess, xtol=1e-4, phitol=phitol,
        thinCutoff=thinCutoff, npoints=npoints,
        rmin=1e-4, rmax=1e4, max_interior_pts=max_interior_pts,
        _MAX_ITERS= _MAX_ITERS
    )
    return profile

def run_all(case: str = "thin",
            xguess: Optional[float] = None,
            phitol: float = 1e-5,
            save_dir: Optional[str] = None):
    """
    Execute all examples A..F in sequence for the chosen case ("thin", "thick" or "mine").

    Parameters
    ----------
    case : {"thin","thick", "mine"}
        Which benchmark potential to use.
    xguess : float or None
        Optional initial guess for the internal shooting parameter used by findProfile.
    phitol : float
        Fractional tolerance for integration in findProfile (smaller = tighter).
    save_dir : str or None
        If provided, figures are saved under this folder.
    """
    save_dir = ensure_dir(save_dir)
    inst, label = make_inst(case)
    print(f"=== Running complete showcase on: {label} potential ===")

    # Solve once and reuse the profile for all examples
    profile = compute_profile(inst, xguess=xguess, phitol=phitol, npoints=800)

    # A) Potential geometry & inverted view with φ0
    example_A_potential_geometry(inst, profile, save_dir=save_dir)

    # B) Local quadratic at φ0
    example_B_local_quadratic_at_phi0(inst, profile, save_dir=save_dir)

    # C) Inverted potential with path, and V(φ) with points
    example_C_inverted_path(inst, profile, save_dir=save_dir)

    # D) φ(r) with interior shading and markers
    example_D_phi_of_r(inst, profile, save_dir=save_dir)

    # E) 2D spherical visualizations (Cartesian & polar)
    example_E_spherical_maps(inst, profile, save_dir=save_dir)

    # F) ODE terms decomposition along the profile
    example_F_ode_terms(inst, profile, save_dir=save_dir)

    print("=== Showcase complete. ===")
    if save_dir:
        print(f"Figures saved under: {os.path.abspath(save_dir)}")

# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Thin-wall demo (set save_dir to a folder to save images)
    run_all(case="thin", xguess=None, phitol=1e-5, save_dir=None)

    # Uncomment to also run thick-wall in one go:
    #run_all(case="thick", xguess=None, phitol=1e-5, save_dir=None)
