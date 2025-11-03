# ----------------------------------------------------------------------
# Lot_SF_2.py  — Barrier & scales tests for SingleFieldInstanton
#
# What this file covers:
#   1) For the "thin-wall" and "thick-wall" demo potentials:
#      - Plot V(φ) with markers for φ_top (barrier top) and φ_bar (edge),
#        plus a horizontal reference at V(φ_meta).
#      - Print barrier diagnostics and the characteristic scales:
#          * rscale_cubic (legacy/robust)
#          * rscale_curv  (from V''(φ_top), when defined)
#   2) (Optional) Negative test: a potential with NO barrier -> clean error.
#
# Notes:
# - This test assumes the modernized SingleFieldInstanton from tunneling1D.py,
#   with improved findBarrierLocation / findRScale and cached diagnostics:
#       ._barrier_info: dict with keys ["phi_bar","phi_top","V_top_minus_Vmeta",...]
#       ._scale_info  : dict with keys ["rscale_cubic","rscale_curv","d2V_top",...]
# - If you run from repository root, matplotlib windows will pop up. Close each
#   to continue, or set SAVE_FIGS=True to save PNGs next to this script.
# ----------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# --- Flexible import to match your project layout --------------------------------

from CosmoTransitions import SingleFieldInstanton, PotentialError

# ----------------------------------------------------------------------
# Potentials: thin-wall and thick-wall demos (with analytic dV, d2V)
# ----------------------------------------------------------------------
def V_thin(phi):
    # 0.25 φ^4 - 0.49 φ^3 + 0.235 φ^2
    return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2

def dV_thin(phi):
    # φ (φ - 0.47) (φ - 1)
    return phi * (phi - 0.47) * (phi - 1.0)

def d2V_thin(phi):
    # derivative of dV_thin: 3 φ^2 - 2.94 φ + 0.47
    return 3.0*phi**2 - 2.94*phi + 0.47


def V_thick(phi):
    # 0.25 φ^4 - 0.4 φ^3 + 0.1 φ^2
    return 0.25*phi**4 - 0.4*phi**3 + 0.1*phi**2

def dV_thick(phi):
    # φ (φ - 0.2) (φ - 1)
    return phi * (phi - 0.2) * (phi - 1.0)

def d2V_thick(phi):
    # derivative of dV_thick: 3 φ^2 - 2.4 φ + 0.2
    return 3.0*phi**2 - 2.4*phi + 0.2


# (Optional) a no-barrier potential for negative test
def V_nobar(phi):
    # Single well at 0, monotonic rise thereafter -> no barrier between 0 and 1
    return 0.5*(phi - 0.0)**2

def dV_nobar(phi):
    return (phi - 0.0)

def d2V_nobar(phi):
    return np.ones_like(phi, dtype=float)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
SAVE_FIGS = False  # set True to save images instead of (or in addition to) show
FIG_DIR = os.path.dirname(os.path.abspath(__file__))

def _maybe_save(fig, name):
    if SAVE_FIGS:
        path = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"[saved] {path}")


def run_case(label, V, dV, d2V, phi_absMin=1.0, phi_metaMin=0.0, x_range=(-0.2, 1.2)):
    """
    Build an instanton object, print barrier & scale diagnostics,
    and plot V(φ) with φ_top, φ_bar, and V(φ_meta) guide.
    """
    print("\n" + "="*72)
    print(f"CASE: {label}")
    print("="*72)

    try:
        inst = SingleFieldInstanton(
            phi_absMin=phi_absMin,
            phi_metaMin=phi_metaMin,
            V=V,
            dV=dV,
            d2V=d2V,
            # alpha, phi_eps use defaults; barrier/scale will be computed
        )
    except PotentialError as e:
        print(f"[ERROR] Could not construct SingleFieldInstanton: {e}")
        return

    # --- Barrier diagnostics
    binfo = getattr(inst, "_barrier_info", {}) or {}
    phi_top = binfo.get("phi_top", None)
    phi_bar = binfo.get("phi_bar", None)
    V_meta = binfo.get("V_meta", V(phi_metaMin))
    dVtag = binfo.get("V_top_minus_Vmeta", None)

    print("Barrier diagnostics:")
    print(f"  phi_metaMin = {phi_metaMin:.12g},  V(phi_metaMin) = {V_meta:.12g}")
    if phi_top is not None:
        print(f"  phi_top     = {phi_top:.12g},  ΔV_top ≡ V(top)-V(meta) = {dVtag:.12g}")
    else:
        print("  phi_top     = (not available)")
    if phi_bar is not None:
        print(f"  phi_bar     = {phi_bar:.12g}  (V equals V(phi_metaMin) on downhill side)")
    else:
        print("  phi_bar     = (not available)")

    # --- Scale diagnostics
    sinfo = getattr(inst, "_scale_info", {}) or {}
    r_cubic = sinfo.get("rscale_cubic", np.nan)
    r_curv  = sinfo.get("rscale_curv", np.inf)
    d2Vtop  = sinfo.get("d2V_top", np.nan)

    print("Scale diagnostics:")
    print(f"  rscale_cubic (legacy/robust) = {r_cubic:.12g}")
    if np.isfinite(r_curv):
        print(f"  rscale_curv  (from V'' at top)= {r_curv:.12g}  with  V''(top) = {d2Vtop:.12g}")
    else:
        print(f"  rscale_curv  (from V'' at top)= ∞  (flat or non-negative curvature at top)")

    # --- Plot: V(φ) with φ_top, φ_bar, and V(φ_meta) line
    phi = np.linspace(x_range[0], x_range[1], 600)
    Vv = V(phi)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(phi, Vv, lw=2, label=r"$V(\phi)$")

    # Horizontal guide at V(phi_meta)
    ax.axhline(V_meta, ls="--", lw=1.2, alpha=0.8, label=r"$V(\phi_{\rm meta})$")

    # Vertical guides: phi_top and phi_bar (if available)
    if phi_top is not None:
        ax.axvline(phi_top, ls="-.", lw=1.2, alpha=0.9, label=r"$\phi_{\rm top}$")
    if phi_bar is not None:
        ax.axvline(phi_bar, ls=":", lw=1.6, alpha=0.9, label=r"$\phi_{\rm bar}$")

    # Also show the two minima for context
    ax.axvline(phi_metaMin, ls=":", lw=1.0, alpha=0.5)
    ax.axvline(phi_absMin,  ls=":", lw=1.0, alpha=0.5)

    ax.set_title(f"{label}: Potential with barrier markers")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V(\phi)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    _maybe_save(fig, f"Lot_SF_2__{label.replace(' ', '_').lower()}__V_with_markers")
    plt.show()


def negative_test_no_barrier():
    """
    (Optional) Show that we get a clear PotentialError when no barrier exists.
    """
    print("\n" + "="*72)
    print("NEGATIVE TEST: no barrier")
    print("="*72)
    try:
        _ = SingleFieldInstanton(
            phi_absMin=1.0,
            phi_metaMin=0.0,
            V=V_nobar,
            dV=dV_nobar,
            d2V=d2V_nobar,
        )
    except PotentialError as e:
        print(f"[expected] Raised PotentialError (no barrier): {e}")
    else:
        print("[warning] Unexpectedly constructed an instanton for a no-barrier potential.")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Thin-wall: sharper barrier (nearly degenerate minima -> thinner wall)
    run_case(
        label="Thin-wall demo",
        V=V_thin, dV=dV_thin, d2V=d2V_thin,
        phi_absMin=1.0, phi_metaMin=0.0,
        x_range=(-0.1, 1.1),
    )

    # Thick-wall: broader barrier
    run_case(
        label="Thick-wall demo",
        V=V_thick, dV=dV_thick, d2V=d2V_thick,
        phi_absMin=1.0, phi_metaMin=0.0,
        x_range=(-0.1, 1.1),
    )

    # Negative test (uncomment to run)
    negative_test_no_barrier()
