# --------------------------------------------------------------------------------------
# Lot SF-3 — Quadratic local solution & initial conditions (SingleFieldInstanton)
# --------------------------------------------------------------------------------------
# What this file does:
# 1) Local quadratic solution near the true minimum (stable curvature, d2V>0):
#    - Evaluate phi(r), phi'(r) from exactSolution at small r for thin/thick potentials.
#    - Visualize regularity at the origin and the near-quadratic start.
# 2) Local solution near the barrier top (unstable curvature, d2V<0):
#    - Locate phi_top via 1D maximize of -V on [phi_metaMin, phi_bar]; compute d2V<0.
#    - Plot the oscillatory/“inverted” behavior for small r.
# 3) Flat-curvature branch (d2V = 0):
#    - Call exactSolution with d2V=0 and compare against the closed-form polynomial
#      phi(r) = phi0 + [dV/(2*(alpha+1))] * r^2. Print the max absolute difference.
# 4) initialConditions: find (r0, phi(r0), phi'(r0)) for a practical thin-wall style setup.
#    - Plot the short ‘pre-integration’ track from r=0 to r=r0; mark the starting point.
# 5) initialConditions error handling:
#    - Choose delta_phi0=0 exactly (at the true minimum) with a positive cutoff,
#      which cannot be reached by the local model. Show the IntegrationError message.
#
# Notes:
# - The potentials V1/V2 match the thin/thick examples used earlier.
# - We pass analytic dV,d2V to SingleFieldInstanton to avoid finite-difference noise.
# - Keep grids modest; these are *didactic* demos.
# --------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Import the modernized class and errors
from src.CosmoTransitions import SingleFieldInstanton, PotentialError
from src.CosmoTransitions.helper_functions import IntegrationError


np.set_printoptions(precision=6, suppress=True)


# --------------------------------------------------------------------------------------
# Potentials (same as in lots SF-1/SF-2)
# --------------------------------------------------------------------------------------
def V1(phi):  # Thin-walled example
    return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2

def dV1(phi):
    # d/dphi [phi*(phi - 0.47)*(phi - 1)]
    g  = phi*phi - 1.47*phi + 0.47
    gp = 2.0*phi - 1.47
    return phi*g

def d2V1(phi):
    g  = phi*phi - 1.47*phi + 0.47
    gp = 2.0*phi - 1.47
    return g + phi*gp

def V2(phi):  # Thick-walled example
    return 0.25*phi**4 - 0.4*phi**3 + 0.1*phi**2

def dV2(phi):
    # d/dphi [phi*(phi - 0.2)*(phi - 1)]
    g  = phi*phi - 1.2*phi + 0.2
    gp = 2.0*phi - 1.2
    return phi*g

def d2V2(phi):
    g  = phi*phi - 1.2*phi + 0.2
    gp = 2.0*phi - 1.2
    return g + phi*gp


def make_SFI(V, dV, d2V, alpha=2):
    # absMin=1.0, metaMin=0.0 matches our examples
    return SingleFieldInstanton(
        phi_absMin=1.0,
        phi_metaMin=0.0,
        V=V,
        dV=dV,
        d2V=d2V,
        alpha=alpha,          # friction power (alpha = D-1)
        phi_eps=1e-3,         # only used by fallback finite-differences (we provided dV/d2V)
        phi_bar=None,         # auto-locate barrier edge
        rscale=None           # auto-scale using barrier shape
    )


print("---------- TESTS: Lot SF-3 (exactSolution & initialConditions) ----------\n")

# --------------------------------------------------------------------------------------
# Test 1 — Stable curvature near the true minimum (thin & thick): small-r behavior
# --------------------------------------------------------------------------------------
print("=== Test 1: Local quadratic solution near abs minimum (d2V > 0) ===")
for name, (V, dV, d2V) in {"Thin": (V1, dV1, d2V1), "Thick": (V2, dV2, d2V2)}.items():
    sfi = make_SFI(V, dV, d2V, alpha=2)
    Dphi = sfi.phi_metaMin - sfi.phi_absMin  # negative number here
    phi0 = sfi.phi_absMin + 1e-3*abs(Dphi)   # very close to absMin
    dV0  = sfi.dV(phi0)
    d2V0 = sfi.d2V(phi0)                   # should be > 0
    assert d2V0 > 0, f"{name}: expected positive curvature at/near true minimum."

    # Small-r grid
    rmax_small = min(0.2*sfi.rscale, 0.5)
    r = np.linspace(0.0, rmax_small, 200)
    phi = np.empty_like(r)
    dphi = np.empty_like(r)
    for i, ri in enumerate(r):
        sol = sfi.exactSolution(ri, phi0, dV0, d2V0)
        phi[i], dphi[i] = sol.phi, sol.dphi

    print(f" {name}: dV(phi0)={dV0:.3e}, d2V(phi0)={d2V0:.3e}, rscale≈{sfi.rscale:.3e}")
    print("  Expectation: phi'(0)=0, smooth quadratic rise; numerical curve should be regular.\n")

    plt.figure(figsize=(8.0,4.6))
    plt.plot(r, phi - phi0, label=r"$\phi(r)-\phi_0$")
    plt.plot(r, dphi, label=r"$\phi'(r)$")
    plt.title(f"Test 1 ({name}) — Near abs minimum (d2V>0)")
    plt.xlabel("r"); plt.ylabel("value")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------------------
# Test 2 — Unstable curvature near the barrier top (d2V < 0)
# --------------------------------------------------------------------------------------
print("=== Test 2: Local solution near barrier top (d2V < 0) ===")
for name, (V, dV, d2V) in {"Thin": (V1, dV1, d2V1), "Thick": (V2, dV2, d2V2)}.items():
    sfi = make_SFI(V, dV, d2V, alpha=2)
    # Reconstruct barrier top: maximize V between meta and barrier-edge.
    x1 = min(sfi.phi_metaMin, sfi.phi_bar)
    x2 = max(sfi.phi_metaMin, sfi.phi_bar)
    tol = abs(x2 - x1) * 1e-8
    phi_top = optimize.fminbound(lambda x: -sfi.V(x), x1, x2, xtol=tol)
    dVt  = sfi.dV(phi_top)
    d2Vt = sfi.d2V(phi_top)
    print(f" {name}: phi_top={phi_top:.6f}, dV(phi_top)={dVt:.3e}, d2V(phi_top)={d2Vt:.3e}")
    if d2Vt >= 0:
        print("  Warning: curvature at the found top is not negative; the potential may be nearly flat.")
    # Small-r grid around the top
    r = np.linspace(0.0, min(0.2*sfi.rscale, 0.5), 220)
    phi = np.empty_like(r)
    dphi = np.empty_like(r)
    for i, ri in enumerate(r):
        sol = sfi.exactSolution(ri, phi_top, dVt, d2Vt)
        phi[i], dphi[i] = sol.phi, sol.dphi

    print("  Expectation: d2V < 0 → oscillatory (J_ν) behavior; still regular at r=0 with phi'(0)=0.\n")

    plt.figure(figsize=(8.0,4.6))
    plt.plot(r, phi - phi_top, label=r"$\phi(r)-\phi_{\rm top}$")
    plt.plot(r, dphi, label=r"$\phi'(r)$")
    plt.title(f"Test 2 ({name}) — Near barrier top (d2V<0)")
    plt.xlabel("r"); plt.ylabel("value")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------------------
# Test 3 — Flat-curvature branch (d2V = 0): compare with polynomial closed form
# --------------------------------------------------------------------------------------
print("=== Test 3: Flat-curvature branch (d2V = 0) vs polynomial ===")
for name, (V, dV, d2V) in {"Thin": (V1, dV1, d2V1), "Thick": (V2, dV2, d2V2)}.items():
    sfi = make_SFI(V, dV, d2V, alpha=2)
    Δφ = sfi.phi_metaMin - sfi.phi_absMin
    phi0 = sfi.phi_absMin + 2e-3*abs(Δφ)
    dV0  = sfi.dV(phi0)
    # Force the flat-curvature branch by passing d2V=0 explicitly:
    r = np.linspace(0.0, min(0.25*sfi.rscale, 0.75), 180)
    phi_exact = np.empty_like(r)
    for i, ri in enumerate(r):
        sol = sfi.exactSolution(ri, phi0, dV0, 0.0)  # <— flat curvature branch
        phi_exact[i] = sol.phi
    # Polynomial closed form: phi(r) = phi0 + [dV/(2*(alpha+1))] r^2
    poly = phi0 + (dV0/(2.0*(sfi.alpha+1.0)))*r*r
    maxdiff = np.max(np.abs(phi_exact - poly))
    print(f" {name}: dV(phi0)={dV0:.3e}, alpha={sfi.alpha}, max|exact - polynomial| = {maxdiff:.3e}")
    print("  Expectation: differences at machine epsilon (both are exact for d2V=0).\n")

    plt.figure(figsize=(7.8,4.4))
    plt.plot(r, phi_exact, label="exactSolution(d2V=0)")
    plt.plot(r, poly, "--", label="polynomial closed form")
    plt.title(f"Test 3 ({name}) — Flat curvature branch")
    plt.xlabel("r"); plt.ylabel(r"$\phi(r)$")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------------------
# Test 4 — initialConditions: pick practical deltas, find r0 and mark the start point
# --------------------------------------------------------------------------------------
print("=== Test 4: initialConditions (r0, phi(r0), phi'(r0)) ===")
for name, (V, dV, d2V) in {"Thin": (V1, dV1, d2V1), "Thick": (V2, dV2, d2V2)}.items():
    sfi = make_SFI(V, dV, d2V, alpha=2)
    Δφ = sfi.phi_metaMin - sfi.phi_absMin
    # thinCutoff ~ 1% of |Δφ|, central offset ~ e^{-4} |Δφ|
    delta_phi0        = np.exp(-4.0) * abs(Δφ)
    delta_phi_cutoff  = 0.01 * abs(Δφ)
    rmin = 1e-3 * sfi.rscale

    r0, phi0, dphi0 = sfi.initialConditions(delta_phi0, rmin, delta_phi_cutoff)
    print(f" {name}: r0={r0:.6e}, phi(r0)={phi0:.6e}, phi'(r0)={dphi0:.6e}")
    print("  Expectation: |phi(r0)-phi_absMin| ≳ cutoff, and phi'(r0) has the same sign as delta_phi0.\n")

    # Visualize the short track from r=0 to r=r0 with the local model we used
    # (same phi0_at_center and local dV/d2V inferred there)
    phi_center = sfi.phi_absMin + delta_phi0
    dV_center  = sfi.dV(phi_center)
    d2V_center = sfi.d2V(phi_center)
    r = np.linspace(0.0, r0, 150)
    phi_local = np.empty_like(r)
    for i, ri in enumerate(r):
        sol = sfi.exactSolution(ri, phi_center, dV_center, d2V_center)
        phi_local[i] = sol.phi

    plt.figure(figsize=(7.8,4.4))
    plt.plot(r, phi_local, label="local quadratic solution")
    plt.axvline(r0, ls="--", alpha=0.6, label=r"$r_0$")
    plt.scatter([r0], [phi0], s=45, zorder=5, label=r"$(r_0,\phi(r_0))$")
    plt.title(f"Test 4 ({name}) — initialConditions path to $r_0$")
    plt.xlabel("r"); plt.ylabel(r"$\phi(r)$")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------------------
# Test 5 — initialConditions error case: delta_phi0 = 0 → cannot exceed cutoff
# --------------------------------------------------------------------------------------
print("=== Test 5: initialConditions error (unreachable cutoff) ===")
for name, (V, dV, d2V) in {"Thin": (V1, dV1, d2V1), "Thick": (V2, dV2, d2V2)}.items():
    sfi = make_SFI(V, dV, d2V, alpha=2)
    Δφ = sfi.phi_metaMin - sfi.phi_absMin
    delta_phi0        = 0.0                   # exactly at the true minimum → dV=0, stays put
    delta_phi_cutoff  = 1e-3 * abs(Δφ)        # strictly positive target offset
    rmin = 1e-3 * sfi.rscale

    try:
        _ = sfi.initialConditions(delta_phi0, rmin, delta_phi_cutoff)
        print(f" {name}: Unexpected success; was expecting an IntegrationError.")
    except IntegrationError as e:
        print(f" {name}: Caught expected IntegrationError:\n   {e}\n")

print("---------- END OF TESTS: Lot SF-3 ----------")
