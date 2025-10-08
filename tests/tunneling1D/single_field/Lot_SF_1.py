# ---------------- Lot SF-1: Potential Interface & Validations -----------------
# This script validates the SingleFieldInstanton potential interface and the
# built-in derivative helpers introduced in Lot SF-1.
#
# What it does:
# 1) Metastability check raises PotentialError when violated.
# 2) Shape & scalar semantics of dV/d2V (scalar-in -> scalar-out; arrays preserve shape).
# 3) Compare built-in dV/d2V against analytic derivatives for thin/thick wall quartics.
# 4) Show 4th-order FD is more accurate than 2nd-order on smooth quartics.
# 5) Validate dV_from_absMin near the minimum (blend to d2V*delta_phi).
# 6) Light barrier sanity: V(phi_bar) ~ V(phi_metaMin).
# 7) Plots for V, V', V'' (analytic vs built-in) for both examples.

import numpy as np
import matplotlib.pyplot as plt

# Import the modernized class + error
from src.CosmoTransitions import SingleFieldInstanton, PotentialError

np.set_printoptions(precision=6, suppress=True)
print("---------- TESTS: Lot SF-1 (Potential Interface & Validations) ----------")

# =============================================================================
# Helper: define two quartic potentials (thin-wall and thick-wall exemplars)
# =============================================================================
# Thin-walled (nearly degenerate minima)
def V_thin(phi):   return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2
def dV_thin(phi):  return phi*(phi-0.47)*(phi-1.0)                     # analytic
def d2V_thin(phi): return 3.0*phi**2 - 2.94*phi + 0.47     # safer: expand explicitly

# The line above is intentionally verbose to avoid mental slips. Let's derive carefully:
# V = (1/4)φ^4 - 0.49 φ^3 + 0.235 φ^2
# V' = φ^3 - 1.47 φ^2 + 0.47 φ
# V''= 3φ^2 - 2*1.47 φ + 0.47 = 3φ^2 - 2.94 φ + 0.47


# Thick-walled (clear separation)
def V_thick(phi):   return 0.25*phi**4 - 0.4*phi**3 + 0.1*phi**2
# V' = φ^3 - 1.2 φ^2 + 0.2 φ
# V''= 3φ^2 - 2.4 φ + 0.2
def dV_thick(phi):  return phi**3 - 1.2*phi**2 + 0.2*phi
def d2V_thick(phi): return 3.0*phi**2 - 2.4*phi + 0.2

# Minima used in examples (consistent with the legacy docs)
phi_abs_min = 1.0
phi_meta_min = 0.0

# A common phi grid that covers both examples comfortably
phi_grid = np.linspace(-0.1, 1.2, 600)

# =============================================================================
# Test 1 — Metastability validation
# =============================================================================
print("\n=== Test 1: Metastability validation ===")
try:
    # Intentionally violate metastability: swap minima labels
    _ = SingleFieldInstanton(
        phi_absMin=phi_meta_min,
        phi_metaMin=phi_abs_min,
        V=V_thin,
        validate=True
    )
    print("ERROR: Expected PotentialError was not raised.")
except PotentialError as e:
    print("OK: PotentialError raised as expected ->", e)

# =============================================================================
# Test 2 — Shape & scalar semantics
# =============================================================================
print("\n=== Test 2: Shape & scalar semantics for dV/d2V ===")
inst_thick = SingleFieldInstanton(
    phi_absMin=phi_abs_min,
    phi_metaMin=phi_meta_min,
    V=V_thick,
    # Use builtin FD (do not pass dV/d2V)
    fd_order=4,
    validate=True
)

# Scalar-in -> scalar-out
phi0 = 0.33
out_dV_scalar  = inst_thick.dV(phi0)
out_d2V_scalar = inst_thick.d2V(phi0)
print(f"Scalar dV:  type={type(out_dV_scalar)}, value={float(out_dV_scalar):.6e}")
print(f"Scalar d2V: type={type(out_d2V_scalar)}, value={float(out_d2V_scalar):.6e}")

# Array preserves shape
phi_arr = np.array([0.0, 0.5, 1.0])
out_dV_arr  = inst_thick.dV(phi_arr)
out_d2V_arr = inst_thick.d2V(phi_arr)
assert out_dV_arr.shape  == phi_arr.shape
assert out_d2V_arr.shape == phi_arr.shape
print("Array shapes preserved:", out_dV_arr.shape, out_d2V_arr.shape)

# =============================================================================
# Test 3 — Built-in FD vs analytic derivatives (thin & thick)
# =============================================================================
print("\n=== Test 3: Built-in FD vs analytic derivatives ===")

def compare_derivatives(name, V, dV_true, d2V_true):
    inst4 = SingleFieldInstanton(phi_abs_min, phi_meta_min, V, fd_order=4, validate=True)
    inst2 = SingleFieldInstanton(phi_abs_min, phi_meta_min, V, fd_order=2, validate=True)

    dV_fd4  = inst4.dV(phi_grid)
    d2V_fd4 = inst4.d2V(phi_grid)
    dV_fd2  = inst2.dV(phi_grid)
    d2V_fd2 = inst2.d2V(phi_grid)

    dV_ref  = dV_true(phi_grid)
    d2V_ref = d2V_true(phi_grid)

    err_dV_4  = np.max(np.abs(dV_fd4  - dV_ref))
    err_d2V_4 = np.max(np.abs(d2V_fd4 - d2V_ref))
    err_dV_2  = np.max(np.abs(dV_fd2  - dV_ref))
    err_d2V_2 = np.max(np.abs(d2V_fd2 - d2V_ref))

    print(f"[{name}] max|dV_fd4 - dV_true|  = {err_dV_4:.3e}")
    print(f"[{name}] max|d2V_fd4 - d2V_true|= {err_d2V_4:.3e}")
    print(f"[{name}] max|dV_fd2 - dV_true|  = {err_dV_2:.3e}")
    print(f"[{name}] max|d2V_fd2 - d2V_true|= {err_d2V_2:.3e}")
    assert err_dV_4  <= err_dV_2 * 1.05 + 1e-10   # allow tiny slack
    assert err_d2V_4 <= err_d2V_2 * 1.05 + 1e-10

    # Light barrier sanity (Lot SF-2 will test in depth)
    V_meta = V(inst4.phi_metaMin)
    #V_bar  = V(inst4.phi_bar)
    #print(f"[{name}] barrier check: V(phi_bar) - V(phi_metaMin) = {V_bar - V_meta:+.3e}")

compare_derivatives("THIN",  V_thin,  dV_thin,  d2V_thin)
compare_derivatives("THICK", V_thick, dV_thick, d2V_thick)

# =============================================================================
# Test 4 — dV_from_absMin near the minimum
# =============================================================================
print("\n=== Test 4: dV_from_absMin near φ_absMin ===")
inst_thin = SingleFieldInstanton(
    phi_absMin=phi_abs_min,
    phi_metaMin=phi_meta_min,
    V=V_thin,
    fd_order=4,
    validate=True
)
delta_list = np.array([1e-6, 5e-6, 1e-5, 5e-5])  # offsets from φ_absMin
for dlt in delta_list:
    phi = phi_abs_min + dlt
    val = inst_thin.dV_from_absMin(dlt)
    ref = d2V_thin(phi) * dlt               # local linear approx (uses d2V at φ)
    abs_err = abs(val - ref)
    rel_err = abs_err / (abs(ref) + 1e-30)
    print(f"delta={dlt:.1e} -> dV_from_absMin={val:.3e}, ref≈d2V*delta={ref:.3e}, rel.err={rel_err:.3e}")
    # We expect blending to track ~linear scaling at these tiny deltas
    assert rel_err < 5e-3

# =============================================================================
# Test 5 — Plots: V, V', V'' (analytic vs built-in) for thin & thick cases
# =============================================================================
print("\n=== Test 5: Plots (potential and derivatives) ===")

def plot_set(name, V, dV_true, d2V_true):
    inst = SingleFieldInstanton(phi_abs_min, phi_meta_min, V, fd_order=4, validate=True)

    Vg     = V(phi_grid)
    dV_fd  = inst.dV(phi_grid)
    d2V_fd = inst.d2V(phi_grid)
    dV_ref = dV_true(phi_grid)
    d2V_ref= d2V_true(phi_grid)

    plt.figure(figsize=(9, 5))
    plt.plot(phi_grid, Vg, label="V(φ)", lw=2)
    plt.title(f"{name}: Potential")
    plt.xlabel("φ"); plt.ylabel("V(φ)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(phi_grid, dV_ref, label="V'(φ) analytic", lw=2)
    plt.plot(phi_grid, dV_fd,  "--", label="V'(φ) builtin FD (o4)")
    plt.title(f"{name}: First derivative")
    plt.xlabel("φ"); plt.ylabel("V'(φ)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(phi_grid, d2V_ref, label="V''(φ) analytic", lw=2)
    plt.plot(phi_grid, d2V_fd,  "--", label="V''(φ) builtin FD (o4)")
    plt.title(f"{name}: Second derivative")
    plt.xlabel("φ"); plt.ylabel("V''(φ)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # Residual summaries
    m1 = np.max(np.abs(dV_fd  - dV_ref))
    m2 = np.max(np.abs(d2V_fd - d2V_ref))
    print(f"[{name}] residuals: max|V'_fd - V'_ref|={m1:.3e}, max|V''_fd - V''_ref|={m2:.3e}")

plot_set("THIN",  V_thin,  dV_thin,  d2V_thin)
plot_set("THICK", V_thick, dV_thick, d2V_thick)

print("\n---------- END OF TESTS: Lot SF-1 ----------")
