# tests/finiteT/test_dispatch_wrappers.py

import numpy as np
import matplotlib.pyplot as plt

from src.CosmoTransitions import Jb, Jf  # dispatchers


np.set_printoptions(precision=6, suppress=True)
print("---------- TESTS: Dispatcher (Jb, Jf) ----------")

# -----------------------------
# Helper: convert spline θ-derivs to x-derivs
# θ = x^2; d/dx = (dθ/dx) d/dθ = 2x d/dθ
# d²/dx² = 2 d/dθ + 4x² d²/dθ²
# -----------------------------
def spline_dJdx(x, species="b"):
    x = np.asarray(x, dtype=float)
    theta = x**2
    if species == "b":
        dJ_dtheta = Jb(theta, approx="spline", deriv=1)
    else:
        dJ_dtheta = Jf(theta, approx="spline", deriv=1)
    return 2.0 * x * dJ_dtheta

def spline_d2Jdx2(x, species="b"):
    x = np.asarray(x, dtype=float)
    theta = x**2
    if species == "b":
        J1 = Jb(theta, approx="spline", deriv=1)   # dJ/dθ
        J2 = Jb(theta, approx="spline", deriv=2)   # d²J/dθ²
    else:
        J1 = Jf(theta, approx="spline", deriv=1)
        J2 = Jf(theta, approx="spline", deriv=2)
    return 2.0 * J1 + 4.0 * x**2 * J2


# =============================================================================
# Test A — Exact J_b and J_f on [0, 10]
# =============================================================================
print("\n=== Test A: exact J_b, J_f on [0,10] ===")
x = np.linspace(0.0, 10.0, 120)
Jb_ex = Jb(x, approx="exact")
Jf_ex = Jf(x, approx="exact").real  # exact fermion preserves legacy complex dtype

plt.figure(figsize=(9,4.6))
plt.plot(x, Jb_ex, label=r"$J_b$ (exact)", lw=2)
plt.plot(x, Jf_ex, label=r"$J_f$ (exact)", lw=2)
plt.title(r"Exact $J_b(x)$ and $J_f(x)$")
plt.xlabel("x"); plt.ylabel("J(x)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print(f"At x=0: J_b={Jb_ex[0]:.12e}, J_f={Jf_ex[0]:.12e} (should match -π^4/45 and -7π^4/360).")


# =============================================================================
# Test B — Spline J_b and J_f on [0, 10] (θ = x^2)
# =============================================================================
print("\n=== Test B: spline J_b, J_f on [0,10] (θ = x^2) ===")
theta = x**2
Jb_spl = Jb(theta, approx="spline", deriv=0)
Jf_spl = Jf(theta, approx="spline", deriv=0)

plt.figure(figsize=(9,4.6))
plt.plot(x, Jb_spl, "--", label=r"$J_b$ (spline, θ=x^2)")
plt.plot(x, Jf_spl, "--", label=r"$J_f$ (spline, θ=x^2)")
plt.title(r"Spline $J_b$ and $J_f$ (input is $\theta=x^2$)")
plt.xlabel("x"); plt.ylabel("J(x)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# quick sanity vs exact at same x-grid
rel = lambda num, den: np.abs(num - den) / np.maximum(np.abs(den), 1e-12)
rb = np.nanmax(rel(Jb_spl, Jb_ex))
rf = np.nanmax(rel(Jf_spl, Jf_ex))
print(f"Max relative diff spline vs exact on [0,10]:  J_b: {rb:.3e},  J_f: {rf:.3e}")


# =============================================================================
# Test C — Exact first derivatives dJ/dx on [0, 10]
# =============================================================================
print("\n=== Test C: exact derivatives dJ/dx on [0,10] ===")
dJb_ex = Jb(x, approx="exact", deriv=1)
dJf_ex = Jf(x, approx="exact", deriv=1)

plt.figure(figsize=(9,4.6))
plt.plot(x, dJb_ex, label=r"$ dJ_b/ dx$ (exact)", lw=2)
plt.plot(x, dJf_ex, label=r"$ dJ_f/ dx$ (exact)", lw=2)
plt.title(r"Exact first derivatives")
plt.xlabel("x"); plt.ylabel(r"$ dJ/ dx$")
plt.grid(True, alpha=0.3); plt.legend();plt.tight_layout(); plt.show()

print(f"dJ_b/dx at x=0: {dJb_ex[0]:.3e} (expected 0);  dJ_f/dx at x=0: {dJf_ex[0]:.3e} (expected 0)")


# =============================================================================
# Test D — Spline first derivative mapped to dJ/dx (chain rule)
# =============================================================================
print("\n=== Test D: spline first derivative (chain rule to dJ/dx) on [0,10] ===")
dJb_spl_dx = spline_dJdx(x, species="b")
dJf_spl_dx = spline_dJdx(x, species="f")

plt.figure(figsize=(9,4.6))
plt.plot(x, dJb_spl_dx, "--", label=r"$dJ_b/ dx$ (spline→x)")
plt.plot(x, dJf_spl_dx, "--", label=r"$ dJ_f/dx$ (spline→x)")
plt.title(r"Spline first derivatives mapped to $x$ via chain rule")
plt.xlabel("x"); plt.ylabel(r"$ dJ/dx$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# compare to exact derivatives
rb_dx = np.nanmax(rel(dJb_spl_dx, dJb_ex))
rf_dx = np.nanmax(rel(dJf_spl_dx, dJf_ex))
print(f"Max relative diff (spline→x) vs exact dJ/dx:  J_b: {rb_dx:.3e},  J_f: {rf_dx:.3e}")


# =============================================================================
# Test E — Spline second derivative mapped to d²J/dx² (chain rule)
# =============================================================================
print("\n=== Test E: spline second derivative d²J/dx² on [0,10] ===")
d2Jb_spl = spline_d2Jdx2(x, species="b")
d2Jf_spl = spline_d2Jdx2(x, species="f")

plt.figure(figsize=(9,4.6))
plt.plot(x, d2Jb_spl, "--", label=r"$ d^2J_b/dx^2$ (spline→x)")
plt.plot(x, d2Jf_spl, "--", label=r"$ d^2J_f/dx^2$ (spline→x)")
plt.title(r"Spline second derivatives mapped to $x$")
plt.xlabel("x"); plt.ylabel(r"$d^2J/dx^2$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# =============================================================================
# Test F — LOW-x series vs exact with relative error
#          J_b on [0,7], J_f on [0,3.7]
# =============================================================================
print("\n=== Test F: low-x series vs exact (with relative error) ===")

# Boson low range
xb = np.linspace(0.0, 7.0, 160)
Jb_low = Jb(xb, approx="low", n=20)
Jb_ex_low = Jb(xb, approx="exact")

plt.figure(figsize=(9,4.6))
plt.plot(xb, Jb_ex_low, label=r"$J_b$ (exact)", lw=2)
plt.plot(xb, Jb_low, "--", label=r"$J_b$ (low-x n=20)")
plt.title(r"Boson: low-x series vs exact")
plt.xlabel("x"); plt.ylabel(r"$J_b$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

rel_b = rel(Jb_low, Jb_ex_low)
print(f"Boson low-x: max rel err = {np.nanmax(rel_b):.3e}")
plt.figure(figsize=(8.6,3.8))
plt.semilogy(xb, np.maximum(rel_b, 1e-18))
plt.title(r"Boson: relative error (low-x vs exact)")
plt.xlabel("x"); plt.ylabel("relative error")
plt.grid(True, which="both", ls=":"); plt.tight_layout(); plt.show()

# Fermion low range
xf = np.linspace(0.0, 3.7, 130)
Jf_low = Jf(xf, approx="low", n=20)
Jf_ex_low = Jf(xf, approx="exact").real

plt.figure(figsize=(9,4.6))
plt.plot(xf, Jf_ex_low, label=r"$J_f$ (exact)", lw=2)
plt.plot(xf, Jf_low, "--", label=r"$J_f$ (low-x n=20)")
plt.title(r"Fermion: low-x series vs exact")
plt.xlabel("x"); plt.ylabel(r"$J_f$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

rel_f = rel(Jf_low, Jf_ex_low)
print(f"Fermion low-x: max rel err = {np.nanmax(rel_f):.3e}")
plt.figure(figsize=(8.6,3.8))
plt.semilogy(xf, np.maximum(rel_f, 1e-18))
plt.title(r"Fermion: relative error (low-x vs exact)")
plt.xlabel("x"); plt.ylabel("relative error")
plt.grid(True, which="both", ls=":"); plt.tight_layout(); plt.show()


# =============================================================================
# Test G — HIGH-x series vs exact on [1, 10] with relative error
# =============================================================================
print("\n=== Test G: high-x series vs exact (x in [1,10]) ===")
xh = np.linspace(1.0, 10.0, 120)
Jb_hi = Jb(xh, approx="high", n=8)
Jf_hi = Jf(xh, approx="high", n=8)
Jb_ex_hi = Jb(xh, approx="exact")
Jf_ex_hi = Jf(xh, approx="exact").real

plt.figure(figsize=(9,4.6))
plt.semilogy(xh, -Jb_ex_hi,  label=r"$-J_b$ (exact)", lw=2)
plt.semilogy(xh, -Jb_hi,  "--", label=r"$-J_b$ (high-x n=8)")
plt.semilogy(xh, -Jf_ex_hi,  label=r"$-J_f$ (exact)", lw=2)
plt.semilogy(xh, -Jf_hi,  "--", label=r"$-J_f$ (high-x n=8)")
plt.title("High-x comparison (semilog)")
plt.xlabel("x"); plt.ylabel("magnitude")
plt.grid(True, which="both", ls=":"); plt.legend(); plt.tight_layout(); plt.show()

rel_bh = rel(Jb_hi, Jb_ex_hi)
rel_fh = rel(Jf_hi, Jf_ex_hi)
print(f"High-x: max rel err  J_b: {np.nanmax(rel_bh):.3e},  J_f: {np.nanmax(rel_fh):.3e}")

plt.figure(figsize=(8.8,4.0))
plt.plot(xh, rel_bh, label=r"boson rel. err")
plt.plot(xh, rel_fh, label=r"fermion rel. err")
plt.title("High-x relative error (n=8)")
plt.xlabel("x"); plt.ylabel("relative error")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("\n---------- END OF TESTS: Dispatcher (Jb, Jf) ----------")
