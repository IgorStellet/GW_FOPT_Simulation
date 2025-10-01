#---------------- First round of tests and modifications ------------------------
####################################
# Exact Thermal Integrals (J_b, J_f)
####################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

"""
Progressive tests for the Exact Thermal Integrals (Sub-block A):
Jb_exact, Jf_exact, Jb_exact2 (theta = x^2), Jf_exact2, dJb_exact, dJf_exact.

What this file does:
1) Sanity + small-x physics: J_b(0) = -pi^4/45, J_f(0) = -7*pi^4/360.
2a) Consistency: J(x) vs J(theta=x^2) for x>=0.
2b) Negative-theta check: J(theta<0) vs J(x=i*sqrt(|theta|)) (dashed).
3) Derivatives: sign/shape check and positivity (both should move toward 0 as x grows).
4) Cross-check dJ/dx via finite-difference gradientFunction (order=4).
5a) Global trend: x from 0 to large; show approach to 0 with y=0 reference.
5b) Large-x tail: semilog decay; empirical comparison with x^2 K2(x).
6) (physical demo) Thermal piece V_T ~ T^4/(2π^2) * n * J(x=m/T).

Notes:
- We keep grids modest to avoid heavy quadrature times.
- All plots are didactic; numbers printed give quick validation targets.
"""

# Import the freshly modernized exact functions
from src.CosmoTransitions import (
    Jb_exact, Jf_exact, Jb_exact2, Jf_exact2,
    dJb_exact, dJf_exact,
    gradientFunction )

np.set_printoptions(precision=6, suppress=True)
print("---------- TESTS: Exact Thermal Integrals (J_b, J_f) ----------")

# =============================================================================
# Test 1 — Small-x physics sanity: constants at x=0 and nearby behavior
# =============================================================================
print("\n=== Test 1: Small-x sanity (x → 0) ===")
x0 = 0.0
Jb0_expected = -np.pi**4 / 45.0
Jf0_expected = -7.0 * np.pi**4 / 360.0

Jb0 = float(Jb_exact(x0))
Jf0 = float(Jf_exact(x0).real)  # legacy returns complex for Jf; take real
print(f"J_b(0): num={Jb0:.12e}, expected={Jb0_expected:.12e}, |Δ|={abs(Jb0-Jb0_expected):.2e}")
print(f"J_f(0): num={Jf0:.12e}, expected={Jf0_expected:.12e}, |Δ|={abs(Jf0-Jf0_expected):.2e}")

# A small grid to visualize the approach from x=0 upwards
x_small = np.linspace(0.0, 2.0, 60)
Jb_small = Jb_exact(x_small)
Jf_small = Jf_exact(x_small).real

plt.figure(figsize=(8.5,4.5))
plt.plot(x_small, Jb_small, label=r"$J_b(x)$", lw=2)
plt.plot(x_small, Jf_small, label=r"$J_f(x)$", lw=2)
plt.axhline(Jb0_expected, ls="--", label=r"$J_b(0)=-\pi^4/45$", alpha=0.7)
plt.axhline(Jf0_expected, ls="--", label=r"$J_f(0)=-7\pi^4/360$", alpha=0.7)
plt.title("Small-x behavior")
plt.xlabel("x"); plt.ylabel("J(x)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

print("Expectation: For both bosons and fermions, J(x) starts negative and increases toward 0 as x grows.")

# =============================================================================
# Test 2a — Consistency check: J(x) vs J(theta=x^2) for x>=0
# =============================================================================
print("\n=== Test 2: Consistency J(x) vs J(theta=x^2) (x ≥ 0) ===")
x_pos = np.linspace(0.0, 3.0, 50)
theta = x_pos**2

Jb_by_x  = Jb_exact(x_pos)
Jb_by_th = Jb_exact2(theta)
Jf_by_x  = Jf_exact(x_pos).real
Jf_by_th = Jf_exact2(theta)

err_b = np.max(np.abs(Jb_by_x - Jb_by_th))
err_f = np.max(np.abs(Jf_by_x - Jf_by_th))
print(f"Max |J_b(x) - J_b(theta)| over grid: {err_b:.3e}")
print(f"Max |J_f(x) - J_f(theta)| over grid: {err_f:.3e}")

plt.figure(figsize=(8.0,4.2))
plt.plot(x_pos, Jb_by_x - Jb_by_th, label=r"$J_b(x)-J_b(\theta)$")
plt.plot(x_pos, Jf_by_x - Jf_by_th, label=r"$J_f(x)-J_f(\theta)$")
plt.title("Consistency: direct x vs theta=x^2")
plt.xlabel("x"); plt.ylabel("difference")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("Expectation: differences ~ 0 within quadrature noise.")

# =============================================================================
# Test 2b — θ < 0: compare θ-API against x = i μ (imaginary) trace
# =============================================================================
print("\n=== Test 2b: Negative-θ branch and imaginary-x consistency ===")
mu = np.linspace(0.1, 2.5, 40)            # μ > 0
theta_neg = -(mu**2)                       # θ = -μ^2
x_imag = 1j * mu                           # x = i μ

Jb_theta_neg = Jb_exact2(theta_neg)
Jf_theta_neg = Jf_exact2(theta_neg)
Jb_x_imag = Jb_exact(x_imag).real          # legacy returns real via split
Jf_x_imag = Jf_exact(x_imag).real

err_b_neg = np.max(np.abs(Jb_theta_neg - Jb_x_imag))
err_f_neg = np.max(np.abs(Jf_theta_neg - Jf_x_imag))
print(f"Max |J_b(θ<0) - J_b(x=iμ)| over grid: {err_b_neg:.3e}")
print(f"Max |J_f(θ<0) - J_f(x=iμ)| over grid: {err_f_neg:.3e}")

plt.figure(figsize=(8.6,4.6))
plt.plot(mu, Jb_theta_neg, label=r"$J_b(\theta=-\mu^2)$ (θ-API)", lw=2)
plt.plot(mu, Jb_x_imag,  "--", label=r"$J_b(x=i\mu)$ (x-API, dashed)", lw=2)
plt.plot(mu, Jf_theta_neg, label=r"$J_f(\theta=-\mu^2)$ (θ-API)", lw=2)
plt.plot(mu, Jf_x_imag,  "--", label=r"$J_f(x=i\mu)$ (x-API, dashed)", lw=2)
plt.title(r"Negative-$\theta$ branch vs $x=i\mu$ (consistency)")
plt.xlabel(r"$\mu$"); plt.ylabel(r"$J(\cdot)$")
plt.grid(True, alpha=0.3); plt.legend(ncol=2); plt.tight_layout(); plt.show()

print("Expectation: solid and dashed curves overlap within quadrature accuracy.")

# =============================================================================
# Test 3 — Derivatives: shape and sign (dJ/dx >= 0 for x≥0)
# =============================================================================
print("\n=== Test 3: Derivative sign/shape (dJ/dx ≥ 0 for x ≥ 0) ===")
x_d = np.linspace(0.0, 3.0, 50)
dJb = dJb_exact(x_d)
dJf = dJf_exact(x_d)

print(f"Min dJ_b/dx on grid: {np.min(dJb):.3e}  (expected ≥ 0)")
print(f"Min dJ_f/dx on grid: {np.min(dJf):.3e}  (expected ≥ 0)")

plt.figure(figsize=(8.5,4.5))
plt.plot(x_d, dJb, label=r"$dJ_b/dx$", lw=2)
plt.plot(x_d, dJf, label=r"$dJ_f/dx$", lw=2)
plt.title("Derivatives vs x")
plt.xlabel("x"); plt.ylabel("dJ/dx")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

print("Expectation: derivatives are positive (curves move up toward 0).")

# =============================================================================
# Test 4 — Cross-check dJ/dx with finite-difference gradientFunction (order=4)
# =============================================================================
print("\n=== Test 4: Cross-check dJ/dx using gradientFunction (order=4) ===")

# Wrap the scalar functions as 1D fields for gradientFunction
def fJb(X):
    # X shape (..., 1) -> scalar per point
    x = np.asarray(X, dtype=float)[..., 0]
    return Jb_exact(x)

def fJf(X):
    x = np.asarray(X, dtype=float)[..., 0]
    return Jf_exact(x).real

# Build gradient operators (Ndim=1)
gf_b = gradientFunction(fJb, eps=2e-3, Ndim=1, order=4)
gf_f = gradientFunction(fJf, eps=2e-3, Ndim=1, order=4)

x_chk = np.linspace(0.05, 3.0, 40)[..., None]  # avoid exactly x=0 for FD
dJb_fd = gf_b(x_chk)[..., 0]
dJf_fd = gf_f(x_chk)[..., 0]
dJb_ex = dJb_exact(x_chk[..., 0])
dJf_ex = dJf_exact(x_chk[..., 0])

err_b_abs = np.max(np.abs(dJb_fd - dJb_ex))
err_f_abs = np.max(np.abs(dJf_fd - dJf_ex))
print(f"Max |dJ_b (FD) - dJ_b (exact)|: {err_b_abs:.3e}")
print(f"Max |dJ_f (FD) - dJ_f (exact)|: {err_f_abs:.3e}")

plt.figure(figsize=(9,4.4))
plt.plot(x_chk[..., 0], dJb_ex, label="dJ_b/dx (exact)", lw=2)
plt.plot(x_chk[..., 0], dJb_fd, "--", label="dJ_b/dx (FD, order=4)")
plt.title("Derivative cross-check — boson")
plt.xlabel("x"); plt.ylabel("dJ_b/dx")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(9,4.4))
plt.plot(x_chk[..., 0], dJf_ex, label="dJ_f/dx (exact)", lw=2)
plt.plot(x_chk[..., 0], dJf_fd, "--", label="dJ_f/dx (FD, order=4)")
plt.title("Derivative cross-check — fermion")
plt.xlabel("x"); plt.ylabel("dJ_f/dx")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("Expectation: order-4 finite differences should track the exact derivative closely (small max absolute error).")

# =============================================================================
# Test 5A — Global trend from x=0 to large: approach to 0 (y=0 guide)
# =============================================================================
print("\n=== Test 5A: Global trend (x from 0 to large) with y=0 reference ===")
x_wide = np.linspace(0.0, 10.0, 120)
Jb_wide = Jb_exact(x_wide)
Jf_wide = Jf_exact(x_wide).real

plt.figure(figsize=(8.8,4.8))
plt.plot(x_wide, Jb_wide, label=r"$J_b(x)$", lw=2)
plt.plot(x_wide, Jf_wide, label=r"$J_f(x)$", lw=2)
plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.5, label="expected $\\to 0$ at large $x$")
plt.title("From x=0 to large x — approach to 0")
plt.xlabel("x"); plt.ylabel("J(x)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("Expectation: as x increases (heavier over T), both J_b and J_f approach 0 exponentially.")

# =============================================================================
# Test 5B — Large-x tail: exponential decay; empirical comparison to x^2 K2(x)
# =============================================================================
print("\n=== Test 5B: Large-x tail (semilog plots) ===")
x_big = np.linspace(2.5, 10.0, 40)
Jb_big = Jb_exact(x_big)
Jf_big = Jf_exact(x_big).real

# First Bessel term proxy (empirical, for intuition)
K2 = special.kv(2, x_big)
proxy = - (x_big**2) * K2

# Ratios (avoid division by tiny numbers)
ratio_b = Jb_big / proxy
ratio_f = Jf_big / proxy

print("Tail check (medians):")
print(f" median[ J_b / ( -x^2 K2 ) ] = {np.nanmedian(ratio_b):.3f}")
print(f" median[ J_f / ( -x^2 K2 ) ] = {np.nanmedian(ratio_f):.3f}")

plt.figure(figsize=(8.8,4.6))
plt.semilogy(x_big, -Jb_big, "o-", label=r"$-J_b(x)$")
plt.semilogy(x_big, -Jf_big, "s-", label=r"$-J_f(x)$")
plt.semilogy(x_big, -proxy,  "--", label=r"$x^2 K_2(x)$ (proxy)")
plt.title("Large-x tail (semilog)")
plt.xlabel("x"); plt.ylabel("magnitude")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8.0,4.2))
plt.plot(x_big, ratio_b, "o-", label=r"$J_b / [ -x^2 K_2(x) ]$")
plt.plot(x_big, ratio_f, "s-", label=r"$J_f / [ -x^2 K_2(x) ]$")
plt.axhline(1.0, color="k", alpha=0.3, lw=1)
plt.title("Empirical tail ratio vs. Bessel proxy")
plt.xlabel("x"); plt.ylabel("ratio")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("Expectation: both |J| decay ~exp(-x); the Bessel proxy captures the trend qualitatively.")

# =============================================================================
# Test 6 —  Physical demo: thermal piece V_T using J(x)
# =============================================================================
print("\n=== Test 6: Thermal contribution V_T ∝ J(x) with x = m/T ===")
# For illustration only (not a strict unit test):
T = 1.0  # set T=1 to focus on x=m/T
deg_b, deg_f = 2.0, 4.0  # example degeneracies
x_phys = np.linspace(0.0, 10.0, 160)
Vb = (T**4 / (2*np.pi**2)) * deg_b * Jb_exact(x_phys)
Vf = (T**4 / (2*np.pi**2)) * deg_f * Jf_exact(x_phys).real

plt.figure(figsize=(8.6,4.6))
plt.plot(x_phys, Vb, label=r"boson: $n_b\,J_b(x)$", lw=2)
plt.plot(x_phys, Vf, label=r"fermion: $n_f\,J_f(x)$", lw=2)
plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.5, label="expected $\\to 0$ at large $x$")
plt.title(r"Thermal piece $V_T \propto \sum_i n_i\,J_{\pm}(m_i/T)$ (illustrative)")
plt.xlabel(r"$x=m/T$"); plt.ylabel(r"$V_T$ (arb. units)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

print("Expectation: contributions are largest near x≈0 and vanish exponentially for x≫1.")
print("\n---------- END OF TESTS: Exact Thermal Integrals ----------")
