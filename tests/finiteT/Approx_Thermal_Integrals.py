# ------------------ Tests: Approx Thermal Integrals -------------------
# Block Approx thermal integrals: Low-x (high-T) and High-x (low-T) approximations
#
# What this file does:
# 1) LOW x window (x in [0, 1.5]): compare J_b/J_f exact vs Jb_low/Jf_low.
#    - Plots exact (solid) vs low-x approximation (dashed) for each species.
#    - Prints max absolute and relative errors; highlights the regime where
#      the low-x series is typically reliable (x ≲ 0.5–0.8). Diverges at x>=3 for J_f and at x>= 5.5 for J_b
# 2) HIGH x window (x in [2, 10]): compare J_b/J_f exact vs Jb_high/Jf_high.
#    - Shows convergence with number of exponential terms n = 4, 8, 12.
#    - Plots |J| in semilog to emphasize exponential tails.
#    - Plots relative error vs x for each n.
#
# Notes:
# - All approximations here are for REAL x (physical x = m/T).
# - You can tweak the grids/ranges as needed for tighter/looser comparisons.
# ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from src.CosmoTransitions import (
    Jb_exact, Jf_exact,
    Jb_low,   Jf_low,
    Jb_high,  Jf_high,
)

np.set_printoptions(precision=6, suppress=True)

print("---------- TESTS: Approx Thermal Integrals (Low-x & High-x) ----------")

# Small utility to build robust relative errors
def rel_err(approx, exact, eps=1e-14):
    denom = np.maximum(np.abs(exact), eps)
    return np.abs(approx - exact) / denom


# =============================================================================
# Test 1 — LOW-x regime: exact vs low-x expansions
# =============================================================================
print("\n=== Test 1: LOW-x (high-T) comparison: exact vs low-x series ===")

# Go beyond the ideal small-x window intentionally to see where it degrades
x_low = np.linspace(0.0, 1.5, 200)

Jb_ex_lo = Jb_exact(x_low)
Jf_ex_lo = Jf_exact(x_low).real

# Use a moderate tail length; increase/decrease as desired
n_tail = 20
Jb_lo = Jb_low(x_low, n=n_tail)
Jf_lo = Jf_low(x_low, n=n_tail)

# Error stats
eb_abs = np.max(np.abs(Jb_lo - Jb_ex_lo))
ef_abs = np.max(np.abs(Jf_lo - Jf_ex_lo))
eb_rel = np.max(rel_err(Jb_lo, Jb_ex_lo))
ef_rel = np.max(rel_err(Jf_lo, Jf_ex_lo))

print(f"Boson  (low-x)  n={n_tail:2d}: max abs err={eb_abs:.3e}, max rel err={eb_rel:.3e}")
print(f"Fermion(low-x)  n={n_tail:2d}: max abs err={ef_abs:.3e}, max rel err={ef_rel:.3e}")

# Where is rel error below thresholds?
for thr in (1e-3, 1e-4):
    b_ok = x_low[rel_err(Jb_lo, Jb_ex_lo) < thr]
    f_ok = x_low[rel_err(Jf_lo, Jf_ex_lo) < thr]
    b_max = b_ok.max() if b_ok.size else np.nan
    f_max = f_ok.max() if f_ok.size else np.nan
    print(f"  threshold {thr:>.0e}: max x with rel err < thr →  J_b: {b_max:.3f},  J_f: {f_max:.3f}")

# Plots: values
plt.figure(figsize=(8.8, 4.8))
plt.plot(x_low, Jb_ex_lo, label=r"$J_b$ exact", lw=2)
plt.plot(x_low, Jb_lo,  "--", label=rf"$J_b$ low-$x$ (n={n_tail})")
plt.axvspan(0.0, 0.8, color="tab:blue", alpha=0.06, label="typical small-$x$ window")
plt.title("Boson: exact vs low-$x$ expansion")
plt.xlabel("x"); plt.ylabel("J_b(x)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8.8, 4.8))
plt.plot(x_low, Jf_ex_lo, label=r"$J_f$ exact", lw=2)
plt.plot(x_low, Jf_lo,  "--", label=rf"$J_f$ low-$x$ (n={n_tail})")
plt.axvspan(0.0, 0.8, color="tab:orange", alpha=0.06, label="typical small-$x$ window")
plt.title("Fermion: exact vs low-$x$ expansion")
plt.xlabel("x"); plt.ylabel("J_f(x)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# Plots: relative errors
plt.figure(figsize=(8.6, 4.4))
plt.plot(x_low, rel_err(Jb_lo, Jb_ex_lo), label=r"$J_b$ rel. err")
plt.plot(x_low, rel_err(Jf_lo, Jf_ex_lo), label=r"$J_f$ rel. err")
plt.axhline(1e-3, color="k", ls=":", lw=1, alpha=0.6)
plt.axhline(1e-4, color="k", ls=":", lw=1, alpha=0.6)
plt.title("LOW-$x$: relative error vs x")
plt.xlabel("x"); plt.ylabel("relative error")
plt.yscale("log"); plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(); plt.tight_layout(); plt.show()

print("Expectation: The low-x series is excellent near x≈0 and degrades gradually;")
print("             truncation at n≈20 is typically sufficient up to x~0.6–0.8.")


# =============================================================================
# Test 2 — HIGH-x regime: exact vs high-x series (Bessel-K sums)
# =============================================================================
print("\n=== Test 2: HIGH-x (low-T) comparison: exact vs high-x series ===")

x_hi = np.linspace(1.0, 10.0, 140)
Jb_ex_hi = Jb_exact(x_hi)
Jf_ex_hi = Jf_exact(x_hi).real

n_list = [4, 8, 12]  # show convergence with more terms

# Values plot (semilog magnitude)
plt.figure(figsize=(9.2, 5.0))
plt.semilogy(x_hi, -Jb_ex_hi, label=r"$-J_b$ exact", lw=2)
for n in n_list:
    plt.semilogy(x_hi, -Jb_high(x_hi, deriv=0, n=n), "--", label=rf"$-J_b$ high-$x$ (n={n})")
plt.title("Boson: |J_b|, exact vs high-$x$ series (semilog)")
plt.xlabel("x"); plt.ylabel("magnitude")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(9.2, 5.0))
plt.semilogy(x_hi, -Jf_ex_hi, label=r"$-J_f$ exact", lw=2)
for n in n_list:
    plt.semilogy(x_hi, -Jf_high(x_hi, deriv=0, n=n), "--", label=rf"$-J_f$ high-$x$ (n={n})")
plt.title("Fermion: |J_f|, exact vs high-$x$ series (semilog)")
plt.xlabel("x"); plt.ylabel("magnitude")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

# Relative error plots
plt.figure(figsize=(9.0, 4.6))
for n in n_list:
    r = rel_err(Jb_high(x_hi, deriv=0, n=n), Jb_ex_hi)
    plt.plot(x_hi, r, label=rf"n={n}")
plt.title("Boson: HIGH-$x$ relative error vs x")
plt.xlabel("x"); plt.ylabel("relative error")
plt.yscale("log"); plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(9.0, 4.6))
for n in n_list:
    r = rel_err(Jf_high(x_hi, deriv=0, n=n), Jf_ex_hi)
    plt.plot(x_hi, r, label=rf"n={n}")
plt.title("Fermion: HIGH-$x$ relative error vs x")
plt.xlabel("x"); plt.ylabel("relative error")
plt.yscale("log"); plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(); plt.tight_layout(); plt.show()

# Print aggregate stats for high-x region
for species, J_ex, J_hi_fun in (
    ("boson",   Jb_ex_hi, Jb_high),
    ("fermion", Jf_ex_hi, Jf_high),
):
    print(f"\nHigh-x error summary ({species}):")
    for n in n_list:
        approx = J_hi_fun(x_hi, deriv=0, n=n)
        re = rel_err(approx, J_ex)
        print(f"  n={n:2d}: max rel err={np.max(re):.3e}, median rel err={np.median(re):.3e}")

print("\nExpectation: High-x sums converge exponentially fast with n and x;")
print("             even n≈8–12 is typically very accurate for x≳2.")

print("\n---------- END OF TESTS: Approx Thermal Integrals ----------")
