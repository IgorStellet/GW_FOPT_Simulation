# ---------------- Spline Thermal Integrals (J_b, J_f) — Tests ----------------
# What this file checks:
# 1) Spline vs exact for x in [0, 10] (J_b and J_f). Collocation points as dots,
#    spline dashed, exact solid.
# 2) Derivatives: compare dJ/dtheta (spline) vs exact via chain rule from dJ/dx
#    (only theta > 0). Same plotting style, with dots at collocation nodes.
# 3) Compatibility with helper_functions.Nbspl: rebuild BSpline evaluation from
#    (t, c, k) and compare (two plots: J_b, J_f).
# 4) Negative-theta behavior: compare spline vs exact for theta < 0 (both J_b, J_f).
#
# Notes:
# - J*_spline expects theta = (m/T)^2 as input; exact routines J*_exact expect x.
# - Jf_exact returns complex (legacy); we compare its real part.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import public APIs (and a few internals for test convenience) ---
from src.CosmoTransitions import (
    Jb_exact, Jf_exact, Jb_exact2, Jf_exact2,
    dJb_exact, dJf_exact,
    Jb_spline, Jf_spline,
)

# Internals: to fetch BSpline (t, c, k) and dataset thetas (if needed)
from src.CosmoTransitions import (
    _ensure_Jf_spline, _ensure_Jb_spline,
    _build_Jf_dataset, _build_Jb_dataset,
    _JF_CACHE_FILE, _JB_CACHE_FILE
)


from src.CosmoTransitions import Nbspl


np.set_printoptions(precision=6, suppress=True)
print("---------- TESTS: Spline Thermal Integrals (J_b, J_f) ----------")


# -------------------------------------------------------------------
# Utilities: collocation nodes and BSpline-reconstruction via Nbspl
# -------------------------------------------------------------------

def _get_spline_and_thetas(which: str):
    """
    Return (spl, theta_nodes) for which in {'f','b'}.
    spl is a scipy.interpolate.BSpline.
    theta_nodes are the dataset points used to fit (for plotting dots).
    """
    # With internals available:
    if which == 'f':
        spl = _ensure_Jf_spline()
        # Try cache first to get the original theta nodes
        theta = None
        if Path(_JF_CACHE_FILE).exists():
            try:
                data = np.load(_JF_CACHE_FILE, allow_pickle=False)
                theta = data.get("theta", None)
            except Exception:
                theta = None
        if theta is None:
            theta, _ = _build_Jf_dataset()
        return spl, theta
    else:
        spl = _ensure_Jb_spline()
        theta = None
        if Path(_JB_CACHE_FILE).exists():
            try:
                data = np.load(_JB_CACHE_FILE, allow_pickle=False)
                theta = data.get("theta", None)
            except Exception:
                theta = None
        if theta is None:
            theta, _ = _build_Jb_dataset()
        return spl, theta


def _reconstruct_with_Nbspl(spl, theta_eval: np.ndarray) -> np.ndarray:
    """Rebuild BSpline evaluation via helper_functions.Nbspl using s(x) = sum c_i N_{i,k}(x)."""
    t, c, k = spl.t, spl.c, spl.k
    # Nbspl returns basis matrix shape (n_eval, m-k-1). c has shape (m-k-1,)
    N = Nbspl(t, theta_eval, k=k)
    return N @ c


# =============================================================================
# Test 1 — Spline vs Exact for x in [0,10] (J_b and J_f). Collocation dots.
# =============================================================================
print("\n=== Test 1: Spline vs Exact for x ∈ [0, 10] ===")

# Build grids
x = np.linspace(0.0, 10.0, 200)
theta = x**2

# Ensure splines exist & fetch collocation nodes (for dots)
spl_f, theta_nodes_f = _get_spline_and_thetas('f')
spl_b, theta_nodes_b = _get_spline_and_thetas('b')

# Only show positive-theta nodes on the x-plot
nodes_x_f = np.sqrt(np.clip(theta_nodes_f, a_min=0.0, a_max=None))
nodes_x_b = np.sqrt(np.clip(theta_nodes_b, a_min=0.0, a_max=None))

# limits of axis-x
x_limit = x.max()  # = 10.0
sel_b = (nodes_x_b > 0) & (nodes_x_b <= x_limit)
sel_f = (nodes_x_f > 0) & (nodes_x_f <= x_limit)

# Evaluate
Jb_ex = Jb_exact(x)
Jf_ex = Jf_exact(x).real  # legacy complex -> real
Jb_sp = Jb_spline(theta)
Jf_sp = Jf_spline(theta)

# Collocation values (at nodes >=0)
Jb_nodes = Jb_spline(nodes_x_b**2)
Jf_nodes = Jf_spline(nodes_x_f**2)

# Errors
err_b = np.max(np.abs(Jb_sp - Jb_ex))
err_f = np.max(np.abs(Jf_sp - Jf_ex))
print(f"Max |J_b (spline) - J_b (exact)| over x∈[0,10]: {err_b:.3e}")
print(f"Max |J_f (spline) - J_f (exact)| over x∈[0,10]: {err_f:.3e}")

# Plots
plt.figure(figsize=(8.8,4.6))
plt.plot(x, Jb_ex, label=r"exact $J_b(x)$", lw=2)
plt.plot(x, Jb_sp, "--", label=r"spline $J_b(\theta=x^2)$")
# dots at nodes (positive theta only)
plt.plot(nodes_x_b[sel_b], Jb_nodes[sel_b], "o", ms=3, label="collocation nodes")
plt.title("Boson: exact vs spline (with nodes)")
plt.xlabel("x")
plt.ylabel(r"$J_b$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8.8,4.6))
plt.plot(x, Jf_ex, label=r"exact $J_f(x)$", lw=2)
plt.plot(x, Jf_sp, "--", label=r"spline $J_f(\theta=x^2)$")
sel = nodes_x_f > 0
plt.plot(nodes_x_f[sel_f], Jf_nodes[sel_f], "o", ms=3, label="collocation nodes")
plt.title("Fermion: exact vs spline (with nodes)")
plt.xlabel("x")
plt.ylabel(r"$J_f$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# Test 2 — Derivatives: dJ/dtheta (spline) vs exact via chain rule
# =============================================================================
print("\n=== Test 2: Derivatives — spline dJ/dtheta vs exact (chain rule) ===")
# Avoid theta=0 (singularity in 1/(2x))
x_d = np.linspace(1e-3, 10.0, 180)
theta_d = x_d**2

# Spline derivative wrt theta
dJb_dth_sp = Jb_spline(theta_d, n=1)
dJf_dth_sp = Jf_spline(theta_d, n=1)

# Exact dJ/dtheta via chain rule from dJ/dx
dJb_dx = dJb_exact(x_d)
dJf_dx = dJf_exact(x_d)
dth_dx = 2.0 * x_d  # theta = x^2
dJb_dth_ex = dJb_dx / dth_dx
dJf_dth_ex = dJf_dx / dth_dx

# Errors
err_b_d = np.max(np.abs(dJb_dth_sp - dJb_dth_ex))
err_f_d = np.max(np.abs(dJf_dth_sp - dJf_dth_ex))
print(f"Max |dJ_b/dθ (spline) - dJ_b/dθ (exact)| over x∈(1e-3,10]: {err_b_d:.3e}")
print(f"Max |dJ_f/dθ (spline) - dJ_f/dθ (exact)| over x∈(1e-3,10]: {err_f_d:.3e}")

# Collocation nodes for derivative: evaluate derivative-spline at node thetas (>=0)
theta_limit = theta_d.max()  # = (10.0)**2
nodes_th_b_pos = theta_nodes_b[(theta_nodes_b >= 0) & (theta_nodes_b <= theta_limit)]
nodes_th_f_pos = theta_nodes_f[(theta_nodes_f >= 0) & (theta_nodes_f <= theta_limit)]
dJb_nodes = Jb_spline(nodes_th_b_pos, n=1)
dJf_nodes = Jf_spline(nodes_th_f_pos, n=1)

plt.figure(figsize=(9.0,4.6))
plt.plot(x_d, dJb_dth_ex, label=r"exact $dJ_b/d\theta$", lw=2)
plt.plot(x_d, dJb_dth_sp, "--", label=r"spline $dJ_b/d\theta$")
plt.plot(np.sqrt(nodes_th_b_pos), dJb_nodes, "o", ms=3, label="nodes (θ≥0)")
plt.title("Boson: derivative wrt θ")
plt.xlabel("x")
plt.ylabel(r"$\partial J_b/\partial \theta$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9.0,4.6))
plt.plot(x_d, dJf_dth_ex, label=r"exact $dJ_f/d\theta$", lw=2)
plt.plot(x_d, dJf_dth_sp, "--", label=r"spline $dJ_f/d\theta$")
plt.plot(np.sqrt(nodes_th_f_pos), dJf_nodes, "o", ms=3, label="nodes (θ≥0)")
plt.title("Fermion: derivative wrt θ")
plt.xlabel("x")
plt.ylabel(r"$\partial J_f/\partial \theta$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# Test 3 — Compatibility with helper_functions.Nbspl (basis reconstruction)
# =============================================================================
print("\n=== Test 3: Compatibility with helper_functions.Nbspl ===")
# Evaluate on a moderate θ-grid inside the domain
theta_c = np.linspace(0.0, 100.0, 200)  # safe inside, avoids clamp/zero edges

# Ensure splines
spl_f, _ = _get_spline_and_thetas('f')
spl_b, _ = _get_spline_and_thetas('b')

# Our spline evaluation
Jf_sp_c = Jf_spline(theta_c)
Jb_sp_c = Jb_spline(theta_c)

# Rebuild using Nbspl(t, θ, k) @ c
Jf_rebuilt = _reconstruct_with_Nbspl(spl_f, theta_c)
Jb_rebuilt = _reconstruct_with_Nbspl(spl_b, theta_c)

# Errors
err_f_nb = np.max(np.abs(Jf_sp_c - Jf_rebuilt))
err_b_nb = np.max(np.abs(Jb_sp_c - Jb_rebuilt))
print(f"Max |J_f (spline) - J_f (Nbspl rebuild)| on θ∈[0,100]: {err_f_nb:.3e}")
print(f"Max |J_b (spline) - J_b (Nbspl rebuild)| on θ∈[0,100]: {err_b_nb:.3e}")

# Plots
x_c = np.sqrt(theta_c)
plt.figure(figsize=(8.6,4.6))
plt.plot(x_c, Jb_sp_c, lw=2, label="J_b: spline")
plt.plot(x_c, Jb_rebuilt, "--", label="J_b: Nbspl rebuild")
plt.title("Boson: BSpline vs Nbspl reconstruction")
plt.xlabel("x"); plt.ylabel(r"$J_b$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8.6,4.6))
plt.plot(x_c, Jf_sp_c, lw=2, label="J_f: spline")
plt.plot(x_c, Jf_rebuilt, "--", label="J_f: Nbspl rebuild")
plt.title("Fermion: BSpline vs Nbspl reconstruction")
plt.xlabel("x"); plt.ylabel(r"$J_f$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# =============================================================================
# Test 4 — Negative theta: spline vs exact (J_b and J_f)
# =============================================================================
print("\n=== Test 4: Negative-theta behavior (θ < 0) ===")
# Pick a negative-theta window safely inside domain
theta_neg = np.linspace(-5.0, -0.1, 120)

Jb_ex_neg = Jb_exact2(theta_neg)
Jf_ex_neg = Jf_exact2(theta_neg)
Jb_sp_neg = Jb_spline(theta_neg)
Jf_sp_neg = Jf_spline(theta_neg)

err_b_neg = np.max(np.abs(Jb_sp_neg - Jb_ex_neg))
err_f_neg = np.max(np.abs(Jf_sp_neg - Jf_ex_neg))
print(f"Max |J_b (spline) - J_b (exact)| on θ∈[-5,-0.1]: {err_b_neg:.3e}")
print(f"Max |J_f (spline) - J_f (exact)| on θ∈[-5,-0.1]: {err_f_neg:.3e}")

plt.figure(figsize=(8.6,4.6))
plt.plot(theta_neg, Jb_ex_neg, lw=2, label="exact J_b(θ)")
plt.plot(theta_neg, Jb_sp_neg, "--", label="spline J_b(θ)")
plt.title("Boson: negative θ")
plt.xlabel(r"$\theta$"); plt.ylabel(r"$J_b(\theta)$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8.6,4.6))
plt.plot(theta_neg, Jf_ex_neg, lw=2, label="exact J_f(θ)")
plt.plot(theta_neg, Jf_sp_neg, "--", label="spline J_f(θ)")
plt.title("Fermion: negative θ")
plt.xlabel(r"$\theta$"); plt.ylabel(r"$J_f(\theta)$")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

print("\n---------- END OF TESTS: Spline Thermal Integrals ----------")
