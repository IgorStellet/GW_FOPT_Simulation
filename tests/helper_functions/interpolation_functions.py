#---------------- Fourth round of tests and modifications ------------------------
# ============================================================
# Interpolation Functions — Tests / Examples (1–7)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from src.CosmoTransitions import makeInterpFuncs, cubicInterpFunction, Nbspl, Nbspld1, Nbspld2

np.set_printoptions(precision=6, suppress=True)

# ------------------------------------------------------------
# Utilities for the B-spline tests
# ------------------------------------------------------------
def open_uniform_knots(a, b, k, nb):
    """
    Build an open-uniform knot vector on [a,b] for degree k and nb basis functions.
    """
    m = nb + k + 1
    # First/last knots repeated k+1 times
    t_start = np.full(k+1, a)
    t_end   = np.full(k+1, b)
    # Interior knots equally spaced
    n_interior = m - 2*(k+1)
    if n_interior > 0:
        interior = np.linspace(a, b, n_interior + 2)[1:-1]
        t = np.concatenate([t_start, interior, t_end])
    else:
        t = np.concatenate([t_start, t_end])
    return t

def greville_abscissae(t, k, *, nudge=True, eps_factor=1e-9):
    """
    Greville abscissae: Xi_i = (t_{i+1} + ... + t_{i+k}) / k  (k>=1).
    If nudge=True, move endpoints slightly inside (a,b) to avoid degeneracy
    with right-closed degree-0 basis when t has repeated end knots.
    """
    t = np.asarray(t, dtype=float)
    nb = len(t) - k - 1
    if k == 0:
        Xi = 0.5*(t[:-1] + t[1:])
    else:
        Xi = np.array([np.mean(t[i+1:i+k+1]) for i in range(nb)], dtype=float)

    if nudge:
        uniq = np.unique(t)
        gaps = np.diff(uniq)
        # smallest positive gap; fallback to span if all equal (degenerate)
        pos = gaps[gaps > 0]
        span = pos.min() if pos.size else max(t[-1] - t[0], 1.0)
        eps = eps_factor * span
        Xi = np.clip(Xi, t[0] + eps, t[-1] - eps)
    return Xi


# ============================================================
# 1) Quintic two-point interpolation: exact endpoint constraints
# ============================================================
print("\n=== Test 1: Quintic interpolation with value/1st/2nd derivatives at both ends ===")
# Define endpoint data on x in [0,1]
y0, dy0, d2y0 = 1.0, -0.5, 0.75
y1, dy1, d2y1 = 2.0,  0.8, -0.25
f, df = makeInterpFuncs(y0, dy0, d2y0, y1, dy1, d2y1)

# Check constraints
print("Endpoint checks (should match exactly within fp precision):")
print(f"  f(0)={f(0): .12f} vs {y0: .12f}")
print(f" df(0)={df(0): .12f} vs {dy0: .12f}")
# finite diff for d2 at 0
h = 1e-6
d2_num_0 = (df(h) - df(0)) / h
print(f"d2f(0)≈{d2_num_0: .12f} vs {d2y0: .12f}")

print(f"  f(1)={f(1): .12f} vs {y1: .12f}")
print(f" df(1)={df(1): .12f} vs {dy1: .12f}")
d2_num_1 = (df(1) - df(1-h)) / h
print(f"d2f(1)≈{d2_num_1: .12f} vs {d2y1: .12f}")

# Plot shape
xs = np.linspace(0, 1, 400)
plt.figure(figsize=(7,4))
plt.plot(xs, f(xs), label="Quintic interpolant")
plt.scatter([0,1], [y0,y1], color='k', zorder=3, label="endpoints")
plt.title("Quintic two-point interpolant (value, slope, curvature matched)")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# ============================================================
# 2) Cubic two-point interpolation (Bezier/Hermite): slope control
# ============================================================
print("\n=== Test 2: Cubic interpolation (Bezier/Hermite) — slope control at ends ===")
y0, y1 = 0.0, 1.0
dy0, dy1 = 2.0, -1.0  # strong slopes at ends
cubic = cubicInterpFunction(y0, dy0, y1, dy1)

# Visualize
ts = np.linspace(0, 1, 400)
ys = cubic(ts)

print("Endpoint checks (value only; slopes are encoded by control points):")
print(f"  y(0)={ys[0]: .12f} vs {y0: .12f}")
print(f"  y(1)={ys[-1]: .12f} vs {y1: .12f}")
print("Expected: curve starts rising fast (positive dy0) and ends with negative slope (dy1).")

plt.figure(figsize=(7,4))
plt.plot(ts, ys, label="Cubic Bezier/Hermite")
plt.scatter([0,1], [y0,y1], color='k', zorder=3, label="endpoints")
# approximate tangent lines near ends for intuition
eps = 0.02
plt.plot([0, eps], [y0, y0 + dy0*eps], 'r--', lw=1, label="start tangent (~)")
plt.plot([1-eps, 1], [y1 - dy1*eps, y1], 'g--', lw=1, label="end tangent (~)")
plt.title("Cubic two-point interpolant (slope control at ends)")
plt.xlabel("t"); plt.ylabel("y(t)"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# ============================================================
# 3) B-spline bases: partition of unity, non-negativity, local support
# ============================================================
print("\n=== Test 3: B-spline bases (partition of unity & local support) ===")
a, b, k, nb = 0.0, 1.0, 3, 8  # cubic, 8 bases
t = open_uniform_knots(a, b, k, nb)
x = np.linspace(a, b, 400)
N = Nbspl(t, x, k=k)  # shape (400, nb)

# Checks:
sumN = np.sum(N, axis=1)
print("Partition of unity: max|sum_i N_i(x)-1| =",
      np.max(np.abs(sumN - 1.0)))
print("Non-negativity: min(N) =", np.min(N), " (should be >= 0)")

# Plot a subset of bases to avoid clutter
plt.figure(figsize=(8,4))
for i in range(nb):
    plt.plot(x, N[:, i], lw=1.5)
plt.title(f"B-spline bases (degree={k}) — open-uniform knots")
plt.xlabel("x"); plt.ylabel("N_i,k(x)")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ============================================================
# 4) First/Second derivative of bases: compare vs finite differences
# ============================================================
print("\n=== Test 4: dN/dx and d²N/dx² vs finite differences (sanity) ===")
N, dN = Nbspld1(t, x, k=k)
N2, dN2, d2N = Nbspld2(t, x, k=k)
assert np.allclose(N, N2)  # same basis from both functions

# Finite differences (central) in interior to avoid boundary effects
h_idx = 1  # step in index
dx = x[1] - x[0]
dN_fd  = (N[3:-1, :] - N[1:-3, :]) / (2*dx)
d2N_fd = (N[3:-1, :] - 2*N[2:-2, :] + N[1:-3, :]) / (dx*dx)
# Compare on matched rows: dN[1:-1] vs dN_fd, d2N[1:-1] vs d2N_fd
err_dN  = np.max(np.abs(dN[2:-2, :]  - dN_fd))
err_d2N = np.max(np.abs(d2N[2:-2, :] - d2N_fd))
print(f"Max |dN - FD|  = {err_dN:.3e}")
print(f"Max |d2N - FD| = {err_d2N:.3e}")
print("Expected: small (method consistent); edges less accurate (central FD not applicable).")

# Plot a couple of bases and their derivatives
idxs = [2, 3, 4]
fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
for i in idxs:
    axes[0].plot(x, N[:, i], label=f"N[{i}]")
    axes[0].plot(x, dN[:, i], '--', label=f"N'[{i}]")
axes[0].set_title("Selected basis & first derivative")
axes[0].set_xlabel("x"); axes[0].grid(True); axes[0].legend(ncol=2, fontsize=8)
for i in idxs:
    axes[1].plot(x, dN[:, i], label=f"N'[{i}]")
    axes[1].plot(x, d2N[:, i], '--', label=f"N''[{i}]")
axes[1].set_title("First vs second derivative")
axes[1].set_xlabel("x"); axes[1].grid(True); axes[1].legend(ncol=2, fontsize=8)
plt.show()

# ============================================================
# 5) Interpolate exact data with B-splines by solving N(xi) c = y
#     (use Greville points as collocation for a well-conditioned system)
# ============================================================
print("\n=== Test 5: Exact interpolation via B-splines (solve for coefficients) ===")
k, nb = 3, 10
t = open_uniform_knots(0.0, 1.0, k, nb)
Xi = greville_abscissae(t, k)   # collocation points (nb points)
# target function
f = lambda x: np.sin(2*np.pi*x) + 0.2*x
y = f(Xi)
# Build system and solve
N_colloc = Nbspl(t, Xi, k=k)    # (nb, nb)
c = np.linalg.solve(N_colloc, y)

# Evaluate spline on a fine grid and compare to f
xf = np.linspace(0, 1, 600)
Nf = Nbspl(t, xf, k=k)          # (600, nb)
Sf = Nf @ c
res_colloc = np.max(np.abs((N_colloc @ c) - y))
res_fine   = np.max(np.abs(Sf - f(xf)))
print(f"Max residual at collocation points (should be ~1e-12): {res_colloc:.3e}")
print(f"Max abs error on fine grid (approx. quality, not necessarily tiny): {res_fine:.3e}")

plt.figure(figsize=(8,4))
plt.plot(xf, f(xf), 'k-', lw=2, label="target f(x)")
plt.plot(xf, Sf, 'r--', lw=2, label="B-spline fit")
plt.scatter(Xi, y, s=30, c='b', zorder=3, label="collocation points")
plt.title("B-spline interpolation via linear system (exact at Greville points)")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.show()

# ============================================================
# 6) Effect of repeated interior knot: continuity drop (k=3 -> C^2 to C^1/C^0)
# ============================================================
print("\n=== Test 6: Repeated interior knot reduces smoothness (see jump in derivatives) ===")
k, nb = 3, 12
# Start with open-uniform
t = open_uniform_knots(0.0, 1.0, k, nb)
# Repeat an interior knot (e.g., near 0.5) to reduce continuity
mid = 0.5
# Insert the knot 2 extra times (multiplicity r=3 total) -> continuity C^{k-r}=C^{0} at x=0.5
t = np.sort(np.concatenate([t, [mid, mid]]))

# Build a simple smooth function via coefficients
# We'll fit constants for demo: c_i = sin(i/nb * 2pi) gives some variation
c = np.sin(np.linspace(0, 2*np.pi, len(t)-k-1, endpoint=False))
# Evaluate spline and its derivative
x = np.linspace(0, 1, 1000)
N, dN = Nbspld1(t, x, k=k)
S  = N @ c
Sd = dN @ c

# Locate indices around the repeated knot and measure left/right derivative
ix = np.argmin(np.abs(x - mid))
left_slope  = Sd[ix-1]
right_slope = Sd[ix+1]
print(f"Interior knot at x≈{mid}: left slope={left_slope:.6f}, right slope={right_slope:.6f}")
print("Expected: visible change in slope at the repeated knot (reduced continuity).")

fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
axes[0].plot(x, S); axes[0].axvline(mid, color='k', ls=':', label='repeated knot')
axes[0].set_title("Spline S(x) with repeated interior knot")
axes[0].set_xlabel("x"); axes[0].set_ylabel("S(x)"); axes[0].grid(True); axes[0].legend()
axes[1].plot(x, Sd); axes[1].axvline(mid, color='k', ls=':')
axes[1].set_title("First derivative S'(x) — note slope mismatch at knot")
axes[1].set_xlabel("x"); axes[1].set_ylabel("S'(x)"); axes[1].grid(True)
plt.show()

# ============================================================
# 7) Expected error cases (robustness)
# ============================================================
print("\n=== Test 7: Expected error cases ===")
# 20.1 k too large for knot vector
try:
    Nbspl([0,0,0,0], x=[0,0.5,1], k=4)
except ValueError as e:
    print("k too large (Nbspl):", e)

# 20.2 x not 1D (pass a 2D array accidentally)
try:
    X2D = np.array([[0.0, 0.5, 1.0],
                    [0.2, 0.7, 0.9]])
    Nbspl(open_uniform_knots(0,1,3,6), x=X2D, k=3)
except Exception as e:
    print("x not 1D (may raise/behave unexpectedly):", repr(e))

# 20.3 Knot vector too short for requested degree
try:
    Nbspl([0,0,1,1], x=[0.1, 0.2], k=3)  # len(t)=4, need k<=len(t)-2=2
except ValueError as e:
    print("knot length invalid (Nbspl):", e)

print("\n---------- END OF INTERPOLATION TESTS ---------")