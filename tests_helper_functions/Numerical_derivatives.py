#---------------- Third round of tests and modifications ------------------------
#######################
# Numerical derivatives
#######################
import numpy as np
import matplotlib.pyplot as plt
from CosmoTranstions_2  import rkqs
from CosmoTranstions_2 import deriv14, deriv14_const_dx, deriv23, deriv23_const_dx, deriv1n

# ==========================================
# Tests for Numerical Derivatives + RKQS
# ==========================================
# What this file does:
# 1) Validates first and second derivatives on uniform grids (closed-form accuracy, O(h^4) inside).
# 2) Validates first and second derivatives on non-uniform grids (Fornberg weights).
# 3) Shows how accuracy changes with stencil size using deriv1n.
# 4) Exercises boundary behavior explicitly.
# 5) Demonstrates expected errors (ValueError) for invalid inputs.
# 6) Bonus: uses rkqs on a physical system (harmonic oscillator) and plots results.
#
# Expected outcomes (roughly):
# - For smooth functions (sin, exp), max abs errors for 5-point uniform interior:
#   first derivative: ~O(h^4), second derivative: ~O(h^4) (interior).
# - On non-uniform grids, accuracy remains high; exact order depends on local spacing.
# - Increasing stencil size in deriv1n (m=n+1) typically reduces error for smooth functions.
# - Error cases should raise ValueError with clear messages.

print("---------- TESTS NUMERICAL DERIVATIVES---------")
np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(42)

# -------------------------------------------------
# 1) UNIFORM GRID – FIRST DERIVATIVE (sin profile)
# -------------------------------------------------
print("\n=== Test derivative: Uniform grid, first derivative (sin profile) ===")
N = 201
L = 2*np.pi
x = np.linspace(0.0, L, N)
dx = x[1] - x[0]
k = 1.0

y = np.sin(k*x)
dy_exact = k*np.cos(k*x)

dy_num = deriv14_const_dx(y, dx=dx)
max_err = np.max(np.abs(dy_num - dy_exact))

print(f"Grid: N={N}, dx={dx:.3e}, function: sin(kx), k={k}")
print(f"Max abs error (dy): {max_err:.3e}  (expected ~O(h^4) ~ {(dx**4):.1e})")

plt.figure(figsize=(9,4))
plt.plot(x, dy_exact, label="exact dy/dx", linewidth=2)
plt.plot(x, dy_num, '--', label="numerical dy/dx (deriv14_const_dx)")
plt.title("First derivative on uniform grid (sin)")
plt.xlabel("x")
plt.ylabel("dy/dx")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,3))
plt.plot(x, dy_num - dy_exact)
plt.title("Error: numerical - exact (dy)")
plt.xlabel("x")
plt.ylabel("error")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 2) UNIFORM GRID – SECOND DERIVATIVE (sin profile)
# -------------------------------------------------
print("\n=== Test 2: Uniform grid, second derivative (sin profile) ===")
d2y_exact = -k**2 * np.sin(k*x)

d2y_num = deriv23_const_dx(y, dx=dx)
max_err2 = np.max(np.abs(d2y_num - d2y_exact))
print(f"Max abs error (d2y): {max_err2:.3e}  (expected ~O(h^3) at edges ~ {(dx**3):.1e})")

plt.figure(figsize=(9,4))
plt.plot(x, d2y_exact, label="exact d2y/dx2", linewidth=2)
plt.plot(x, d2y_num, '--', label="numerical d2y/dx2 (deriv23_const_dx)")
plt.title("Second derivative on uniform grid (sin)")
plt.xlabel("x")
plt.ylabel("d2y/dx2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,3))
plt.plot(x, d2y_num - d2y_exact)
plt.title("Error: numerical - exact (d2y)")
plt.xlabel("x")
plt.ylabel("error")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 3) NON-UNIFORM GRID – FIRST & SECOND DERIVATIVES (exp profile)
# -------------------------------------------------
print("\n=== Test 3: Non-uniform grid, first & second derivatives (exp profile) ===")
Nn = 161
# Build a strictly increasing non-uniform grid on [0, 1]
x_nonuni = np.linspace(0, 1, Nn)
x_nonuni = x_nonuni**1.7  # make it non-uniform but monotonic increasing
y_nonuni = np.exp(x_nonuni)

dy_exact_nonuni = np.exp(x_nonuni)            # d/dx exp(x) = exp(x)
d2y_exact_nonuni = np.exp(x_nonuni)           # d2/dx2 exp(x) = exp(x)

dy_num_nonuni = deriv14(y_nonuni, x_nonuni)
d2y_num_nonuni = deriv23(y_nonuni, x_nonuni)

err1 = np.max(np.abs(dy_num_nonuni - dy_exact_nonuni))
err2 = np.max(np.abs(d2y_num_nonuni - d2y_exact_nonuni))
print(f"Non-uniform grid size: N={Nn}")
print(f"Max abs error (dy):  {err1:.3e}")
print(f"Max abs error (d2y): {err2:.3e}")

plt.figure(figsize=(9,4))
plt.plot(x_nonuni, dy_exact_nonuni, label="exact dy/dx", linewidth=2)
plt.plot(x_nonuni, dy_num_nonuni, '.', ms=3, label="numerical dy/dx (deriv14)")
plt.title("First derivative on non-uniform grid (exp)")
plt.xlabel("x")
plt.ylabel("dy/dx")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,4))
plt.plot(x_nonuni, d2y_exact_nonuni, label="exact d2y/dx2", linewidth=2)
plt.plot(x_nonuni, d2y_num_nonuni, '.', ms=3, label="numerical d2y/dx2 (deriv23)")
plt.title("Second derivative on non-uniform grid (exp)")
plt.xlabel("x")
plt.ylabel("d2y/dx2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 4) ACCURACY VS STENCIL SIZE (deriv1n on sin)
# -------------------------------------------------
print("\n=== Test 4: Accuracy vs stencil size (deriv1n on sin) ===")
N2 = 181
x2 = np.linspace(0, 2*np.pi, N2)
y2 = np.sin(x2)
dy2_exact = np.cos(x2)

for n in (4, 6, 8):  # stencil sizes m=n+1 -> 5, 7, 9 points
    dy2_num = deriv1n(y2, x2, n=n)
    err = np.max(np.abs(dy2_num - dy2_exact))
    print(f"n={n} (stencil m={n+1}): max abs error = {err:.3e}")
print("Expected: error generally decreases as stencil size increases (for smooth functions).")

# -------------------------------------------------
# 5) BOUNDARY BEHAVIOR CHECK (compare first/last points)
# -------------------------------------------------
print("\n=== Test 5: Boundary behavior (first/last points) ===")
# Use sin on uniform grid; exact derivative known at boundaries:
dy_num_u = deriv14_const_dx(y, dx=dx)
print(f"dy at x[0]:  num={dy_num_u[0]: .6e}, exact={dy_exact[0]: .6e}")
print(f"dy at x[-1]: num={dy_num_u[-1]: .6e}, exact={dy_exact[-1]: .6e}")
print("Note: boundary stencils are one-sided; error is typically larger at the ends than in the interior.")


# -------------------------------------------------
# 6) EXPECTED ERROR CASES
# -------------------------------------------------
print("\n=== Test 6: Expected error cases ===")
# 6.1: Too few points
try:
    deriv14(np.array([1,2,3,4.0]), np.array([0,1,2,3.0]))
except ValueError as e:
    print("Too few points (deriv14):", e)

# 6.2: Non-monotonic x
try:
    x_bad = np.array([0.0, 1.0, 0.5, 1.5, 2.0])  # not strictly monotonic
    y_bad = np.sin(x_bad)
    deriv23(y_bad, x_bad)
except ValueError as e:
    print("Non-monotonic x (deriv23):", e)

# 6.3: x not 1D
try:
    x_2d = np.vstack([x, x])  # 2D
    deriv14(y, x_2d)
except ValueError as e:
    print("x not 1D (deriv14):", e)

# ================================
# Tests for gradient & Hessian
# (continuation of previous suite)
# ================================

from CosmoTranstions_2 import gradientFunction, hessianFunction
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(123)

print("---------- TESTS GRADIENT & HESSIAN ---------")

# -------------------------------------------------
# 7) GRADIENT on a smooth 2D scalar field (sin* cos)
# -------------------------------------------------
print("\n=== Test 7: Gradient on V(x,y) = sin(x) * cos(y) (order=4) ===")

def V_sin_cos(X):
    # X shape (..., 2)
    x, y = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
    return np.sin(x) * np.cos(y)

# analytic gradient
def gradV_sin_cos_exact(X):
    x, y = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
    dVdx = np.cos(x) * np.cos(y)
    dVdy = -np.sin(x) * np.sin(y)
    return np.moveaxis(np.stack([dVdx, dVdy], axis=0), 0, -1)

# grid
nx = ny = 101
x = np.linspace(-2*np.pi, 2*np.pi, nx)
y = np.linspace(-2*np.pi, 2*np.pi, ny)
Xg, Yg = np.meshgrid(x, y, indexing='xy')
Pts = np.stack([Xg, Yg], axis=-1)            # shape (ny, nx, 2)

gf = gradientFunction(V_sin_cos, eps=[1e-4, 1e-4], Ndim=2, order=4)
g_num = gf(Pts)                               # (ny, nx, 2)
g_exa = gradV_sin_cos_exact(Pts)

err = np.linalg.norm(g_num - g_exa, axis=-1)  # magnitude error per point
print(f"Grid: {nx}x{ny}, eps=1e-4 (per axis), order=4")
print(f"Max |grad error| = {err.max():.3e},  Mean |grad error| = {err.mean():.3e}")

# 3D surface of V colored by |∇V|
Vval = V_sin_cos(Pts)
grad_mag = np.linalg.norm(g_num, axis=-1)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# color by gradient magnitude (|∇V|)
norm = plt.Normalize(grad_mag.min(), grad_mag.max())
colors = plt.cm.viridis(norm(grad_mag))

surf = ax.plot_surface(Xg, Yg, Vval, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)

mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(grad_mag)

cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("|∇V|")
ax.set_title("V(x,y) = sin(x) cos(y) — surface colored by |∇V|")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("V")
plt.tight_layout(); plt.show()

# 2D quiver of the gradient (for intuition)
step = 6
plt.figure(figsize=(6,5))
plt.contour(Xg, Yg, Vval, levels=15, linewidths=0.8, colors='k', alpha=0.3)
plt.quiver(Xg[::step, ::step], Yg[::step, ::step],
           g_num[::step, ::step, 0], g_num[::step, ::step, 1],
           grad_mag[::step, ::step], cmap='viridis', scale=40)
plt.title("Gradient field ∇V (colored by |∇V|)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()

# -------------------------------------------------
# 8) HESSIAN exactness on a quadratic: V = 1/2 x^T A x + b·x + c
# -------------------------------------------------
print("\n=== Test 8: Hessian exactness on quadratic form (order=4) ===")

Ndim = 3
# Random symmetric positive-definite A
M = rng.standard_normal((Ndim, Ndim))
A = (M + M.T)/2
A += Ndim * np.eye(Ndim)  # make it SPD-ish
b = rng.standard_normal(Ndim)
c = 0.3

def V_quad(X):
    X = np.asarray(X, dtype=float)
    return 0.5*np.einsum('...i,ij,...j->...', X, A, X) + np.einsum('i,...i->...', b, X) + c

hf = hessianFunction(V_quad, eps=1e-4, Ndim=Ndim, order=4)

# sample points
P = rng.standard_normal((200, Ndim))
H_num = hf(P)                         # (200, Ndim, Ndim)

# exact Hessian is constant = A
errH = np.abs(H_num - A).reshape(-1, Ndim, Ndim)
max_abs = np.max(np.abs(H_num - A))
mean_abs = np.mean(np.abs(H_num - A))
print(f"Max |H - A| = {max_abs:.3e},  Mean |H - A| = {mean_abs:.3e}")
print("Expected: near machine precision for smooth quadratics with small eps.")

# -------------------------------------------------
# 9) HESSIAN with mixed terms: V(x,y) = x^2 y + y^3
#    H = [[ 2y, 2x], [2x, 6y]]
# -------------------------------------------------
print("\n=== Test 9: Hessian with mixed terms (order=4) ===")

def V_mixed(X):
    x, y = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
    return x*x*y + y**3

def H_mixed_exact(X):
    x, y = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
    H11 = 2*y
    H22 = 6*y
    H12 = H21 = 2*x
    H = np.stack([[H11, H12], [H21, H22]], axis=0)   # (2,2,...) then move axes
    return np.moveaxis(H, (0,1), (-1,-2))            # (...,2,2)

hf_mixed = hessianFunction(V_mixed, eps=[1e-4, 3e-4], Ndim=2, order=4)

xx = np.linspace(-2, 2, 60)
yy = np.linspace(-2, 2, 60)
Xg, Yg = np.meshgrid(xx, yy, indexing='xy')
Pts2 = np.stack([Xg, Yg], axis=-1)
Hn = hf_mixed(Pts2)               # (ny, nx, 2, 2)
He = H_mixed_exact(Pts2)          # (ny, nx, 2, 2)
errH = np.max(np.abs(Hn - He))
print(f"Grid 60x60, eps=[1e-4,3e-4]: max |H_num - H_ex| = {errH:.3e}")

# Quick visualization: Hxy numeric vs exact (shared color scale + single colorbar)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

Hxy_num = Hn[..., 0, 1]
Hxy_ex  = He[..., 0, 1]

# Use the same symmetric color limits for fair comparison
vabs = np.nanmax(np.abs(np.stack([Hxy_num, Hxy_ex])))

im0 = axes[0].imshow(Hxy_num, extent=[xx.min(), xx.max(), yy.min(), yy.max()],origin='lower', aspect='auto', cmap='coolwarm',
    vmin=-vabs, vmax=vabs)
axes[0].set_title("Hxy numeric")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(Hxy_ex,extent=[xx.min(), xx.max(), yy.min(), yy.max()],origin='lower', aspect='auto', cmap='coolwarm',
    vmin=-vabs, vmax=vabs)
axes[1].set_title("Hxy exact")
axes[1].set_xlabel("x")

# One shared colorbar for both panels
cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
cbar.set_label("Hxy")

plt.show()

# -------------------------------------------------
# 10) ORDER CHECK: rate vs eps for order=2 and 4 (1D function)
# -------------------------------------------------
print("\n=== Test 10: Convergence rate vs eps (gradientFunction) ===")

def f1d(X):
    x = np.asarray(X, dtype=float)[..., 0]
    return np.sin(2.3*x)

def df1d_exact(X):
    x = np.asarray(X, dtype=float)[..., 0]
    return 2.3*np.cos(2.3*x)[..., None]

# sample points
xpts = np.linspace(-2, 2, 200)[..., None]   # shape (200, 1)
eps_list = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])

def rate(order):
    errs = []
    for e in eps_list:
        gf_local = gradientFunction(f1d, eps=e, Ndim=1, order=order)
        err = np.max(np.abs(gf_local(xpts) - df1d_exact(xpts)))
        errs.append(err)
    # slope of log(err) vs log(eps)
    p = np.polyfit(np.log(eps_list), np.log(np.array(errs)), 1)[0]
    return p, np.array(errs)

p2, errs2 = rate(2)
p4, errs4 = rate(4)
print(f"Estimated order (order=2 stencil):  {p2:.2f} (expected ~2)")
print(f"Estimated order (order=4 stencil):  {p4:.2f} (expected ~4)")

plt.figure(figsize=(6,5))
plt.loglog(eps_list, errs2, 'o-', label='order=2')
plt.loglog(eps_list, errs4, 's-', label='order=4')
plt.gca().invert_xaxis()
plt.xlabel("eps"); plt.ylabel("max abs error")
plt.title("Convergence with eps (1D test)")
plt.grid(True, which='both', ls=':')
plt.legend(); plt.tight_layout(); plt.show()

# -------------------------------------------------
# 11) PHYSICS: Electrostatic-like potential (softened) and E = -∇V
#     Two opposite charges; 3D surface colored by |E| and 2D quiver.
# -------------------------------------------------
print("\n=== Test 11: Electrostatic potential (softened) and field via gradientFunction ===")

k_c = 1.0
q1, q2 = +1.0, -1.0
a = 0.6
r1 = np.array([-a, 0.0])
r2 = np.array([+a, 0.0])
soft2 = 0.05**2  # softening^2 to avoid singularities in FD evaluations

def V_elec(X):
    X = np.asarray(X, dtype=float)
    dx1 = X - r1
    dx2 = X - r2
    r1s = np.sum(dx1*dx1, axis=-1) + soft2
    r2s = np.sum(dx2*dx2, axis=-1) + soft2
    return k_c*( q1 / np.sqrt(r1s) + q2 / np.sqrt(r2s) )

def E_exact(X):
    X = np.asarray(X, dtype=float)
    dx1 = X - r1
    dx2 = X - r2
    r1s = np.sum(dx1*dx1, axis=-1) + soft2
    r2s = np.sum(dx2*dx2, axis=-1) + soft2
    # Expand scalars to (..., 1) so they broadcast against the last dim (2)
    inv_r1_3 = 1.0 / (r1s[..., None] ** 1.5)   # (..., 1)
    inv_r2_3 = 1.0 / (r2s[..., None] ** 1.5)   # (..., 1)

    # grad V = - sum q r / (r^2+soft2)^(3/2); E = -grad V
    gV = (-k_c * q1 * dx1 * inv_r1_3) + (-k_c * q2 * dx2 * inv_r2_3)
    E = -gV
    return E

gf_e = gradientFunction(V_elec, eps=[1e-3, 1e-3], Ndim=2, order=4)

# grid for plotting
nxy = 121
x = np.linspace(-2.0, 2.0, nxy)
y = np.linspace(-1.5, 1.5, nxy)
Xg, Yg = np.meshgrid(x, y, indexing='xy')
Pts = np.stack([Xg, Yg], axis=-1)
Vg = V_elec(Pts)
gradV = gf_e(Pts)
E_num = -gradV
E_exa = E_exact(Pts)

# relative error (mask near charges)
rel = np.linalg.norm(E_num - E_exa, axis=-1) / np.maximum(np.linalg.norm(E_exa, axis=-1), 1e-12)
mask_near = (np.hypot(Xg - r1[0], Yg - r1[1]) < 0.2) | (np.hypot(Xg - r2[0], Yg - r2[1]) < 0.2)
rel_masked = np.where(mask_near, np.nan, rel)
print(f"Rel. error stats away from charges: max={np.nanmax(rel_masked):.3e}, mean={np.nanmean(rel_masked):.3e}")

# 3D surface colored by |E|
Emag = np.linalg.norm(E_num, axis=-1)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(Emag.min(), np.nanmax(Emag))
colors = plt.cm.inferno(norm(Emag))
ax.plot_surface(Xg, Yg, Vg, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)
m = plt.cm.ScalarMappable(cmap='inferno', norm=norm); m.set_array(Emag)
cb = fig.colorbar(m,ax=ax, shrink=0.6, pad=0.1); cb.set_label("|E| = |-∇V|")
ax.set_title("Softened dipole potential V — surface colored by |E|")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("V")
plt.tight_layout(); plt.show()

# 2D quiver + potential contours
plt.figure(figsize=(7,5))
plt.contour(Xg, Yg, Vg, levels=30, colors='k', linewidths=0.5, alpha=0.5)
step = 6
plt.quiver(Xg[::step, ::step], Yg[::step, ::step],
           E_num[::step, ::step, 0], E_num[::step, ::step, 1],
           np.linalg.norm(E_num[::step, ::step], axis=-1),
           cmap='inferno', scale=40)
plt.scatter([r1[0], r2[0]], [r1[1], r2[1]], c=['b','r'], s=60, marker='o', label='charges')
plt.title("Electric field E = -∇V (colored by |E|) with equipotentials")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.axis('equal'); plt.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()

# -------------------------------------------------
# 12) Dynamics in a 2D anisotropic harmonic potential using rkqs + gradientFunction
#     V(x,y) = 0.5 * (kx x^2 + ky y^2), force = -∇V, m=1
# -------------------------------------------------
print("\n=== Test 12 (bonus): 2D anisotropic harmonic oscillator with rkqs and ∇V ===")

kx, ky = 1.0, 2.0

def V_harm(X):
    x, y = np.moveaxis(np.asarray(X, dtype=float), -1, 0)
    return 0.5*(kx*x*x + ky*y*y)

gf_h = gradientFunction(V_harm, eps=[1e-5, 1e-5], Ndim=2, order=4)

def eom(state, t):
    # state = [x, y, vx, vy]
    pos = state[:2]
    vel = state[2:]
    gradV = np.asarray(gf_h(pos)).reshape(-1)      # shape (2,) (thanks to broadcasting in our class)
    acc = -gradV                # m=1
    return np.array([vel[0], vel[1], acc[0], acc[1]], dtype=float)

# integrate
y = np.array([1.0, 0.0, 0.0, 1.0])  # initial [x,y,vx,vy]
t, t_end, dt_try = 0.0, 30.0, 0.05
epsfrac, epsabs = 1e-7, 1e-10

ts = [t]; traj = [y.copy()]; dts = []
while t < t_end:
    dydt = eom(y, t)
    dy, dt_used, dt_next = rkqs(y, dydt, t, eom, dt_try, epsfrac, epsabs)
    y = y + dy; t = t + dt_used
    ts.append(t); traj.append(y.copy()); dts.append(dt_used)
    dt_try = dt_next

ts = np.array(ts); traj = np.array(traj)
x, y_, vx, vy = traj.T
E = 0.5*(vx**2 + vy**2) + V_harm(np.stack([x, y_], axis=-1))

print(f"Energy stats: Emin={E.min():.6f}, Emax={E.max():.6f}, ΔE={E.max()-E.min():.3e}")

plt.figure(figsize=(6,5))
plt.plot(x, y_, '-')
plt.title("Orbit in anisotropic harmonic potential")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure(figsize=(9,3))
plt.plot(ts[1:], dts, '-o', ms=3)
plt.title("Adaptive time steps (rkqs)")
plt.xlabel("t"); plt.ylabel("dt used"); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure(figsize=(9,3))
plt.plot(ts, E)
plt.title("Total energy (should stay nearly constant)")
plt.xlabel("t"); plt.ylabel("E(t)"); plt.grid(True)
plt.tight_layout(); plt.show()

# -------------------------------------------------
# 13) EXPECTED ERROR CASES (robustness checks)
# -------------------------------------------------
print("\n=== Test 13: Expected error cases (gradient/hessian) ===")

# 13.1 Invalid order
try:
    gradientFunction(V_sin_cos, eps=1e-3, Ndim=2, order=3)
except ValueError as e:
    print("Invalid order (gradientFunction):", e)

try:
    hessianFunction(V_sin_cos, eps=1e-3, Ndim=2, order=6)
except ValueError as e:
    print("Invalid order (hessianFunction):", e)

# 13.2 eps wrong shape
try:
    gradientFunction(V_sin_cos, eps=[1e-3], Ndim=2, order=4)
except ValueError as e:
    print("Wrong eps shape (gradientFunction):", e)

# 13.3 x last-axis length ≠ Ndim
try:
    gf_bad = gradientFunction(V_sin_cos, eps=[1e-3,1e-3], Ndim=2, order=4)
    gf_bad(np.zeros((10, 3)))  # last axis 3 instead of 2
except ValueError as e:
    print("x last axis mismatch (gradientFunction):", e)

# 13.4 f not scalar (returns vector) -> broadcasting/sum should fail
def f_not_scalar(X):
    # deliberately returns a vector per point (illegal for our classes)
    X = np.asarray(X, dtype=float)
    return X  # shape (..., Ndim)

try:
    gf_ns = gradientFunction(f_not_scalar, eps=[1e-3,1e-3], Ndim=2, order=4)
    gf_ns(np.zeros((4,2)))
except Exception as e:
    print("Non-scalar f (gradientFunction):", repr(e))

try:
    hf_ns = hessianFunction(f_not_scalar, eps=[1e-3,1e-3], Ndim=2, order=4)
    hf_ns(np.zeros((4,2)))
except Exception as e:
    print("Non-scalar f (hessianFunction):", repr(e))

print("\n---------- END OF GRADIENT & HESSIAN TESTS ---------")