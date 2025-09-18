from CosmoTranstions_2 import set_default_args, monotonic_indices, clamp_val, rkqs, _rkck ,deriv14, deriv14_const_dx, deriv23,deriv23_const_dx ,deriv1n
import numpy as np
import matplotlib.pyplot as plt

#---------------- First round of tests and modifications ------------------------
#########################
# Miscellaneous functions
#########################

print("---------- TESTS MISCELLANEOUS FUNCTIONS ---------")
"""
The purpose of set_default_args is to modify the default parameter values of a function,
either by changing the original function or by creating a wrapper.
"""

# Usage examples:

# example 1: positional + keyword-only (to the right of * only keyword-only params can be passed by name, e.g., d=20 and not f(20))
def f(a, b=2, c=3, *, d=4):
    return a, b, c, d

print(f(10))

# in-place
set_default_args(f, b=20, d=40)  # Changes the previous function's defaults to b=20 and d=40 | Using inplace mutates the original function to the desired new defaults

print(f(10))


# non-inplace
def g(a, b=2, c=3, *, d=4):
    return a, b, c, d

g2 = set_default_args(g, inplace=False, b=99, d=111)  # Creates a new function with different base defaults from the original

print(g(1))
print(g2(1))

# example 2: Errors when parameters do not exist or lack a default
def h(a, b, c=3):
    return a, b, c

try:
    set_default_args(h, x=1)  # No corresponding parameter
except ValueError as e:
    print("Error: ", e)

try:
    set_default_args(h, b=10)  # 'b' has no default
except ValueError as e:
    print("Error: ", e)  # Parameter without an initial default value


"""
The purpose of monotonic_indices is to provide the indices of elements that form a strictly increasing subsequence.
Use case: if the sequence has one or another point that breaks strict monotonic increase, that point is removed, which can help with small unwanted deviations.
"""

x = [1, 2, 3, -1, 20, 19, 50]  # Example with a broken value in the middle, overall increasing
y = []
for i in monotonic_indices(x):
    y.append(x[i])

print(y)

k = [50, 19, 20, -1, 3, 2, 1]  # Example with a broken value in the middle, overall decreasing
print(monotonic_indices(k))  # indices mapped back for the decreasing case

"""
The purpose of clamp_val is to transform the values of a list that are not within a given interval [a, b]
so that they fall within that interval. If a value is larger, it becomes the maximum (b); if smaller, it becomes the minimum (a).
This can be useful to remove non-physical results from simulations/calculations.
"""

x = [1, 2, 3, -1, 20, 19, 50]

y = clamp_val(x, a=1, b=20)

print(y)


#---------------- Second round of tests and modifications ------------------------
#######################
# Numerical integration
#######################
print("---------- TESTS NUMERICAL INTEGRATION ---------")

"""
The purpose of rkqs is to use the 5th-order Runge–Kutta method with adaptive error control to solve ODEs
that require function integration. Below are two simple examples.
"""

# Numerical integrals tests

def f(y, t):
    return y  # dy/dt = y → exact solution y = exp(t)

y0 = np.array([1.0])
t0 = 0.0
dydt0 = f(y0, t0)

result = rkqs(y0, dydt0, t0, f, dt_try=0.1, epsfrac=1e-6, epsabs=1e-9)
print(result)

# It should give Delta_y ~ 0.105 (since exp(0.1)-1 ≈ 0.105).
#-------------------------------------------------

# -------------------------------------------------
# PHYSICAL EXAMPLE: Harmonic Oscillator with rkqs
# -------------------------------------------------
# Test with harmonic oscillator
print("\n=== Test: Harmonic oscillator with rkqs ===")

def harmonic_oscillator(y, t, omega):
    # y = [x, v], dx/dt = v, dv/dt = -omega^2 x
    return np.array([y[1], -omega**2 * y[0]], dtype=float)

def integrate_oscillator(y0, t0, t_end, dt_try, omega, epsfrac=1e-6, epsabs=1e-9):
    y = np.array(y0, dtype=float)
    t = float(t0)
    dt = float(dt_try)

    xs = [y[0]]
    vs = [y[1]]
    ts = [t]
    dts = []

    while t < t_end:
        dydt = harmonic_oscillator(y, t, omega)
        dy, dt_used, dt_next = rkqs(
            y, dydt, t, harmonic_oscillator,
            dt, epsfrac, epsabs, args=(omega,)
        )
        y = y + dy
        t = t + dt_used

        xs.append(y[0]); vs.append(y[1]); ts.append(t); dts.append(dt_used)
        dt = dt_next

    return np.array(ts), np.array(xs), np.array(vs), np.array(dts)

omega = 1.0
y0 = [1.0, 0.0]   # x(0)=1, v(0)=0
t0, t_end = 0.0, 20.0
dt_try = 0.1

ts, xs, vs, dts = integrate_oscillator(y0, t0, t_end, dt_try, omega)

# Check approximate energy conservation (not symplectic, so small oscillations expected)
E = 0.5*(vs**2 + (omega**2)*(xs**2))
print(f"Energy stats (should be nearly constant; small oscillations expected):")
print(f"  E min={E.min():.6f}, E max={E.max():.6f}, ΔE={E.max()-E.min():.3e}")

plt.figure(figsize=(9,4))
plt.plot(ts, xs, label="x(t)")
plt.plot(ts, vs, label="v(t)")
plt.title("Harmonic oscillator solved with rkqs (adaptive RKCK)")
plt.xlabel("t")
plt.ylabel("state")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,3))
plt.plot(ts[1:], dts, marker='o', ms=3, linestyle='-')
plt.title("Adaptive time steps chosen by rkqs")
plt.xlabel("t")
plt.ylabel("dt used")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,3))
plt.plot(ts, E)
plt.title("Total energy (expected nearly constant)")
plt.xlabel("t")
plt.ylabel("E(t)")
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------- Third round of tests and modifications ------------------------
#######################
# Numerical derivatives
#######################

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


