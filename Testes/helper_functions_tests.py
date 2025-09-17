from CosmoTranstions_2 import set_default_args, monotonic_indices, clamp_val, rkqs, _rkck ,deriv14, deriv14_const_dx, deriv23, deriv1n
import numpy as np
import matplotlib.pyplot as plt

#---------------- First round of tests and modifications ------------------------
#########################
# Miscellaneous functions
#########################


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

# Test with harmonic oscillator

def harmonic_oscillator(y, t, omega):
    """
    Derivative for the simple harmonic oscillator.
    y[0] = position x
    y[1] = velocity v
    """
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def integrate_oscillator(y0, t0, t_end, dt_try, omega, epsfrac=1e-6, epsabs=1e-9):
    """
    Integrates the harmonic oscillator using rkqs.
    """
    y = np.array(y0, dtype=float)
    t = t0
    dt = dt_try

    positions = [y[0]]
    velocities = [y[1]]
    times = [t]

    while t < t_end:
        dydt = harmonic_oscillator(y, t, omega)
        dy, dt_used, dt_next = rkqs(
            y, dydt, t, harmonic_oscillator,
            dt, epsfrac, epsabs, args=(omega,)
        )
        # Update solution
        y = y + dy
        t = t + dt_used

        positions.append(y[0])
        velocities.append(y[1])
        times.append(t)

        dt = dt_next  # use adaptive step

    return np.array(times), np.array(positions), np.array(velocities)

# Parameters
omega = 1.0       # natural frequency
y0 = [1.0, 0.0]   # initial position=1, velocity=0
t0, t_end = 0.0, 20.0
dt_try = 0.1

# Run integration
times, positions, velocities = integrate_oscillator(y0, t0, t_end, dt_try, omega)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(times, positions, label="Position x(t)")
plt.plot(times, velocities, label="Velocity v(t)")
plt.xlabel("Time t")
plt.ylabel("State")
plt.legend()
plt.grid(True)
plt.title("Harmonic Oscillator (Runge–Kutta with adaptive step size)")
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

