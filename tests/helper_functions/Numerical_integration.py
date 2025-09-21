#---------------- Second round of tests and modifications ------------------------
#######################
# Numerical integration
#######################
import numpy as np
import matplotlib.pyplot as plt
from src.CosmoTransitions import rkqs
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