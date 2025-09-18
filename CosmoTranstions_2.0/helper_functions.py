# New version of helper_functions

import numpy as np
import inspect
import functools
from typing import Callable, Any, Dict, Tuple, Iterable, Union, Optional
from collections import namedtuple

###############################################################
# Miscellaneous functions - Functions to help others in general
###############################################################

def set_default_args(func: Callable, inplace: bool = True, **kwargs) -> Callable:
    """
     Update the default values of parameters of `func`.

     Parameters
     ----------
     func : Callable
         The function (or unbound method) whose defaults will be updated.
     inplace : bool, optional (default=True)
         If True, modifies the given function (changes __defaults__ / __kwdefaults__).
         If False, returns a *wrapper* that applies the defaults without touching `func`.
     **kwargs
         Mapping parameter_name=new_default_value

     Behavior / Notes
     ----------------
     - Supports both positional parameters with defaults and keyword-only parameters.
     - Raises ValueError if:
         * a name passed in kwargs does not exist in the function’s signature.
         * the parameter exists but does NOT have a default value (cannot replace what doesn’t exist).
     - For *args and **kwargs parameters, defaults cannot be set.
     - For bound methods, pass the underlying function (Class.method) or use __func__.

     Returns
     -------
     Callable
         The modified function itself (if inplace=True) or a wrapper (if inplace=False).

     Example
     -------
     >>> def f(a, b=2, c=3, *, d=4):
     ...     return a, b, c, d
     >>> set_default_args(f, b=20, d=40)
     >>> f(1)
     (1, 20, 3, 40)

     >>> g = set_default_args(f, inplace=False, b=99)
     >>> g(1)
     (1, 99, 3, 40)
     """
    if not callable(func):
        raise TypeError("`func` must be callable")

    # If the user passed a bound method (e.g. instance.method), get the actual function
    original_obj = func
    is_bound_method = hasattr(func, "__func__") and inspect.ismethod(func)
    if is_bound_method:
        func = func.__func__

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    param_map: Dict[str, inspect.Parameter] = {p.name: p for p in params}

    # Validate names and if they have default
    for name in kwargs:
        if name not in param_map:
            raise ValueError(f"Function '{func.__name__}' doesn't have '{name}' parameter")
        if param_map[name].default is inspect._empty:
            raise ValueError(f"Parameter '{name}' doesn't have any default to be changed")

    # Separate updates by parameter type
    pos_updates: Dict[str, Any] = {}
    kwonly_updates: Dict[str, Any] = {}
    for name, val in kwargs.items():
        kind = param_map[name].kind
        if kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos_updates[name] = val
        elif kind == inspect.Parameter.KEYWORD_ONLY:
            kwonly_updates[name] = val
        else:
            # VAR_POSITIONAL or VAR_KEYWORD
            raise ValueError(f"It's not possible to give a default for the parameter of the type {kind} ('{name}')")

    # --- In-place update: handle __defaults__ and __kwdefaults__ ---
    def _apply_inplace(f: Callable) -> Callable:
        # keyword-only defaults (dict)
        current_kwdefaults = getattr(f, "__kwdefaults__", None) or {}
        current_kwdefaults = dict(current_kwdefaults)  # make a mutable copy
        current_kwdefaults.update(kwonly_updates)
        if current_kwdefaults:
            f.__kwdefaults__ = current_kwdefaults
        else:
            # if it is empty, clear kwdefaults to avoid stale state
            if hasattr(f, "__kwdefaults__"):
                f.__kwdefaults__ = None

        # positional defaults: build list in order of positional parameters that have default
        pos_params = [p for p in params
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        pos_defaults_params = [p for p in pos_params if p.default is not inspect._empty]

        if pos_defaults_params:
            new_defaults = []
            for p in pos_defaults_params:
                if p.name in pos_updates:
                    new_defaults.append(pos_updates[p.name])
                else:
                    new_defaults.append(p.default)
            f.__defaults__ = tuple(new_defaults)
        else:
            # there are no positional defaults -> ensure attribute None
            f.__defaults__ = None

        return f

    # --- Non-inplace: create wrapper that applies defaults without touching the original func ---
    def _make_wrapper(f: Callable) -> Callable:
        # prepare new default values for signing
        new_params = []
        for p in params:
            if p.name in kwargs:
                # replace only the default (do not change the kind)
                new_params.append(p.replace(default=kwargs[p.name]))
            else:
                new_params.append(p)
        new_sig = sig.replace(parameters=new_params)

        @functools.wraps(f)
        def wrapper(*a, **kw):
            # Ensure defaults are applied if not passed by the caller
            for name, val in kwargs.items():
                if name not in kw:
                    kw[name] = val
            return f(*a, **kw)

        # adjust wrapper signature to reflect new defaults
        wrapper.__signature__ = new_sig

        # also set __kwdefaults__ for compatibility (optional)
        kwdefaults_for_wrapper = {k: kwargs[k] for k in kwonly_updates} if kwonly_updates else {}
        if kwdefaults_for_wrapper:
            wrapper.__kwdefaults__ = kwdefaults_for_wrapper

        return wrapper

    if inplace:
        # apply directly to the function (or underlying function, if it is a method)
        modified = _apply_inplace(func)
        # if it is a bound method, return the bound method original_obj (which already references the modified function)
        return original_obj if is_bound_method else modified
    else:
        return _make_wrapper(func)



def monotonic_indices(x):
    """
    Return the indices of a strictly increasing subsequence of `x`
    starting at the first element and ending at the last element.

    Parameters
    ----------
    x : array_like
        Input sequence.

    Returns
    -------
    ndarray
        Indices of elements forming a monotonic increasing subsequence.

    Notes
    -----
    - If the sequence decreases from start to end, it is temporarily reversed
      and indices are mapped back before returning.
    - Guarantees inclusion of the first and last index.
    """
    x = np.asarray(x)

    # If sequence decreases overall, reverse it to simplify logic
    if x[0] > x[-1]:
        x = x[::-1]
        reversed_order = True
    else:
        reversed_order = False

    indices = [0]
    for i in range(1, len(x) - 1):
        if x[i] > x[indices[-1]] and x[i] < x[-1]:
            indices.append(i)
    indices.append(len(x) - 1)

    indices = np.array(indices)
    if reversed_order:
        indices = (len(x) - 1) - indices

    return indices


def clamp_val(x, a, b):
    """
    Clamp values of `x` to be within the closed interval [a, b].

    Parameters
    ----------
    x : array_like
        Input values.
    a, b : array_like
        Interval bounds. Can be scalars or arrays, must be broadcastable with `x`.

    Returns
    -------
    ndarray
        Values of `x` clipped to lie between min(a, b) and max(a, b).

    Examples
    --------
    >>> clamp_val([1, 5, 10], 3, 8)
    array([3, 5, 8])
    >>> clamp_val([1, 5, 10], 8, 3)  # reversed bounds
    array([3, 5, 8])
    """
    x = np.asarray(x)
    lower = np.minimum(a, b)
    upper = np.maximum(a, b)
    return np.clip(x, lower, upper)


####################################################################################
# Numerical integration - Functions to evaluate the integrals and solve EDO problems
####################################################################################

def _rkck(y: np.ndarray, dydt: np.ndarray, t: float,f: Callable, dt: float, args: tuple = ()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Cash-Karp 5th-order Runge-Kutta step.

    Parameters
    ----------
    y : array_like
        Current state at time `t`.
    dydt : array_like
        Derivative `dy/dt` at (y, t), usually `f(y, t)`.
    t : float
        Current integration variable.
    f : callable
        Derivative function with signature f(y, t, *args).
    dt : float
        Step size.
    args : tuple, optional
        Extra arguments for f.

    Returns
    -------
    dyout : array_like
        Increment in `y` after one RKCK(5) step.
    yerr : array_like
        Estimated local truncation error (difference between 5th and 4th order).
    """

    # Coefficients (Cash–Karp)
    a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
    b21 = 0.2
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 0.3, -0.9, 1.2
    b51, b52, b53, b54 = -11/54, 2.5, -70/27, 35/27
    b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096

    c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771

    # Error coefficients (difference between 5th and 4th order)
    dc1 = c1 - 2825/27648
    dc3 = c3 - 18575/48384
    dc4 = c4 - 13525/55296
    dc5 = -277/14336
    dc6 = c6 - 0.25

    # Runge-Kutta stages
    k2 = f(y + dt*b21*dydt, t + a2*dt, *args)
    k3 = f(y + dt*(b31*dydt + b32*k2), t + a3*dt, *args)
    k4 = f(y + dt*(b41*dydt + b42*k2 + b43*k3), t + a4*dt, *args)
    k5 = f(y + dt*(b51*dydt + b52*k2 + b53*k3 + b54*k4), t + a5*dt, *args)
    k6 = f(y + dt*(b61*dydt + b62*k2 + b63*k3 + b64*k4 + b65*k5), t + a6*dt, *args)

    # 5th order solution
    dyout = dt * (c1*dydt + c3*k3 + c4*k4 + c6*k6)

    # Error estimate (difference 5th - 4th order)
    yerr = dt * (dc1*dydt + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6)

    return dyout, yerr


class IntegrationError(Exception):
    """Custom error for numerical integration failures (used in rkqs)."""
    pass

_rkqs_rval = namedtuple("rkqs_rval", ["Delta_y", "Delta_t", "dtnxt"])

def rkqs(y: np.ndarray, dydt: np.ndarray, t: float, f: callable, dt_try: float, epsfrac: float, epsabs: float, args: tuple = ()) -> _rkqs_rval:
    """
    Perform one adaptive 5th-order Runge-Kutta-Cash-Karp step with error control.

    Parameters
    ----------
    y : array_like
        Current state at time `t`.
    dydt : array_like
        Derivative `dy/dt` at (y, t), usually `f(y, t)`.
    t : float
        Current integration variable.
    f : callable
        Derivative function, must have signature f(y, t, *args).
    dt_try : float
        Initial guess for the step size.
    epsfrac : float
        Relative error tolerance.
    epsabs : float
        Absolute error tolerance.
    args : tuple, optional
        Extra arguments to pass to f.

    Returns
    -------
    _rkqs_rval
        Named tuple with:
        - Delta_y : array_like, increment in y
        - Delta_t : float, actual step size taken
        - dtnxt   : float, suggested next step size

    Raises
    ------
    IntegrationError
        If step size underflows (too small to represent).
    """
    dt = dt_try
    eps = np.finfo(float).eps  # machine epsilon

    while True:
        # Single RKCK step
        dy, yerr = _rkck(y, dydt, t, f, dt, args)

        # Compute normalized error (max over components)
        denom = np.maximum(np.abs(y), eps) * epsfrac
        err_ratio = np.abs(yerr) / np.maximum(epsabs, denom)
        errmax = np.max(err_ratio)

        if errmax < 1.0:
            # Step succeeded
            break

        # Reduce step size and retry
        dttemp = 0.9 * dt * errmax**-0.25
        dt = max(dttemp, 0.1 * dt) if dt > 0 else min(dttemp, 0.1 * dt)

        if t + dt == t:
            raise IntegrationError(
                f"Step size underflow at t={t:.6e}, dt={dt:.6e}"
            )

    # Estimate next step
    if errmax > 1.89e-4:
        dtnext = 0.9 * dt * errmax**-0.2
    else:
        dtnext = 5.0 * dt

    return _rkqs_rval(dy, dt, dtnext)

###############################################################
# Numerical derivatives - Functions to evaluate the derivatives
###############################################################

# -----------------------------
# Finite-difference weight core
# -----------------------------
def fd_weights_1d(x_nodes: np.ndarray, x0: float, der: int) -> np.ndarray:
    """
    Compute 1D finite-difference weights for the `der`-th derivative at `x0`,
    given arbitrary (distinct) stencil nodes `x_nodes`, using Fornberg's algorithm.

    Parameters
    ----------
    x_nodes : (m, ) array_like
        Stencil nodes (distinct x-values), not necessarily uniform or ordered.
    x0 : float
        Expansion point where the derivative is approximated.
    der : int
        Derivative order (1 for first derivative, 2 for second derivative).

    Returns
    -------
    w : (m,) ndarray
        Weights such that f^(der)(x0) ≈ sum_j w[j] * f(x_nodes[j]).

    Notes
    -----
    - This is the standard Fornberg algorithm (see B. Fornberg, 1988, 1998).
    - Exact for all polynomials up to degree m-1; accuracy on smooth functions
      is typically O(h^{m-der}) on near-uniform meshes.
    """
    x = np.asarray(x_nodes, dtype=float)
    m = x.size
    if der < 0:
        raise ValueError("`der` must be nonnegative (1 or 2).")
    if m < der + 1:
        raise ValueError("Need at least der+1 stencil nodes.")
    # c[j,k] -> coefficient for node j, derivative order k (k=0..der)
    c = np.zeros((m, der + 1), dtype=float)
    c[0, 0] = 1.0
    c1 = 1.0
    c4 = x[0] - x0
    for i in range(1, m):
        mn = min(i, der)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            if c3 == 0.0:
                raise ZeroDivisionError("Stencil nodes must be distinct.")
            c2 *= c3
            # update the new row i
            for k in range(mn, 0, -1):
                c[i, k] = (c1 * (k * c[i-1, k-1] - c5 * c[i-1, k])) / c2
            c[i, 0] = (-c1 * c5 * c[i-1, 0]) / c2
            # update previous rows j
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k-1]) / c3
            c[j, 0] = (c4 * c[j, 0]) / c3
        c1 = c2
    return c[:, der]

# -----------------------------
# Helper: pick a length-m stencil around index k
# -----------------------------
def _stencil_indices(n: int, k: int, m: int) -> np.ndarray:
    """
    Choose a length-m stencil around index k within [0, n-1], preferably centered.
    Falls back to left-/right-sided windows near boundaries.

    Returns
    -------
    idx : (m,) ndarray of ints
    """
    half = m // 2
    start = k - half
    # For even m, this centers slightly to the left; acceptable and symmetric enough.
    if start < 0:
        start = 0
    if start + m > n:
        start = n - m
    return np.arange(start, start + m)


# -------------------------------------------------------------
# First derivative: 5-point (order ~4 inside), non-uniform x
# -------------------------------------------------------------
def deriv14(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    First derivative along the last axis using a 5-point finite-difference stencil.
    Fourth-order accurate in the interior on non-uniform grids; one-sided high-order
    stencils near boundaries.

    Parameters
    ----------
    y : array_like
        Values sampled at x; derivative is taken along the last axis (..., n).
    x : (n,) array_like
        Sample locations (strictly monotonic). At least 5 points.

    Returns
    -------
    dy : ndarray
        Same shape as y; dy/dx along the last axis.

    Notes
    -----
    - Uses Fornberg weights for each local 5-point stencil.
    - Interior points (k=2...n-3) use centered [k-2...k+2].
    - Boundaries (k=0,1 and k=n-2,n-1) use one-sided windows.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    n = x.size
    if n < 5:
        raise ValueError("deriv14 requires at least 5 samples.")
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        raise ValueError("x must be strictly monotonic (increasing or decreasing).")

    dy = np.empty_like(y, dtype=float)

    # Boundaries: one-sided 5-point stencils
    left = np.arange(5)
    right = np.arange(n-5, n)
    # k = 0, 1
    for k in (0, 1):
        w = fd_weights_1d(x[left], x[k], der=1)
        dy[..., k] = np.tensordot(y[..., left], w, axes=([-1], [0]))
    # k = n-2, n-1
    for k in (n-2, n-1):
        w = fd_weights_1d(x[right], x[k], der=1)
        dy[..., k] = np.tensordot(y[..., right], w, axes=([-1], [0]))

    # Interior: centered 5-point stencils
    for k in range(2, n-2):
        idx = np.arange(k-2, k+3)
        w = fd_weights_1d(x[idx], x[k], der=1)
        dy[..., k] = np.tensordot(y[..., idx], w, axes=([-1], [0]))

    return dy


# -------------------------------------------------------------
# First derivative: 5-point (order ~4), uniform spacing fast-path
# -------------------------------------------------------------
def deriv14_const_dx(y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    First derivative along the last axis with a uniform grid using 5-point
    fourth-order formulas (fast path).

    Parameters
    ----------
    y : array_like
        Values sampled on a uniform grid along the last axis (..., n).
    dx : float, optional
        Uniform spacing.

    Returns
    -------
    dy : ndarray
        Same shape as y; ∂y/∂x along the last axis.

    Notes
    -----
    - Interior (k=2..n-3): central 5-point stencil
      f'(x_k) ≈ (f_{k-2} - 8 f_{k-1} + 8 f_{k+1} - f_{k+2}) / (12 h)
    - Boundaries: one-sided 5-point stencils (standard coefficients).
    """
    y = np.asarray(y, dtype=float)
    if y.shape[-1] < 5:
        raise ValueError("deriv14_const_dx requires at least 5 samples along the last axis.")
    h = float(dx)
    dy = np.empty_like(y, dtype=float)

    # Interior (vectorized along the last axis)
    dy[..., 2:-2] = -(
        - y[..., :-4] + 8.0 * y[..., 1:-3]
        - 8.0 * y[..., 3:-1] + y[..., 4:]
    ) / (12.0 * h)

    # Left boundary (k=0,1)
    dy[..., 0] = (
        -25.0 * y[..., 0] + 48.0 * y[..., 1] - 36.0 * y[..., 2]
        + 16.0 * y[..., 3] - 3.0 * y[..., 4]
    ) / (12.0 * h)
    dy[..., 1] = (
        -3.0 * y[..., 0] - 10.0 * y[..., 1] + 18.0 * y[..., 2]
        - 6.0 * y[..., 3] + 1.0 * y[..., 4]
    ) / (12.0 * h)

    # Right boundary (k=n-2, n-1)
    dy[..., -2] = (
         3.0 * y[..., -1] + 10.0 * y[..., -2] - 18.0 * y[..., -3]
        + 6.0 * y[..., -4] - 1.0 * y[..., -5]
    ) / (12.0 * h)
    dy[..., -1] = (
         25.0 * y[..., -1] - 48.0 * y[..., -2] + 36.0 * y[..., -3]
        - 16.0 * y[..., -4] + 3.0 * y[..., -5]
    ) / (12.0 * h)

    return dy

# -------------------------------------------------------------
# Second derivative: 5-point (order ~3/4), non-uniform x
# -------------------------------------------------------------
def deriv23(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Second derivative along the last axis using a 5-point finite-difference stencil.
    Third- to fourth-order accurate depending on local spacing; exact for polynomials
    up to degree 4 on uniform meshes (interior).

    Parameters
    ----------
    y : array_like
        Values sampled at x; second derivative along the last axis (..., n).
    x : (n,) array_like
        Sample locations (strictly monotonic). At least 5 points.

    Returns
    -------
    d2y : ndarray
        Same shape as y; ∂²y/∂x² along the last axis.

    Notes
    -----
    - Uses Fornberg weights with `der=2`.
    - Interior (k=2..n-3): centered [k-2..k+2]; boundaries: one-sided.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    n = x.size
    if n < 5:
        raise ValueError("deriv23 requires at least 5 samples.")
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        raise ValueError("x must be strictly monotonic (increasing or decreasing).")

    d2y = np.empty_like(y, dtype=float)

    # Boundaries
    left = np.arange(5)
    right = np.arange(n-5, n)
    for k in (0, 1):
        w = fd_weights_1d(x[left], x[k], der=2)
        d2y[..., k] = np.tensordot(y[..., left], w, axes=([-1], [0]))
    for k in (n-2, n-1):
        w = fd_weights_1d(x[right], x[k], der=2)
        d2y[..., k] = np.tensordot(y[..., right], w, axes=([-1], [0]))

    # Interior
    for k in range(2, n-2):
        idx = np.arange(k-2, k+3)
        w = fd_weights_1d(x[idx], x[k], der=2)
        d2y[..., k] = np.tensordot(y[..., idx], w, axes=([-1], [0]))

    return d2y

# -------------------------------------------------------------
# Second derivative: 5-point (order ~4), uniform spacing fast-path
# -------------------------------------------------------------
def deriv23_const_dx(y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Second derivative along the last axis with a uniform grid using 5-point
    high-order formulas (fast path).

    Parameters
    ----------
    y : array_like
        Values on a uniform grid along the last axis (..., n).
    dx : float, optional
        Uniform spacing.

    Returns
    -------
    d2y : ndarray
        Same shape as y; ∂²y/∂x² along the last axis.

    Notes
    -----
    - Interior (k=2..n-3): central 5-point stencil
      f''(x_k) ≈ (-f_{k-2} + 16 f_{k-1} - 30 f_k + 16 f_{k+1} - f_{k+2}) / (12 h^2)
    - Boundaries: one-sided 5-point stencils (standard coefficients).
    - FIX compared to legacy: divide by 12*dx**2 (legacy code divided by 12*dx).
    """
    y = np.asarray(y, dtype=float)
    if y.shape[-1] < 5:
        raise ValueError("deriv23_const_dx requires at least 5 samples along the last axis.")
    h = float(dx)
    d2y = np.empty_like(y, dtype=float)

    # Interior
    d2y[..., 2:-2] = (
        - y[..., :-4] + 16.0 * y[..., 1:-3] - 30.0 * y[..., 2:-2]
        + 16.0 * y[..., 3:-1] - y[..., 4:]
    ) / (12.0 * h * h)

    # Left boundary (k=0,1)
    d2y[..., 0] = (
        35.0 * y[..., 0] - 104.0 * y[..., 1] + 114.0 * y[..., 2]
        - 56.0 * y[..., 3] + 11.0 * y[..., 4]
    ) / (12.0 * h * h)
    d2y[..., 1] = (
        11.0 * y[..., 0] - 20.0 * y[..., 1] + 6.0 * y[..., 2]
        + 4.0 * y[..., 3] - 1.0 * y[..., 4]
    ) / (12.0 * h * h)

    # Right boundary (k=n-2, n-1)
    d2y[..., -2] = (
        11.0 * y[..., -1] - 20.0 * y[..., -2] + 6.0 * y[..., -3]
        + 4.0 * y[..., -4] - 1.0 * y[..., -5]
    ) / (12.0 * h * h)
    d2y[..., -1] = (
        35.0 * y[..., -1] - 104.0 * y[..., -2] + 114.0 * y[..., -3]
        - 56.0 * y[..., -4] + 11.0 * y[..., -5]
    ) / (12.0 * h * h)

    return d2y

# -------------------------------------------------------------
# General first derivative with (n+1)-point stencil on non-uniform x
# -------------------------------------------------------------
def deriv1n(y: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """
    First derivative along the last axis using an (n+1)-point stencil
    on a non-uniform grid (Fornberg weights). Interior uses (approximately)
    centered windows; near boundaries uses one-sided windows.

    Parameters
    ----------
    y : array_like
        Values sampled at x; derivative along the last axis (..., N).
    x : (N,) array_like
        Strictly monotonic sample locations.
    n : int
        Desired accuracy order in Δx; equivalently, stencil size m = n+1 (m >= 5 recommended).
        For derivative order 1, polynomial exactness is up to degree m-1 and the local
        truncation error is typically O(h^{m-1-1}) = O(h^{n-1}) on near-uniform grids.

    Returns
    -------
    dy : ndarray
        Same shape as y; ∂y/∂x along the last axis.

    Notes
    -----
    - For n=4, this reduces to the 5-point case (like deriv14).
    - Large n implies large stencils; Fornberg may become ill-conditioned on widely spaced nodes.
      Values in the range 4 ≤ n ≤ 8 are usually safe and effective.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    N = x.size
    if n < 2:
        raise ValueError("n must be at least 2 (stencil of size n+1 >= 3).")
    m = n + 1
    if N < m:
        raise ValueError("Not enough points: need at least n+1 samples in x.")
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        raise ValueError("x must be strictly monotonic (increasing or decreasing).")

    dy = np.empty_like(y, dtype=float)
    for k in range(N):
        idx = _stencil_indices(N, k, m)
        w = fd_weights_1d(x[idx], x[k], der=1)
        dy[..., k] = np.tensordot(y[..., idx], w, axes=([-1], [0]))

    return dy

# ----------------------
# Class GradientFunction
# ----------------------

ArrayLike = Union[np.ndarray, float]

class gradientFunction:
    """
    Create a callable that returns the gradient of a scalar function f: R^N -> R
    using finite differences of order 2 or 4, with per-dimension steps `eps`.

    Parameters
    ----------
    f : callable
        Scalar function. It must accept an array of points with shape (..., Ndim)
        and return an array of shape (...) (scalar per point).
    eps : float or array_like
        Finite-difference step. If scalar, it is broadcast to all Ndim.
        If array-like, length must be Ndim.
    Ndim : int
        Number of dimensions of the input points.
    order : {2, 4}, optional
        Finite-difference accuracy order (default 4).

    Notes
    -----
    - Evaluates f in a batched way at `order * Ndim` displaced points per call.
    - The gradient is computed along the last axis of the input x.

    Example
    -------
    >>> def f(X):  # X shape (..., 2) -> scalar
    ...     x, y = np.moveaxis(X, -1, 0)
    ...     return (x*x + x*y + 3*y*y*y)
    >>> df = gradientFunction(f, eps=1e-3, Ndim=2, order=4)
    >>> df([[0,0],[0,1],[1,0],[1,1]])
    array([[ 0.,  0.],
           [ 1.,  9.],
           [ 2.,  1.],
           [ 3., 10.]])
    """

    def __init__(self, f: Callable, eps: ArrayLike, Ndim: int, order: int = 4):
        if order not in (2, 4):
            raise ValueError("order must be 2 or 4")
        self.f = f
        self.Ndim = int(Ndim)
        # normalize eps to shape (Ndim,)
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.ndim == 0:
            eps_arr = np.full(self.Ndim, float(eps_arr))
        if eps_arr.shape != (self.Ndim,):
            raise ValueError(f"`eps` must be scalar or have shape ({self.Ndim},)")
        self.eps = eps_arr

        # Offsets and coefficients for the 1st derivative
        if order == 2:
            offsets = np.array([-1.0, 1.0])     # positions
            coeffs  = np.array([-0.5, 0.5])     # central diff / (1*eps)
        else:  # order == 4
            offsets = np.array([-2.0, -1.0, 1.0, 2.0])
            coeffs  = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0

        # Build shift tensor dx with shape (order, Ndim, Ndim)
        # Only the diagonal across the last two axes is non-zero:
        # dx[k, i, i] = offsets[k] * eps[i]
        order_len = offsets.size
        dx = np.zeros((order_len, self.Ndim, self.Ndim), dtype=float)
        dx[:, np.arange(self.Ndim), np.arange(self.Ndim)] = offsets[:, None] * self.eps[None, :]
        self._dx = dx  # shape (order, Ndim, Ndim)

        # Coefficients per (order, dimension): coeffs[k]/eps[i]
        self._coef = (coeffs[:, None] / self.eps[None, :])  # shape (order, Ndim)
        self.order = order_len

    def __call__(self, x: ArrayLike, *args, **kwargs) -> np.ndarray:
        """
        Compute the gradient at points x.

        Parameters
        ----------
        x : array_like, shape (..., Ndim)
            Points where the gradient is evaluated.

        Returns
        -------
        grad : ndarray, shape (..., Ndim)
            Gradient ∇f evaluated at x.
        """
        x = np.asarray(x, dtype=float)
        if x.shape == (self.Ndim,):
            x = x[None, ...]  # promote to (1, Ndim)
        if x.shape[-1] != self.Ndim:
            raise ValueError(f"Last axis of x must have length Ndim={self.Ndim}")

        # Broadcast x against all displaced points: (..., 1, 1, Ndim) + (order, Ndim, Ndim)
        x_exp = x[..., None, None, :]  # (..., 1, 1, Ndim)
        vals = self.f(x_exp + self._dx, *args, **kwargs)  # -> shape (..., order, Ndim)
        # Combine along the stencil axis (order)
        grad = np.sum(vals * self._coef, axis=-2)  # sum over 'order' axis -> (..., Ndim)
        return grad


# ----------------------
# Class hessianFunction
# ----------------------


class hessianFunction:
    """
    Create a callable that returns the Hessian matrix (second derivatives) of
    a scalar function f: R^N -> R using finite differences of order 2 or 4.

    Parameters
    ----------
    f : callable
        Scalar function. It must accept an array of points with shape (..., Ndim)
        and return an array of shape (...).
    eps : float or array_like
        Finite-difference step. If scalar, broadcast to all Ndim. If array-like,
        length must be Ndim.
    Ndim : int
        Number of dimensions of the input points.
    order : {2, 4}, optional
        Finite-difference accuracy order (default 4).

    Notes
    -----
    - For off-diagonal entries (i != j), uses the tensor product of two 1st-derivative
      stencils (in directions i and j).
    - For diagonal entries (i == i), uses 1D 2nd-derivative stencils.
    - Evaluations are batched to reduce Python overhead.
    """

    def __init__(self, f: Callable, eps: ArrayLike, Ndim: int, order: int = 4):
        if order not in (2, 4):
            raise ValueError("order must be 2 or 4")
        self.f = f
        self.Ndim = int(Ndim)
        eps_arr = np.asarray(eps, dtype=float)
        if eps_arr.ndim == 0:
            eps_arr = np.full(self.Ndim, float(eps_arr))
        if eps_arr.shape != (self.Ndim,):
            raise ValueError(f"`eps` must be scalar or have shape ({self.Ndim},)")
        self.eps = eps_arr
        self.order = order

        # First-derivative stencil (used to build cross-derivatives)
        if order == 2:
            off1 = np.array([-1.0,  1.0])
            c1   = np.array([-0.5,  0.5])         # / eps
            # Second-derivative 1D stencil (diagonal)
            off2 = np.array([-1.0, 0.0, 1.0])
            c2   = np.array([1.0, -2.0, 1.0])     # / eps^2
        else:
            off1 = np.array([-2.0, -1.0, 1.0, 2.0])
            c1   = np.array([1.0, -8.0, 8.0, -1.0]) / 12.0
            off2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            c2   = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0

        m1 = off1.size           # stencil length for 1st-derivative
        m2 = off2.size           # stencil length for 2nd-derivative

        # Precompute diagonal stencils (i==i): shifts and weights
        diag_shifts = []
        diag_weights = []
        for i in range(self.Ndim):
            shifts = np.zeros((m2, self.Ndim), dtype=float)
            shifts[:, i] = off2 * self.eps[i]
            # weights include division by eps^2
            w = c2 / (self.eps[i] * self.eps[i])
            diag_shifts.append(shifts)   # shape (m2, Ndim)
            diag_weights.append(w)       # shape (m2,)

        # Precompute off-diagonal stencils (i>j): tensor products of 1st-derivative stencils
        off_shifts = [[None]*self.Ndim for _ in range(self.Ndim)]
        off_weights = [[None]*self.Ndim for _ in range(self.Ndim)]
        for i in range(self.Ndim):
            for j in range(i):
                # shifts on a (m1, m1, Ndim) grid, then flatten to (m1*m1, Ndim)
                shifts = np.zeros((m1, m1, self.Ndim), dtype=float)
                shifts[:, :, i] = off1[:, None] * self.eps[i]
                shifts[:, :, j] = off1[None, :] * self.eps[j]
                shifts = shifts.reshape(m1*m1, self.Ndim)

                # weights are outer product of 1st-derivative coeffs in i and j, each divided by eps
                wi = c1 / self.eps[i]          # (m1,)
                wj = c1 / self.eps[j]          # (m1,)
                w  = np.outer(wi, wj).reshape(m1*m1)  # (m1*m1,)
                off_shifts[i][j]  = shifts
                off_weights[i][j] = w

        self._diag_shifts  = diag_shifts
        self._diag_weights = diag_weights
        self._off_shifts   = off_shifts
        self._off_weights  = off_weights

    def __call__(self, x: ArrayLike, *args, **kwargs) -> np.ndarray:
        """
        Compute the Hessian at points x.

        Parameters
        ----------
        x : array_like, shape (..., Ndim)
            Points where the Hessian is evaluated.

        Returns
        -------
        H : ndarray, shape (..., Ndim, Ndim)
            Hessian matrix at x.
        """
        x = np.asarray(x, dtype=float)
        if x.shape == (self.Ndim,):
            x = x[None, ...]
        if x.shape[-1] != self.Ndim:
            raise ValueError(f"Last axis of x must have length Ndim={self.Ndim}")

        out_shape = x.shape[:-1] + (self.Ndim, self.Ndim)
        H = np.empty(out_shape, dtype=float)

        # Off-diagonal terms (i > j), then symmetrize
        for i in range(self.Ndim):
            for j in range(i):
                shifts = self._off_shifts[i][j]      # (m1*m1, Ndim)
                w      = self._off_weights[i][j]     # (m1*m1,)
                vals   = self.f(x[..., None, :] + shifts, *args, **kwargs)  # (..., P)
                hij    = np.sum(vals * w, axis=-1)   # (...)
                H[..., i, j] = H[..., j, i] = hij

        # Diagonal terms
        for i in range(self.Ndim):
            shifts = self._diag_shifts[i]            # (m2, Ndim)
            w      = self._diag_weights[i]           # (m2,)
            vals   = self.f(x[..., None, :] + shifts, *args, **kwargs)      # (..., m2)
            H[..., i, i] = np.sum(vals * w, axis=-1)

        return H

######################################################################################
# Interpolation Functions - functions to approximate path between points by a function
######################################################################################

# -------------------------------------------------------------
# Two-point interpolation: quintic with value/1st/2nd derivatives
# -------------------------------------------------------------
def makeInterpFuncs(y0, dy0, d2y0, y1, dy1, d2y1) -> Tuple[Callable, Callable]:
    """
    Build a 5th-degree polynomial on x in [0, 1] that matches:
      f(0)=y0, f'(0)=dy0, f''(0)=d2y0,  f(1)=y1, f'(1)=dy1, f''(1)=d2y1.

    Returns
    -------
    f  : callable
        Evaluates the interpolant at x (scalar or array).
    df : callable
        Evaluates the derivative at x.

    Notes
    -----
    The polynomial is p(x) = a0 + a1 x + ... + a5 x^5.
    We set a0=y0, a1=dy0, a2=d2y0/2 and solve a 3x3 linear system for (a3,a4,a5)
    from the constraints at x=1.
    """
    # Coefficients known from x=0 constraints
    a0 = y0
    a1 = dy0
    a2 = 0.5 * d2y0

    # Right-hand side remainders at x=1 after subtracting known (a0,a1,a2)
    r1 = y1  - (a0 + a1 + a2)
    r2 = dy1 - (a1 + 2.0*a2)
    r3 = d2y1 - (2.0*a2)

    # System for [a3, a4, a5]:
    # [ 1  1   1 ] [a3] = r1
    # [ 3  4   5 ] [a4] = r2
    # [ 6 12  20 ] [a5] = r3
    A = np.array([[1.0, 1.0, 1.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 12.0, 20.0]], dtype=float)
    b = np.array([r1, r2, r3], dtype=float)
    a3, a4, a5 = np.linalg.solve(A, b)

    coefs = np.array([a0, a1, a2, a3, a4, a5], dtype=float)

    def f(x, c=coefs):
        x = np.asarray(x, dtype=float)
        # Horner for numerical stability
        return (((((c[5]*x + c[4])*x + c[3])*x + c[2])*x + c[1])*x + c[0])

    def df(x, c=coefs):
        x = np.asarray(x, dtype=float)
        # Derivative via Horner on p'(x)
        d = np.array([c[1], 2*c[2], 3*c[3], 4*c[4], 5*c[5]], dtype=float)
        return ((((d[4]*x + d[3])*x + d[2])*x + d[1])*x + d[0])

    return f, df


# -------------------------------------------------------------
# Two-point interpolation: cubic Bézier/Hermite (values + slopes)
# -------------------------------------------------------------
class cubicInterpFunction:
    """
    Cubic interpolant between two points using value and 1st derivative at the ends.

    Parameters
    ----------
    y0, dy0 : array_like
        Value and slope at t=0.
    y1, dy1 : array_like
        Value and slope at t=1.

    Notes
    -----
    Uses the Bezier representation equivalent to the Hermite form:
      P0 = y0
      P1 = y0 + (1/3) dy0
      P2 = y1 - (1/3) dy1
      P3 = y1
    Then: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    """

    def __init__(self, y0, dy0, y1, dy1):
        y0 = np.asarray(y0, dtype=float)
        dy0 = np.asarray(dy0, dtype=float)
        y1 = np.asarray(y1, dtype=float)
        dy1 = np.asarray(dy1, dtype=float)

        P0 = y0
        P1 = y0 + dy0/3.0
        P2 = y1 - dy1/3.0
        P3 = y1
        self._ctrl = (P0, P1, P2, P3)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        P0, P1, P2, P3 = self._ctrl
        mt = 1.0 - t
        return (P0*(mt**3)
              + 3.0*P1*(mt*mt*t)
              + 3.0*P2*(mt*t*t)
              + P3*(t**3))


# -------------------------------------------------------------
# B-spline basis functions and derivatives (Cox–de Boor recursion)
# -------------------------------------------------------------
def _safe_div(num, den):
    """
    Divide `num` by `den` with broadcasting, returning 0 where `den == 0`.

    Shapes:
      - `num`: any shape
      - `den`: any shape broadcastable to `num`
    Returns:
      array with broadcasted shape
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    shape = np.broadcast(num,den).shape
    out = np.zeros(shape, dtype=float)

    np.divide(num, den, out=out, where=(den !=0))

    return out

def Nbspl(t, x, k=3):
    """
    Evaluate B-spline basis functions of degree k for knot vector `t` at points `x`.

    Parameters
    ----------
    t : array_like, shape (m,)
        Knot vector (non-decreasing). The number of basis functions is m - k - 1.
    x : array_like, shape (n,)
        Evaluation points.
    k : int, optional
        Spline degree (order). Must satisfy k <= len(t) - 2.

    Returns
    -------
    N : ndarray, shape (n, m-k-1)
        Basis functions N_{i,k}(x_j).

    Notes
    -----
    - Uses Cox–de Boor recursion with a right-closed convention at each interval
      (i.e. N_{i,0}(x)=1 on (t_i, t_{i+1}] ), to mirror the legacy behavior.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if k > len(t) - 2:
        raise ValueError("Nbspl: require k <= len(t)-2")

    n = x.size
    m = t.size
    n0 = m - 1                  # number of 0-degree pieces
    nb = m - k - 1              # number of degree-k basis functions

    # Degree 0: indicator on (t_i, t_{i+1}]
    # Build matrix N0 with shape (n, n0)
    x_col = x[:, None]          # (n,1)
    N_prev = ((x_col > t[:-1]) & (x_col <= t[1:])).astype(float)  # (n, n0)

    # Recursively elevate degree up to k
    for p in range(1, k+1):
        # For degree p, there are m - p - 1 bases
        ncols = m - p - 1
        N = np.zeros((n, ncols), dtype=float)

        # denominators for left/right fractions
        left_den  = (t[p:  p+ncols] - t[:ncols])          # (ncols,)
        right_den = (t[p+1:p+1+ncols] - t[1:1+ncols])     # (ncols,)

        # broadcast x against knot vectors
        left_num  = x_col - t[:ncols]                     # (n, ncols)
        right_num = t[p+1:p+1+ncols] - x_col              # (n, ncols)

        N_left  = _safe_div(left_num,  left_den) * N_prev[:, :ncols]
        N_right = _safe_div(right_num, right_den) * N_prev[:, 1:ncols+1]
        N = N_left + N_right
        N_prev = N

    # N_prev is degree-k basis: shape (n, nb)
    return N_prev


def Nbspld1(t, x, k=3):
    """
    Same as `Nbspl` but also returns first derivatives dN/dx.

    Returns
    -------
    N  : ndarray, shape (n, m-k-1)
    dN : ndarray, shape (n, m-k-1)
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if k > len(t) - 2:
        raise ValueError("Nbspld1: require k <= len(t)-2")

    n = x.size
    m = t.size
    x_col = x[:, None]

    # Store N for all degrees up to k (we need degree k-1 for the derivative)
    N_list = []

    # Degree 0
    N0 = ((x_col > t[:-1]) & (x_col <= t[1:])).astype(float)  # (n, m-1)
    N_list.append(N0)

    # Build up to degree k
    for p in range(1, k+1):
        ncols = m - p - 1
        left_den  = (t[p:  p+ncols] - t[:ncols])
        right_den = (t[p+1:p+1+ncols] - t[1:1+ncols])
        left_num  = x_col - t[:ncols]
        right_num = t[p+1:p+1+ncols] - x_col
        N_prev = N_list[-1]
        Np = _safe_div(left_num,  left_den) * N_prev[:, :ncols] \
           + _safe_div(right_num, right_den) * N_prev[:, 1:ncols+1]
        N_list.append(Np)

    N = N_list[-1]                        # degree k
    if k == 0:
        dN = np.zeros_like(N)
        return N, dN

    Nk_1 = N_list[-2]                     # degree k-1
    nb = N.shape[1]                       # m-k-1

    # dN via closed form: dN_{i,k} = k/(t_{i+k}-t_i) N_{i,k-1} - k/(t_{i+k+1}-t_{i+1}) N_{i+1,k-1}
    a = _safe_div(k*np.ones(nb), (t[k: k+nb] - t[:nb]))                # (nb,)
    b = _safe_div(k*np.ones(nb), (t[k+1:k+1+nb] - t[1:1+nb]))          # (nb,)
    dN = (Nk_1[:, :nb] * a) - (Nk_1[:, 1:nb+1] * b)

    return N, dN


def Nbspld2(t, x, k=3):
    """
    Same as `Nbspld1` but also returns the second derivatives d²N/dx².

    Returns
    -------
    N   : ndarray, shape (n, m-k-1)
    dN  : ndarray, shape (n, m-k-1)
    d2N : ndarray, shape (n, m-k-1)
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if k > len(t) - 2:
        raise ValueError("Nbspld2: require k <= len(t)-2")

    n = x.size
    m = t.size
    x_col = x[:, None]

    # Build and store N for all degrees 0..k
    N_list = []
    N0 = ((x_col > t[:-1]) & (x_col <= t[1:])).astype(float)
    N_list.append(N0)
    for p in range(1, k+1):
        ncols = m - p - 1
        left_den  = (t[p:  p+ncols] - t[:ncols])
        right_den = (t[p+1:p+1+ncols] - t[1:1+ncols])
        left_num  = x_col - t[:ncols]
        right_num = t[p+1:p+1+ncols] - x_col
        N_prev = N_list[-1]
        Np = _safe_div(left_num,  left_den) * N_prev[:, :ncols] \
           + _safe_div(right_num, right_den) * N_prev[:, 1:ncols+1]
        N_list.append(Np)

    N = N_list[-1]
    nb = N.shape[1]

    # First derivatives for all degrees 0..k
    dN_list = [np.zeros_like(N_list[0])]
    for p in range(1, k+1):
        # dN_p from N_{p-1}
        ncols = m - p - 1
        a = _safe_div(p*np.ones(ncols), (t[p: p+ncols] - t[:ncols]))
        b = _safe_div(p*np.ones(ncols), (t[p+1:p+1+ncols] - t[1:1+ncols]))
        dN_p = (N_list[p-1][:, :ncols] * a) - (N_list[p-1][:, 1:ncols+1] * b)
        dN_list.append(dN_p)

    dN = dN_list[-1]

    # Second derivative: d2N_k from dN_{k-1}
    if k == 0:
        d2N = np.zeros_like(N)
    elif k == 1:
        # d2N_1 uses dN_0=0 → zero everywhere
        d2N = np.zeros_like(N)
    else:
        # general: d2N_{i,k} = k/(t_{i+k}-t_i) dN_{i,k-1} - k/(t_{i+k+1}-t_{i+1}) dN_{i+1,k-1}
        a2 = _safe_div(k*np.ones(nb), (t[k: k+nb] - t[:nb]))
        b2 = _safe_div(k*np.ones(nb), (t[k+1:k+1+nb] - t[1:1+nb]))
        dN_km1 = dN_list[-2]
        d2N = (dN_km1[:, :nb] * a2) - (dN_km1[:, 1:nb+1] * b2)

    return N, dN, d2N