# New version of finiteT


import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Union
from scipy import integrate, special


####################################
# Exact Thermal Integrals (J_b, J_f)
####################################



# Back-compat imports/constants (from legacy)
pi = np.pi
euler_gamma = 0.577215661901532
log, exp, sqrt = np.log, np.exp, np.sqrt
array = np.array
spline_data_path = os.path.dirname(__file__)

Number = Union[float, complex, np.floating, np.complexfloating]

# --------------------------
# Internal helpers
# --------------------------

def _asarray(x: ArrayLike) -> np.ndarray:
    """Convert to ndarray without copying unnecessarily."""
    return np.asarray(x)

def _is_scalar(x: ArrayLike) -> bool:
    """True if `x` is a scalar (0-d array or python/scalar numpy types)."""
    try:
        return np.ndim(x) == 0
    except Exception:
        return False

def _apply_elementwise(
    f: Callable[[Number], Number], x: ArrayLike,*,dtype: np.dtype = np.float64,) -> Union[Number, np.ndarray]:
    """
    Apply scalar function `f` element-wise to `x` (scalar or array),
    returning NaN for elements that raise exceptions (legacy behavior).
    Preserves scalar-in → scalar-out.
    """
    if _is_scalar(x):
        try:
            return f(x)  # may return float or complex
        except Exception:
            return dtype.type(np.nan) if dtype != np.complex128 else np.nan + 0j

    x_arr = _asarray(x)
    out = np.empty(x_arr.shape, dtype=dtype)
    it = np.nditer(x_arr, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        idx = it.multi_index
        try:
            out[idx] = f(x_arr[idx])
        except Exception:
            out[idx] = np.nan
        it.iternext()
    return out

# ---------------------------------------------------------------------
# Exact integrands (stable forms). E = sqrt(y^2 + x^2)
# ---------------------------------------------------------------------

def _Jf_exact_scalar(x: Number) -> Number:
    """
    Exact J_f(x) for scalar x (real or complex).
    Uses stable log1p form; handles complex branch by splitting domain
    as in the legacy implementation.
    """
    # If purely real -> integrate directly with stable log1p
    if getattr(x, "imag", 0.0) == 0:
        xr = abs(x)
        def f(y: float) -> float:
            E = sqrt(y*y + xr*xr)
            # - y^2 * log(1 + e^{-E})
            return -y*y * np.log1p(np.exp(-E))
        val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        return val

    # Complex branch: split integral as in the legacy code
    ax2 = abs(x*x)  # positive real number
    ax = np.sqrt(ax2)
    def f1(y: float) -> float:
        # -y^2 * log(1 + exp(-i*...)) -> legacy uses log(2*|cos(.)/2|)
        # Keep same identity for consistency with legacy behavior
        return -y*y * np.log(2.0 * abs(np.cos(np.sqrt(ax2 - y*y)/2.0)))
    def f2(y: float) -> float:
        E = float(np.sqrt(max(y*y - ax*ax, 0.0)))
        return -y*y * np.log1p(np.exp(-E))
    v1, _ = integrate.quad(f1, 0.0, ax, epsabs=1e-10, epsrel=1e-8, limit=200)
    v2, _ = integrate.quad(f2, ax, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
    return v1 + v2

def _Jb_exact_scalar(x: Number) -> Number:
    """
    Exact J_b(x) for scalar x (real or complex).
    Uses stable log1p(-exp(-E)) form; handles complex branch as in legacy.
    """
    if getattr(x, "imag", 0.0) == 0:
        xr = abs(x)
        def f(y: float) -> float:
            E = sqrt(y*y + xr*xr)
            # Stable: log(1 - exp(-E)) = log1p(-exp(-E))
            return y*y * np.log1p(-np.exp(-E))
        val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        return val

    ax2 = abs(x*x)
    ax = np.sqrt(ax2)
    def f1(y: float) -> float:
        return y*y * np.log(2.0 * abs(np.sin(np.sqrt(ax2 - y*y)/2.0)))
    def f2(y: float) -> float:
        E = float(np.sqrt(max(y*y - ax*ax, 0.0)))
        return y*y * np.log1p(-np.exp(-E))
    v1, _ = integrate.quad(f1, 0.0, ax, epsabs=1e-10, epsrel=1e-8, limit=200)
    v2, _ = integrate.quad(f2, ax, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
    return v1 + v2

def _Jf_exact2_scalar(theta: Number) -> float:
    """
    Exact J_f as a function of theta = x^2 (scalar).
    Mirrors legacy piecewise definition; returns real part.
    """
    th = float(np.real(theta))
    def f(y: float) -> float:
        E = sqrt(y*y + th)
        return -y*y * np.log1p(np.exp(-E))
    if th >= 0.0:
        val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        return float(val)
    else:
        # negative theta branch (legacy identity)
        def f1(y: float) -> float:
            return -y*y * np.log(2.0 * abs(np.cos(np.sqrt(-th - y*y)/2.0)))
        v2, _ = integrate.quad(f,  np.sqrt(abs(th)), np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        v1, _ = integrate.quad(f1, 0.0, np.sqrt(abs(th)), epsabs=1e-10, epsrel=1e-8, limit=200)
        return float(v1 + v2)

def _Jb_exact2_scalar(theta: Number) -> float:
    """
    Exact J_b as a function of theta = x^2 (scalar).
    Mirrors legacy piecewise definition; returns real part.
    """
    th = float(np.real(theta))
    def f(y: float) -> float:
        E = sqrt(y*y + th)
        return y*y * np.log1p(-np.exp(-E))
    if th >= 0.0:
        val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        return float(val)
    else:
        def f1(y: float) -> float:
            return y*y * np.log(2.0 * abs(np.sin(np.sqrt(-th - y*y)/2.0)))
        v2, _ = integrate.quad(f,  np.sqrt(abs(th)), np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        v1, _ = integrate.quad(f1, 0.0, np.sqrt(abs(th)), epsabs=1e-10, epsrel=1e-8, limit=200)
        return float(v1 + v2)

def _dJf_exact_scalar(x: Number) -> float:
    """
    Exact derivative dJ_f/dx (scalar).
    Uses logistic expit for numerical stability.
    """
    # by symmetry, derivative at x=0 is 0
    if x == 0 or (getattr(x, "real", x) == 0 and getattr(x, "imag", 0.0) == 0.0):
        return 0.0
    xr = float(abs(x))
    def f(y: float) -> float:
        E = sqrt(y*y + xr*xr)
        nF = special.expit(-E)       # 1/(e^E + 1)
        return y*y * nF * (xr / E)
    val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
    return float(val)

def _dJb_exact_scalar(x: Number) -> float:
    """
    Exact derivative dJ_b/dx (scalar).
    Uses 1/expm1(E) for small-E stability (bosons).
    """
    if x == 0 or (getattr(x, "real", x) == 0 and getattr(x, "imag", 0.0) == 0.0):
        return 0.0
    xr = float(abs(x))
    E_sw = 40.0  # switch where expm1(E) would overflow anyway
    def f(y: float) -> float:
        E = sqrt(y*y + xr*xr)
        if E <= E_sw:
            nB = 1.0 / np.expm1(E)       # 1/(e^E - 1)
        else:
            nB = np.exp(-E)  # asymptotic form; avoids overflow
        return y*y * nB * (xr / E)
    val, _ = integrate.quad(f, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
    return float(val)

# ---------------------------------------------------------
# Public API (names preserved). Docstrings modernized.
# ---------------------------------------------------------

def Jf_exact(x: ArrayLike) -> Union[Number, np.ndarray]:
    """
    Compute the exact one-loop fermionic thermal integral J_f(x).

    Parameters
    ----------
    x : float, complex, or array-like
        Argument(s) x. Real inputs are treated as |x| (legacy behavior).
        Complex inputs follow the legacy branch-splitting prescription.

    Returns
    -------
    out : float, complex or ndarray
        J_f(x) evaluated element-wise. Scalar-in → scalar-out.
        For array inputs, any element that fails to evaluate returns NaN.

    Notes
    -----
    Stable evaluation using log1p; complex branch matches the legacy
    formulation that uses a cosine identity on [0, |x|].
    """
    # complex dtype because the legacy version used complex for Jf_exact
    return _apply_elementwise(_Jf_exact_scalar, x, dtype=np.complex128)

def Jf_exact2(theta: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the exact J_f as a function of theta = x^2 (legacy API).

    Parameters
    ----------
    theta : float or array-like
        May be negative; in that case we use the legacy trigonometric identity.

    Returns
    -------
    out : float or ndarray
        Real part of J_f(theta). Scalar-in → scalar-out.
    """
    return _apply_elementwise(_Jf_exact2_scalar, theta, dtype=np.float64)

def Jb_exact(x: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the exact one-loop bosonic thermal integral J_b(x).

    Parameters
    ----------
    x : float, complex, or array-like
        Argument(s) x. Real inputs are treated as |x| (legacy behavior).
        Complex inputs follow the legacy branch-splitting prescription.

    Returns
    -------
    out : float or ndarray
        J_b(x) evaluated element-wise. Scalar-in → scalar-out.
        For array inputs, elements that fail return NaN.
    """
    return _apply_elementwise(_Jb_exact_scalar, x, dtype=np.float64)

def Jb_exact2(theta: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the exact J_b as a function of theta = x^2 (legacy API).

    Parameters
    ----------
    theta : float or array-like
        May be negative; in that case we use the legacy trigonometric identity.

    Returns
    -------
    out : float or ndarray
        Real part of J_b(theta). Scalar-in → scalar-out.
    """
    return _apply_elementwise(_Jb_exact2_scalar, theta, dtype=np.float64)

def dJf_exact(x: ArrayLike) -> Union[float, np.ndarray]:
    """
    Exact derivative dJ_f/dx by direct integration.

    Parameters
    ----------
    x : float or array-like

    Returns
    -------
    out : float or ndarray
        dJ_f/dx (scalar-in → scalar-out). For array inputs, elements that
        fail return NaN.

    Notes
    -----
    Uses logistic `expit(-E)` for the Fermi factor to avoid overflow.
    Returns exactly 0.0 for x == 0 to avoid a removable singularity.
    """
    return _apply_elementwise(_dJf_exact_scalar, x, dtype=np.float64)

def dJb_exact(x: ArrayLike) -> Union[float, np.ndarray]:
    """
    Exact derivative dJ_b/dx by direct integration.

    Parameters
    ----------
    x : float or array-like

    Returns
    -------
    out : float or ndarray
        dJ_b/dx (scalar-in → scalar-out). For array inputs, elements that
        fail return NaN.

    Notes
    -----
    Uses 1/expm1(E) for the Bose factor to improve stability near E≈0.
    Returns exactly 0.0 for x == 0 to avoid a removable singularity.
    """
    return _apply_elementwise(_dJb_exact_scalar, x, dtype=np.float64)

