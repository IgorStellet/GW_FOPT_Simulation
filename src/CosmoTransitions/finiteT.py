# New version of finiteT


import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Union, Tuple
from scipy import integrate, special
from scipy.interpolate import BSpline, make_interp_spline


####################################
# Exact Thermal Integrals (J_b, J_f)
####################################



# Back-compat imports/constants (from legacy)
pi = np.pi
euler_gamma = 0.577215661901532
log, exp, sqrt = np.log, np.exp, np.sqrt
array = np.array

Number = Union[float, complex, np.floating, np.complexfloating]
ArrayLike = Union[float, np.ndarray]

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

#####################################
# Spline Thermal Integrals (J_b, J_f)
#####################################

# ============================================================
# Spline Thermal Integrals (J_f): fast interpolated evaluator
# ============================================================

# Domain used historically for theta = (m/T)^2
_THETA_MIN_F = -6.82200203
_THETA_MAX_F = 1.35e3

# Where to try to cache the spline parameters (optional)
spline_data_path = os.path.dirname(__file__)
_JF_CACHE_FILE = os.path.join(spline_data_path, "Jf_spline_v1.npz")

# Internal singleton for the compiled BSpline
__JF_SPLINE: Union[None, BSpline] = None


def _build_Jf_dataset(n_neg: int = 420, n_pos_lin: int = 380, n_pos_log: int = 300
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a nonuniform theta-grid and exact values Jf_exact2(theta) for spline fit.
    - Negative branch: dense linear grid in [THETA_MIN, 0]
    - Positive small:  linear grid in [0, 50]
    - Positive tail:   log grid in (50, THETA_MAX]
    """
    th_neg = np.linspace(_THETA_MIN_F, 0.0, n_neg, dtype=float)
    th_pos_lin = np.linspace(0.0, 50.0, n_pos_lin, dtype=float)
    # to avoid 0 in logspace start slightly above 50
    th_pos_log = np.geomspace(50.0 + 1e-6, _THETA_MAX_F, n_pos_log, dtype=float)

    theta = np.unique(np.concatenate([th_neg, th_pos_lin, th_pos_log], axis=0))
    # Exact values from our scalar exact routine in theta
    y = Jf_exact2(theta).astype(float)  # returns real values for theta ∈ R
    return theta, y


def _save_Jf_cache(theta: np.ndarray, coeffs: np.ndarray, t: np.ndarray, k: int) -> None:
    try:
        np.savez(_JF_CACHE_FILE, theta=theta, coeffs=coeffs, t=t, k=np.array([k], dtype=int))
    except Exception:
        # Not fatal: just skip disk cache if path not writable
        pass


def _load_Jf_cache() -> Union[None, Tuple[np.ndarray, np.ndarray, int]]:
    if not os.path.exists(_JF_CACHE_FILE):
        return None
    try:
        data = np.load(_JF_CACHE_FILE, allow_pickle=False)
        t = data["t"]
        c = data["coeffs"]
        k = int(data["k"][0])
        return t, c, k
    except Exception:
        return None


def _ensure_Jf_spline() -> BSpline:
    """
    Ensure the global BSpline for Jf(theta) exists.
    Try to load from cache; otherwise build from exact data and cache.
    """
    global __JF_SPLINE

    if __JF_SPLINE is not None:
        return __JF_SPLINE

    # Try to load a previously cached spline
    cached = _load_Jf_cache()
    if cached is not None:
        t, c, k = cached
        __JF_SPLINE = BSpline(t, c, k, extrapolate=False)
        return __JF_SPLINE

    # Build dataset (this calls Jf_exact2 on a few hundred points)
    theta, y = _build_Jf_dataset()

    # Interpolating cubic spline (k=3) with reasonable behavior across knots
    spl: BSpline = make_interp_spline(theta, y, k=3)
    __JF_SPLINE = spl

    # Save cache (optional, best-effort)
    try:
        _save_Jf_cache(theta, spl.c, spl.t, spl.k)
    except Exception:
        pass

    return __JF_SPLINE


def Jf_spline(X: ArrayLike, n: int = 0) -> ArrayLike:
    """
    Interpolated J_f via a precomputed cubic B-spline in theta = (m/T)^2.

    Parameters
    ----------
    X : float or array-like
        Input theta values (can be negative). Scalar-in → scalar-out.
    n : int, optional (default=0)
        Derivative order with respect to theta.

    Returns
    -------
    out : float or ndarray
        Spline-evaluated J_f(theta) or its n-th theta-derivative.
        For theta < THETA_MIN: returns the clamp value at THETA_MIN.
        For theta > THETA_MAX: returns 0 (legacy behavior), also for derivatives.

    Notes
    -----
    - This is a fast surrogate for `Jf_exact2(theta)` based on an interpolating
      cubic spline fit on a non-uniform grid. It preserves legacy API/behavior.
    - Physically meaningful inputs are real theta. Complex x is not physical here.
    """
    spl = _ensure_Jf_spline()

    X_arr = np.asarray(X)
    scalar = np.ndim(X_arr) == 0
    x = X_arr.reshape(-1).astype(float)

    # Derivative spline if requested
    if n == 0:
        s = spl
    else:
        s = spl.derivative(n)

    # Evaluate where inside domain; handle clamps/extrapolation to match legacy
    out = np.empty_like(x, dtype=float)

    # Mask regions
    m_lo = x < _THETA_MIN_F
    m_hi = x > _THETA_MAX_F
    m_in = ~(m_lo | m_hi)

    # Inside: normal evaluation (no extrapolation)
    # We guard against NaN from the spline by forcing extrapolate=False above.
    out[m_in] = s(x[m_in])

    # Left clamp: constant value at theta_min (or derivative at theta_min)
    th_min_val = s(_THETA_MIN_F)
    out[m_lo] = th_min_val

    # Right tail: zero (J → 0), derivative also zero (legacy behavior)
    out[m_hi] = 0.0

    return float(out[0]) if scalar else out.reshape(X_arr.shape)

# ============================================================
# Spline Thermal Integrals (J_b): fast interpolated evaluator
# ============================================================

# Domain for theta = (m/T)^2 (legacy choices)
_THETA_MIN_B = -3.72402637
_THETA_MAX_B = 1.41e3

# Cache file (best-effort) ao lado do módulo (mesmo padrão do Jf)
_JB_CACHE_FILE = os.path.join(spline_data_path, "Jb_spline_v1.npz")

# Singleton interno com o spline compilado
__JB_SPLINE: Union[None, BSpline] = None


def _build_Jb_dataset(n_neg: int = 360, n_pos_lin: int = 360, n_pos_log: int = 300
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a nonuniform theta-grid and exact values Jb_exact2(theta) for spline fit.
    - Negative branch: dense linear grid in [THETA_MIN_B, 0]
    - Positive small:  linear grid in [0, 50]
    - Positive tail:   log grid in (50, THETA_MAX_B]
    """
    th_neg = np.linspace(_THETA_MIN_B, 0.0, n_neg, dtype=float)
    th_pos_lin = np.linspace(0.0, 50.0, n_pos_lin, dtype=float)
    th_pos_log = np.geomspace(50.0 + 1e-6, _THETA_MAX_B, n_pos_log, dtype=float)

    theta = np.unique(np.concatenate([th_neg, th_pos_lin, th_pos_log], axis=0))
    y = Jb_exact2(theta).astype(float)  # returns real values for theta ∈ R
    return theta, y


def _save_Jb_cache(theta: np.ndarray, coeffs: np.ndarray, t: np.ndarray, k: int) -> None:
    try:
        np.savez(_JB_CACHE_FILE, theta=theta, coeffs=coeffs, t=t, k=np.array([k], dtype=int))
    except Exception:
        # Not fatal: just skip disk cache if path not writable
        pass


def _load_Jb_cache() -> Union[None, Tuple[np.ndarray, np.ndarray, int]]:
    if not os.path.exists(_JB_CACHE_FILE):
        return None
    try:
        data = np.load(_JB_CACHE_FILE, allow_pickle=False)
        t = data["t"]
        c = data["coeffs"]
        k = int(data["k"][0])
        return t, c, k
    except Exception:
        return None


def _ensure_Jb_spline() -> BSpline:
    """
    Ensure the global BSpline for Jb(theta) exists.
    Try to load from cache; otherwise build from exact data and cache.
    """
    global __JB_SPLINE
    if __JB_SPLINE is not None:
        return __JB_SPLINE

    cached = _load_Jb_cache()
    if cached is not None:
        t, c, k = cached
        __JB_SPLINE = BSpline(t, c, k, extrapolate=False)
        return __JB_SPLINE

    theta, y = _build_Jb_dataset()

    # Interpolating cubic spline (k=3) with reasonable behavior across knots
    spl: BSpline = make_interp_spline(theta, y, k=3)
    __JB_SPLINE = spl

    try:
        _save_Jb_cache(theta, spl.c, spl.t, spl.k)
    except Exception:
        pass
    return __JB_SPLINE


def Jb_spline(X: ArrayLike, n: int = 0) -> ArrayLike:
    """
    Jb interpolated from a precomputed cubic B-spline in theta = (m/T)^2.
    (Legacy behavior preserved.)

    Parameters
    ----------
    X : float or array-like
        Input theta values (can be negative). Scalar-in → scalar-out.
    n : int, optional (default=0)
        Derivative order with respect to theta.

    Returns
    -------
    out : float or ndarray
        Spline-evaluated J_b(theta) or its n-th theta-derivative.
        For theta < THETA_MIN_B: returns the clamp value at THETA_MIN_B.
        For theta > THETA_MAX_B: returns 0 (and derivatives 0), legacy-consistent.

    Notes
    -----
    - Backend: scipy.interpolate.BSpline (k=3), derivatives via .derivative(n).
    - Physically meaningful inputs are real theta. For analytic-continuation
      questions, use the exact routines.
    """
    spl = _ensure_Jb_spline()

    X_arr = np.asarray(X)
    scalar = (X_arr.ndim == 0)
    x = X_arr.reshape(-1).astype(float)

    # Derivative spline if requested
    s = spl if n == 0 else spl.derivative(n)

    out = np.empty_like(x, dtype=float)

    m_lo = x < _THETA_MIN_B
    m_hi = x > _THETA_MAX_B
    m_in = ~(m_lo | m_hi)

    # Inside: normal evaluation (no extrapolation)
    out[m_in] = s(x[m_in])

    # Left clamp: constant value at theta_min (or derivative at theta_min)
    th_min_val = s(_THETA_MIN_B)
    out[m_lo] = th_min_val

    # Right clamp: constant value at theta_max (or derivative at theta_max)
    out[m_hi] = 0.0

    return float(out[0]) if scalar else out.reshape(X_arr.shape)
