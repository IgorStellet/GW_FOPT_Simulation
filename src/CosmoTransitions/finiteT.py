# New version of finiteT


import os
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Union, Tuple
from scipy import integrate, special
from scipy.interpolate import BSpline, make_interp_spline
from scipy.special import factorial as fac

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


#####################################
# Approx Thermal Integrals (J_b, J_f)
#####################################

# -------------------------------
# Low-x (high-T) asymptotic series
# -------------------------------

# ---- Precomputed coefficients (50 terms) ----
# Bosons
_a_b = -pi**4/45.0
_b_b =  pi*pi/12.0
_c_b = -pi/6.0
_d_b = -1.0/32.0
_logab = 1.5 - 2.0*euler_gamma + 2.0*np.log(4.0*pi)

_l = np.arange(50, dtype=int) + 1
_g_b = (-2.0 * pi**3.5 * (-1.0)**_l * (1.0 + special.zetac(2*_l + 1))
        * special.gamma(_l + 0.5) / (fac(_l + 2) * (2.0*pi)**(2*_l + 4)))
_g_b = np.asarray(_g_b, dtype=np.float64)

# Fermions
_a_f = -7.0*pi**4/360.0
_b_f =  pi*pi/24.0
_d_f =  1.0/32.0
_logaf = 1.5 - 2.0*euler_gamma + 2.0*np.log(pi)

_g_f = (0.25 * pi**3.5 * (-1.0)**_l * (1.0 + special.zetac(2*_l + 1))
        * special.gamma(_l + 0.5) * (1.0 - 0.5**(2*_l + 1))
        / (fac(_l + 2) * pi**(2*_l + 4)))
_g_f = np.asarray(_g_f, dtype=np.float64)

_MAX_LOW_TERMS = int(_g_b.size)  # = 50


def _series_tail_sum(g: np.ndarray, x2: np.ndarray, n: int) -> np.ndarray:
    """
    Compute sum_{i=1..n} g[i-1] * x^(2i+4) in a vectorized, stable way:
    x^(2i+4) = x^4 * (x^2)^i.
    """
    if n <= 0:
        return np.zeros_like(x2, dtype=np.float64)
    n = int(n)
    n = min(max(n, 0), _MAX_LOW_TERMS)
    x4 = (x2 * x2)
    # powers of x2: (x2^1, x2^2, ..., x2^n)
    pows = x2[..., None] ** np.arange(1, n + 1, dtype=np.int64)  # shape (..., n)
    # dot along last axis with g[:n] -> shape (...)
    return x4 * (pows @ g[:n])


def Jb_low(x, n: int = 20):
    r"""
    Low-x (high-T) expansion for the bosonic thermal integral \(J_b(x)\).

    Series (truncated at n terms in the tail):
    \[
    J_b(x) = -\frac{\pi^4}{45}
             + \frac{\pi^2}{12}x^2
             - \frac{\pi}{6}x^3
             - \frac{1}{32}x^4\big(\log x^2 - \mathrm{const}_b\big)
             + \sum_{i=1}^{n} g^{(b)}_i\, x^{2i+4},
    \]
    with \(\mathrm{const}_b = 1.5 - 2\gamma_E + 2\log(4\pi)\).

    Parameters
    ----------
    x : float or array-like
        Argument \(x=m/T\). Intended for \(|x|\ll 1\).
    n : int, optional (default=20)
        Number of tail terms \(\sum_{i=1}^n g_i x^{2i+4}\).
        Clipped to the available maximum (50). Must be >= 0.

    Returns
    -------
    y : float or ndarray
        Approximation to \(J_b(x)\) with the chosen truncation.

    Notes
    -----
    * Fully vectorized; scalar-in → scalar-out.
    * The \(x^4\log x^2\) term is handled with a removable-singularity
      convention: its contribution is set to 0 exactly at \(x=0\).
    * Use the **exact** or **spline** implementations outside the small-x regime.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if np.iscomplexobj(x):
        raise TypeError("Jb_low expects real x (use exact/spline for complex cases).")

    x = _asarray(x).astype(np.float64, copy=False)
    x2 = x * x

    # Polynomial core up to x^3 plus x^4 log-term
    y = (_a_b
         + _b_b * x2
         + _c_b * (x * x2))  # x^3

    # log-term: d * x^4 * (log(x^2) - const_b), with safe handling at x=0
    logx2 = np.empty_like(x, dtype=np.float64)
    mask = (x != 0.0)
    logx2[mask] = np.log(x2[mask])
    logx2[~mask] = 0.0  # will be multiplied by x^4=0 => net 0 at x=0
    y += _d_b * (x2 * x2) * (logx2 - _logab)

    # Tail sum
    y += _series_tail_sum(_g_b, x2, n)

    return float(y) if _is_scalar(x) else y


def Jf_low(x, n: int = 20):
    r"""
    Low-x (high-T) expansion for the fermionic thermal integral \(J_f(x)\).

    Series (truncated at n terms in the tail):
    \[
    J_f(x) = -\frac{7\pi^4}{360}
             + \frac{\pi^2}{24}x^2
             + \frac{1}{32}x^4\big(\log x^2 - \mathrm{const}_f\big)
             + \sum_{i=1}^{n} g^{(f)}_i\, x^{2i+4},
    \]
    with \(\mathrm{const}_f = 1.5 - 2\gamma_E + 2\log(\pi)\).

    Parameters
    ----------
    x : float or array-like
        Argument \(x=m/T\). Intended for \(|x|\ll 1\).
    n : int, optional (default=20)
        Number of tail terms \(\sum_{i=1}^n g_i x^{2i+4}\).
        Clipped to the available maximum (50). Must be >= 0.

    Returns
    -------
    y : float or ndarray
        Approximation to \(J_f(x)\) with the chosen truncation.

    Notes
    -----
    * Fully vectorized; scalar-in → scalar-out.
    * The \(x^4\log x^2\) term is handled with a removable-singularity
      convention: its contribution is set to 0 exactly at \(x=0\).
    * Use the **exact** or **spline** implementations outside the small-x regime.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if np.iscomplexobj(x):
        raise TypeError("Jf_low expects real x (use exact/spline for complex cases).")

    x = _asarray(x).astype(np.float64, copy=False)
    x2 = x * x

    # Polynomial core up to x^2 (no x^3 term for fermions)
    y = (_a_f
         + _b_f * x2)

    # log-term: d * x^4 * (log(x^2) - const_f), with safe handling at x=0
    logx2 = np.empty_like(x, dtype=np.float64)
    mask = (x != 0.0)
    logx2[mask] = np.log(x2[mask])
    logx2[~mask] = 0.0  # x=0 => contribution 0
    y += _d_f * (x2 * x2) * (logx2 - _logaf)

    # Tail sum
    y += _series_tail_sum(_g_f, x2, n)

    return float(y) if _is_scalar(x) else y


# ----------------------------------------------
# High-x (low-T) asymptotics via Bessel K
# J_b^high and J_f^high with up to 3 derivatives
#------------------------------------------------


# ---- Single-k term contributions (even/odd symmetry handled) ----
def x2K2(k: int, x):
    """
    Term:  - x^2 * K_2(k|x|) / k^2
    Limit x→0:  -2 / k^4
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("x2K2 expects real x.")
    xa = np.abs(_asarray(x)).astype(np.float64, copy=False)
    z = k * xa
    val = - (xa * xa) * special.kv(2, z) / (k * k)
    if _is_scalar(x):
        return float(-2.0 / (k**4)) if xa == 0.0 else float(val)
    out = val
    m0 = (xa == 0.0)
    if np.any(m0):
        out = out.copy()
        out[m0] = -2.0 / (k**4)
    return out

def dx2K2(k: int, x):
    """
    First derivative wrt x:
      d/dx [ -x^2 K_2(k|x|)/k^2 ]  =  sign(x) * |x|^2 * K_1(k|x|) / k
    Implemented as x*|x|/k * K_1(k|x|), which is odd in x and 0 at x=0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("dx2K2 expects real x.")
    x_arr = _asarray(x).astype(np.float64, copy=False)
    xa = np.abs(x_arr)
    z = k * xa
    val = (x_arr * xa) * special.kv(1, z) / k
    return float(val) if _is_scalar(x) else val


def d2x2K2(k: int, x):
    """
    Second derivative wrt x (even):
      d^2/dx^2 [ -x^2 K_2(k|x|)/k^2 ] = |x| * ( K_1(k|x|)/k - |x| * K_0(k|x|) )
    Limit x→0:  1 / k^2
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("d2x2K2 expects real x.")
    xa = np.abs(_asarray(x)).astype(np.float64, copy=False)
    z = k * xa
    val = xa * (special.kv(1, z) / k - xa * special.kv(0, z))
    if _is_scalar(x):
        return float(1.0 / (k**2)) if xa == 0.0 else float(val)
    out = val
    m0 = (xa == 0.0)
    if np.any(m0):
        out = out.copy()
        out[m0] = 1.0 / (k**2)
    return out


def d3x2K2(k: int, x):
    """
    Third derivative wrt x (odd):
      d^3/dx^3 [ -x^2 K_2(k|x|)/k^2 ] = x * ( |x|*k*K_1(k|x|) - 3*K_0(k|x|) )
    This is identically 0 at x=0 (factor x).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("d3x2K2 expects real x.")
    x_arr = _asarray(x).astype(np.float64, copy=False)
    xa = np.abs(x_arr)
    z = k * xa
    val = x_arr * (xa * k * special.kv(1, z) - 3.0 * special.kv(0, z))
    return float(val) if _is_scalar(x) else val


# ---- High-x sums for bosons/fermions ----
def _select_K(deriv: int):
    if deriv not in (0, 1, 2, 3):
        raise ValueError("`deriv` must be 0, 1, 2, or 3.")
    return (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]


def Jb_high(x, deriv: int = 0, n: int = 8):
    """
    Bosonic high-x (low-T) expansion:
      J_b(x) ≈ Σ_{k=1..n} T_k(x), where T_k is the k-th Bessel-K term (or its derivatives).
    Parameters
    ----------
    x : float or array-like (real)
    deriv : {0,1,2,3}, optional
        0 → J itself, 1 → dJ/dx, 2 → d²J/dx², 3 → d³J/dx³
    n : int, optional (default 8)
        Number of exponential terms to sum (positive).
    Returns
    -------
    float or ndarray
        Truncated high-x sum; scalar-in → scalar-out.
    Notes
    -----
    * Each term decays ~ e^{-k|x|}, so the series is rapidly convergent for large |x|.
    * Uses |x| inside Bessel arguments to maintain the even/odd symmetry of derivatives.
    """
    if n <= 0:
        raise ValueError("`n` must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("Jb_high expects real x.")
    K = _select_K(deriv)
    # small n (default 8): simple loop is fast, keeps memory footprint tiny
    acc = 0.0
    for k in range(1, n + 1):
        acc = acc + K(k, x)
    return float(acc) if _is_scalar(x) else _asarray(acc)


def Jf_high(x, deriv: int = 0, n: int = 8):
    """
    Fermionic high-x (low-T) expansion:
      J_f(x) ≈ Σ_{k=1..n} (-1)^{k-1} T_k(x), with the same T_k used for bosons.
    Parameters
    ----------
    x : float or array-like (real)
    deriv : {0,1,2,3}, optional
        0 → J itself, 1 → dJ/dx, 2 → d²J/dx², 3 → d³J/dx³
    n : int, optional (default 8)
        Number of exponential terms to sum (positive).
    Returns
    -------
    float or ndarray
        Truncated high-x sum; scalar-in → scalar-out.
    Notes
    -----
    * Alternating signs reflect Fermi–Dirac statistics.
    * Rapid exponential convergence for large |x|.
    """
    if n <= 0:
        raise ValueError("`n` must be a positive integer.")
    if np.iscomplexobj(x):
        raise TypeError("Jf_high expects real x.")
    K = _select_K(deriv)
    acc = 0.0
    s = 1.0
    for k in range(1, n + 1):
        acc = acc + s * K(k, x)
        s = -s
    return float(acc) if _is_scalar(x) else _asarray(acc)

#######################
# Short Hand (J_b, J_f)
#######################

# ---------------------------------------------------------
# Public convenience wrappers (preserve legacy API)
# ---------------------------------------------------------

def Jb(x, approx: str = "high", deriv: int = 0, n: int = 8):
    """
    Shorthand dispatcher for the bosonic thermal integral.

    Parameters
    ----------
    x : float, array-like
        Input argument. Interpretation depends on `approx`:
        - approx == "exact" | "low" | "high": `x` is the usual real mass ratio x = m/T.
        - approx == "spline": `x` is **theta = (m/T)^2** (can be negative to model tachyonic curvature).
          This matches the legacy behavior, where the spline was built in θ.
    approx : {"exact","high","low","spline"}, optional
        Which evaluator to use. Default: "high".
        - "exact": numerical quadrature of the defining integral (supports d/dx).
        - "low":   small-x (high-T) series (function value only).
        - "high":  large-x (low-T) Bessel-K sum (supports up to 3 derivatives).
        - "spline": cubic spline in θ (supports up to 3 derivatives; input is θ).
    deriv : int, optional
        Derivative order w.r.t. x for "exact"/"high", or w.r.t. θ for "spline".
        Allowed ranges per mode:
        - exact: 0 or 1
        - low:   0 only
        - high:  0..3
        - spline:0..3
        Default: 0.
    n : int, optional
        Truncation parameter:
        - "low": number of tail terms in the small-x series (max 50).
        - "high": number of exponential terms ∑_{k=1}^n (…).
        Ignored for "exact" and "spline". Default: 8.

    Returns
    -------
    out : float or ndarray
        J_b evaluated element-wise (scalar-in → scalar-out).

    Notes
    -----
    • The "spline" mode expects θ, not x, by design (legacy API). This allows θ<0.
    • Complex x has no physical meaning here and is not supported in this wrapper.
      Use the exact scalar routines directly if you really need complex analysis.
    """
    mode = str(approx).lower()
    if mode == "exact":
        if deriv == 0:
            return Jb_exact(x)
        if deriv == 1:
            return dJb_exact(x)
        raise ValueError("For approx='exact', 'deriv' must be 0 or 1.")
    elif mode == "spline":
        if not (0 <= int(deriv) <= 3):
            raise ValueError("For approx='spline', 'deriv' must be in {0,1,2,3}.")
        # In spline mode, x is *theta* by legacy convention
        return Jb_spline(x, n=int(deriv))
    elif mode == "low":
        if deriv != 0:
            raise ValueError("For approx='low', 'deriv' must be 0 (series gives function value only).")
        if int(n) > 50:
            raise ValueError("For approx='low', 'n' must be ≤ 50 (series length).")
        return Jb_low(x, n=int(n))
    elif mode == "high":
        if not (0 <= int(deriv) <= 3):
            raise ValueError("For approx='high', 'deriv' must be in {0,1,2,3}.")
        return Jb_high(x, deriv=int(deriv), n=int(n))
    else:
        raise ValueError("Invalid 'approx'. Choose one of: 'exact', 'high', 'low', 'spline'.")


def Jf(x, approx: str = "high", deriv: int = 0, n: int = 8):
    """
    Shorthand dispatcher for the fermionic thermal integral.

    Parameters
    ----------
    x : float, array-like
        Input argument. Interpretation depends on `approx`:
        - approx == "exact" | "low" | "high": `x` is the usual real mass ratio x = m/T.
        - approx == "spline": `x` is **theta = (m/T)^2** (can be negative to model tachyonic curvature).
          This matches the legacy behavior, where the spline was built in θ.
    approx : {"exact","high","low","spline"}, optional
        Which evaluator to use. Default: "high".
        - "exact": numerical quadrature of the defining integral (supports d/dx).
        - "low":   small-x (high-T) series (function value only).
        - "high":  large-x (low-T) Bessel-K alternating sum (supports up to 3 derivatives).
        - "spline": cubic spline in θ (supports up to 3 derivatives; input is θ).
    deriv : int, optional
        Derivative order w.r.t. x for "exact"/"high", or w.r.t. θ for "spline".
        Allowed ranges per mode:
        - exact: 0 or 1
        - low:   0 only
        - high:  0..3
        - spline:0..3
        Default: 0.
    n : int, optional
        Truncation parameter:
        - "low": number of tail terms in the small-x series (max 50).
        - "high": number of exponential terms ∑_{k=1}^n (with alternating sign).
        Ignored for "exact" and "spline". Default: 8.

    Returns
    -------
    out : float, complex, or ndarray
        J_f evaluated element-wise (scalar-in → scalar-out).
        Note: in "exact" mode we preserve the legacy complex dtype; take `.real`
        if you only need the physical (real) value for real x.

    Notes
    -----
    • The "spline" mode expects θ, not x, by design (legacy API). This allows θ<0.
    • Complex x has no physical meaning here and is not supported in this wrapper.
      Use the exact scalar routines directly if you really need complex analysis.
    """
    mode = str(approx).lower()
    if mode == "exact":
        if deriv == 0:
            return Jf_exact(x)
        if deriv == 1:
            return dJf_exact(x)
        raise ValueError("For approx='exact', 'deriv' must be 0 or 1.")
    elif mode == "spline":
        if not (0 <= int(deriv) <= 3):
            raise ValueError("For approx='spline', 'deriv' must be in {0,1,2,3}.")
        # In spline mode, x is *theta* by legacy convention
        return Jf_spline(x, n=int(deriv))
    elif mode == "low":
        if deriv != 0:
            raise ValueError("For approx='low', 'deriv' must be 0 (series gives function value only).")
        if int(n) > 50:
            raise ValueError("For approx='low', 'n' must be ≤ 50 (series length).")
        return Jf_low(x, n=int(n))
    elif mode == "high":
        if not (0 <= int(deriv) <= 3):
            raise ValueError("For approx='high', 'deriv' must be in {0,1,2,3}.")
        return Jf_high(x, deriv=int(deriv), n=int(n))
    else:
        raise ValueError("Invalid 'approx'. Choose one of: 'exact', 'high', 'low', 'spline'.")

#################################### End of FiniteT modifications ####################################