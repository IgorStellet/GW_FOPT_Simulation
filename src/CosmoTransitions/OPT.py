"""
OPT.py
================

Objective
---------
Build a clean and reliable Python module for the OPT effective potential
of the model studied in the Mathematica notebook.

Final goal
----------
Given model parameters and thermodynamic inputs (T, mu), this module should:

1. evaluate the off-shell OPT effective potential
       V_eff(phi, eta^2; T, mu),

2. solve the OPT gap equation for the variational mass
       eta^2 = eta^2(phi, T, mu),

3. build the on-shell effective potential
       V_eff(phi; T, mu) = V_eff(phi, eta^2(phi,T,mu); T, mu),

4. scan V(phi) for fixed T and mu,

5. identify the minimum/minima of the potential,

6. reproduce the Mathematica notebook results, in particular
   the final plot of section 3.

Physical notes
--------------
- The physical backbone of the module comes from section 1 of the notebook:
  the OPT effective potential, the physical gap equation for eta, and the
  stationary condition in phi.

- The numerical backbone comes from section 2:
  continuation in temperature, careful seed updates, and the distinction
  between symmetric and broken branches.

- Section 3 provides the concrete prototype for evaluating V(phi) at fixed
  T and mu through the solution of the gap equation.

Important conceptual distinction
--------------------------------
This module works with two related objects:

1. Off-shell potential:
       V_eff(phi, eta^2; T, mu)

2. On-shell potential:
       V_eff(phi; T, mu)
   obtained only after solving the gap equation for eta^2.

Implementation strategy
-----------------------
We will implement the module in layers:

1. Model parameters and numerical options
2. Thermal backend
3. Core physics equations
4. Root-solving utilities
5. Branch solvers
6. On-shell potential evaluation
7. Continuation/scans
8. Physical observables
9. Plotting/testing helpers

Notebook references
-------------------
- Section 1: defines the physical equations
- Section 2: defines the useful numerical strategy
- Section 3: provides the direct V(phi) scan prototype
"""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import mpmath as mp
from scipy.optimize import root, root_scalar
import matplotlib.pyplot as plt

try:
    from CosmoTransitions import Jb, Jf
    _HAS_CT_THERMAL = True
except Exception:
    Jb = None
    Jf = None
    _HAS_CT_THERMAL = False


# ============================================================================
# 1. Model parameters and numerical options
# ============================================================================
@dataclass(frozen=True)
class OPTPhaseState:
    """
    Stationary phase solution of the OPT effective potential.

    Parameters
    ----------
    branch : {"symmetric", "broken"}
        Name of the branch.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    phi : float
        Field value of the stationary point.
    phi2 : float
        Squared field value.
    eta2 : float
        Variational mass squared.
    veff : float
        Effective potential evaluated at the stationary point.
    F_eta : float
        Residual of the eta-gap equation at the solution.
    F_phi : float
        Residual of the factored phi-stationary equation at the solution.
        For the symmetric branch this is stored as np.nan because the
        factored broken-branch residual is not the relevant quantity at phi=0.
    """
    branch: Literal["symmetric", "broken"]
    T: float
    mu: float
    phi: float
    phi2: float
    eta2: float
    veff: float
    F_eta: float
    F_phi: float


@dataclass(frozen=True)
class OPTModelParams:
    """
    Physical parameters of the OPT model.

    Parameters
    ----------
    m2 : float
        Bare mass parameter m^2.
    lam : float
        Quartic coupling lambda.
    M : float
        Renormalization scale.
    """
    m2: float = -1.0
    lam: float = 1.0
    M: float = 1.0


@dataclass(frozen=True)
class SolverOptions:
    """
    Numerical controls used by the root solvers and scans.

    Parameters
    ----------
    root_tol : float
        Root-finding tolerance.
    max_iter : int
        Maximum number of root iterations.
    continuation : bool
        If True, reuse the previous solution as seed for the next point.
    """
    root_tol: float = 1e-12
    max_iter: int = 200
    continuation: bool = True


@dataclass(frozen=True)
class ThermalOptions:
    """
    Controls for the thermal backend.

    Parameters
    ----------
    mode_mu0 : {"auto", "notebook_highT", "ct_backend"}
        Strategy for mu = 0.
        - "notebook_highT": use the notebook high-T formulas directly.
        - "ct_backend": use the CosmoTransitions bosonic thermal backend.
        - "auto": prefer the CosmoTransitions backend when available;
          otherwise fall back to the notebook high-T formulas.
    mode_muneq0 : {"notebook_highT"}
        Strategy for mu != 0. For now, this remains the notebook
        high-T implementation.
    ct_mu0_approx : {"exact", "high"}
        Approximation passed to CosmoTransitions.Jb for mu = 0.
    ct_mu0_n : int
        Truncation parameter passed to CosmoTransitions.Jb when relevant.
    mu_zero_tol : float
        Numerical tolerance used to decide whether mu should be treated as zero.
    """
    mode_mu0: Literal["auto", "notebook_highT", "ct_backend"] = "auto"
    mode_muneq0: Literal["notebook_highT"] = "notebook_highT"
    ct_mu0_approx: Literal["exact", "high"] = "exact"
    ct_mu0_n: int = 20
    mu_zero_tol: float = 1e-14


# ============================================================================
# 2. Low-level utilities
# ============================================================================

def effective_mass_sq(eta2: float, params: OPTModelParams) -> float:
    """
    Return the effective squared mass

        Omega^2 = m^2 + eta^2.

    This quantity is the natural argument of the thermal functions and
    logarithmic terms in the notebook.

    Parameters
    ----------
    eta2 : float
        Variational squared mass eta^2.
    params : OPTModelParams
        Model parameters.

    Returns
    -------
    float
        Effective squared mass Omega^2.

    Raises
    ------
    ValueError
        If `eta2` or `params.m2` is not finite.
    """
    eta2 = float(eta2)
    m2 = float(params.m2)

    if not np.isfinite(eta2):
        raise ValueError(f"`eta2` must be finite, got eta2={eta2!r}.")
    if not np.isfinite(m2):
        raise ValueError(f"`params.m2` must be finite, got m2={m2!r}.")

    return m2 + eta2


def validate_effective_mass_sq(eta2: float, params: OPTModelParams) -> None:
    """
    Check whether Omega^2 = m^2 + eta^2 is inside the allowed domain.

    This is important because the notebook formulas involve sqrt(Omega^2)
    and log(M / sqrt(Omega^2)).

    Parameters
    ----------
    eta2 : float
        Variational squared mass eta^2.
    params : OPTModelParams
        Model parameters.

    Raises
    ------
    ValueError
        If Omega^2 <= 0, or if M <= 0, or if any relevant quantity is not finite.
    """
    m2_eff = effective_mass_sq(eta2, params)
    M = float(params.M)

    if not np.isfinite(M):
        raise ValueError(f"`params.M` must be finite, got M={M!r}.")
    if M <= 0.0:
        raise ValueError(
            f"`params.M` must be strictly positive because log(M/sqrt(Omega^2)) "
            f"is used later. Got M={M}."
        )

    if not np.isfinite(m2_eff):
        raise ValueError(
            f"Effective mass squared Omega^2 must be finite, got Omega^2={m2_eff!r}."
        )
    if m2_eff <= 0.0:
        raise ValueError(
            f"Effective mass squared Omega^2 must be strictly positive for the "
            f"notebook formulas. Got Omega^2 = m^2 + eta^2 = {m2_eff} "
            f"(with m2={params.m2}, eta2={eta2})."
        )


def thermal_variables(m2_eff: float, T: float, mu: float) -> tuple[float, float]:
    """
    Return the dimensionless variables used in the notebook high-T formulas:

        y = sqrt(m2_eff) / T,
        r = mu / sqrt(m2_eff).

    Parameters
    ----------
    m2_eff : float
        Effective squared mass Omega^2.
    T : float
        Temperature.
    mu : float
        Chemical potential.

    Returns
    -------
    tuple[float, float]
        Pair (y, r) with
        - y = sqrt(m2_eff)/T
        - r = mu/sqrt(m2_eff)

    Raises
    ------
    ValueError
        If `m2_eff <= 0`, `T <= 0`, or if any input is not finite.
    """
    m2_eff = float(m2_eff)
    T = float(T)
    mu = float(mu)

    if not np.isfinite(m2_eff):
        raise ValueError(f"`m2_eff` must be finite, got m2_eff={m2_eff!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")

    if m2_eff <= 0.0:
        raise ValueError(
            f"`m2_eff` must be strictly positive because sqrt(m2_eff) is used. "
            f"Got m2_eff={m2_eff}."
        )
    if T <= 0.0:
        raise ValueError(
            f"`T` must be strictly positive in the thermal variables. Got T={T}."
        )

    sqrt_m2_eff = np.sqrt(m2_eff)
    y = sqrt_m2_eff / T
    r = mu / sqrt_m2_eff

    return float(y), float(r)

# ============================================================================
# 3. Thermal backend
# ============================================================================

def h_e_odd(l: int, y: float, r: float) -> float:
    """
    Global notebook formula for h^(e)_{2l+1}(l, y, r).

    This is the mother formula used in section 3 to build H1, H3 and H5
    in the high-temperature expansion with explicit mu dependence.

    Parameters
    ----------
    l : int
        Integer label. In this module we only need l = 0, 1, 2 corresponding
        to H1, H3 and H5.
    y : float
        Dimensionless ratio sqrt(m2_eff) / T.
    r : float
        Dimensionless ratio mu / sqrt(m2_eff).

    Returns
    -------
    float
        Real value of the notebook high-T expression.

    Raises
    ------
    ValueError
        If the inputs are not finite, if y <= 0, or if |r| >= 1.
        The latter condition is imposed because the notebook expression
        contains powers of (1 - r^2) and is used here in its real-valued regime.
    """
    if int(l) != l or l < 0:
        raise ValueError(f"`l` must be a non-negative integer, got l={l!r}.")
    l = int(l)

    y = float(y)
    r = float(r)

    if not np.isfinite(y):
        raise ValueError(f"`y` must be finite, got y={y!r}.")
    if not np.isfinite(r):
        raise ValueError(f"`r` must be finite, got r={r!r}.")
    if y <= 0.0:
        raise ValueError(f"`y` must be strictly positive, got y={y}.")
    if abs(r) >= 1.0:
        raise ValueError(
            f"The notebook high-T formula is being used in its real-valued regime, "
            f"which requires |r| < 1. Got r={r}."
        )

    y_mp = mp.mpf(y)
    r_mp = mp.mpf(r)

    # Term 1
    term1 = (
        ((-1) ** l)
        * mp.pi
        * (1 - r_mp**2) ** (-mp.mpf("0.5") + l)
        * y_mp ** (-1 + 2 * l)
        / (2 * mp.gamma(1 + 2 * l))
    )

    # Term 2
    pfq = mp.hyper([1, 1, 1 - l], [mp.mpf("1.5"), 2], r_mp**2)
    term2 = (
        ((-1) ** l)
        * mp.power(2, -2 * l)
        * y_mp ** (2 * l)
        / (2 * mp.gamma(1 + l) ** 2)
    ) * (
        l * r_mp**2 * pfq
        + mp.log(y_mp / (4 * mp.pi))
        + mp.mpf("0.5") * (mp.euler - mp.digamma(1 + l))
    )

    # Finite sum: k = 0 ... l-1
    sum1 = mp.mpf("0")
    for k in range(0, l):
        sum1 += (
            1 / mp.gamma(1 + k)
            * ((-1) ** k)
            * mp.power(2, -2 * k)
            * y_mp ** (2 * k)
            * mp.gamma(-k + l)
            * mp.hyp2f1(-k, -k + l, mp.mpf("0.5"), r_mp**2)
            * mp.zeta(-2 * k + 2 * l)
        )
    sum1 *= 1 / (2 * mp.gamma(1 + l))

    # Finite sum: k = 1 ... 3
    sum2 = mp.mpf("0")
    for k in range(1, 4):
        coeff = (
            ((-1) ** k)
            * (y_mp / (4 * mp.pi)) ** (2 * k)
            * mp.gamma(2 * k + 1)
            * mp.zeta(2 * k + 1)
            / (mp.gamma(1 + k) * mp.gamma(1 + l + k))
        )
        sum2 += coeff * mp.hyp2f1(-k, -l - k, mp.mpf("0.5"), r_mp**2)

    term3 = (
        ((-1) ** l)
        * (y_mp / 2) ** (2 * l)
        / (2 * mp.gamma(1 + l))
        * sum2
    )

    value = term1 + term2 + sum1 + term3
    return float(mp.re(value))


def H1_notebook_highT(m2_eff: float, T: float, mu: float) -> float:
    """
    Notebook section 3 high-T approximation for H1.

    Notes
    -----
    The notebook defines H1EHT by applying a series expansion to the
    mother expression and then substituting y = sqrt(m2_eff)/T and
    r = mu/sqrt(m2_eff). Here we evaluate the corresponding high-T
    expression directly in the same variables.
    """
    y, r = thermal_variables(m2_eff, T, mu)
    return h_e_odd(0, y, r)


def H3_notebook_highT(m2_eff: float, T: float, mu: float) -> float:
    """
    Notebook section 3 high-T approximation for H3.
    """
    y, r = thermal_variables(m2_eff, T, mu)
    return h_e_odd(1, y, r)


def H5_notebook_highT(m2_eff: float, T: float, mu: float) -> float:
    """
    Notebook section 3 high-T approximation for H5.
    """
    y, r = thermal_variables(m2_eff, T, mu)
    return h_e_odd(2, y, r)


def H3_ct_mu0(
    m2_eff: float,
    T: float,
    approx: Literal["exact", "high"] = "exact",
    n: int = 8,
) -> float:
    """
    mu = 0 backend for H3 using the bosonic CosmoTransitions thermal integral.

    Physical mapping
    ----------------
    The notebook states that H5 = J0, H3 = J1, H1 = J2 in the notation
    used there. Matching the small-x expansion and the derivative identity

        dH5/d(m2_eff) = - H3 / (8 T^2)

    with the standard bosonic thermal function Jb(x), x = sqrt(m2_eff)/T,
    gives the practical relation

        H5 = - Jb(x) / 8,
        H3 = Jb'(x) / (2 x).

    Parameters
    ----------
    m2_eff : float
        Effective squared mass.
    T : float
        Temperature.
    approx : {"exact", "high"}, optional
        Approximation mode passed to CosmoTransitions.Jb.
    n : int, optional
        Truncation parameter passed to CosmoTransitions.Jb when relevant.

    Returns
    -------
    float
        H3 evaluated via the mu = 0 CosmoTransitions backend.

    Raises
    ------
    RuntimeError
        If CosmoTransitions.Jb is not available.
    ValueError
        If the inputs are outside the allowed domain.
    """
    if not _HAS_CT_THERMAL:
        raise RuntimeError(
            "CosmoTransitions thermal backend is not available in this Python "
            "environment. Install/import `CosmoTransitions` or use "
            "`mode_mu0='notebook_highT'`."
        )

    if approx not in {"exact", "high"}:
        raise ValueError(
            f"`approx` must be either 'exact' or 'high', got approx={approx!r}."
        )

    y, _ = thermal_variables(m2_eff, T, 0.0)
    x = y  # CosmoTransitions uses x = m/T

    if x <= 0.0:
        raise ValueError(f"`x = sqrt(m2_eff)/T` must be > 0, got x={x}.")

    jb_prime = Jb(x, approx=approx, deriv=1, n=int(n))
    jb_prime = float(np.real(jb_prime))

    return float(jb_prime / (2.0 * x))


def H5_ct_mu0(
    m2_eff: float,
    T: float,
    approx: Literal["exact", "high"] = "exact",
    n: int = 8,
) -> float:
    """
    mu = 0 backend for H5 using the bosonic CosmoTransitions thermal integral.

    Physical mapping
    ----------------
        H5 = - Jb(x) / 8,
    with x = sqrt(m2_eff)/T.

    Parameters
    ----------
    m2_eff : float
        Effective squared mass.
    T : float
        Temperature.
    approx : {"exact", "high"}, optional
        Approximation mode passed to CosmoTransitions.Jb.
    n : int, optional
        Truncation parameter passed to CosmoTransitions.Jb when relevant.

    Returns
    -------
    float
        H5 evaluated via the mu = 0 CosmoTransitions backend.

    Raises
    ------
    RuntimeError
        If CosmoTransitions.Jb is not available.
    ValueError
        If the inputs are outside the allowed domain.
    """
    if not _HAS_CT_THERMAL:
        raise RuntimeError(
            "CosmoTransitions thermal backend is not available in this Python "
            "environment. Install/import `CosmoTransitions` or use "
            "`mode_mu0='notebook_highT'`."
        )

    if approx not in {"exact", "high"}:
        raise ValueError(
            f"`approx` must be either 'exact' or 'high', got approx={approx!r}."
        )

    y, _ = thermal_variables(m2_eff, T, 0.0)
    x = y  # CosmoTransitions uses x = m/T

    jb_val = Jb(x, approx=approx, deriv=0, n=int(n))
    jb_val = float(np.real(jb_val))

    return float(-jb_val / 8.0)


def H3(m2_eff: float, T: float, mu: float, thermal: ThermalOptions) -> float:
    """
    Unified public interface for H3.

    Dispatch logic
    --------------
    - for mu != 0: use the notebook high-T formula,
    - for mu = 0:
        * if mode_mu0 == "notebook_highT", use the notebook backend,
        * if mode_mu0 == "ct_backend", use the CosmoTransitions Jb-based backend,
        * if mode_mu0 == "auto", prefer the CosmoTransitions backend when
          available, otherwise fall back to the notebook high-T backend.
    """
    mu = float(mu)

    if abs(mu) > thermal.mu_zero_tol:
        if thermal.mode_muneq0 != "notebook_highT":
            raise ValueError(
                f"Unsupported `mode_muneq0={thermal.mode_muneq0!r}`. "
                "Currently only 'notebook_highT' is implemented."
            )
        return H3_notebook_highT(m2_eff, T, mu)

    mode = thermal.mode_mu0
    if mode == "auto":
        mode = "ct_backend" if _HAS_CT_THERMAL else "notebook_highT"

    if mode == "notebook_highT":
        return H3_notebook_highT(m2_eff, T, 0.0)
    if mode == "ct_backend":
        return H3_ct_mu0(
            m2_eff,
            T,
            approx=thermal.ct_mu0_approx,
            n=thermal.ct_mu0_n,
        )

    raise ValueError(
        f"Unsupported `mode_mu0={thermal.mode_mu0!r}`. "
        "Choose one of: 'auto', 'notebook_highT', 'ct_backend'."
    )


def H5(m2_eff: float, T: float, mu: float, thermal: ThermalOptions) -> float:
    """
    Unified public interface for H5.
    """
    mu = float(mu)

    if abs(mu) > thermal.mu_zero_tol:
        if thermal.mode_muneq0 != "notebook_highT":
            raise ValueError(
                f"Unsupported `mode_muneq0={thermal.mode_muneq0!r}`. "
                "Currently only 'notebook_highT' is implemented."
            )
        return H5_notebook_highT(m2_eff, T, mu)

    mode = thermal.mode_mu0
    if mode == "auto":
        mode = "ct_backend" if _HAS_CT_THERMAL else "notebook_highT"

    if mode == "notebook_highT":
        return H5_notebook_highT(m2_eff, T, 0.0)
    if mode == "ct_backend":
        return H5_ct_mu0(
            m2_eff,
            T,
            approx=thermal.ct_mu0_approx,
            n=thermal.ct_mu0_n,
        )

    raise ValueError(
        f"Unsupported `mode_mu0={thermal.mode_mu0!r}`. "
        "Choose one of: 'auto', 'notebook_highT', 'ct_backend'."
    )

# ============================================================================
# 4. Core physics equations from section 1
# ============================================================================

def opt_veff_off_shell(
    phi: float,
    eta2: float,
    T: float,
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
) -> float:
    """
    Off-shell OPT effective potential

        V_eff(phi, eta^2; T, mu)

    using the physical expression extracted from section 1 of the notebook.

    Notes
    -----
    This is the raw object before solving the gap equation.

    Parameters
    ----------
    phi : float
        Background field value.
    eta2 : float
        Variational mass squared eta^2.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    params : OPTModelParams
        Physical model parameters.
    thermal : ThermalOptions
        Thermal backend options.

    Returns
    -------
    float
        Off-shell OPT effective potential.

    Raises
    ------
    ValueError
        If the physical or numerical domain is invalid.
    """
    phi = float(phi)
    eta2 = float(eta2)
    T = float(T)
    mu = float(mu)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(params.lam):
        raise ValueError(f"`params.lam` must be finite, got lam={params.lam!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    validate_effective_mass_sq(eta2, params)

    lam = float(params.lam)
    M = float(params.M)
    X = effective_mass_sq(eta2, params)
    log_term = np.log(M / np.sqrt(X))

    h3 = H3(X, T, mu, thermal)
    h5 = H5(X, T, mu, thermal)

    # Common notebook combination:
    # A = X - 16 T^2 H3 + 2 X log(M/sqrt(X))
    A = X - 16.0 * T**2 * h3 + 2.0 * X * log_term

    part1 = eta2 * A / (16.0 * np.pi**2)

    part2 = -(
        3.0 * X**2
        + 512.0 * T**4 * h5
        + 4.0 * X**2 * log_term
    ) / (64.0 * np.pi**2)

    part3 = (
        lam
        / (768.0 * np.pi**4)
        * (
            X**2
            - 32.0 * T**2 * X * h3
            + 256.0 * T**4 * h3**2
            + 4.0 * X**2 * log_term
            - 64.0 * T**2 * X * h3 * log_term
            + 4.0 * X**2 * log_term**2
        )
    )

    tree_phi2 = 0.5 * params.m2 * phi**2 - 0.5 * mu**2 * phi**2
    vertex_phi2 = -(lam * A * phi**2) / (48.0 * np.pi**2)
    tree_phi4 = lam * phi**4 / 24.0

    veff = part1 + part2 + part3 + tree_phi2 + vertex_phi2 + tree_phi4

    return float(veff)


def eta_gap_residual(
    phi: float,
    eta2: float,
    T: float,
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
) -> float:
    """
    Physical gap equation for eta^2, written as a residual:

        F_eta(phi, eta^2; T, mu) = 0.

    This follows the physical final form adopted in the notebook, rather than
    the naive symbolic PMS route.

    Parameters
    ----------
    phi : float
        Background field value.
    eta2 : float
        Variational mass squared eta^2.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    params : OPTModelParams
        Physical model parameters.
    thermal : ThermalOptions
        Thermal backend options.

    Returns
    -------
    float
        Residual of the physical eta-gap equation.

    Raises
    ------
    ValueError
        If the physical or numerical domain is invalid.
    """
    phi = float(phi)
    eta2 = float(eta2)
    T = float(T)
    mu = float(mu)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(params.lam):
        raise ValueError(f"`params.lam` must be finite, got lam={params.lam!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    validate_effective_mass_sq(eta2, params)

    lam = float(params.lam)
    M = float(params.M)
    X = effective_mass_sq(eta2, params)
    log_term = np.log(M / np.sqrt(X))
    h3 = H3(X, T, mu, thermal)

    residual = (
        24.0 * np.pi**2 * eta2
        + params.m2 * lam
        + eta2 * lam
        - 16.0 * T**2 * lam * h3
        + 2.0 * params.m2 * lam * log_term
        + 2.0 * eta2 * lam * log_term
        - 8.0 * np.pi**2 * lam * phi**2
    )

    return float(residual)


def phi_stationary_residual(
    phi: float,
    eta2: float,
    T: float,
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
) -> float:
    """
    Stationary condition in phi, written as the factored notebook residual:

        F_phi(phi, eta^2; T, mu) = 0.

    Important
    ---------
    This function implements the nontrivial branch equation used in the notebook,
    i.e. the residual after factoring out the overall phi from dV/dphi.

    Therefore:
    - phi = 0 is the symmetric stationary branch and should be handled
      separately when appropriate;
    - for phi != 0, this residual is the relevant equation used to search
      for the broken branch.

    The relation with the full derivative is

        dV/dphi = phi * F_phi / 24.

    Parameters
    ----------
    phi : float
        Background field value.
    eta2 : float
        Variational mass squared eta^2.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    params : OPTModelParams
        Physical model parameters.
    thermal : ThermalOptions
        Thermal backend options.

    Returns
    -------
    float
        Factored stationary residual for the nontrivial phi-branch.

    Raises
    ------
    ValueError
        If the physical or numerical domain is invalid.
    """
    phi = float(phi)
    eta2 = float(eta2)
    T = float(T)
    mu = float(mu)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(params.lam):
        raise ValueError(f"`params.lam` must be finite, got lam={params.lam!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    validate_effective_mass_sq(eta2, params)

    lam = float(params.lam)
    M = float(params.M)
    X = effective_mass_sq(eta2, params)
    log_term = np.log(M / np.sqrt(X))
    h3 = H3(X, T, mu, thermal)

    residual = (
        24.0 * params.m2
        - 24.0 * mu**2
        + (
            16.0 * T**2 * lam * h3
            - X * lam * (1.0 + 2.0 * log_term)
        ) / (np.pi**2)
        + 4.0 * lam * phi**2
    )

    return float(residual)

# ============================================================================
# 5. Root-solving helpers
# ============================================================================
# ============================================================================
# 5. Root-solving helpers
# ============================================================================

def _eta2_physical_lower_bound(
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    safety: float = 1e-12,
) -> float:
    """
    Return the strict lower bound for eta^2 imposed by the currently active
    thermal backend.

    Physical logic
    --------------
    1. eta^2 is a squared quantity, so eta^2 >= 0.
    2. The notebook formulas require Omega^2 = m^2 + eta^2 > 0.
    3. For the notebook high-T backend with explicit mu dependence, the
       implementation of h_e_odd is used in its real-valued regime, which
       requires

           |r| = |mu| / sqrt(Omega^2) < 1
           <=> Omega^2 > mu^2
           <=> eta^2 > mu^2 - m^2.

    Parameters
    ----------
    mu : float
        Chemical potential.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    safety : float, optional
        Small positive offset to keep the lower bound strict.

    Returns
    -------
    float
        Strict lower bound for eta^2.
    """
    mu = float(mu)
    m2 = float(params.m2)

    lower = max(0.0, -m2)

    mu_is_zero = abs(mu) <= thermal.mu_zero_tol
    if not mu_is_zero and thermal.mode_muneq0 == "notebook_highT":
        lower = max(lower, mu**2 - m2)

    return float(lower + safety)

#################################################################
def _default_eta2_seed(
    phi: float,
    T: float,
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
) -> float:
    """
    Build a deterministic default seed for eta^2.

    This is not meant to be a physical statement by itself. It is only a
    robust numerical starting point when the user does not provide a seed.

    Parameters
    ----------
    phi : float
        Background field value.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.

    Returns
    -------
    float
        A positive eta^2 seed safely inside the physical domain.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)

    eta2_min = _eta2_physical_lower_bound(mu, params, thermal)

    thermal_scale = max(1.0, 0.25 * T**2)
    field_scale = 0.5 * abs(params.lam) * phi**2
    mass_scale = abs(params.m2) + mu**2

    guess = eta2_min + thermal_scale + field_scale + mass_scale + 1e-6
    return float(max(guess, eta2_min + 1e-6))


def _default_phi2_seed(
    mu: float,
    params: OPTModelParams,
    floor: float = 1e-4,
) -> float:
    """
    Build a deterministic default seed for phi^2 on the broken branch.

    The estimate is based on the tree-level broken minimum of

        V_tree(phi) = 1/2 (m2 - mu^2) phi^2 + lam/24 phi^4,

    which gives

        phi^2 ~ 6 (mu^2 - m2) / lam

    whenever the quadratic coefficient favors symmetry breaking.

    Parameters
    ----------
    mu : float
        Chemical potential.
    params : OPTModelParams
        Model parameters.
    floor : float, optional
        Minimum positive seed.

    Returns
    -------
    float
        Positive seed for phi^2.
    """
    mu = float(mu)
    lam = float(params.lam)

    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(lam):
        raise ValueError(f"`params.lam` must be finite, got lam={lam!r}.")
    if lam == 0.0:
        raise ValueError("`params.lam` must be nonzero to build a broken-branch seed.")

    tree_phi2 = 6.0 * max(mu**2 - params.m2, 0.0) / abs(lam)
    return float(max(floor, tree_phi2, 1.0))

def _build_eta2_search_grid(
    eta2_min: float,
    center: float,
    upper: float,
    n_near: int = 500,
    n_far: int = 700,
) -> np.ndarray:
    """
    Build a search grid for eta^2 with extra resolution near the physical lower bound.
    """
    eta2_min = float(eta2_min)
    center = float(center)
    upper = float(upper)

    if upper <= eta2_min:
        upper = eta2_min + 1.0

    near_span = max(1.0, 2.0 * max(center - eta2_min, 1e-8))
    near = eta2_min + np.geomspace(1e-12, near_span, int(n_near))

    far_start = max(near[-1], eta2_min + 1e-6)
    if upper <= far_start:
        far = np.array([far_start], dtype=float)
    else:
        far = np.linspace(far_start, upper, int(n_far), dtype=float)

    grid = np.unique(np.concatenate([near, far]))
    return grid


def _deduplicate_eta2_roots(
    roots: list[float],
    rel_tol: float = 1e-8,
) -> list[float]:
    """
    Remove numerically duplicated eta^2 roots.
    """
    if not roots:
        return []

    roots_sorted = sorted(float(r) for r in roots if np.isfinite(r))
    unique_roots: list[float] = [roots_sorted[0]]

    for root in roots_sorted[1:]:
        scale = max(1.0, abs(root), abs(unique_roots[-1]))
        if abs(root - unique_roots[-1]) > rel_tol * scale:
            unique_roots.append(root)

    return unique_roots


def _find_eta2_root_candidates(
    phi: float,
    T: float,
    mu: float,
    eta2_reference: float | None,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> list[float]:
    """
    Find physical real candidate roots of the eta-gap equation.

    Strategy
    --------
    1. Try a local search around a reference point.
    2. If needed, perform a broader scan on an adaptive eta^2 grid.
    3. If needed, try a few secant solves.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)

    eta2_min = _eta2_physical_lower_bound(mu, params, thermal)
    center = (
        float(eta2_reference)
        if eta2_reference is not None and np.isfinite(eta2_reference)
        else _default_eta2_seed(phi, T, mu, params, thermal)
    )
    center = max(center, eta2_min + 1e-10)

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))

    def f_eta(e2: float) -> float:
        e2 = float(e2)
        if e2 <= eta2_min:
            return np.nan
        try:
            return float(eta_gap_residual(phi, e2, T, mu, params, thermal))
        except ValueError:
            return np.nan

    candidates: list[float] = []

    # --------------------------------------------------
    # 1. Local search around the reference point
    # --------------------------------------------------
    f_center = f_eta(center)
    if np.isfinite(f_center) and abs(f_center) < residual_tol:
        return [float(center)]

    step = max(1e-3, 0.05 * max(1.0, center))
    n_local = max(12, solver.max_iter // 8)

    for _ in range(n_local):
        left = max(eta2_min + 1e-10, center - step)
        right = center + step

        f_left = f_eta(left)
        f_right = f_eta(right)

        if np.isfinite(f_left) and np.isfinite(f_center) and f_left * f_center < 0.0:
            sol = root_scalar(
                f_eta,
                bracket=(left, center),
                method="brentq",
                xtol=solver.root_tol,
                maxiter=solver.max_iter,
            )
            if sol.converged:
                candidates.append(float(sol.root))
                break

        if np.isfinite(f_center) and np.isfinite(f_right) and f_center * f_right < 0.0:
            sol = root_scalar(
                f_eta,
                bracket=(center, right),
                method="brentq",
                xtol=solver.root_tol,
                maxiter=solver.max_iter,
            )
            if sol.converged:
                candidates.append(float(sol.root))
                break

        step *= 1.8

    if candidates:
        return _deduplicate_eta2_roots(candidates)

    # --------------------------------------------------
    # 2. Global adaptive scan
    # --------------------------------------------------
    upper = max(center + 10.0, eta2_min + 20.0)

    for _ in range(8):
        x_grid = _build_eta2_search_grid(
            eta2_min=eta2_min,
            center=center,
            upper=upper,
            n_near=max(500, 2 * solver.max_iter),
            n_far=max(700, 3 * solver.max_iter),
        )

        f_grid = np.array([f_eta(x) for x in x_grid], dtype=float)

        for i in range(len(x_grid) - 1):
            x0 = float(x_grid[i])
            x1 = float(x_grid[i + 1])
            f0 = float(f_grid[i])
            f1 = float(f_grid[i + 1])

            if np.isfinite(f0) and abs(f0) < residual_tol:
                candidates.append(x0)

            if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 < 0.0:
                try:
                    sol = root_scalar(
                        f_eta,
                        bracket=(x0, x1),
                        method="brentq",
                        xtol=solver.root_tol,
                        maxiter=solver.max_iter,
                    )
                    if sol.converged:
                        candidates.append(float(sol.root))
                except Exception:
                    pass

        if np.isfinite(f_grid[-1]) and abs(f_grid[-1]) < residual_tol:
            candidates.append(float(x_grid[-1]))

        candidates = _deduplicate_eta2_roots(candidates)
        if candidates:
            return candidates

        upper = eta2_min + 2.5 * (upper - eta2_min) + 5.0

    # --------------------------------------------------
    # 3. Final secant fallback
    # --------------------------------------------------
    secant_pairs = [
        (eta2_min + 1e-10, max(center, eta2_min + 1e-5)),
        (max(center, eta2_min + 1e-6), max(center + 0.25, eta2_min + 1e-4)),
        (max(center, eta2_min + 1e-4), max(center + 1.0, eta2_min + 1e-3)),
        (max(center, eta2_min + 1e-2), max(center + 5.0, eta2_min + 1e-1)),
    ]

    for x0, x1 in secant_pairs:
        try:
            sol = root_scalar(
                f_eta,
                x0=float(x0),
                x1=float(x1),
                method="secant",
                xtol=solver.root_tol,
                maxiter=solver.max_iter,
            )
        except Exception:
            continue

        if not sol.converged:
            continue

        root_val = float(sol.root)
        if not np.isfinite(root_val):
            continue
        if root_val <= eta2_min:
            continue

        f_root = f_eta(root_val)
        if np.isfinite(f_root) and abs(f_root) < residual_tol:
            candidates.append(root_val)

    return _deduplicate_eta2_roots(candidates)


def _eta2_root_is_disconnected(
    candidate: float,
    reference: float | None,
    T: float,
    phi: float,
) -> bool:
    """
    Heuristic continuity check used to reject clearly disconnected eta^2 roots.

    In the Mathematica notebook the relevant eta branch is followed by
    continuation. When that branch disappears, a global search may jump to a
    remote large-eta solution that is not continuously connected to the branch
    being tracked. Such jumps are precisely what can destroy the broken phase
    and incorrectly favor the symmetric branch.

    The present check is intentionally conservative: only simultaneously large
    absolute and relative jumps are rejected.
    """
    if reference is None:
        return False

    reference = float(reference)
    candidate = float(candidate)
    T = float(T)
    phi = float(phi)

    if not np.isfinite(reference) or reference <= 0.0:
        return False
    if not np.isfinite(candidate) or candidate <= 0.0:
        return True

    abs_jump = abs(candidate - reference)
    rel_jump = max(candidate / reference, reference / candidate)

    abs_cap = max(25.0, 5.0 + 2.0 * T + 2.0 * abs(phi))
    rel_cap = 25.0

    return bool(abs_jump > abs_cap and rel_jump > rel_cap)


def _select_eta2_root_candidate(
    candidates: list[float],
    phi: float,
    T: float,
    mu: float,
    eta2_reference: float | None,
    params: OPTModelParams,
    thermal: ThermalOptions,
) -> float:
    """
    Select one candidate eta^2 root.

    Selection rule
    --------------
    1. If a reference seed is available, prefer the candidate closest to it.
    2. Reject candidates that are clearly disconnected from that reference.
    3. Without a reference, choose the smallest physical real root.

    The last rule is deliberate: for seedless calls, picking the smallest root
    is much closer to the continuation logic used in the Mathematica notebook
    than minimizing the off-shell potential over eta^2, which tends to favor
    remote large-eta solutions.
    """
    if not candidates:
        raise RuntimeError("No eta^2 candidates were supplied for selection.")

    clean = sorted(float(c) for c in candidates if np.isfinite(c))
    if not clean:
        raise RuntimeError("All eta^2 candidates were non-finite.")

    if eta2_reference is not None and np.isfinite(eta2_reference):
        ordered = sorted(clean, key=lambda x: abs(float(x) - float(eta2_reference)))
        for cand in ordered:
            if not _eta2_root_is_disconnected(cand, eta2_reference, T, phi):
                return float(cand)
        raise RuntimeError(
            "All candidate eta^2 roots are disconnected from the supplied reference. "
            f"reference={eta2_reference}, candidates={ordered}."
        )

    return float(clean[0])


#################################################################


def solve_eta2_given_phi(
    phi: float,
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> float:
    """
    Solve the physical gap equation for eta^2 at fixed (phi, T, mu).

    This public version supports a seedless workflow:
    if `eta2_seed` is omitted, the routine builds a deterministic reference
    point and then searches globally for physical real candidates.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")
    if solver.root_tol <= 0.0:
        raise ValueError(f"`solver.root_tol` must be > 0, got {solver.root_tol}.")
    if solver.max_iter <= 0:
        raise ValueError(f"`solver.max_iter` must be > 0, got {solver.max_iter}.")

    eta2_min = _eta2_physical_lower_bound(mu, params, thermal)

    eta2_reference = (
        eta2_seed
        if eta2_seed is not None
        else _default_eta2_seed(phi, T, mu, params, thermal)
    )
    eta2_reference = max(float(eta2_reference), eta2_min + 1e-10)

    candidates = _find_eta2_root_candidates(
        phi=phi,
        T=T,
        mu=mu,
        eta2_reference=eta2_reference,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    if not candidates:
        raise RuntimeError(
            "Could not find a physical real eta^2 root. "
            f"Inputs: phi={phi}, T={T}, mu={mu}, "
            f"eta2_seed={eta2_seed}, eta2_min={eta2_min}."
        )

    eta2_root = _select_eta2_root_candidate(
        candidates=candidates,
        phi=phi,
        T=T,
        mu=mu,
        eta2_reference=eta2_seed,
        params=params,
        thermal=thermal,
    )

    if eta2_root <= eta2_min:
        raise RuntimeError(
            f"Selected eta^2 root outside the physical domain: "
            f"eta2={eta2_root}, eta2_min={eta2_min}."
        )

    if eta2_seed is not None and _eta2_root_is_disconnected(eta2_root, eta2_seed, T, phi):
        raise RuntimeError(
            "The eta^2 solve jumped to a disconnected root. "
            f"seed={eta2_seed}, selected={eta2_root}, phi={phi}, T={T}, mu={mu}."
        )

    f_root = eta_gap_residual(phi, eta2_root, T, mu, params, thermal)
    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))
    if not np.isfinite(f_root) or abs(f_root) >= residual_tol:
        raise RuntimeError(
            "Selected eta^2 root does not satisfy the requested residual level. "
            f"eta2={eta2_root}, F_eta={f_root:.3e}, tol={residual_tol:.3e}."
        )

    return float(eta2_root)


def solve_stationary_system(
    T: float,
    mu: float,
    eta2_seed: float,
    phi2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[float, float]:
    """
    Solve the coupled stationary system on the broken branch:

        F_eta(phi, eta^2; T, mu) = 0
        F_phi(phi, eta^2; T, mu) = 0

    solving for (eta^2, phi^2) in the spirit of section 2.

    Numerical strategy
    ------------------
    We solve for transformed variables

        eta^2 = eta2_min + exp(u),
        phi^2 = exp(v),

    so that:
    - eta^2 automatically stays in the physical domain;
    - phi^2 automatically stays positive;
    - the broken branch is naturally selected.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
    phi2_seed : float
        Initial guess for phi^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    tuple[float, float]
        (eta2, phi2) on the broken stationary branch.

    Raises
    ------
    RuntimeError
        If the coupled solve does not converge to a physical broken-branch root.
    ValueError
        If the input values are invalid.
    """
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)
    phi2_seed = float(phi2_seed)

    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
    if not np.isfinite(phi2_seed):
        raise ValueError(f"`phi2_seed` must be finite, got phi2_seed={phi2_seed!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")
    if solver.root_tol <= 0.0:
        raise ValueError(f"`solver.root_tol` must be > 0, got {solver.root_tol}.")
    if solver.max_iter <= 0:
        raise ValueError(f"`solver.max_iter` must be > 0, got {solver.max_iter}.")

    eta2_min = _eta2_physical_lower_bound(mu, params, thermal)
    eta2_seed = max(eta2_seed, eta2_min + 1e-8)
    phi2_seed = max(phi2_seed, 1e-10)

    def unpack(z: np.ndarray) -> tuple[float, float]:
        u = float(z[0])
        v = float(z[1])

        # Clip only to avoid numerical overflow in exp during iterations.
        u_clip = np.clip(u, -100.0, 100.0)
        v_clip = np.clip(v, -100.0, 100.0)

        eta2_val = eta2_min + np.exp(u_clip)
        phi2_val = np.exp(v_clip)
        return float(eta2_val), float(phi2_val)

    def system(z: np.ndarray) -> np.ndarray:
        eta2_val, phi2_val = unpack(z)
        phi_val = float(np.sqrt(phi2_val))

        try:
            f1 = eta_gap_residual(phi_val, eta2_val, T, mu, params, thermal)
            f2 = phi_stationary_residual(phi_val, eta2_val, T, mu, params, thermal)
        except ValueError:
            return np.array([1e30, 1e30], dtype=float)

        return np.array([float(f1), float(f2)], dtype=float)

    def make_seed(e2: float, p2: float) -> np.ndarray:
        return np.array(
            [
                np.log(max(e2 - eta2_min, 1e-12)),
                np.log(max(p2, 1e-12)),
            ],
            dtype=float,
        )

    seed_list = [
        make_seed(eta2_seed, phi2_seed),
        make_seed(max(1.10 * eta2_seed, eta2_min + 1e-6), max(0.50 * phi2_seed, 1e-6)),
        make_seed(max(0.95 * eta2_seed, eta2_min + 1e-6), max(1.50 * phi2_seed, 1e-6)),
    ]

    best_norm = np.inf
    best_eta2 = None
    best_phi2 = None
    best_success = False

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))

    for x0 in seed_list:
        sol = root(
            system,
            x0,
            method="hybr",
            options={"xtol": solver.root_tol, "maxfev": solver.max_iter},
        )

        eta2_val, phi2_val = unpack(sol.x)
        res = system(sol.x)
        norm = float(np.linalg.norm(res, ord=2))

        if norm < best_norm:
            best_norm = norm
            best_eta2 = eta2_val
            best_phi2 = phi2_val
            best_success = bool(sol.success)

        if sol.success and norm < residual_tol:
            best_eta2 = eta2_val
            best_phi2 = phi2_val
            best_success = True
            break

    if best_eta2 is None or best_phi2 is None:
        raise RuntimeError(
            "Broken-branch stationary solve failed before producing any candidate."
        )

    if best_phi2 <= 1e-10:
        raise RuntimeError(
            "The stationary solve collapsed to phi^2 ~ 0. "
            "This indicates the symmetric branch should be used instead."
        )

    phi_best = float(np.sqrt(best_phi2))
    f_eta_best = eta_gap_residual(phi_best, best_eta2, T, mu, params, thermal)
    f_phi_best = phi_stationary_residual(phi_best, best_eta2, T, mu, params, thermal)
    final_norm = float(np.linalg.norm([f_eta_best, f_phi_best], ord=2))

    if final_norm >= residual_tol:
        raise RuntimeError(
            "Broken-branch stationary solve did not reach the requested residual level. "
            f"Best candidate: eta2={best_eta2}, phi2={best_phi2}, "
            f"||F||={final_norm:.3e}, success={best_success}."
        )

    return float(best_eta2), float(best_phi2)


# ============================================================================
# 6. Branch solvers from section 2 logic
# ============================================================================
def solve_symmetric_branch(
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> float:
    """
    Solve the symmetric branch, defined by phi = 0.

    This public version is seedless by default.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float or None, optional
        Optional eta^2 reference for continuity. If None, an internal default is used.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    float
        Physical eta^2 solution on the symmetric branch.
    """
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    eta2_sym = solve_eta2_given_phi(
        phi=0.0,
        T=T,
        mu=mu,
        eta2_seed=eta2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    residual = eta_gap_residual(
        phi=0.0,
        eta2=eta2_sym,
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
    )

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))
    if not np.isfinite(residual) or abs(residual) >= residual_tol:
        raise RuntimeError(
            "Symmetric-branch eta solve did not reach the requested residual level. "
            f"eta2={eta2_sym}, F_eta={residual:.3e}, tol={residual_tol:.3e}."
        )

    return float(eta2_sym)


def solve_broken_branch(
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> tuple[float, float]:
    """
    Solve the broken branch using the coupled system for (eta^2, phi^2).

    This public version is seedless by default. If no seeds are provided,
    deterministic internal seeds are constructed automatically.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float or None, optional
        Initial eta^2 seed. If None, an internal default is built.
    phi2_seed : float or None, optional
        Initial phi^2 seed. If None, an internal default is built.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    tuple[float, float]
        (eta2, phi2) on the broken branch.
    """
    T = float(T)
    mu = float(mu)

    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")
    if solver.root_tol <= 0.0:
        raise ValueError(f"`solver.root_tol` must be > 0, got {solver.root_tol}.")
    if solver.max_iter <= 0:
        raise ValueError(f"`solver.max_iter` must be > 0, got {solver.max_iter}.")

    if phi2_seed is None:
        phi2_seed = _default_phi2_seed(mu, params)
    else:
        phi2_seed = float(phi2_seed)
        if not np.isfinite(phi2_seed):
            raise ValueError(f"`phi2_seed` must be finite when provided, got {phi2_seed!r}.")

    phi2_seed = max(float(phi2_seed), 1e-8)

    if eta2_seed is None:
        eta2_seed = _default_eta2_seed(
            phi=float(np.sqrt(phi2_seed)),
            T=T,
            mu=mu,
            params=params,
            thermal=thermal,
        )
    else:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))
    eta2_floor = _eta2_physical_lower_bound(mu, params, thermal) + 1e-6

    phi2_trials = [
        phi2_seed,
        max(0.25 * phi2_seed, 1e-4),
        max(0.50 * phi2_seed, 1e-4),
        max(1.50 * phi2_seed, 1e-4),
        max(2.00 * phi2_seed, 1e-4),
        0.1,
        1.0,
        4.0,
        9.0,
    ]

    trial_seeds: list[tuple[float, float]] = []
    for p2 in phi2_trials:
        p2 = max(float(p2), 1e-8)
        e2_ref = _default_eta2_seed(
            phi=float(np.sqrt(p2)),
            T=T,
            mu=mu,
            params=params,
            thermal=thermal,
        )
        e2_base = max(float(eta2_seed), e2_ref, eta2_floor)

        for fac in (0.85, 1.00, 1.15):
            trial_seeds.append((max(fac * e2_base, eta2_floor), p2))

    best_candidate = None
    best_norm = np.inf
    last_error = None

    for e2_seed_try, p2_seed_try in trial_seeds:
        try:
            eta2_broken, phi2_broken = solve_stationary_system(
                T=T,
                mu=mu,
                eta2_seed=e2_seed_try,
                phi2_seed=p2_seed_try,
                params=params,
                thermal=thermal,
                solver=solver,
            )

            phi_broken = float(np.sqrt(phi2_broken))
            f_eta = eta_gap_residual(
                phi=phi_broken,
                eta2=eta2_broken,
                T=T,
                mu=mu,
                params=params,
                thermal=thermal,
            )
            f_phi = phi_stationary_residual(
                phi=phi_broken,
                eta2=eta2_broken,
                T=T,
                mu=mu,
                params=params,
                thermal=thermal,
            )

            norm = float(np.linalg.norm([f_eta, f_phi], ord=2))

            if norm < best_norm:
                best_norm = norm
                best_candidate = (float(eta2_broken), float(phi2_broken))

            if phi2_broken > 1e-10 and norm < residual_tol:
                return float(eta2_broken), float(phi2_broken)

        except RuntimeError as exc:
            last_error = exc

    if best_candidate is not None:
        eta2_best, phi2_best = best_candidate
        if phi2_best > 1e-10 and best_norm < 10.0 * residual_tol:
            return eta2_best, phi2_best

    message = (
        "Broken-branch solve failed to produce a sufficiently accurate physical "
        f"solution at T={T}, mu={mu}, eta2_seed={eta2_seed}, phi2_seed={phi2_seed}."
    )
    if last_error is not None:
        message += f" Last error: {last_error}"

    raise RuntimeError(message)


def solve_phase_candidates_at_T(
    T: float,
    mu: float,
    sym_eta2_seed: float | None = None,
    broken_eta2_seed: float | None = None,
    broken_phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> dict[str, OPTPhaseState | None]:
    """
    Build the stationary phase candidates at fixed (T, mu).

    The routine attempts both the symmetric and broken branches. If a branch
    cannot be solved consistently, its entry is returned as None instead of
    forcing a disconnected or numerically unstable solution.
    """
    T = float(T)
    mu = float(mu)

    sym_state: OPTPhaseState | None = None
    broken_state: OPTPhaseState | None = None

    try:
        eta2_sym = solve_symmetric_branch(
            T=T,
            mu=mu,
            eta2_seed=sym_eta2_seed,
            params=params,
            thermal=thermal,
            solver=solver,
        )

        veff_sym = opt_veff_off_shell(
            phi=0.0,
            eta2=eta2_sym,
            T=T,
            mu=mu,
            params=params,
            thermal=thermal,
        )

        sym_state = OPTPhaseState(
            branch="symmetric",
            T=float(T),
            mu=float(mu),
            phi=0.0,
            phi2=0.0,
            eta2=float(eta2_sym),
            veff=float(veff_sym),
            F_eta=float(eta_gap_residual(0.0, eta2_sym, T, mu, params, thermal)),
            F_phi=float(np.nan),
        )
    except RuntimeError:
        sym_state = None

    try:
        eta2_broken, phi2_broken = solve_broken_branch(
            T=T,
            mu=mu,
            eta2_seed=broken_eta2_seed,
            phi2_seed=broken_phi2_seed,
            params=params,
            thermal=thermal,
            solver=solver,
        )

        phi_broken_val = float(np.sqrt(phi2_broken))
        veff_broken_val = opt_veff_off_shell(
            phi=phi_broken_val,
            eta2=eta2_broken,
            T=T,
            mu=mu,
            params=params,
            thermal=thermal,
        )

        broken_state = OPTPhaseState(
            branch="broken",
            T=float(T),
            mu=float(mu),
            phi=float(phi_broken_val),
            phi2=float(phi2_broken),
            eta2=float(eta2_broken),
            veff=float(veff_broken_val),
            F_eta=float(eta_gap_residual(phi_broken_val, eta2_broken, T, mu, params, thermal)),
            F_phi=float(phi_stationary_residual(phi_broken_val, eta2_broken, T, mu, params, thermal)),
        )
    except RuntimeError:
        broken_state = None

    return {
        "symmetric": sym_state,
        "broken": broken_state,
    }


def solve_physical_phase_at_T(
    T: float,
    mu: float,
    sym_eta2_seed: float | None = None,
    broken_eta2_seed: float | None = None,
    broken_phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> OPTPhaseState:
    """
    Return the physical stationary phase at fixed (T, mu).

    Seedless autonomous mode
    ------------------------
    When all seeds are omitted, the routine does *not* hard-code any critical
    temperature or branch switch. Instead it reconstructs a small temperature
    context around the requested T and uses the same continuation logic as
    `trace_physical_phases_over_T(...)`:

    - broken branch traced from low to high temperature;
    - symmetric branch traced from high to low temperature;
    - physical phase selected as the available stationary branch with the
      lowest effective potential at the requested temperature.

    This mirrors the Mathematica notebook strategy much better than an isolated
    one-shot comparison of seedless stationary candidates at a single T.

    Returns
    -------
    OPTPhaseState
        Physical stationary phase.
    """
    T = float(T)
    mu = float(mu)

    if (
        sym_eta2_seed is None
        and broken_eta2_seed is None
        and broken_phi2_seed is None
    ):
        try:
            return _solve_physical_phase_from_temperature_context(
                T=T,
                mu=mu,
                params=params,
                thermal=thermal,
                solver=solver,
            )
        except RuntimeError:
            pass

    candidates = solve_phase_candidates_at_T(
        T=T,
        mu=mu,
        sym_eta2_seed=sym_eta2_seed,
        broken_eta2_seed=broken_eta2_seed,
        broken_phi2_seed=broken_phi2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    available = [state for state in candidates.values() if state is not None]
    if not available:
        raise RuntimeError(
            f"No stationary phase candidates could be constructed at T={T}, mu={mu}."
        )

    return min(available, key=lambda state: state.veff)



# ============================================================================
# 7. On-shell potential layer
# ============================================================================

def veff_on_shell(
    phi: float,
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> tuple[float, float]:
    """
    Evaluate the on-shell effective potential at fixed phi:

        V_eff(phi; T, mu)

    by solving eta^2(phi, T, mu) first.

    This public version is seedless by default.

    Returns
    -------
    tuple[float, float]
        (veff, eta2).
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    eta2_sol = solve_eta2_given_phi(
        phi=phi,
        T=T,
        mu=mu,
        eta2_seed=eta2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    veff = opt_veff_off_shell(
        phi=phi,
        eta2=eta2_sol,
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
    )

    return float(veff), float(eta2_sol)

def veff_difference_from_origin(
    phi: float,
    T: float,
    mu: float,
    eta2_seed_phi: float | None = None,
    eta2_seed_zero: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> tuple[float, float, float]:
    """
    Compute

        Delta V(phi) = V_eff(phi; T, mu) - V_eff(0; T, mu)

    This public version is seedless by default.

    Returns
    -------
    tuple[float, float, float]
        (dV, eta2_phi, eta2_zero).
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)

    if eta2_seed_phi is not None:
        eta2_seed_phi = float(eta2_seed_phi)
        if not np.isfinite(eta2_seed_phi):
            raise ValueError(
                f"`eta2_seed_phi` must be finite when provided, got {eta2_seed_phi!r}."
            )

    if eta2_seed_zero is not None:
        eta2_seed_zero = float(eta2_seed_zero)
        if not np.isfinite(eta2_seed_zero):
            raise ValueError(
                f"`eta2_seed_zero` must be finite when provided, got {eta2_seed_zero!r}."
            )

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    veff_phi, eta2_phi = veff_on_shell(
        phi=phi,
        T=T,
        mu=mu,
        eta2_seed=eta2_seed_phi,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    eta2_zero = solve_symmetric_branch(
        T=T,
        mu=mu,
        eta2_seed=eta2_seed_zero,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    veff_zero = opt_veff_off_shell(
        phi=0.0,
        eta2=eta2_zero,
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
    )

    dV = veff_phi - veff_zero
    return float(dV), float(eta2_phi), float(eta2_zero)

# ============================================================================
# 8. Scans and continuation
# ============================================================================
def scan_potential(
    phi_grid: np.ndarray,
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
    subtract_origin: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan the on-shell potential over a grid in phi.

    Numerical strategy
    ------------------
    - If `eta2_seed` is provided, sweep through `phi_grid` in the order given,
      using continuation in the same spirit as the Mathematica notebook.
    - If `eta2_seed` is omitted, determine the *physical phase at this T
      autonomously* and use that phase as the anchor of the phi scan.
      Therefore, low-temperature scans naturally start on the broken branch,
      while high-temperature scans naturally start on the symmetric branch.

    Important detail
    ----------------
    When `subtract_origin=True`, the origin value is taken from the *same
    branch-connected scan* whenever possible. This prevents a low-temperature
    broken-phase scan from being contaminated by an unrelated large-eta
    symmetric root at phi=0.
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if phi_grid.ndim != 1:
        raise ValueError(f"`phi_grid` must be one-dimensional, got shape={phi_grid.shape}.")
    if phi_grid.size == 0:
        raise ValueError("`phi_grid` must contain at least one point.")
    if not np.all(np.isfinite(phi_grid)):
        raise ValueError("`phi_grid` must contain only finite values.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    n = phi_grid.size
    values_abs = np.empty_like(phi_grid, dtype=float)
    eta2_vals = np.empty_like(phi_grid, dtype=float)

    if eta2_seed is not None:
        anchor_index = 0
        anchor_seed = float(eta2_seed)
    else:
        try:
            anchor_state = solve_physical_phase_at_T(
                T=T,
                mu=mu,
                sym_eta2_seed=None,
                broken_eta2_seed=None,
                broken_phi2_seed=None,
                params=params,
                thermal=thermal,
                solver=solver,
            )
            anchor_seed = float(anchor_state.eta2)
            anchor_phi = float(anchor_state.phi)
            anchor_index = int(np.argmin(np.abs(phi_grid - anchor_phi)))
        except RuntimeError:
            anchor_index = int(np.argmax(np.abs(phi_grid)))
            anchor_seed = _default_eta2_seed(
                phi=float(phi_grid[anchor_index]),
                T=T,
                mu=mu,
                params=params,
                thermal=thermal,
            )

    veff0, eta20 = veff_on_shell(
        phi=float(phi_grid[anchor_index]),
        T=T,
        mu=mu,
        eta2_seed=float(anchor_seed),
        params=params,
        thermal=thermal,
        solver=solver,
    )
    values_abs[anchor_index] = float(veff0)
    eta2_vals[anchor_index] = float(eta20)

    def _scan_step(phi_value: float, primary_seed: float | None, fallback_seed: float | None) -> tuple[float, float]:
        try:
            return veff_on_shell(
                phi=phi_value,
                T=T,
                mu=mu,
                eta2_seed=primary_seed,
                params=params,
                thermal=thermal,
                solver=solver,
            )
        except RuntimeError:
            if fallback_seed is not None and (
                primary_seed is None or abs(float(fallback_seed) - float(primary_seed)) > 1e-14
            ):
                try:
                    return veff_on_shell(
                        phi=phi_value,
                        T=T,
                        mu=mu,
                        eta2_seed=float(fallback_seed),
                        params=params,
                        thermal=thermal,
                        solver=solver,
                    )
                except RuntimeError:
                    pass

            return veff_on_shell(
                phi=phi_value,
                T=T,
                mu=mu,
                eta2_seed=None,
                params=params,
                thermal=thermal,
                solver=solver,
            )

    current_seed = float(eta20)
    for i in range(anchor_index + 1, n):
        primary_seed = current_seed if solver.continuation else float(anchor_seed)
        veff_i, eta2_i = _scan_step(
            phi_value=float(phi_grid[i]),
            primary_seed=primary_seed,
            fallback_seed=float(anchor_seed),
        )
        values_abs[i] = float(veff_i)
        eta2_vals[i] = float(eta2_i)
        if solver.continuation:
            current_seed = float(eta2_i)

    current_seed = float(eta20)
    for i in range(anchor_index - 1, -1, -1):
        primary_seed = current_seed if solver.continuation else float(anchor_seed)
        veff_i, eta2_i = _scan_step(
            phi_value=float(phi_grid[i]),
            primary_seed=primary_seed,
            fallback_seed=float(anchor_seed),
        )
        values_abs[i] = float(veff_i)
        eta2_vals[i] = float(eta2_i)
        if solver.continuation:
            current_seed = float(eta2_i)

    i_zero = int(np.argmin(np.abs(phi_grid)))
    phi_zero_grid = float(phi_grid[i_zero])

    if np.isclose(phi_zero_grid, 0.0, atol=1e-14):
        eta2_zero = float(eta2_vals[i_zero])
        veff_zero = float(values_abs[i_zero])
    else:
        veff_zero, eta2_zero = veff_on_shell(
            phi=0.0,
            T=T,
            mu=mu,
            eta2_seed=float(eta2_vals[i_zero]),
            params=params,
            thermal=thermal,
            solver=solver,
        )

    if subtract_origin:
        values = values_abs - veff_zero
        return {
            "phi": phi_grid.copy(),
            "values": values,
            "eta2": eta2_vals,
            "eta2_zero": np.full_like(phi_grid, float(eta2_zero), dtype=float),
            "quantity": np.array(["deltaV"], dtype=object),
            "subtract_origin": np.array([True], dtype=bool),
            "T": np.array([T], dtype=float),
            "mu": np.array([mu], dtype=float),
        }

    return {
        "phi": phi_grid.copy(),
        "values": values_abs,
        "eta2": eta2_vals,
        "quantity": np.array(["veff"], dtype=object),
        "subtract_origin": np.array([False], dtype=bool),
        "T": np.array([T], dtype=float),
        "mu": np.array([mu], dtype=float),
    }


def trace_branches_over_T(
    mu: float,
    T_grid: np.ndarray,
    eta2_seed: float,
    phi2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> dict[str, np.ndarray]:
    """
    Follow the solutions as temperature changes, using continuation.

    This function implements the section 2 logic:
    - reuse previous solutions as seeds,
    - follow the broken branch while phi^2 > 0,
    - switch naturally to the symmetric branch when phi^2 -> 0.

    Notes
    -----
    The order of `T_grid` matters. This routine follows the branch sequence
    in the order provided by the user.

    The implemented logic is intentionally one-way:
    once the algorithm switches from the broken branch to the symmetric branch,
    it remains on the symmetric branch for the rest of the scan. This mirrors
    the practical branch-following strategy used in the notebook.

    Parameters
    ----------
    mu : float
        Chemical potential.
    T_grid : ndarray
        One-dimensional temperature grid in the order to be traced.
    eta2_seed : float
        Initial guess for eta^2.
    phi2_seed : float
        Initial guess for phi^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing:
        - "T": temperature grid,
        - "eta2": eta^2(T),
        - "phi2": phi^2(T),
        - "phi": sqrt(phi^2),
        - "is_broken": boolean mask,
        - "branch": object array with labels {"broken", "symmetric"}.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    mu = float(mu)
    eta2_seed = float(eta2_seed)
    phi2_seed = float(phi2_seed)

    if T_grid.ndim != 1:
        raise ValueError(f"`T_grid` must be one-dimensional, got shape={T_grid.shape}.")
    if T_grid.size == 0:
        raise ValueError("`T_grid` must contain at least one point.")
    if not np.all(np.isfinite(T_grid)):
        raise ValueError("`T_grid` must contain only finite values.")
    if np.any(T_grid <= 0.0):
        raise ValueError("All temperatures in `T_grid` must be strictly positive.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
    if not np.isfinite(phi2_seed):
        raise ValueError(f"`phi2_seed` must be finite, got phi2_seed={phi2_seed!r}.")

    eta2_out = np.empty_like(T_grid, dtype=float)
    phi2_out = np.empty_like(T_grid, dtype=float)
    phi_out = np.empty_like(T_grid, dtype=float)
    is_broken_out = np.empty(T_grid.shape, dtype=bool)
    branch_out = np.empty(T_grid.shape, dtype=object)

    current_eta2_seed = float(eta2_seed)
    current_phi2_seed = max(float(phi2_seed), 1e-8)

    # If the initial phi2 seed is essentially zero, start directly on the symmetric branch.
    on_broken_branch = bool(phi2_seed > 1e-10)

    for i, T in enumerate(T_grid):
        T = float(T)

        if on_broken_branch:
            try:
                eta2_val, phi2_val = solve_broken_branch(
                    T=T,
                    mu=mu,
                    eta2_seed=current_eta2_seed,
                    phi2_seed=current_phi2_seed,
                    params=params,
                    thermal=thermal,
                    solver=solver,
                )

                if phi2_val <= 1e-10:
                    raise RuntimeError("Broken branch collapsed to phi^2 ~ 0.")

                branch_label = "broken"
                is_broken = True

            except RuntimeError:
                eta2_val = solve_symmetric_branch(
                    T=T,
                    mu=mu,
                    eta2_seed=current_eta2_seed,
                    params=params,
                    thermal=thermal,
                    solver=solver,
                )
                phi2_val = 0.0
                branch_label = "symmetric"
                is_broken = False
                on_broken_branch = False

        else:
            eta2_val = solve_symmetric_branch(
                T=T,
                mu=mu,
                eta2_seed=current_eta2_seed,
                params=params,
                thermal=thermal,
                solver=solver,
            )
            phi2_val = 0.0
            branch_label = "symmetric"
            is_broken = False

        eta2_out[i] = float(eta2_val)
        phi2_out[i] = float(phi2_val)
        phi_out[i] = float(np.sqrt(max(phi2_val, 0.0)))
        is_broken_out[i] = bool(is_broken)
        branch_out[i] = branch_label

        if solver.continuation:
            current_eta2_seed = float(eta2_val)
            current_phi2_seed = max(float(phi2_val), 1e-8)
        else:
            current_eta2_seed = float(eta2_seed)
            current_phi2_seed = max(float(phi2_seed), 1e-8)

    return {
        "T": T_grid.copy(),
        "eta2": eta2_out,
        "phi2": phi2_out,
        "phi": phi_out,
        "is_broken": is_broken_out,
        "branch": branch_out,
        "mu": np.array([mu], dtype=float),
    }

def trace_physical_phases_over_T(
    mu: float,
    T_grid: np.ndarray,
    sym_eta2_seed: float | None = None,
    broken_eta2_seed: float | None = None,
    broken_phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> dict[str, np.ndarray]:
    """
    Trace the physical OPT phase over temperature.

    Numerical strategy
    ------------------
    The symmetric and broken branches are not reconstructed independently at
    each temperature. Instead, each one is followed by continuation in the
    spirit of the Mathematica notebook:

    - the symmetric branch is seeded from the highest temperature and traced
      downward, where disconnected large-eta jumps are rejected;
    - the broken branch is seeded from the lowest temperature and traced upward
      until it collapses to the symmetric phase.

    The physical phase at each temperature is then chosen as the available
    stationary branch with the lowest effective potential.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    mu = float(mu)

    if T_grid.ndim != 1:
        raise ValueError(f"`T_grid` must be one-dimensional, got shape={T_grid.shape}.")
    if T_grid.size == 0:
        raise ValueError("`T_grid` must contain at least one point.")
    if not np.all(np.isfinite(T_grid)):
        raise ValueError("`T_grid` must contain only finite values.")
    if np.any(T_grid <= 0.0):
        raise ValueError("All temperatures in `T_grid` must be strictly positive.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")

    if sym_eta2_seed is not None:
        sym_eta2_seed = float(sym_eta2_seed)
        if not np.isfinite(sym_eta2_seed):
            raise ValueError(f"`sym_eta2_seed` must be finite when provided, got {sym_eta2_seed!r}.")
    if broken_eta2_seed is not None:
        broken_eta2_seed = float(broken_eta2_seed)
        if not np.isfinite(broken_eta2_seed):
            raise ValueError(f"`broken_eta2_seed` must be finite when provided, got {broken_eta2_seed!r}.")
    if broken_phi2_seed is not None:
        broken_phi2_seed = float(broken_phi2_seed)
        if not np.isfinite(broken_phi2_seed):
            raise ValueError(f"`broken_phi2_seed` must be finite when provided, got {broken_phi2_seed!r}.")

    nT = len(T_grid)
    order_up = np.argsort(T_grid)
    order_down = order_up[::-1]

    sym_states: list[OPTPhaseState | None] = [None] * nT
    broken_states: list[OPTPhaseState | None] = [None] * nT

    current_sym_seed = sym_eta2_seed
    sym_active = True
    for idx in order_down:
        T = float(T_grid[idx])
        if not sym_active:
            continue
        try:
            eta2_sym_val = solve_symmetric_branch(
                T=T, mu=mu, eta2_seed=current_sym_seed,
                params=params, thermal=thermal, solver=solver,
            )
            veff_sym_val = opt_veff_off_shell(
                phi=0.0, eta2=eta2_sym_val, T=T, mu=mu, params=params, thermal=thermal,
            )
            sym_states[idx] = OPTPhaseState(
                branch="symmetric", T=T, mu=float(mu), phi=0.0, phi2=0.0,
                eta2=float(eta2_sym_val), veff=float(veff_sym_val),
                F_eta=float(eta_gap_residual(0.0, eta2_sym_val, T, mu, params, thermal)),
                F_phi=float(np.nan),
            )
            if solver.continuation:
                current_sym_seed = float(eta2_sym_val)
        except RuntimeError:
            sym_active = False

    current_broken_eta2_seed = broken_eta2_seed
    current_broken_phi2_seed = broken_phi2_seed
    broken_active = True
    for idx in order_up:
        T = float(T_grid[idx])
        if not broken_active:
            continue
        try:
            eta2_broken_val, phi2_broken_val = solve_broken_branch(
                T=T, mu=mu, eta2_seed=current_broken_eta2_seed, phi2_seed=current_broken_phi2_seed,
                params=params, thermal=thermal, solver=solver,
            )
            phi_broken_val = float(np.sqrt(phi2_broken_val))
            veff_broken_val = opt_veff_off_shell(
                phi=phi_broken_val, eta2=eta2_broken_val, T=T, mu=mu, params=params, thermal=thermal,
            )
            broken_states[idx] = OPTPhaseState(
                branch="broken", T=T, mu=float(mu), phi=float(phi_broken_val), phi2=float(phi2_broken_val),
                eta2=float(eta2_broken_val), veff=float(veff_broken_val),
                F_eta=float(eta_gap_residual(phi_broken_val, eta2_broken_val, T, mu, params, thermal)),
                F_phi=float(phi_stationary_residual(phi_broken_val, eta2_broken_val, T, mu, params, thermal)),
            )
            if solver.continuation:
                current_broken_eta2_seed = float(eta2_broken_val)
                current_broken_phi2_seed = float(phi2_broken_val)
        except RuntimeError:
            broken_active = False

    phi_phys = np.empty(nT, dtype=float)
    phi2_phys = np.empty(nT, dtype=float)
    eta_phys = np.empty(nT, dtype=float)
    eta2_phys = np.empty(nT, dtype=float)
    veff_phys = np.empty(nT, dtype=float)
    branch_phys = np.empty(nT, dtype=object)
    is_broken_phys = np.empty(nT, dtype=bool)

    eta2_sym = np.full(nT, np.nan, dtype=float)
    eta_sym = np.full(nT, np.nan, dtype=float)
    veff_sym = np.full(nT, np.nan, dtype=float)
    F_eta_sym = np.full(nT, np.nan, dtype=float)

    broken_exists = np.zeros(nT, dtype=bool)
    phi_broken = np.full(nT, np.nan, dtype=float)
    phi2_broken = np.full(nT, np.nan, dtype=float)
    eta2_broken = np.full(nT, np.nan, dtype=float)
    eta_broken = np.full(nT, np.nan, dtype=float)
    veff_broken = np.full(nT, np.nan, dtype=float)
    F_eta_broken = np.full(nT, np.nan, dtype=float)
    F_phi_broken = np.full(nT, np.nan, dtype=float)

    for i in range(nT):
        sym_state = sym_states[i]
        broken_state = broken_states[i]

        if sym_state is not None:
            eta2_sym[i] = float(sym_state.eta2)
            eta_sym[i] = float(np.sqrt(max(sym_state.eta2, 0.0)))
            veff_sym[i] = float(sym_state.veff)
            F_eta_sym[i] = float(sym_state.F_eta)

        if broken_state is not None:
            broken_exists[i] = True
            phi_broken[i] = float(broken_state.phi)
            phi2_broken[i] = float(broken_state.phi2)
            eta2_broken[i] = float(broken_state.eta2)
            eta_broken[i] = float(np.sqrt(max(broken_state.eta2, 0.0)))
            veff_broken[i] = float(broken_state.veff)
            F_eta_broken[i] = float(broken_state.F_eta)
            F_phi_broken[i] = float(broken_state.F_phi)

        available = [state for state in (sym_state, broken_state) if state is not None]
        if not available:
            raise RuntimeError(f"No stationary phase candidates could be constructed at T={T_grid[i]}, mu={mu}.")

        physical_state = min(available, key=lambda state: state.veff)
        phi_phys[i] = float(physical_state.phi)
        phi2_phys[i] = float(physical_state.phi2)
        eta2_phys[i] = float(physical_state.eta2)
        eta_phys[i] = float(np.sqrt(max(physical_state.eta2, 0.0)))
        veff_phys[i] = float(physical_state.veff)
        branch_phys[i] = str(physical_state.branch)
        is_broken_phys[i] = bool(physical_state.branch == "broken")

    return {
        "T": T_grid.copy(),
        "mu": np.array([mu], dtype=float),
        "branch": branch_phys,
        "is_broken": is_broken_phys,
        "phi": phi_phys,
        "phi2": phi2_phys,
        "eta": eta_phys,
        "eta2": eta2_phys,
        "veff": veff_phys,
        "eta_sym": eta_sym,
        "eta2_sym": eta2_sym,
        "veff_sym": veff_sym,
        "F_eta_sym": F_eta_sym,
        "broken_exists": broken_exists,
        "phi_broken": phi_broken,
        "phi2_broken": phi2_broken,
        "eta_broken": eta_broken,
        "eta2_broken": eta2_broken,
        "veff_broken": veff_broken,
        "F_eta_broken": F_eta_broken,
        "F_phi_broken": F_phi_broken,
    }


def _solve_physical_phase_from_temperature_context(
    T: float,
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> OPTPhaseState:
    """
    Reconstruct the physical phase at one temperature using an automatic
    temperature context.

    The routine adaptively builds a temperature interval containing the
    requested T, traces both branches over that interval, and then reads off
    the physical phase exactly at T. No critical temperature is supplied by
    hand; the broken-to-symmetric change is inferred from the branch
    competition itself.
    """
    T = float(T)
    mu = float(mu)

    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")

    base_scale = float(np.sqrt(abs(params.m2) + mu**2 + 1.0))
    T_low = max(5e-2, min(T, 0.35 * base_scale, 0.35 * T if T > 0.0 else 5e-2, 0.5))
    T_high = max(T + 1.0, 3.0 * base_scale + 1.0, 4.0)

    last_trace = None

    for _ in range(6):
        left = np.linspace(T_low, T, 16, dtype=float)
        right = np.linspace(T, T_high, 16, dtype=float)
        T_context = np.unique(np.concatenate([left, right, np.array([T], dtype=float)]))

        trace = trace_physical_phases_over_T(
            mu=mu,
            T_grid=T_context,
            params=params,
            thermal=thermal,
            solver=solver,
        )
        last_trace = trace

        idx = int(np.argmin(np.abs(T_context - T)))
        top_is_symmetric = str(trace["branch"][-1]) == "symmetric"
        bottom_has_broken = bool(trace["broken_exists"][0] or trace["branch"][0] == "broken")

        if top_is_symmetric and bottom_has_broken:
            if str(trace["branch"][idx]) == "symmetric":
                return OPTPhaseState(
                    branch="symmetric",
                    T=float(T),
                    mu=float(mu),
                    phi=0.0,
                    phi2=0.0,
                    eta2=float(trace["eta2"][idx]),
                    veff=float(trace["veff"][idx]),
                    F_eta=float(trace["F_eta_sym"][idx]),
                    F_phi=float(np.nan),
                )

            return OPTPhaseState(
                branch="broken",
                T=float(T),
                mu=float(mu),
                phi=float(trace["phi"][idx]),
                phi2=float(trace["phi2"][idx]),
                eta2=float(trace["eta2"][idx]),
                veff=float(trace["veff"][idx]),
                F_eta=float(trace["F_eta_broken"][idx]),
                F_phi=float(trace["F_phi_broken"][idx]),
            )

        T_low = max(1e-3, 0.5 * T_low)
        T_high = T_high + max(2.0, 0.5 * T_high)

    if last_trace is None:
        raise RuntimeError("Automatic temperature-context phase solve produced no trace.")

    idx = int(np.argmin(np.abs(last_trace["T"] - T)))
    if str(last_trace["branch"][idx]) == "symmetric":
        return OPTPhaseState(
            branch="symmetric",
            T=float(T),
            mu=float(mu),
            phi=0.0,
            phi2=0.0,
            eta2=float(last_trace["eta2"][idx]),
            veff=float(last_trace["veff"][idx]),
            F_eta=float(last_trace["F_eta_sym"][idx]),
            F_phi=float(np.nan),
        )

    return OPTPhaseState(
        branch="broken",
        T=float(T),
        mu=float(mu),
        phi=float(last_trace["phi"][idx]),
        phi2=float(last_trace["phi2"][idx]),
        eta2=float(last_trace["eta2"][idx]),
        veff=float(last_trace["veff"][idx]),
        F_eta=float(last_trace["F_eta_broken"][idx]),
        F_phi=float(last_trace["F_phi_broken"][idx]),
    )


def phi_min(
    T: float,
    mu: float,
    phi_grid: np.ndarray,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> tuple[float, float]:
    """
    Return the approximate location of the minimum of V(phi) for fixed T and mu.

    This public version is seedless by default.

    Returns
    -------
    tuple[float, float]
        (phi_star, V_star).
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if phi_grid.ndim != 1:
        raise ValueError(f"`phi_grid` must be one-dimensional, got shape={phi_grid.shape}.")
    if phi_grid.size == 0:
        raise ValueError("`phi_grid` must contain at least one point.")
    if not np.all(np.isfinite(phi_grid)):
        raise ValueError("`phi_grid` must contain only finite values.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    scan_data = scan_potential(
        phi_grid=phi_grid,
        T=T,
        mu=mu,
        eta2_seed=eta2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
        subtract_origin=False,
    )

    values = np.asarray(scan_data["values"], dtype=float)
    idx_min = int(np.argmin(values))

    return float(scan_data["phi"][idx_min]), float(values[idx_min])


def eta2_solution(
    phi: float,
    T: float,
    mu: float,
    eta2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> float:
    """
    Convenience wrapper returning only the on-shell eta^2(phi, T, mu).

    This public version is seedless by default.
    """
    return float(
        solve_eta2_given_phi(
            phi=phi,
            T=T,
            mu=mu,
            eta2_seed=eta2_seed,
            params=params,
            thermal=thermal,
            solver=solver,
        )
    )


def Tc_opt(
    mu: float,
    T_grid: np.ndarray,
    eta2_seed: float,
    phi2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> float:
    """
    Estimate the OPT critical temperature using the branch-tracing logic
    inspired by section 2.

    Current implementation
    ----------------------
    This first implementation defines Tc as the first temperature in the scan
    at which the algorithm has switched from the broken branch to the symmetric
    branch. Therefore, it is a grid-resolved estimate and its accuracy is set
    by the supplied `T_grid`.

    Parameters
    ----------
    mu : float
        Chemical potential.
    T_grid : ndarray
        One-dimensional temperature grid in tracing order.
    eta2_seed : float
        Initial guess for eta^2.
    phi2_seed : float
        Initial guess for phi^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    float
        Estimated OPT critical temperature on the supplied grid.

    Raises
    ------
    RuntimeError
        If the scan never exhibits a broken-to-symmetric transition.
    ValueError
        If the inputs are invalid.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    mu = float(mu)
    eta2_seed = float(eta2_seed)
    phi2_seed = float(phi2_seed)

    if T_grid.ndim != 1:
        raise ValueError(f"`T_grid` must be one-dimensional, got shape={T_grid.shape}.")
    if T_grid.size < 2:
        raise ValueError("`T_grid` must contain at least two points to estimate Tc.")
    if not np.all(np.isfinite(T_grid)):
        raise ValueError("`T_grid` must contain only finite values.")
    if np.any(T_grid <= 0.0):
        raise ValueError("All temperatures in `T_grid` must be strictly positive.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
    if not np.isfinite(phi2_seed):
        raise ValueError(f"`phi2_seed` must be finite, got phi2_seed={phi2_seed!r}.")

    trace_data = trace_branches_over_T(
        mu=mu,
        T_grid=T_grid,
        eta2_seed=eta2_seed,
        phi2_seed=phi2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    is_broken = np.asarray(trace_data["is_broken"], dtype=bool)
    T_vals = np.asarray(trace_data["T"], dtype=float)

    for i in range(1, len(T_vals)):
        if is_broken[i - 1] and not is_broken[i]:
            return float(T_vals[i])

    raise RuntimeError(
        "Could not estimate Tc_opt from the supplied temperature grid: "
        "no broken-to-symmetric branch transition was found."
    )


def Tc_pt(mu: float, params: OPTModelParams) -> float:
    """
    Perturbative critical temperature benchmark from section 2.

    In the notebook normalization used for the benchmark comparison,
    the perturbative estimate is

        Tc^2 = 18 * (mu^2 - m^2).

    For the default notebook choice m^2 = -1 this becomes

        Tc^2 = 18 * (1 + mu^2),

    which matches the section-2 benchmark curve.

    Parameters
    ----------
    mu : float
        Chemical potential.
    params : OPTModelParams
        Model parameters. Only `m2` enters this benchmark formula.

    Returns
    -------
    float
        Perturbative benchmark critical temperature.

    Raises
    ------
    ValueError
        If the benchmark expression is not real/physical.
    """
    mu = float(mu)
    m2 = float(params.m2)

    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(m2):
        raise ValueError(f"`params.m2` must be finite, got m2={m2!r}.")

    Tc2 = 18.0 * (mu**2 - m2)
    if Tc2 < 0.0:
        raise ValueError(
            "The perturbative benchmark Tc_pt is not real for the supplied "
            f"parameters: Tc^2 = {Tc2}."
        )

    return float(np.sqrt(Tc2))


def physical_eta2(
    T: float,
    mu: float,
    sym_eta2_seed: float | None = None,
    broken_eta2_seed: float | None = None,
    broken_phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> float:
    """
    Return the physical eta^2(T, mu), defined from the true stationary phase.
    """
    state = solve_physical_phase_at_T(
        T=T,
        mu=mu,
        sym_eta2_seed=sym_eta2_seed,
        broken_eta2_seed=broken_eta2_seed,
        broken_phi2_seed=broken_phi2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )
    return float(state.eta2)


def physical_eta(
    T: float,
    mu: float,
    sym_eta2_seed: float | None = None,
    broken_eta2_seed: float | None = None,
    broken_phi2_seed: float | None = None,
    params: OPTModelParams = OPTModelParams(),
    thermal: ThermalOptions = ThermalOptions(),
    solver: SolverOptions = SolverOptions(),
) -> float:
    """
    Return the physical eta(T, mu), defined as sqrt(eta^2_phys).
    """
    eta2_val = physical_eta2(
        T=T,
        mu=mu,
        sym_eta2_seed=sym_eta2_seed,
        broken_eta2_seed=broken_eta2_seed,
        broken_phi2_seed=broken_phi2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )
    return float(np.sqrt(max(eta2_val, 0.0)))

# ============================================================================
# 10. Plotting and notebook reproduction helpers
# ============================================================================

def reproduce_section3_scan(
    phi_max: float = 0.5,
    dphi: float = 0.01,
    T: float = 4.733359332047422 + 0.027,
    mu: float = 0.5,
    eta2_seed: float | None = 2.1214029070500127,
    params: OPTModelParams | None = None,
    thermal: ThermalOptions | None = None,
    solver: SolverOptions | None = None,
    subtract_origin: bool = True,
) -> dict[str, np.ndarray]:
    """
    Convenience helper to reproduce the final section 3 scan.

    This public version is seedless by default.
    """
    phi_max = float(phi_max)
    dphi = float(dphi)
    T = float(T)
    mu = float(mu)

    if eta2_seed is not None:
        eta2_seed = float(eta2_seed)
        if not np.isfinite(eta2_seed):
            raise ValueError(f"`eta2_seed` must be finite when provided, got {eta2_seed!r}.")

    if not np.isfinite(phi_max) or phi_max <= 0.0:
        raise ValueError(f"`phi_max` must be finite and > 0, got phi_max={phi_max!r}.")
    if not np.isfinite(dphi) or dphi <= 0.0:
        raise ValueError(f"`dphi` must be finite and > 0, got dphi={dphi!r}.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"`T` must be finite and > 0, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")

    if params is None:
        params = OPTModelParams()
    if thermal is None:
        thermal = ThermalOptions(
            mode_mu0="notebook_highT",
            mode_muneq0="notebook_highT",
        )
    if solver is None:
        solver = SolverOptions(
            root_tol=1e-12,
            max_iter=300,
            continuation=True,
        )

    phi_grid = np.arange(-phi_max, phi_max + 0.5 * dphi, dphi, dtype=float)

    scan_data = scan_potential(
        phi_grid=phi_grid,
        T=T,
        mu=mu,
        eta2_seed=eta2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
        subtract_origin=subtract_origin,
    )

    scan_data["phi_max"] = np.array([phi_max], dtype=float)
    scan_data["dphi"] = np.array([dphi], dtype=float)
    scan_data["eta2_seed"] = np.array(
        [np.nan if eta2_seed is None else float(eta2_seed)],
        dtype=float,
    )

    return scan_data

def plot_potential_scan(
    scan_data: dict[str, np.ndarray],
    show: bool = True,
) -> None:
    """
    Plot V(phi) or Delta V(phi) from the output of scan_potential().

    Parameters
    ----------
    scan_data : dict[str, ndarray]
        Output dictionary produced by `scan_potential(...)` or
        `reproduce_section3_scan(...)`.
    show : bool, optional
        If True, call `plt.show()` at the end. Use False in automated tests.
    """
    if "phi" not in scan_data or "values" not in scan_data:
        raise ValueError("`scan_data` must contain at least the keys 'phi' and 'values'.")

    phi = np.asarray(scan_data["phi"], dtype=float)
    values = np.asarray(scan_data["values"], dtype=float)

    if phi.ndim != 1 or values.ndim != 1:
        raise ValueError("`scan_data['phi']` and `scan_data['values']` must be one-dimensional.")
    if phi.size != values.size:
        raise ValueError(
            "`scan_data['phi']` and `scan_data['values']` must have the same length."
        )

    quantity = "values"
    if "quantity" in scan_data and len(scan_data["quantity"]) > 0:
        quantity = str(scan_data["quantity"][0])

    if quantity.lower() == "deltav":
        ylabel = r"$\Delta V_{\mathrm{eff}}$"
        legend_label = r"$\Delta V_{\mathrm{eff}}$"
    else:
        ylabel = r"$V_{\mathrm{eff}}$"
        legend_label = r"$V_{\mathrm{eff}}$"

    plt.figure(figsize=(8, 5))
    plt.plot(phi, values, color="black", lw=2.0, label=legend_label)
    plt.plot(phi, values, color="blue", lw=1.3, alpha=0.85)

    plt.xlabel(r"$\tilde{\phi}/M$", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.legend(loc="upper center", frameon=False)
    plt.grid(False)
    plt.tight_layout()

    if show:
        plt.show()


def plot_branches(
    trace_data: dict[str, np.ndarray],
    show: bool = True,
) -> None:
    """
    Plot branch data obtained from trace_branches_over_T().

    The function makes two figures:
    1. phi(T)
    2. eta^2(T)

    Parameters
    ----------
    trace_data : dict[str, ndarray]
        Output of `trace_branches_over_T(...)`.
    show : bool, optional
        If True, call `plt.show()` at the end. Use False in automated tests.
    """
    required_keys = {"T", "phi", "eta2", "is_broken"}
    missing = required_keys.difference(trace_data.keys())
    if missing:
        raise ValueError(
            f"`trace_data` is missing required keys: {sorted(missing)}."
        )

    T = np.asarray(trace_data["T"], dtype=float)
    phi = np.asarray(trace_data["phi"], dtype=float)
    eta2 = np.asarray(trace_data["eta2"], dtype=float)
    is_broken = np.asarray(trace_data["is_broken"], dtype=bool)

    if T.ndim != 1 or phi.ndim != 1 or eta2.ndim != 1 or is_broken.ndim != 1:
        raise ValueError("All plotted arrays in `trace_data` must be one-dimensional.")
    if not (len(T) == len(phi) == len(eta2) == len(is_broken)):
        raise ValueError("All plotted arrays in `trace_data` must have the same length.")

    broken_mask = is_broken
    sym_mask = ~is_broken

    # Figure 1: phi(T)
    plt.figure(figsize=(8, 5))
    if np.any(broken_mask):
        plt.plot(T[broken_mask], phi[broken_mask], marker="o", lw=1.8, label="broken")
    if np.any(sym_mask):
        plt.plot(T[sym_mask], phi[sym_mask], marker="o", lw=1.8, label="symmetric")

    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$\phi(T)$", fontsize=13)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()

    # Figure 2: eta^2(T)
    plt.figure(figsize=(8, 5))
    if np.any(broken_mask):
        plt.plot(T[broken_mask], eta2[broken_mask], marker="o", lw=1.8, label="broken")
    if np.any(sym_mask):
        plt.plot(T[sym_mask], eta2[sym_mask], marker="o", lw=1.8, label="symmetric")

    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$\eta^2(T)$", fontsize=13)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()

    if show:
        plt.show()