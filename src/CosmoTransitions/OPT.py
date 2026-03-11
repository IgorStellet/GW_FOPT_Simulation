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


def solve_eta2_given_phi(
    phi: float,
    T: float,
    mu: float,
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> float:
    """
    Solve the physical gap equation for eta^2 at fixed (phi, T, mu).

    This is the key low-level solver needed to construct the on-shell
    potential V_eff(phi; T, mu).

    Strategy
    --------
    We solve strictly on the real axis and enforce the physical eta^2 domain
    implied by the active backend. The solver uses:

    1. a local bracket search around the provided seed;
    2. a robust Brent solve once a sign change is found;
    3. a wider fallback scan if the local search fails.

    Parameters
    ----------
    phi : float
        Background field value.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    float
        Physical real root eta^2(phi, T, mu).

    Raises
    ------
    RuntimeError
        If no real physical root can be bracketed/found.
    ValueError
        If the input values are invalid.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")
    if solver.root_tol <= 0.0:
        raise ValueError(f"`solver.root_tol` must be > 0, got {solver.root_tol}.")
    if solver.max_iter <= 0:
        raise ValueError(f"`solver.max_iter` must be > 0, got {solver.max_iter}.")

    eta2_min = _eta2_physical_lower_bound(mu, params, thermal)

    def f_eta(e2: float) -> float:
        e2 = float(e2)
        if e2 <= eta2_min:
            return np.nan
        try:
            value = eta_gap_residual(phi, e2, T, mu, params, thermal)
        except ValueError:
            return np.nan
        return float(value)

    # Start near the provided seed but always inside the physical domain.
    center = max(eta2_seed, eta2_min + 1e-10)
    f_center = f_eta(center)

    if np.isfinite(f_center) and abs(f_center) < max(1e-10, solver.root_tol):
        return float(center)

    # --------------------------------------------------
    # Local bracket search around the seed
    # --------------------------------------------------
    bracket = None
    step = max(1e-3, 0.05 * max(1.0, center))

    n_local = max(12, solver.max_iter // 8)
    for _ in range(n_local):
        left = max(eta2_min + 1e-8, center - step)
        right = center + step

        f_left = f_eta(left)
        f_right = f_eta(right)

        if np.isfinite(f_left) and np.isfinite(f_center) and f_left * f_center <= 0.0:
            bracket = (left, center)
            break

        if np.isfinite(f_center) and np.isfinite(f_right) and f_center * f_right <= 0.0:
            bracket = (center, right)
            break

        step *= 1.6

    # --------------------------------------------------
    # Wider fallback scan if the local search fails
    # --------------------------------------------------
    if bracket is None:
        upper = max(center + step, eta2_min + 20.0)

        # Use a grid clustered near the physical lower bound, where the root
        # may live in delicate cases.
        offsets = np.geomspace(1e-12, max(upper - eta2_min, 1.0), max(600, 4 * solver.max_iter))
        x_grid = eta2_min + offsets

        x_prev = x_grid[0]
        f_prev = f_eta(x_prev)

        for x_now in x_grid[1:]:
            f_now = f_eta(x_now)
            if np.isfinite(f_prev) and np.isfinite(f_now) and f_prev * f_now <= 0.0:
                bracket = (x_prev, x_now)
                break
            x_prev = x_now
            f_prev = f_now

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))

    if bracket is None:
        secant_pairs = [
            (max(center, eta2_min + 1e-12), center + max(1e-4, 0.01 * max(1.0, center))),
            (eta2_min + 1e-10, max(center, eta2_min + 1e-6)),
            (max(center, eta2_min + 1e-8), max(center + 0.1, eta2_min + 1e-4)),
            (max(center, eta2_min + 1e-6), max(center + 1.0, eta2_min + 1e-3)),
        ]

        for x0, x1 in secant_pairs:
            try:
                sol_sec = root_scalar(
                    f_eta,
                    x0=float(x0),
                    x1=float(x1),
                    method="secant",
                    xtol=solver.root_tol,
                    maxiter=solver.max_iter,
                )
            except Exception:
                continue

            if not sol_sec.converged:
                continue

            eta2_root = float(sol_sec.root)
            if not np.isfinite(eta2_root):
                continue
            if eta2_root <= eta2_min:
                continue

            f_root = f_eta(eta2_root)
            if np.isfinite(f_root) and abs(f_root) < residual_tol:
                return eta2_root

        raise RuntimeError(
            "Could not find a physical real eta^2 root. "
            f"Inputs: phi={phi}, T={T}, mu={mu}, eta2_seed={eta2_seed}, "
            f"eta2_min={eta2_min}."
        )

    sol = root_scalar(
        f_eta,
        bracket=bracket,
        method="brentq",
        xtol=solver.root_tol,
        maxiter=solver.max_iter,
    )

    if not sol.converged:
        raise RuntimeError(
            "Brent root solve for eta^2 did not converge. "
            f"Inputs: phi={phi}, T={T}, mu={mu}, bracket={bracket}."
        )

    eta2_root = float(sol.root)
    if eta2_root <= eta2_min:
        raise RuntimeError(
            f"Found eta^2 root outside the physical domain: eta2={eta2_root}, "
            f"eta2_min={eta2_min}."
        )

    f_root = f_eta(eta2_root)
    if not np.isfinite(f_root):
        raise RuntimeError(
            f"Eta-gap residual became non-finite at the computed root eta2={eta2_root}."
        )

    return eta2_root


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
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> float:
    """
    Solve the symmetric branch, defined by phi = 0.

    This corresponds to the section 2 strategy where only the eta-gap
    equation is solved on the symmetric phase.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
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

    Raises
    ------
    RuntimeError
        If the branch solve does not converge to a valid symmetric solution.
    ValueError
        If the input values are invalid.
    """
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
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
    eta2_seed: float,
    phi2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[float, float]:
    """
    Solve the broken branch using the coupled system for (eta^2, phi^2).

    This follows the section 2 logic for the phase with nonzero order
    parameter.

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
        (eta2, phi2) on the broken branch.

    Raises
    ------
    RuntimeError
        If the branch solve does not converge to a valid broken-branch solution.
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

    # Keep the initial phi^2 seed on the broken side.
    phi2_seed = max(phi2_seed, 1e-8)

    residual_tol = max(1e-8, 10.0 * np.sqrt(solver.root_tol))

    trial_seeds = [
        (eta2_seed, phi2_seed),
        (max(eta2_seed, _eta2_physical_lower_bound(mu, params, thermal) + 1e-6), max(phi2_seed, 0.1)),
        (max(1.10 * eta2_seed, _eta2_physical_lower_bound(mu, params, thermal) + 1e-6), max(2.0 * phi2_seed, 0.5)),
        (max(0.95 * eta2_seed, _eta2_physical_lower_bound(mu, params, thermal) + 1e-6), max(0.5 * phi2_seed, 0.05)),
    ]

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


# ============================================================================
# 7. On-shell potential layer
# ============================================================================

def veff_on_shell(
    phi: float,
    T: float,
    mu: float,
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[float, float]:
    """
    Evaluate the on-shell effective potential at fixed phi:

        V_eff(phi; T, mu)

    by solving eta^2(phi, T, mu) first.

    Parameters
    ----------
    phi : float
        Background field value.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    tuple[float, float]
        (veff, eta2), where
        - veff is the on-shell effective potential at the requested phi,
        - eta2 is the gap-equation solution used in the evaluation.

    Raises
    ------
    RuntimeError
        If the eta-gap solve fails.
    ValueError
        If the inputs are invalid.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
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
    eta2_seed_phi: float,
    eta2_seed_zero: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[float, float, float]:
    """
    Compute the potential difference used in notebook section 3:

        Delta V(phi) = V_eff(phi; T, mu) - V_eff(0; T, mu)

    Parameters
    ----------
    phi : float
        Background field value.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed_phi : float
        Initial guess for eta^2 at the requested phi.
    eta2_seed_zero : float
        Initial guess for eta^2 at phi = 0.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    tuple[float, float, float]
        (dV, eta2_phi, eta2_zero), where
        - dV = V_eff(phi;T,mu) - V_eff(0;T,mu),
        - eta2_phi is the gap solution at the requested phi,
        - eta2_zero is the symmetric-branch gap solution at phi = 0.

    Raises
    ------
    RuntimeError
        If one of the required branch solves fails.
    ValueError
        If the inputs are invalid.
    """
    phi = float(phi)
    T = float(T)
    mu = float(mu)
    eta2_seed_phi = float(eta2_seed_phi)
    eta2_seed_zero = float(eta2_seed_zero)

    if not np.isfinite(phi):
        raise ValueError(f"`phi` must be finite, got phi={phi!r}.")
    if not np.isfinite(T):
        raise ValueError(f"`T` must be finite, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed_phi):
        raise ValueError(
            f"`eta2_seed_phi` must be finite, got eta2_seed_phi={eta2_seed_phi!r}."
        )
    if not np.isfinite(eta2_seed_zero):
        raise ValueError(
            f"`eta2_seed_zero` must be finite, got eta2_seed_zero={eta2_seed_zero!r}."
        )
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
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
    subtract_origin: bool = True,
) -> dict[str, np.ndarray]:
    """
    Scan the on-shell potential over a grid in phi.

    This is the main tool to reproduce the notebook section 3 plot and to
    build V(phi) for arbitrary fixed T and mu.

    Expected outputs
    ----------------
    - phi grid
    - V(phi) or Delta V(phi)
    - eta^2(phi)

    Notes
    -----
    The order of `phi_grid` matters when `solver.continuation=True`, because
    the eta^2 solution found at one point is used as the seed for the next one.
    This mirrors the continuation logic used in the notebook scans.

    Parameters
    ----------
    phi_grid : ndarray
        One-dimensional grid of field values.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.
    subtract_origin : bool, optional
        If True, return Delta V(phi) = V(phi) - V(0). Otherwise return V(phi).

    Returns
    -------
    dict[str, ndarray]
        Dictionary containing at least:
        - "phi": input phi grid,
        - "values": scanned quantity,
        - "eta2": eta^2(phi),
        - "quantity": object array with a descriptive label.

        If `subtract_origin=True`, the dictionary also includes:
        - "eta2_zero": eta^2 at phi = 0 used in the subtraction.
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

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
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
    if T <= 0.0:
        raise ValueError(f"`T` must be strictly positive, got T={T}.")

    values = np.empty_like(phi_grid, dtype=float)
    eta2_vals = np.empty_like(phi_grid, dtype=float)

    current_eta2_phi_seed = float(eta2_seed)
    current_eta2_zero_seed = float(eta2_seed)

    if subtract_origin:
        eta2_zero_vals = np.empty_like(phi_grid, dtype=float)

        for i, phi in enumerate(phi_grid):
            dV, eta2_phi, eta2_zero = veff_difference_from_origin(
                phi=float(phi),
                T=T,
                mu=mu,
                eta2_seed_phi=current_eta2_phi_seed,
                eta2_seed_zero=current_eta2_zero_seed,
                params=params,
                thermal=thermal,
                solver=solver,
            )

            values[i] = float(dV)
            eta2_vals[i] = float(eta2_phi)
            eta2_zero_vals[i] = float(eta2_zero)

            if solver.continuation:
                current_eta2_phi_seed = float(eta2_phi)
                current_eta2_zero_seed = float(eta2_zero)

        return {
            "phi": phi_grid.copy(),
            "values": values,
            "eta2": eta2_vals,
            "eta2_zero": eta2_zero_vals,
            "quantity": np.array(["deltaV"], dtype=object),
            "subtract_origin": np.array([True], dtype=bool),
            "T": np.array([T], dtype=float),
            "mu": np.array([mu], dtype=float),
        }

    for i, phi in enumerate(phi_grid):
        veff, eta2_phi = veff_on_shell(
            phi=float(phi),
            T=T,
            mu=mu,
            eta2_seed=current_eta2_phi_seed,
            params=params,
            thermal=thermal,
            solver=solver,
        )

        values[i] = float(veff)
        eta2_vals[i] = float(eta2_phi)

        if solver.continuation:
            current_eta2_phi_seed = float(eta2_phi)

    return {
        "phi": phi_grid.copy(),
        "values": values,
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


# ============================================================================
# 9. Observables
# ============================================================================

def phi_min(
    T: float,
    mu: float,
    phi_grid: np.ndarray,
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[float, float]:
    """
    Return the approximate location of the minimum of V(phi) for fixed T and mu.

    The minimum is determined on the supplied discrete phi grid using the
    on-shell effective potential scan.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    phi_grid : ndarray
        One-dimensional grid of field values.
    eta2_seed : float
        Initial guess for eta^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    tuple[float, float]
        (phi_star, V_star), where
        - phi_star is the approximate field value of the minimum on the grid,
        - V_star is the corresponding on-shell potential value.
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

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
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")
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
    eta2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> float:
    """
    Convenience wrapper returning only the on-shell eta^2(phi, T, mu).

    Parameters
    ----------
    phi : float
        Background field value.
    T : float
        Temperature.
    mu : float
        Chemical potential.
    eta2_seed : float
        Initial guess for eta^2.
    params : OPTModelParams
        Model parameters.
    thermal : ThermalOptions
        Thermal backend options.
    solver : SolverOptions
        Root-finding configuration.

    Returns
    -------
    float
        On-shell eta^2(phi, T, mu).
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


# ============================================================================
# 10. Plotting and notebook reproduction helpers
# ============================================================================

def reproduce_section3_scan(
    phi_max: float = 0.5,
    dphi: float = 0.01,
    T: float = 4.733359332047422 + 0.027,
    mu: float = 0.5,
    eta2_seed: float = 2.1214029070500127,
    params: OPTModelParams | None = None,
    thermal: ThermalOptions | None = None,
    solver: SolverOptions | None = None,
    subtract_origin: bool = True,
) -> dict[str, np.ndarray]:
    """
    Convenience helper to reproduce the final section 3 scan with the same
    numerical values used in the notebook by default.

    This function is also intentionally generic: the user may change T, mu,
    the phi-range, the step size, or the numerical options to study other
    scans with the same workflow.

    Parameters
    ----------
    phi_max : float, optional
        Maximum absolute field value. The scan is performed on
        [-phi_max, +phi_max].
    dphi : float, optional
        Field step.
    T : float, optional
        Temperature. Default reproduces the notebook section 3 choice.
    mu : float, optional
        Chemical potential. Default reproduces the notebook section 3 choice.
    eta2_seed : float, optional
        Initial eta^2 seed.
    params : OPTModelParams, optional
        Model parameters. If None, use default notebook-like values.
    thermal : ThermalOptions, optional
        Thermal backend options. If None, default to the notebook high-T
        backend used in section 3.
    solver : SolverOptions, optional
        Solver options. If None, use a robust continuation-friendly default.
    subtract_origin : bool, optional
        If True, return Delta V(phi) = V(phi) - V(0). If False, return V(phi).

    Returns
    -------
    dict[str, ndarray]
        Output of `scan_potential(...)`, enriched with metadata describing
        the scan configuration.
    """
    phi_max = float(phi_max)
    dphi = float(dphi)
    T = float(T)
    mu = float(mu)
    eta2_seed = float(eta2_seed)

    if not np.isfinite(phi_max) or phi_max <= 0.0:
        raise ValueError(f"`phi_max` must be finite and > 0, got phi_max={phi_max!r}.")
    if not np.isfinite(dphi) or dphi <= 0.0:
        raise ValueError(f"`dphi` must be finite and > 0, got dphi={dphi!r}.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"`T` must be finite and > 0, got T={T!r}.")
    if not np.isfinite(mu):
        raise ValueError(f"`mu` must be finite, got mu={mu!r}.")
    if not np.isfinite(eta2_seed):
        raise ValueError(f"`eta2_seed` must be finite, got eta2_seed={eta2_seed!r}.")

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
    scan_data["eta2_seed"] = np.array([eta2_seed], dtype=float)

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