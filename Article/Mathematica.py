from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable
import math
import mpmath as mp

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import root_scalar

from CosmoTransitions import Jb

@dataclass(frozen=True)
class OPTParams:
    """
    Parameters of the scalar OPT effective potential.

    Parameters
    ----------
    m2 : float
        Bare mass parameter m^2.
    lam : float
        Quartic coupling lambda.
    M : float
        Renormalization scale.
    thermal_approx : str, optional
        Approximation flag passed to CosmoTransitions.Jb.
    fd_rel_step : float, optional
        Relative step used in numerical derivatives of Jb.
    fd_abs_step : float, optional
        Absolute floor for the derivative step.
    """
    m2: float
    lam: float
    M: float
    thermal_approx: str = "exact"
    fd_rel_step: float = 1e-4
    fd_abs_step: float = 1e-6

class BosonicThermalOPT:
    """
    Thermal integral wrapper for the OPT notation used in the Mathematica notebook.

    Notes
    -----
    We use the notebook mapping

        H5 = J0
        H3 = J1
        H1 = J2

    together with xW = Omega^2 / T^2 and

        H5 = Jb(xW)
        H3 = -8 dJb/dxW
        H1 = 32 d^2Jb/dxW^2

    The parameter `mu` is kept only for API compatibility with the notebook.
    """

    def __init__(
        self,
        approx: str = "exact",
        fd_rel_step: float = 1e-4,
        fd_abs_step: float = 1e-6,
        jb_callable: Callable[[float], float] = None,
    ) -> None:
        self.approx = approx
        self.fd_rel_step = fd_rel_step
        self.fd_abs_step = fd_abs_step
        self.jb = jb_callable or Jb

    def _step(self, x: float) -> float:
        return max(self.fd_abs_step, self.fd_rel_step * max(1.0, abs(x)))

    def _jb0(self, x: float) -> float:
        return float(self.jb(x, approx=self.approx, deriv=0))

    def _jb1(self, x: float) -> float:
        return float(self.jb(x, approx=self.approx, deriv=1))

    def _jb2(self, x: float) -> float:
        """
        Prefer the built-in second derivative when available.
        Fall back to a finite difference of Jb'(x) otherwise.
        """
        try:
            return float(self.jb(x, approx=self.approx, deriv=2))
        except Exception:
            h = self._step(x)
            return (self._jb1(x + h) - self._jb1(x - h)) / (2.0 * h)

    def h5(self, omega2: float, T: float, mu: float = 0.0) -> float:
        if T <= 0.0:
            raise ValueError("Bosonic thermal integrals require T > 0.")
        xw = omega2 / (T * T)
        return self._jb0(xw)

    def h3(self, omega2: float, T: float, mu: float = 0.0) -> float:
        if T <= 0.0:
            raise ValueError("Bosonic thermal integrals require T > 0.")
        xw = omega2 / (T * T)
        return -8.0 * self._jb1(xw)

    def h1(self, omega2: float, T: float, mu: float = 0.0) -> float:
        if T <= 0.0:
            raise ValueError("Bosonic thermal integrals require T > 0.")
        xw = omega2 / (T * T)
        return 32.0 * self._jb2(xw)


class OPTEffectivePotential:
    """
    OPT effective potential and gap equations (Section 1 of the notebook).

    This class is a direct Python translation of the Mathematica block:

    - ring
    - vertex
    - vertexopt
    - twoloopbuble
    - OPTVEF
    - etasolve = dV/deta
    - solphi  = dV/dphi0

    while using a cleaner, high-level API.
    """

    def __init__(self, params: OPTParams, thermal: BosonicThermalOPT | None = None) -> None:
        self.p = params
        self.thermal = thermal or BosonicThermalOPT(
            approx=params.thermal_approx,
            fd_rel_step=params.fd_rel_step,
            fd_abs_step=params.fd_abs_step,
        )

    @staticmethod
    def _require_positive_omega2(omega2: float) -> None:
        if omega2 <= 0.0:
            raise ValueError(
                "This real-valued implementation requires Omega^2 = m^2 + eta^2 > 0 "
                "because of log(M / sqrt(Omega^2))."
            )

    def omega2(self, eta: float) -> float:
        """
        Omega^2 = m^2 + eta^2.
        """
        return self.p.m2 + eta * eta

    def _log_factor(self, omega2: float) -> float:
        self._require_positive_omega2(omega2)
        return math.log(self.p.M / math.sqrt(omega2))

    def _A(self, omega2: float, T: float, mu: float) -> float:
        """
        Common combination appearing in the one-loop vertex terms:

            A = Omega^2 + 2 Omega^2 log(M / sqrt(Omega^2)) - 16 T^2 H3
        """
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)
        return omega2 + 2.0 * omega2 * L - 16.0 * T * T * H3

    def _dA_domega2(self, omega2: float, T: float, mu: float) -> float:
        """
        Derivative of A with respect to Omega^2.

        Using the notebook identity dH3/d(Omega^2) = -(1/(4 T^2)) H1,

            dA/d(Omega^2) = 2 log(M / sqrt(Omega^2)) + 4 H1
        """
        L = self._log_factor(omega2)
        H1 = self.thermal.h1(omega2, T, mu)
        return 2.0 * L + 4.0 * H1

    def ring(self, eta: float, T: float, mu: float) -> float:
        omega2 = self.omega2(eta)
        L = self._log_factor(omega2)
        H5 = self.thermal.h5(omega2, T, mu)

        numerator = (
            3.0 * omega2**2
            + 4.0 * omega2**2 * L
            + 512.0 * T**4 * H5
        )
        return -numerator / (64.0 * math.pi**2)

    def vertex(self, phi0: float, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        omega2 = self.omega2(eta)
        A = self._A(omega2, T, mu)
        return -(delta * self.p.lam * phi0**2 * A) / (48.0 * math.pi**2)

    def vertex_opt(self, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        omega2 = self.omega2(eta)
        A = self._A(omega2, T, mu)
        return +(delta * eta**2 * A) / (16.0 * math.pi**2)

    def two_loop_bubble(self, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        """
        Translation of Mathematica's `twoloopbuble`.
        """
        omega2 = self.omega2(eta)
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)

        bracket = (
            omega2**2
            + 4.0 * omega2**2 * L
            + 4.0 * omega2**2 * L**2
            - 32.0 * T**2 * omega2 * H3
            - 64.0 * T**2 * omega2 * L * H3
            + 256.0 * T**4 * H3**2
        )
        return +(delta * self.p.lam * bracket) / (768.0 * math.pi**4)

    def tree_level(self, phi0: float, mu: float, delta: float = 1.0) -> float:
        return (
            0.5 * self.p.m2 * phi0**2
            - 0.5 * mu**2 * phi0**2
            + (delta * self.p.lam / 24.0) * phi0**4
        )

    def potential(self, phi0: float, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        """
        Full OPT effective potential V_eff(phi0, eta, T, mu; delta).
        """
        return (
            self.tree_level(phi0, mu, delta=delta)
            + self.ring(eta, T, mu)
            + self.vertex(phi0, eta, T, mu, delta=delta)
            + self.vertex_opt(eta, T, mu, delta=delta)
            + self.two_loop_bubble(eta, T, mu, delta=delta)
        )

    def _dring_domega2(self, omega2: float, T: float, mu: float) -> float:
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)

        dnumerator = 4.0 * omega2 * (1.0 + 2.0 * L) - 64.0 * T**2 * H3
        return -dnumerator / (64.0 * math.pi**2)

    def _dvertex_domega2(
        self,
        phi0: float,
        omega2: float,
        T: float,
        mu: float,
        delta: float,
    ) -> float:
        dA = self._dA_domega2(omega2, T, mu)
        return -(delta * self.p.lam * phi0**2 * dA) / (48.0 * math.pi**2)

    def _dtwo_loop_bubble_domega2(
        self,
        omega2: float,
        T: float,
        mu: float,
        delta: float,
    ) -> float:
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)
        H1 = self.thermal.h1(omega2, T, mu)

        dbracket = (
            4.0 * omega2 * L
            + 8.0 * omega2 * L**2
            - 64.0 * T**2 * L * H3
            + 8.0 * omega2 * H1 * (1.0 + 2.0 * L)
            - 128.0 * T**2 * H3 * H1
        )
        return +(delta * self.p.lam * dbracket) / (768.0 * math.pi**4)

    def dV_dphi0(self, phi0: float, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        """
        Derivative of the OPT effective potential with respect to phi0.

        This is the Python counterpart of Mathematica's `solphi`
        (before imposing the equation = 0).
        """
        omega2 = self.omega2(eta)
        A = self._A(omega2, T, mu)

        return (
            (self.p.m2 - mu**2) * phi0
            + (delta * self.p.lam / 6.0) * phi0**3
            - (delta * self.p.lam * phi0 * A) / (24.0 * math.pi**2)
        )

    def dV_deta(self, phi0: float, eta: float, T: float, mu: float, delta: float = 1.0) -> float:
        """
        Derivative of the OPT effective potential with respect to eta.

        This is the Python counterpart of Mathematica's `etasolve`
        (before imposing the equation = 0).
        """
        omega2 = self.omega2(eta)
        dA = self._dA_domega2(omega2, T, mu)
        A = self._A(omega2, T, mu)

        # Terms that depend on eta only through Omega^2 = m^2 + eta^2
        chain_part = 2.0 * eta * (
            self._dring_domega2(omega2, T, mu)
            + self._dvertex_domega2(phi0, omega2, T, mu, delta)
            + self._dtwo_loop_bubble_domega2(omega2, T, mu, delta)
        )

        # vertex_opt has explicit eta^2 dependence in addition to Omega^2(eta)
        explicit_vertexopt = (delta * eta * (A + eta**2 * dA)) / (8.0 * math.pi**2)

        return chain_part + explicit_vertexopt

    def phi_gap_equation(self, phi0: float, eta: float, T: float, mu: float) -> float:
        """
        Physical phi-gap equation used in the notebook.

        This is the nontrivial branch of dV/dphi0 = 0 after factoring out phi0.
        Therefore:
            - phi0 = 0 is always a separate branch;
            - phi_gap_equation = 0 describes the symmetry-broken branch.
        """
        omega2 = self.omega2(eta)
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)
        lam = self.p.lam

        return (
            24.0 * self.p.m2
            - 24.0 * mu**2
            + (16.0 * T**2 * lam * H3 - omega2 * lam * (1.0 + 2.0 * L)) / (math.pi**2)
            + 4.0 * lam * phi0**2
        )

    def eta_gap_equation(self, phi0: float, eta: float, T: float, mu: float) -> float:
        """
        Physical OPT/PMS eta-gap equation used in the notebook.

        This is the branch discussed in the notebook comments as the one
        containing the physical real roots.
        """
        omega2 = self.omega2(eta)
        L = self._log_factor(omega2)
        H3 = self.thermal.h3(omega2, T, mu)
        lam = self.p.lam

        return (
            24.0 * math.pi**2 * eta**2
            + lam * omega2
            - 16.0 * T**2 * lam * H3
            + 2.0 * lam * omega2 * L
            - 8.0 * math.pi**2 * lam * phi0**2
        )

    def thermal_mass_sq_1loop_from_omega2(self, omega2: float, T: float, mu: float) -> float:
        """
        One-loop OPT thermal mass-squared written in terms of Omega^2.
        """
        A = self._A(omega2, T, mu)
        return mu**2 + (self.p.lam * A) / (24.0 * math.pi**2)

    def thermal_mass_sq_1loop(self, eta: float, T: float, mu: float) -> float:
        """
        One-loop OPT thermal mass-squared using Omega^2 = m^2 + eta^2.
        """
        return self.thermal_mass_sq_1loop_from_omega2(self.omega2(eta), T, mu)

    def gap_equations(self, phi0: float, eta: float, T: float, mu: float) -> Dict[str, float]:
        """
        Return the physical Section-1 gap equations at delta = 1.
        """
        return {
            "eta_gap": self.eta_gap_equation(phi0, eta, T, mu),
            "phi_gap": self.phi_gap_equation(phi0, eta, T, mu),
        }

    def evaluate_all(self, phi0: float, eta: float, T: float, mu: float, delta: float = 1.0) -> Dict[str, float]:
        """
        Convenience method returning all Section 1 building blocks.
        """
        return {
            "tree_level": self.tree_level(phi0, mu, delta=delta),
            "ring": self.ring(eta, T, mu),
            "vertex": self.vertex(phi0, eta, T, mu, delta=delta),
            "vertex_opt": self.vertex_opt(eta, T, mu, delta=delta),
            "two_loop_bubble": self.two_loop_bubble(eta, T, mu, delta=delta),
            "Veff": self.potential(phi0, eta, T, mu, delta=delta),
            "raw_dV_deta": self.dV_deta(phi0, eta, T, mu, delta=delta),
            "raw_dV_dphi0": self.dV_dphi0(phi0, eta, T, mu, delta=delta),
            "eta_gap": self.eta_gap_equation(phi0, eta, T, mu),
            "phi_gap": self.phi_gap_equation(phi0, eta, T, mu),
            "thermal_mass_sq_1loop": self.thermal_mass_sq_1loop(eta, T, mu),
        }


# =============================================================================
# Section 2 — Critical point and the best parameters (high-T approximation)
# =============================================================================

import matplotlib.pyplot as plt


class BosonicThermalHighT:
    """
    High-temperature bosonic thermal functions used in Section 2.
    """

    def __init__(
        self,
        mp_dps: int = 40,
        domain_tol: float = 1e-12,
        edge_tol: float = 1e-10,
        cache_digits: int = 12,
    ) -> None:
        self.mp_dps = mp_dps
        self.domain_tol = domain_tol
        self.edge_tol = edge_tol
        self.cache_digits = cache_digits
        self._cache: dict[tuple[int, float, float, float], float] = {}

    def _cache_key(self, l: int, omega2: float, T: float, mu: float) -> tuple[int, float, float, float]:
        return (
            l,
            round(float(omega2), self.cache_digits),
            round(float(T), self.cache_digits),
            round(float(mu), self.cache_digits),
        )

    def _to_real_float(self, z: mp.mpf | mp.mpc) -> float:
        zc = complex(z)
        if abs(zc.imag) > 1e-10 * max(1.0, abs(zc.real)):
            raise ValueError(f"High-T thermal function developed a non-negligible imaginary part: {zc}")
        return float(zc.real)

    def _check_domain(self, omega2: float, T: float, mu: float, l: int) -> None:
        if omega2 <= 0.0:
            raise ValueError("High-T thermal functions require omega2 > 0.")
        if T <= 0.0:
            raise ValueError("High-T thermal functions require T > 0.")

        # For H3 and H5, the r -> 1 limit is physically relevant in Section 2.
        # For H1 it is singular, so we keep the stricter condition there.
        if l == 0:
            if mu * mu >= omega2 - self.domain_tol:
                raise ValueError("H1 high-T is not used at the r -> 1 boundary.")
        else:
            if mu * mu > omega2 + self.domain_tol:
                raise ValueError("High-T thermal functions require mu^2 <= omega2 for H3/H5.")

    def he_odd(self, l: int, omega2: float, T: float, mu: float) -> float:
        """
        Evaluate h^(e)_(2l+1)[l, y, r] with
            y = sqrt(omega2)/T, r = mu/sqrt(omega2).
        """
        if l not in (0, 1, 2):
            raise ValueError("Only l = 0, 1, 2 are needed here.")

        self._check_domain(omega2, T, mu, l=l)

        key = self._cache_key(l, omega2, T, mu)
        if key in self._cache:
            return self._cache[key]

        with mp.workdps(self.mp_dps):
            omega2_mp = mp.mpf(omega2)
            T_mp = mp.mpf(T)
            mu_mp = mp.mpf(mu)

            y = mp.sqrt(omega2_mp) / T_mp
            r = mu_mp / mp.sqrt(omega2_mp)
            r2 = r * r

            # Clamp the boundary r^2 -> 1 for H3/H5, which is relevant at the critical point
            if l >= 1 and abs(float(1 - r2)) < self.edge_tol:
                r2 = mp.mpf("1.0")

            gamma = mp.gamma
            digamma = mp.digamma
            zeta = mp.zeta
            hyp2f1 = mp.hyp2f1
            pfq = lambda a, b, x: mp.hyper(a, b, x)

            term1 = (
                ((-1) ** l)
                * mp.pi
                * (1 - r2) ** (-mp.mpf("0.5") + l)
                * y ** (-1 + 2 * l)
                / (2 * gamma(1 + 2 * l))
            )

            term2 = (
                ((-1) ** l)
                * (mp.mpf(2) ** (-2 * l))
                * y ** (2 * l)
                / (2 * gamma(1 + l) ** 2)
            ) * (
                l * r2 * pfq([1, 1, 1 - l], [mp.mpf("1.5"), 2], r2)
                + mp.log(y / (4 * mp.pi))
                + mp.mpf("0.5") * (mp.euler - digamma(1 + l))
            )

            term3_sum = mp.mpf("0.0")
            for k in range(0, l):
                term3_sum += (
                    1 / gamma(1 + k)
                    * ((-1) ** k)
                    * (mp.mpf(2) ** (-2 * k))
                    * y ** (2 * k)
                    * gamma(-k + l)
                    * hyp2f1(-k, -k + l, mp.mpf("0.5"), r2)
                    * zeta(-2 * k + 2 * l)
                )
            term3 = term3_sum / (2 * gamma(1 + l))

            term4_sum = mp.mpf("0.0")
            for k in range(1, 4):
                term4_sum += (
                    ((-1) ** k)
                    * (y / (4 * mp.pi)) ** (2 * k)
                    * gamma(2 * k + 1)
                    * zeta(2 * k + 1)
                    / (gamma(1 + k) * gamma(1 + l + k))
                    * hyp2f1(-k, -l - k, mp.mpf("0.5"), r2)
                )
            term4 = (
                ((-1) ** l)
                * (y / 2) ** (2 * l)
                / (2 * gamma(1 + l))
                * term4_sum
            )

            value = self._to_real_float(term1 + term2 + term3 + term4)
            self._cache[key] = value
            return value

    def h1(self, omega2: float, T: float, mu: float) -> float:
        return self.he_odd(0, omega2, T, mu)

    def h3(self, omega2: float, T: float, mu: float) -> float:
        return self.he_odd(1, omega2, T, mu)

    def h5(self, omega2: float, T: float, mu: float) -> float:
        return self.he_odd(2, omega2, T, mu)


class OPTHighTAnalysis:
    """
    High-temperature OPT analysis for Section 2.

    This version is optimized for robustness and speed:
    - critical point solved as a 1D problem in T, using the exact critical relation
      eta_c^2 = mu^2 - m^2;
    - broken branch at fixed (T, mu) solved as a 1D problem in eta^2,
      reconstructing phi^2 analytically from solphi = 0;
    - eta-only branch solved as a 1D scalar root.
    """

    def __init__(self, params: OPTParams) -> None:
        self.params = replace(params, thermal_approx="high")
        self.highT = BosonicThermalHighT(mp_dps=40, cache_digits=12)
        self._omega2_eps = 1e-12
        self._t_eps = 1e-10
        self._phi2_eps = 1e-10

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def omega2_from_eta2(self, eta2: float) -> float:
        return self.params.m2 + float(eta2)

    def eta2_lower_bound(self, mu: float = 0.0) -> float:
        return max(0.0, mu * mu - self.params.m2 + self._omega2_eps)

    @staticmethod
    def _sqrt_nonnegative(x: float) -> float:
        return math.sqrt(max(0.0, float(x)))

    @staticmethod
    def _max_abs(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
        a = np.asarray(arr, dtype=float)
        return float(np.max(np.abs(a)))

    # ------------------------------------------------------------------
    # High-T physical gap equations in squared variables
    # ------------------------------------------------------------------

    def eta_gap_sqvars(self, eta2: float, phi2: float, T: float, mu: float) -> float:
        omega2 = self.omega2_from_eta2(eta2)
        if omega2 < mu * mu - self._omega2_eps:
            raise ValueError("Need omega2 >= mu^2 in the high-T domain.")

        L = math.log(self.params.M / math.sqrt(max(omega2, self._omega2_eps)))
        H3 = self.highT.h3(omega2, T, mu)
        lam = self.params.lam

        return (
            24.0 * math.pi**2 * float(eta2)
            + lam * omega2
            - 16.0 * T**2 * lam * H3
            + 2.0 * lam * omega2 * L
            - 8.0 * math.pi**2 * lam * float(phi2)
        )

    def phi_gap_sqvars(self, eta2: float, phi2: float, T: float, mu: float) -> float:
        omega2 = self.omega2_from_eta2(eta2)
        if omega2 < mu * mu - self._omega2_eps:
            raise ValueError("Need omega2 >= mu^2 in the high-T domain.")

        L = math.log(self.params.M / math.sqrt(max(omega2, self._omega2_eps)))
        H3 = self.highT.h3(omega2, T, mu)
        lam = self.params.lam

        return (
            24.0 * self.params.m2
            - 24.0 * mu**2
            + (16.0 * T**2 * lam * H3 - omega2 * lam * (1.0 + 2.0 * L)) / (math.pi**2)
            + 4.0 * lam * float(phi2)
        )

    def mass_sq_background(self, eta2: float, T: float, mu: float) -> float:
        omega2 = self.omega2_from_eta2(eta2)
        if omega2 < mu * mu - self._omega2_eps:
            raise ValueError("Need omega2 >= mu^2 in the high-T domain.")

        L = math.log(self.params.M / math.sqrt(max(omega2, self._omega2_eps)))
        H3 = self.highT.h3(omega2, T, mu)
        lam = self.params.lam

        return (
            self.params.m2
            - mu**2
            + (16.0 * T**2 * lam * H3 - omega2 * lam * (1.0 + 2.0 * L))
            / (24.0 * math.pi**2)
        )

    # ------------------------------------------------------------------
    # Scalar root helpers
    # ------------------------------------------------------------------

    def _scan_bracket(
        self,
        func,
        x_min: float,
        x_max: float,
        npts: int = 60,
    ) -> tuple[float, float] | None:
        xs = np.linspace(x_min, x_max, npts)
        prev_x = None
        prev_f = None

        for x in xs:
            try:
                fx = float(func(float(x)))
                if not np.isfinite(fx):
                    prev_x = None
                    prev_f = None
                    continue
            except Exception:
                prev_x = None
                prev_f = None
                continue

            if prev_x is not None and prev_f is not None:
                if prev_f == 0.0:
                    return prev_x, prev_x
                if fx == 0.0:
                    return float(x), float(x)
                if np.sign(prev_f) != np.sign(fx):
                    return float(prev_x), float(x)

            prev_x = float(x)
            prev_f = fx

        return None

    def _solve_scalar_root(
        self,
        func,
        x0: float,
        x1: float,
        x_min: float,
        x_max: float,
        residual_tol: float = 1e-6,
        maxiter: int = 80,
    ) -> tuple[float, float]:
        # Fast try: secant
        try:
            sol = root_scalar(func, method="secant", x0=x0, x1=x1, maxiter=maxiter)
            if sol.converged:
                x = float(sol.root)
                fx = float(func(x))
                if np.isfinite(fx) and abs(fx) <= residual_tol:
                    return x, fx
        except Exception:
            pass

        # Robust fallback: bracket + brentq
        bracket = self._scan_bracket(func, x_min=x_min, x_max=x_max, npts=80)
        if bracket is None:
            # return best effort around x0
            fx0 = float(func(x0))
            if np.isfinite(fx0) and abs(fx0) <= 10.0 * residual_tol:
                return float(x0), fx0
            raise RuntimeError("No sign change found for scalar root.")

        a, b = bracket
        if a == b:
            fa = float(func(a))
            return float(a), fa

        sol = root_scalar(func, method="brentq", bracket=(a, b), maxiter=maxiter)
        x = float(sol.root)
        fx = float(func(x))
        if not np.isfinite(fx) or abs(fx) > 10.0 * residual_tol:
            raise RuntimeError(f"Scalar root residual too large: {fx:.3e}")
        return x, fx

    # ------------------------------------------------------------------
    # Critical point
    # ------------------------------------------------------------------

    def critical_eta2(self, mu: float) -> float:
        """
        From the two critical equations with phi^2 = 0 one gets directly
            eta_c^2 = mu^2 - m^2.
        """
        return max(0.0, mu * mu - self.params.m2)

    def critical_temperature_sq_pt(self, mu: float) -> float:
        lam = self.params.lam
        if lam == 0.0:
            raise ZeroDivisionError("PT critical temperature is undefined for lambda = 0.")
        return -12.0 * (self.params.m2 - mu**2) / ((2.0 / 3.0) * lam)

    def critical_temperature_pt(self, mu: float) -> float:
        return self._sqrt_nonnegative(self.critical_temperature_sq_pt(mu))

    def _critical_phi_only(self, T: float, mu: float) -> float:
        eta2 = self.critical_eta2(mu)
        return self.phi_gap_sqvars(eta2=eta2, phi2=0.0, T=T, mu=mu)

    def solve_critical_point(
        self,
        mu: float,
        eta2_guess: float = 10.0,
        T2_guess: float = 100.0,
        residual_tol: float = 1e-6,
    ) -> Dict[str, float]:
        eta2 = self.critical_eta2(mu)

        T_guess = max(self._t_eps, self._sqrt_nonnegative(T2_guess))
        T_pt = max(self._t_eps, self.critical_temperature_pt(mu))

        x0 = T_guess
        x1 = max(self._t_eps, 1.05 * T_pt + 1e-3)
        x_min = max(self._t_eps, 0.25 * T_pt)
        x_max = max(2.0 * T_pt, T_guess + 5.0, 8.0)

        T_root, phi_gap = self._solve_scalar_root(
            func=lambda T: self._critical_phi_only(T, mu),
            x0=x0,
            x1=x1,
            x_min=x_min,
            x_max=x_max,
            residual_tol=residual_tol,
            maxiter=80,
        )

        eta_gap = float(self.eta_gap_sqvars(eta2=eta2, phi2=0.0, T=T_root, mu=mu))
        residual_norm = self._max_abs([eta_gap, phi_gap])

        return {
            "mu": float(mu),
            "eta2": eta2,
            "eta": self._sqrt_nonnegative(eta2),
            "T2": T_root * T_root,
            "T": T_root,
            "eta_gap": eta_gap,
            "phi_gap": phi_gap,
            "residual_norm": residual_norm,
            "solver_success": True,
            "solver_status": 0,
            "solver_message": "critical point solved via 1D root in T",
        }

    def critical_temperature_sq_opt(
        self,
        mu: float,
        eta2_guess: float = 10.0,
        T2_guess: float = 100.0,
        **solver_kwargs,
    ) -> float:
        return self.solve_critical_point(
            mu=mu,
            eta2_guess=eta2_guess,
            T2_guess=T2_guess,
            **solver_kwargs,
        )["T2"]

    def critical_temperature_opt(
        self,
        mu: float,
        eta2_guess: float = 10.0,
        T2_guess: float = 100.0,
        **solver_kwargs,
    ) -> float:
        return self._sqrt_nonnegative(
            self.critical_temperature_sq_opt(
                mu=mu,
                eta2_guess=eta2_guess,
                T2_guess=T2_guess,
                **solver_kwargs,
            )
        )

    # ------------------------------------------------------------------
    # Broken branch at fixed (T, mu)
    # ------------------------------------------------------------------

    def phi2_from_phi_gap(self, eta2: float, T: float, mu: float) -> float:
        lam = self.params.lam
        if lam == 0.0:
            raise ZeroDivisionError("Cannot reconstruct phi^2 from solphi when lambda = 0.")

        omega2 = self.omega2_from_eta2(eta2)
        if omega2 < mu * mu - self._omega2_eps:
            raise ValueError("Need omega2 >= mu^2 in the high-T domain.")

        L = math.log(self.params.M / math.sqrt(max(omega2, self._omega2_eps)))
        H3 = self.highT.h3(omega2, T, mu)

        numerator = (
            24.0 * self.params.m2
            - 24.0 * mu**2
            + (16.0 * T**2 * lam * H3 - omega2 * lam * (1.0 + 2.0 * L)) / (math.pi**2)
        )
        return -numerator / (4.0 * lam)

    def broken_branch_residual(self, eta2: float, T: float, mu: float) -> float:
        phi2 = self.phi2_from_phi_gap(eta2, T, mu)
        if phi2 < -self._phi2_eps:
            raise ValueError("This eta^2 is outside the physical broken branch (phi^2 < 0).")
        return self.eta_gap_sqvars(eta2=eta2, phi2=max(0.0, phi2), T=T, mu=mu)

    def solve_best_parameters(
        self,
        T: float,
        mu: float,
        eta2_guess: float = 4.0,
        phi2_guess: float = 100.0,
        residual_tol: float = 1e-6,
    ) -> Dict[str, float]:
        if T <= 0.0:
            raise ValueError("The best-parameter solver requires T > 0.")

        lower = self.eta2_lower_bound(mu) + 1e-8
        x0 = max(lower, eta2_guess if eta2_guess > 0 else lower + 1.0)
        x1 = x0 * 1.03 + 1e-3
        x_min = lower
        x_max = max(lower + 20.0, x0 + 20.0)

        eta2_root, eta_gap = self._solve_scalar_root(
            func=lambda eta2: self.broken_branch_residual(eta2, T, mu),
            x0=x0,
            x1=x1,
            x_min=x_min,
            x_max=x_max,
            residual_tol=residual_tol,
            maxiter=80,
        )

        phi2 = max(0.0, float(self.phi2_from_phi_gap(eta2_root, T, mu)))
        phi_gap = float(self.phi_gap_sqvars(eta2_root, phi2, T, mu))
        residual_norm = self._max_abs([eta_gap, phi_gap])

        return {
            "T": float(T),
            "mu": float(mu),
            "eta2": eta2_root,
            "eta": self._sqrt_nonnegative(eta2_root),
            "phi2": phi2,
            "phi0": self._sqrt_nonnegative(phi2),
            "eta_gap": eta_gap,
            "phi_gap": phi_gap,
            "residual_norm": residual_norm,
            "solver_success": True,
            "solver_status": 0,
            "solver_message": "best parameters solved via 1D broken-branch root",
        }

    def solve_eta_only(
        self,
        T: float,
        mu: float,
        eta2_guess: float,
        residual_tol: float = 1e-6,
    ) -> Dict[str, float]:
        if T <= 0.0:
            raise ValueError("The eta-only solver requires T > 0.")

        lower = self.eta2_lower_bound(mu) + 1e-8
        x0 = max(lower, eta2_guess if eta2_guess > 0 else lower + 1.0)
        x1 = x0 * 1.03 + 1e-3
        x_min = lower
        x_max = max(lower + 20.0, x0 + 20.0)

        eta2_root, eta_gap = self._solve_scalar_root(
            func=lambda eta2: self.eta_gap_sqvars(eta2=eta2, phi2=0.0, T=T, mu=mu),
            x0=x0,
            x1=x1,
            x_min=x_min,
            x_max=x_max,
            residual_tol=residual_tol,
            maxiter=80,
        )

        return {
            "T": float(T),
            "mu": float(mu),
            "eta2": eta2_root,
            "eta": self._sqrt_nonnegative(eta2_root),
            "phi2": 0.0,
            "phi0": 0.0,
            "eta_gap": eta_gap,
            "residual_norm": abs(eta_gap),
            "solver_success": True,
            "solver_status": 0,
            "solver_message": "eta-only branch solved via 1D root",
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_critical_point_solution(self, mu: float, eta2: float, T2: float) -> Dict[str, float]:
        T = self._sqrt_nonnegative(T2)
        return {
            "eta_gap": self.eta_gap_sqvars(eta2=eta2, phi2=0.0, T=T, mu=mu),
            "phi_gap": self.phi_gap_sqvars(eta2=eta2, phi2=0.0, T=T, mu=mu),
        }

    def evaluate_best_parameter_solution(
        self,
        T: float,
        mu: float,
        eta2: float,
        phi2: float,
    ) -> Dict[str, float]:
        return {
            "eta_gap": self.eta_gap_sqvars(eta2=eta2, phi2=phi2, T=T, mu=mu),
            "phi_gap": self.phi_gap_sqvars(eta2=eta2, phi2=phi2, T=T, mu=mu),
        }

    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

    def scan_critical_curve_opt(
        self,
        mu_values: Iterable[float],
        eta2_guess: float = 10.0,
        T2_guess: float = 100.0,
        **solver_kwargs,
    ) -> np.ndarray:
        mu_values = np.asarray(list(mu_values), dtype=float)
        curve = np.empty((len(mu_values), 2), dtype=float)

        current_T2 = float(T2_guess)

        for i, mu in enumerate(mu_values):
            sol = self.solve_critical_point(
                mu=float(mu),
                eta2_guess=eta2_guess,
                T2_guess=current_T2,
                **solver_kwargs,
            )
            curve[i, 0] = float(mu)
            curve[i, 1] = sol["T"]
            current_T2 = sol["T2"]

        return curve

    def scan_critical_curve_pt(self, mu_values: Iterable[float]) -> np.ndarray:
        mu_values = np.asarray(list(mu_values), dtype=float)
        return np.array(
            [[float(mu), self.critical_temperature_pt(float(mu))] for mu in mu_values],
            dtype=float,
        )

    def best_phi_sq(
        self,
        T: float,
        mu: float,
        eta2_guess: float = 2.0,
        phi2_guess: float = 6.0,
        **solver_kwargs,
    ) -> float:
        return self.solve_best_parameters(
            T=T,
            mu=mu,
            eta2_guess=eta2_guess,
            phi2_guess=phi2_guess,
            **solver_kwargs,
        )["phi2"]

    def best_phi(
        self,
        T: float,
        mu: float,
        eta2_guess: float = 2.0,
        phi2_guess: float = 6.0,
        **solver_kwargs,
    ) -> float:
        return self._sqrt_nonnegative(
            self.best_phi_sq(
                T=T,
                mu=mu,
                eta2_guess=eta2_guess,
                phi2_guess=phi2_guess,
                **solver_kwargs,
            )
        )

    # ------------------------------------------------------------------
    # Temperature branch continuation
    # ------------------------------------------------------------------

    def trace_temperature_branch(
        self,
        T_start: float,
        T_stop: float,
        dT: float,
        mu: float,
        eta2_init: float,
        phi2_init: float,
        full_residual_tol: float = 1e-6,
        eta_only_residual_tol: float = 1e-6,
        phi2_floor: float = 1e-10,
        allow_fallback_to_phi0: bool = True,
    ) -> Dict[str, np.ndarray]:
        if T_start <= 0.0 or T_stop < T_start or dT <= 0.0:
            raise ValueError("Require T_start > 0, T_stop >= T_start, and dT > 0.")

        T_values = np.arange(T_start, T_stop + 0.5 * dT, dT, dtype=float)

        eta2_list = []
        eta_list = []
        phi2_list = []
        phi0_list = []
        mass_sq_list = []
        eta_gap_list = []
        phi_gap_list = []

        eta2_current = float(eta2_init)
        phi2_current = float(phi2_init)

        for T in T_values:
            if phi2_current <= phi2_floor:
                sol = self.solve_eta_only(
                    T=T,
                    mu=mu,
                    eta2_guess=max(self.eta2_lower_bound(mu) + 1e-8, eta2_current),
                    residual_tol=eta_only_residual_tol,
                )
                eta2_current = float(sol["eta2"])
                phi2_current = 0.0
                eta_gap_val = float(sol["eta_gap"])
                phi_gap_val = float(self.phi_gap_sqvars(eta2_current, 0.0, T, mu))
            else:
                try:
                    sol = self.solve_best_parameters(
                        T=T,
                        mu=mu,
                        eta2_guess=max(self.eta2_lower_bound(mu) + 1e-8, eta2_current),
                        phi2_guess=max(1e-8, phi2_current),
                        residual_tol=full_residual_tol,
                    )
                    eta2_current = float(sol["eta2"])
                    phi2_current = float(sol["phi2"])
                    eta_gap_val = float(sol["eta_gap"])
                    phi_gap_val = float(sol["phi_gap"])

                    if phi2_current <= phi2_floor:
                        phi2_current = 0.0

                except RuntimeError:
                    if not allow_fallback_to_phi0:
                        raise
                    sol = self.solve_eta_only(
                        T=T,
                        mu=mu,
                        eta2_guess=max(self.eta2_lower_bound(mu) + 1e-8, eta2_current),
                        residual_tol=eta_only_residual_tol,
                    )
                    eta2_current = float(sol["eta2"])
                    phi2_current = 0.0
                    eta_gap_val = float(sol["eta_gap"])
                    phi_gap_val = float(self.phi_gap_sqvars(eta2_current, 0.0, T, mu))

            eta_val = self._sqrt_nonnegative(eta2_current)
            phi0_val = self._sqrt_nonnegative(phi2_current)
            mass_sq_val = float(self.mass_sq_background(eta2_current, T, mu))

            eta2_list.append(eta2_current)
            eta_list.append(eta_val)
            phi2_list.append(phi2_current)
            phi0_list.append(phi0_val)
            mass_sq_list.append(mass_sq_val)
            eta_gap_list.append(eta_gap_val)
            phi_gap_list.append(phi_gap_val)

        return {
            "T": np.asarray(T_values, dtype=float),
            "eta2": np.asarray(eta2_list, dtype=float),
            "eta": np.asarray(eta_list, dtype=float),
            "phi2": np.asarray(phi2_list, dtype=float),
            "phi0": np.asarray(phi0_list, dtype=float),
            "mass_sq": np.asarray(mass_sq_list, dtype=float),
            "eta_gap": np.asarray(eta_gap_list, dtype=float),
            "phi_gap": np.asarray(phi_gap_list, dtype=float),
        }


# ----------------------------------------------------------------------
# Default Section-2 setup matching the Mathematica substitutions
# ----------------------------------------------------------------------

def make_section2_default_model() -> OPTHighTAnalysis:
    return OPTHighTAnalysis(
        OPTParams(
            m2=-1.0,
            lam=1.0,
            M=1.0,
            thermal_approx="high",
        )
    )


# ----------------------------------------------------------------------
# Print reproductions
# ----------------------------------------------------------------------

def print_section2_first_tests() -> Dict[str, Dict[str, float]]:
    analysis = make_section2_default_model()

    critical_sol = analysis.solve_critical_point(mu=0.5, eta2_guess=4.0, T2_guess=10.0)
    critical_check = analysis.evaluate_critical_point_solution(
        mu=0.5,
        eta2=critical_sol["eta2"],
        T2=critical_sol["T2"],
    )

    best_sol = analysis.solve_best_parameters(
        T=math.sqrt(22.0),
        mu=0.5,
        eta2_guess=4.0,
        phi2_guess=100.0,
    )
    best_check = analysis.evaluate_best_parameter_solution(
        T=math.sqrt(22.0),
        mu=0.5,
        eta2=best_sol["eta2"],
        phi2=best_sol["phi2"],
    )

    first_fixed_sol = analysis.solve_best_parameters(
        T=1.0,
        mu=0.0,
        eta2_guess=2.0,
        phi2_guess=6.0,
    )
    first_fixed_check = analysis.evaluate_best_parameter_solution(
        T=1.0,
        mu=0.0,
        eta2=first_fixed_sol["eta2"],
        phi2=first_fixed_sol["phi2"],
    )

    T2c_pt_mu05 = analysis.critical_temperature_sq_pt(mu=0.5)

    print("\n=== Section 2 | First critical-point test (mu = 0.5) ===")
    print(f"eta^2 = {critical_sol['eta2']:.15f}")
    print(f"T^2   = {critical_sol['T2']:.15f}")
    print(f"eta   = {critical_sol['eta']:.15f}")
    print(f"T     = {critical_sol['T']:.15f}")
    print("Residual check:")
    print(f"  eta_gap = {critical_check['eta_gap']:.12e}")
    print(f"  phi_gap = {critical_check['phi_gap']:.12e}")

    print("\n=== Section 2 | Best parameters at (T, mu) = (sqrt(22), 0.5) ===")
    print(f"eta^2 = {best_sol['eta2']:.15f}")
    print(f"phi^2 = {best_sol['phi2']:.15f}")
    print(f"eta   = {best_sol['eta']:.15f}")
    print(f"phi0  = {best_sol['phi0']:.15f}")
    print("Residual check:")
    print(f"  eta_gap = {best_check['eta_gap']:.12e}")
    print(f"  phi_gap = {best_check['phi_gap']:.12e}")

    print("\n=== Section 2 | PT analytic critical temperature squared at mu = 0.5 ===")
    print(f"T_c^2 (PT) = {T2c_pt_mu05:.15f}")
    print(f"T_c   (PT) = {math.sqrt(max(0.0, T2c_pt_mu05)):.15f}")

    print("\n=== Section 2 | First fixed-(T, mu) best-parameter test at (1, 0) ===")
    print(f"eta^2 = {first_fixed_sol['eta2']:.15f}")
    print(f"phi^2 = {first_fixed_sol['phi2']:.15f}")
    print(f"eta   = {first_fixed_sol['eta']:.15f}")
    print(f"phi0  = {first_fixed_sol['phi0']:.15f}")
    print("Residual check:")
    print(f"  eta_gap = {first_fixed_check['eta_gap']:.12e}")
    print(f"  phi_gap = {first_fixed_check['phi_gap']:.12e}")

    return {
        "critical_sol_mu_0p5": critical_sol,
        "critical_check_mu_0p5": critical_check,
        "best_sol_T_sqrt22_mu_0p5": best_sol,
        "best_check_T_sqrt22_mu_0p5": best_check,
        "first_fixed_sol_T1_mu0": first_fixed_sol,
        "first_fixed_check_T1_mu0": first_fixed_check,
        "T2c_pt_mu_0p5": {"T2c_pt": T2c_pt_mu05},
    }


# ----------------------------------------------------------------------
# Plot reproductions
# ----------------------------------------------------------------------

def plot_critical_curves_simple() -> tuple[plt.Figure, plt.Axes]:
    analysis = make_section2_default_model()

    mu_opt = np.arange(0.0, 0.99 + 0.5 * 0.025, 0.025, dtype=float)
    mu_pt = np.arange(0.0, 0.99 + 0.5 * 0.02, 0.02, dtype=float)

    curve_opt = analysis.scan_critical_curve_opt(mu_opt, eta2_guess=10.0, T2_guess=100.0)
    curve_pt = analysis.scan_critical_curve_pt(mu_pt)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_opt[:, 0], curve_opt[:, 1], marker="o", markersize=3, linewidth=1.5, label="OPT-PMS")
    ax.plot(curve_pt[:, 0], curve_pt[:, 1], marker="o", markersize=3, linewidth=1.5, label="PT")
    ax.set_xlabel(r"$\mu / M$")
    ax.set_ylabel(r"$T_c / M$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_critical_curves_styled() -> tuple[plt.Figure, plt.Axes]:
    analysis = make_section2_default_model()

    mu_opt = np.arange(0.001, 0.99 + 0.5 * 0.025, 0.025, dtype=float)
    mu_pt = np.arange(0.0, 0.99 + 0.5 * 0.02, 0.02, dtype=float)

    curve_opt = analysis.scan_critical_curve_opt(mu_opt, eta2_guess=10.0, T2_guess=100.0)
    curve_pt = analysis.scan_critical_curve_pt(mu_pt)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(
        curve_opt[:, 0],
        curve_opt[:, 1],
        marker="o",
        markersize=4,
        markerfacecolor="none",
        linewidth=1.6,
        label="OPT-PMS",
    )
    ax.plot(
        curve_pt[:, 0],
        curve_pt[:, 1],
        marker="o",
        markersize=4,
        markerfacecolor="none",
        linewidth=1.6,
        label="PT",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(4.0, 6.2)
    ax.set_xlabel(r"$\mu/M$", fontsize=14)
    ax.set_ylabel(r"$T_c/M$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_quick_field_comparison() -> tuple[plt.Figure, plt.Axes, Dict[str, np.ndarray]]:
    analysis = make_section2_default_model()

    branch_mu0 = analysis.trace_temperature_branch(
        T_start=1.0,
        T_stop=6.0,
        dT=0.1,
        mu=0.0,
        eta2_init=1.9807424572910532,
        phi2_init=5.88445474374632,
    )

    branch_mu04 = analysis.trace_temperature_branch(
        T_start=1.0,
        T_stop=6.0,
        dT=0.1,
        mu=0.4,
        eta2_init=2.0,
        phi2_init=1.0,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(branch_mu0["T"], branch_mu0["phi0"], marker="o", markersize=3, linewidth=1.5, label=r"$\mu=0$")
    ax.plot(branch_mu04["T"], branch_mu04["phi0"], marker="o", markersize=3, linewidth=1.5, label=r"$\mu=0.4$")
    ax.set_xlabel(r"$T/M$")
    ax.set_ylabel(r"$\tilde{\phi}/M$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax, {"mu0": branch_mu0, "mu04": branch_mu04}


def plot_mu0_temperature_branch() -> tuple[plt.Figure, plt.Axes, Dict[str, np.ndarray]]:
    analysis = make_section2_default_model()

    branch = analysis.trace_temperature_branch(
        T_start=1.0,
        T_stop=7.0,
        dT=0.025,
        mu=0.0,
        eta2_init=1.9807424572910532,
        phi2_init=5.88445474374632,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(
        branch["T"],
        branch["phi0"],
        marker="o",
        markersize=3,
        markerfacecolor="none",
        linewidth=1.5,
        color="black",
        label=r"$\tilde{\phi}/M$",
    )
    ax.set_xlabel(r"$T/M$", fontsize=14)
    ax.set_ylabel(r"$\tilde{\phi}/M$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig, ax, branch


def plot_mu0_mass_sq_branch() -> tuple[plt.Figure, plt.Axes, Dict[str, np.ndarray]]:
    analysis = make_section2_default_model()

    branch = analysis.trace_temperature_branch(
        T_start=1.0,
        T_stop=7.0,
        dT=0.025,
        mu=0.0,
        eta2_init=1.9807424572910532,
        phi2_init=5.88445474374632,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(branch["T"], branch["mass_sq"], marker="o", markersize=2.5, linewidth=1.2)
    ax.set_xlabel(r"$T/M$", fontsize=14)
    ax.set_ylabel(r"$m_{\mathrm{eff}}^2/M^2$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig, ax, branch


# ----------------------------------------------------------------------
# One-call reproduction of all prints and plots sent so far in Section 2
# ----------------------------------------------------------------------

def reproduce_section2_until_here(show: bool = True) -> Dict[str, object]:
    results = print_section2_first_tests()

    fig1, ax1 = plot_critical_curves_simple()
    fig2, ax2 = plot_critical_curves_styled()
    fig3, ax3, quick_branches = plot_quick_field_comparison()
    fig4, ax4, mu0_branch = plot_mu0_temperature_branch()
    fig5, ax5, mu0_mass_branch = plot_mu0_mass_sq_branch()

    if show:
        plt.show()

    return {
        "prints": results,
        "fig_critical_simple": (fig1, ax1),
        "fig_critical_styled": (fig2, ax2),
        "fig_quick_field_comparison": (fig3, ax3),
        "fig_mu0_branch_field": (fig4, ax4),
        "fig_mu0_branch_mass_sq": (fig5, ax5),
        "quick_branches": quick_branches,
        "mu0_branch": mu0_branch,
        "mu0_mass_branch": mu0_mass_branch,
    }

reproduce_section2_until_here(show = True)