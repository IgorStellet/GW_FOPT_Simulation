"""
gravitational_Waves
===================

High-level helpers to extract thermodynamic parameters relevant for
gravitational-wave production from finite-temperature first-order
phase transitions.

This module is meant to be used together with the core modules
:mod:`tunneling1D` and :mod:`transitionFinder`. It starts from a known
thermal phase structure and bounce solutions and provides, in stages:

- geometric/thermodynamic scales such as d/dT [ S3(T) / T ] and β/H_*;
- the strength parameter α from the free-energy difference;
- the sound-wave contribution h² Ω_sw(f) to the GW spectrum;

and is designed to be extended with bubble-collision and turbulence
contributions.

Here S3(T) is the O(3) Euclidean action of the finite-temperature bounce.
"""

from typing import Any, Callable, Hashable, Mapping, Optional

import numpy as np
import numpy.typing as npt

from .transitionFinder import Phase, _solve_bounce

__all__ = ["GravitationalWaveCalculator"]


class GravitationalWaveCalculator:
    r"""
    Compute thermodynamic quantities needed for gravitational-wave forecasts
    from a finite-temperature first-order phase transition.

    This class assumes that the **thermal phase structure** has already been
    constructed with :mod:`transitionFinder`, i.e. you have a dictionary of
    :class:`Phase` objects describing the metastable (high-T) and stable
    (low-T) minima as functions of temperature.

    The current functionality is deliberately minimal:

    - given a pair of phases and a temperature :math:`T`, compute a symmetric
      finite-difference estimate of

      .. math::

          \frac{d}{dT}\left[\frac{S_3(T)}{T}\right],

      where :math:`S_3(T)` is the O(3) Euclidean action of the thermal bounce
      interpolating between the two phases.

    This derivative is the basic building block for the usual
    :math:`\beta / H_* \sim T \, d(S_3/T)/dT` estimate.

    Parameters
    ----------
    V :
        Finite-temperature potential :math:`V(\phi, T)` with signature
        ``V(x, T) -> float``. The first argument is a 1D array of field
        values (even for a single scalar field), the second a scalar
        temperature.
    dV :
        Gradient of the potential with respect to the field(s),
        ``dV(x, T) -> ndarray`` of the same shape as ``x``.
    phases :
        Mapping from phase keys to :class:`Phase` objects, typically the
        output of :func:`transitionFinder.traceMultiMin`.
    high_phase_key :
        Key identifying the **metastable** (false-vacuum) phase from which
        tunneling occurs.
    low_phase_key :
        Key identifying the **stable** (true-vacuum) phase into which the
        system tunnels.
    fullTunneling_params :
        Optional dictionary forwarded to the underlying tunneling backend
        used inside :func:`transitionFinder._solve_bounce` (e.g. options
        for :mod:`pathDeformation`).

    Notes
    -----
    - The field dimension is inferred from the stored phases. If the
      multi-field backend :mod:`pathDeformation` is unavailable, the
      fallback :class:`tunneling1D.SingleFieldInstanton` backend
      requires a single scalar field.
    """

    def __init__(
        self,
        V: Callable[[npt.NDArray[np.float64], float], float],
        dV: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
        dVdT: Callable[[npt.NDArray[np.float64], float], float],
        phases: Mapping[Hashable, Phase],
        high_phase_key: Hashable,
        low_phase_key: Hashable,
        *,
        fullTunneling_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.V = V
        self.dV = dV
        self.dVdT = dVdT
        self.phases: dict[Hashable, Phase] = dict(phases)

        try:
            self.high_phase: Phase = self.phases[high_phase_key]
        except KeyError as exc:
            msg = f"Unknown high_phase_key {high_phase_key!r}."
            raise KeyError(msg) from exc

        try:
            self.low_phase: Phase = self.phases[low_phase_key]
        except KeyError as exc:
            msg = f"Unknown low_phase_key {low_phase_key!r}."
            raise KeyError(msg) from exc

        # Options passed down to the tunneling backend (_solve_bounce).
        self.fullTunneling_params: dict[str, Any] = dict(fullTunneling_params or {})

        # Simple cache for S3(T) evaluations to avoid recomputing identical points.
        self._S3_cache: dict[float, float] = {}

        # Pre-compute the common temperature interval where both phases exist.
        self._T_min, self._T_max = self._common_temperature_range()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _common_temperature_range(self) -> tuple[float, float]:
        """
        Return the temperature interval where both phases are defined.

        Returns
        -------
        Tmin, Tmax : float
            Lower and upper bounds of the overlapping temperature range.

        Raises
        ------
        ValueError
            If the two phases do not share any overlapping temperature range.
        """
        Th = np.asarray(self.high_phase.T, dtype=float)
        Tl = np.asarray(self.low_phase.T, dtype=float)

        Tmin = float(max(Th[0], Tl[0]))
        Tmax = float(min(Th[-1], Tl[-1]))

        if Tmin >= Tmax:
            msg = (
                "High and low phases do not share an overlapping temperature "
                f"interval (Tmin={Tmin:.6g}, Tmax={Tmax:.6g})."
            )
            raise ValueError(msg)
        return Tmin, Tmax

    def _check_temperature_inside_range(self, T: float) -> None:
        """
        Ensure that T lies inside the common temperature range.

        Parameters
        ----------
        T : float
            Temperature to be checked.

        Raises
        ------
        ValueError
            If T is outside the overlapping interval.
        """
        if not (self._T_min <= T <= self._T_max):
            msg = (
                f"T={T:.6g} is outside the overlapping phase range "
                f"[{self._T_min:.6g}, {self._T_max:.6g}]."
            )
            raise ValueError(msg)

    def _S3_at_T(self, T: float) -> float:
        r"""
        Compute the O(3) Euclidean action :math:`S_3(T)` at fixed temperature.

        This helper:

        1. Evaluates the high-T and low-T minima from the stored phases at
           temperature :math:`T` (using :meth:`Phase.valAt`).
        2. Builds fixed-temperature wrappers :math:`V(\phi, T)` and
           :math:`\partial V / \partial\phi`.
        3. Calls :func:`transitionFinder._solve_bounce` to obtain the bounce
           and its action between the two minima.

        The result is cached so that repeated calls at the same T are cheap.

        Parameters
        ----------
        T : float
            Temperature at which the action is evaluated. Must lie inside
            the overlapping phase range.

        Returns
        -------
        float
            The O(3) Euclidean action :math:`S_3(T)`.

        Raises
        ------
        RuntimeError
            If no first-order bounce is found (no barrier or stable phase).
        """
        T_val = float(T)
        self._check_temperature_inside_range(T_val)

        if T_val in self._S3_cache:
            return self._S3_cache[T_val]

        # Minima at this temperature from the phase splines
        x_high = np.asarray(self.high_phase.valAt(T_val), dtype=float)
        x_low = np.asarray(self.low_phase.valAt(T_val), dtype=float)

        # Fixed-T wrappers around V and dV
        def V_fixed(x: npt.NDArray[np.float64]) -> float:
            return float(self.V(np.asarray(x, dtype=float), T_val))

        def dV_fixed(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(self.dV(np.asarray(x, dtype=float), T_val), dtype=float)

        # Use the unified tunneling backend from transitionFinder
        instanton, action, trantype = _solve_bounce(
            x_high=x_high,
            x_low=x_low,
            V_fixed=V_fixed,
            dV_fixed=dV_fixed,
            T=T_val,
            fullTunneling_params=self.fullTunneling_params,
        )

        if trantype != 1 or not np.isfinite(action):
            msg = (
                "Failed to find a first-order bounce between the selected "
                f"phases at T={T_val:.6g} (trantype={trantype}, S3={action})."
            )
            raise RuntimeError(msg)

        S3 = float(action)
        self._S3_cache[T_val] = S3
        return S3

    # ------------------------------------------------------------------
    # Public API (first step): d/dT [ S3(T) / T ]
    # ------------------------------------------------------------------
    def dS_dT(self, T: float, dT: float) -> float:
        r"""
     Fourth-order central finite-difference estimate of

        .. math::

            \frac{d}{dT}\left[\frac{S_3(T)}{T}\right].

        Here :math:`S_3(T)` is the O(3) Euclidean action of the thermal
        bounce between the configured high and low phases. The derivative
        is approximated with the standard 5-point, fourth-order accurate
        central stencil:

        .. math::

            \left.\frac{d}{dT}\left[\frac{S_3(T)}{T}\right]\right|_{T}
            \approx
            \frac{
                \frac{S_3(T - 2\Delta T)}{T - 2\Delta T}
                - 8\,\frac{S_3(T - \Delta T)}{T - \Delta T}
                + 8\,\frac{S_3(T + \Delta T)}{T + \Delta T}
                - \frac{S_3(T + 2\Delta T)}{T + 2\Delta T}
            }{12\,\Delta T}.
        """
        T = float(T)
        dT = float(dT)

        if dT <= 0.0:
            raise ValueError("dS_dT: dT must be positive.")

        T_m2 = T - 2.0 * dT
        T_m1 = T - dT
        T_p1 = T + dT
        T_p2 = T + 2.0 * dT

        # All stencil points must remain inside the common phase range
        if (
            T_m2 < self._T_min
            or T_m1 < self._T_min
            or T_p1 > self._T_max
            or T_p2 > self._T_max
        ):
            msg = (
                "dS_dT: T ± dT and T ± 2 dT must remain inside the overlapping "
                f"phase range [{self._T_min:.6g}, {self._T_max:.6g}], but "
                f"T-2dT={T_m2:.6g}, T-dT={T_m1:.6g}, "
                f"T+dT={T_p1:.6g}, T+2dT={T_p2:.6g}."
            )
            raise ValueError(msg)

        S_m2 = self._S3_at_T(T_m2)/T_m2
        S_m1 = self._S3_at_T(T_m1)/T_m1
        S_p1 = self._S3_at_T(T_p1)/T_p1
        S_p2 = self._S3_at_T(T_p2)/T_p2

        return (S_m2 - 8.0 * S_m1 + 8.0 * S_p1 - S_p2) / (12.0 * dT)


    def beta_over_H(
        self,
        Tn: float,
        dT: float,
        *,
        H: Optional[float] = None,
    ) -> float | tuple[float, float]:
        r"""
        Estimate :math:`\beta/H_*` at (typically) the nucleation temperature.

        This helper glues together the fourth-order finite-difference
        derivative :meth:`dS_dT` with the standard approximation

        .. math::

            \frac{\beta}{H_*}
            \;\simeq\;
            T_n \,
            \left.
            \frac{d}{dT}\left[\frac{S_3(T)}{T}\right]
            \right|_{T = T_n},

        where :math:`T_n` is usually taken as the nucleation temperature,
        and :math:`S_3(T)` is the O(3) Euclidean action of the thermal
        bounce between the chosen high and low phases.

        If a Hubble rate :math:`H_*` is provided, the function also returns
        the dimensional parameter :math:`\beta` via

        .. math::

            \beta \;=\; (\beta/H_*) \, H_*.

        Parameters
        ----------
        Tn : float
            Temperature at which the derivative is evaluated. In practice
            this is typically the nucleation temperature :math:`T_n`, but
            the function does not enforce this interpretation.
        dT : float
            Finite-difference stepsize :math:`\Delta T` used internally in
            :meth:`dS_dT`. Must be positive, and such that
            :math:`T_n \pm \Delta T` and :math:`T_n \pm 2\Delta T` lie inside
            the overlapping phase range.
        H : float, optional
            If provided, interpreted as the Hubble rate :math:`H_*` at the
            epoch of interest (typically the transition time). When given,
            the function returns both :math:`\beta/H_*` and :math:`\beta`.

        Returns
        -------
        beta_over_H : float
            If ``H`` is not provided, the function returns only the
            dimensionless ratio :math:`\beta/H_*`.
        beta_over_H, beta : (float, float)
            If ``H`` is provided, the function returns a tuple with the
            dimensionless :math:`\beta/H_*` and the dimensional
            :math:`\beta = (\beta/H_*) H_*`.

        Raises
        ------
        ValueError
            If ``dT <= 0`` or if the stencil points required by
            :meth:`dS_dT` fall outside the overlapping phase range.
            Also raised if ``H`` is provided but non-positive.
        RuntimeError
            If the bounce action cannot be computed at any of the required
            temperatures.
        """
        Tn = float(Tn)
        dT = float(dT)

        # Delegate all consistency checks on Tn and dT to dS_dT
        dSdT = self.dS_dT(Tn, dT)

        beta_over_H = Tn * dSdT

        if H is None:
            return float(beta_over_H)

        H = float(H)
        if H <= 0.0:
            raise ValueError("beta_over_H: H must be positive if provided.")

        beta = beta_over_H * H
        return float(beta_over_H), float(beta)

    def alpha(
        self,
        T: float,
        g_star: float,
        *,
        return_delta_rho: bool = False,
    ) -> float | tuple[float, float]:
        r"""
        Compute the strength parameter :math:`\alpha` at temperature ``T``.

        We follow the thermodynamic definition where the finite-temperature
        effective potential :math:`V(\phi, T)` is the Helmholtz free-energy
        density of the plasma. For each phase,

        .. math::

            \rho(\phi, T) = V(\phi, T) - T \, \frac{\partial V(\phi, T)}{\partial T},

        so that the energy-density difference between the two phases is

        .. math::

            \Delta\rho(T)
            = \bigl[
                V_{\text{low}}(T) - T\,\partial_T V_{\text{low}}(T)
              \bigr]
            - \bigl[
                V_{\text{high}}(T) - T\,\partial_T V_{\text{high}}(T)
              \bigr].

        The strength parameter is then

        .. math::

            \alpha(T) = \frac{\Delta\rho(T)}{\rho_{\text{rad}}(T)}, \qquad
            \rho_{\text{rad}}(T) = \frac{\pi^2}{30}\, g_*\, T^4.

        Parameters
        ----------
        T : float
            Temperature at which :math:`\alpha` is evaluated (typically the
            nucleation temperature :math:`T_n`). Must lie in the overlapping
            temperature range of the two phases.
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at temperature ``T``.
        return_delta_rho : bool, optional
            If ``True``, also return :math:`\Delta\rho(T)`.

        Returns
        -------
        alpha : float
            The strength parameter :math:`\alpha(T)`.
        (alpha, delta_rho) : tuple of float
            If ``return_delta_rho=True``, returns both :math:`\alpha(T)` and
            :math:`\Delta\rho(T)`.

        Raises
        ------
        AttributeError
            If ``self.dVdT`` is not defined. In that case you must supply the
            temperature derivative of the potential when constructing the
            calculator.
        ValueError
            If ``T`` lies outside the overlapping temperature range, or if
            ``g_star <= 0``.
        """
        T_val = float(T)
        self._check_temperature_inside_range(T_val)

        if g_star <= 0.0:
            raise ValueError("alpha: g_star must be positive.")

        if not hasattr(self, "dVdT"):
            msg = (
                "alpha: this instance has no 'dVdT' attribute.\n"
                "You must provide a callable dVdT(phi, T) when constructing "
                "GravitationalWaveCalculator if you want to use 'alpha'."
            )
            raise AttributeError(msg)

        # Minima at this temperature
        x_high = np.asarray(self.high_phase.valAt(T_val), dtype=float)
        x_low = np.asarray(self.low_phase.valAt(T_val), dtype=float)

        # Free-energy densities at the minima
        V_high = float(self.V(x_high, T_val))
        V_low = float(self.V(x_low, T_val))

        # Temperature derivatives at fixed minima positions
        dVdT_high = float(self.dVdT(x_high, T_val))
        dVdT_low = float(self.dVdT(x_low, T_val))

        # Energy densities in each phase
        rho_high = V_high - T_val * dVdT_high
        rho_low = V_low - T_val * dVdT_low

        delta_rho = rho_low - rho_high

        # Radiation energy density
        rho_rad = (np.pi**2 / 30.0) * g_star * T_val**4

        alpha_val = delta_rho / rho_rad

        if return_delta_rho:
            return alpha_val, delta_rho
        return alpha_val


    # ------------------------------------------------------------------
    # Gravitational-wave spectrum: sound waves
    # ------------------------------------------------------------------
    def _f_sw_peak(
        self,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float,
    ) -> float:
        r"""
        Peak frequency of the sound-wave GW signal today, in Hz.

        This implements the usual fit

        .. math::

            f_{\rm sw} \simeq 1.9\times 10^{-5}\,{\rm Hz}\,
                \frac{1}{v_w}
                \left(\frac{\beta}{H_*}\right)
                \left(\frac{T_*}{100~{\rm GeV}}\right)
                \left(\frac{g_*}{100}\right)^{1/6},

        where :math:`T_*` is the reference temperature for GW production
        (often :math:`T_n` or :math:`T_{\rm perc}`).

        Parameters
        ----------
        beta_over_H : float
            Dimensionless ratio :math:`\beta/H_*` at :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV.
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at :math:`T_*`.
        v_w : float
            Bubble wall velocity as a fraction of the speed of light.

        Returns
        -------
        float
            Peak frequency :math:`f_{\rm sw}` in Hz.
        """
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if beta_over_H <= 0.0:
            raise ValueError("_f_sw_peak: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("_f_sw_peak: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("_f_sw_peak: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("_f_sw_peak: v_w must be positive.")

        # 1.9×10^{-2} mHz = 1.9×10^{-5} Hz
        prefactor_hz = 1.9e-5  # Hz

        return (
            prefactor_hz
            * (beta_over_H)
            * (T_star / 100.0)
            * (g_star / 100.0) ** (1.0 / 6.0)
            / v_w
        )

    def omega_sw_h2(
        self,
        f: npt.ArrayLike,
        *,
        alpha: float,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float = 1.0,
        shape: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
        y_sup: Optional[float] = None,
        kappa_sw: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        r"""
        Present-day sound-wave contribution :math:`h^2 \Omega_{\rm sw}(f)`.

        This implements the standard phenomenological fit

        .. math::

            h^2 \Omega_{\rm sw}(f)
            = h^2 \Omega_{\rm sw}^{\rm peak} \, S_{\rm sw}(f),

        with

        .. math::

            h^2 \Omega_{\rm sw}^{\rm peak}
            &= 2.65\times 10^{-6}\,
               \frac{v_w}{\beta/H_*}
               \left(\frac{\kappa_{\rm sw}\,\alpha}{1+\alpha}\right)^2
               \left(\frac{100}{g_*}\right)^{1/3} Y_{\rm sup}, \\[0.7em]
            \kappa_{\rm sw}(\alpha)
            &= \frac{\alpha}{0.73 + 0.83\sqrt{\alpha} + \alpha}, \\[0.7em]
            S_{\rm sw}(f)
            &= \left(\frac{f}{f_{\rm sw}}\right)^3
               \left[
                    \frac{7}{4 + 3\,(f/f_{\rm sw})^2}
               \right]^{7/2}.

        The peak frequency :math:`f_{\rm sw}` is given by :meth:`_f_sw_peak`.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz (scalar or array) at which to evaluate
            :math:`h^2 \Omega_{\rm sw}(f)`.
        alpha : float
            Strength parameter :math:`\alpha` of the transition at :math:`T_*`,
            i.e. ratio of released vacuum energy to radiation energy density.
        beta_over_H : float
            Dimensionless parameter :math:`\beta/H_*` evaluated at the
            chosen reference temperature :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV (e.g. :math:`T_n` or
            percolation temperature).
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at :math:`T_*`.
        v_w : float, optional
            Bubble wall velocity as a fraction of the speed of light.
            Default is ``1.0`` (ultra-relativistic walls).
        shape : callable, optional
            Optional custom shape function. If provided, it must have
            signature ``shape(f_array, f_peak) -> ndarray`` and return
            a dimensionless shape factor :math:`S(f)`. If ``None``, the
            default :math:`S_{\rm sw}(f)` above is used.
        y_sup : float, optional
            Suppression factor :math:`Y_{\rm sup} \le 1` accounting for
            the finite lifetime of the sound-wave source. If ``None``,
            no suppression is applied (i.e. ``Y_sup = 1``).
        kappa_sw : float, optional
            Efficiency factor :math:`\kappa_{\rm sw}`. If ``None``, the
            standard fit :math:`\kappa_{\rm sw}(\alpha)` above is used.

        Returns
        -------
        ndarray
            Array with the same shape as ``f`` containing
            :math:`h^2 \Omega_{\rm sw}(f)`.

        Notes
        -----
        - This method does *not* attempt to decide what the appropriate
          reference temperature :math:`T_*` should be; you should pass
          :math:`\alpha` and :math:`\beta/H_*` evaluated at your preferred
          choice (:math:`T_n`, percolation, etc.).
        - For more refined treatments, especially in strongly supercooled
          transitions, you may want to provide your own ``shape`` and
          ``y_sup`` based on hydrodynamic simulations.
        """
        f_arr = np.asarray(f, dtype=float)
        alpha = float(alpha)
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if alpha <= 0.0:
            raise ValueError("omega_sw_h2: alpha must be positive.")
        if beta_over_H <= 0.0:
            raise ValueError("omega_sw_h2: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("omega_sw_h2: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("omega_sw_h2: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("omega_sw_h2: v_w must be positive.")

        if kappa_sw is None:
            # Standard fit for the efficiency of converting vacuum energy
            # into bulk kinetic energy of the plasma.
            kappa_sw = alpha / (0.73 + 0.83 * np.sqrt(alpha) + alpha)
        kappa_sw = float(kappa_sw)

        if y_sup is None:
            U_f = np.sqrt(0.75 * kappa_sw * alpha / (1.0 + alpha))
            tau_sw_Hstar = (8.0 * np.pi) ** (1.0 / 3.0) * v_w / (
                    beta_over_H * U_f)
            y_sup = float(min(1.0, tau_sw_Hstar))
        y_sup = float(y_sup)
        if y_sup <= 0.0:
            raise ValueError("omega_sw_h2: y_sup must be positive if provided.")

        # Peak frequency
        f_peak = self._f_sw_peak(
            beta_over_H=beta_over_H,
            T_star=T_star,
            g_star=g_star,
            v_w=v_w,
        )

        # Shape function S(f)
        if shape is None:
            x = f_arr / f_peak
            S = x**3 * (7.0 / (4.0 + 3.0 * x**2)) ** (7.0 / 2.0)
        else:
            S = np.asarray(shape(f_arr, f_peak), dtype=float)

        # Peak amplitude
        amp_peak = (
            2.65e-6
            * (v_w / beta_over_H)
            * (kappa_sw * alpha / (1.0 + alpha)) ** 2
            * (100.0 / g_star) ** (1.0 / 3.0)
            * y_sup
        )

        return amp_peak * S


    # ------------------------------------------------------------------
    # Gravitational-wave spectrum: MHD turbulence
    # ------------------------------------------------------------------
    def _h_star_Hz(self, T_star: float, g_star: float) -> float:
        r"""
        Hubble frequency today corresponding to the transition epoch, in Hz.

        Implements

        .. math::

            h_* \simeq 16.5\times 10^{-3}\,{\rm mHz}\,
                \left(\frac{T_*}{100~{\rm GeV}}\right)
                \left(\frac{g_*}{100}\right)^{1/6},

        where :math:`T_*` is the characteristic temperature of GW production.
        """
        T_star = float(T_star)
        g_star = float(g_star)

        if T_star <= 0.0:
            raise ValueError("_h_star_Hz: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("_h_star_Hz: g_star must be positive.")

        # 16.5 × 10^{-3} mHz = 16.5 × 10^{-6} Hz = 1.65×10^{-5} Hz
        prefactor_hz = 1.65e-5  # Hz

        return (
            prefactor_hz
            * (T_star / 100.0)
            * (g_star / 100.0) ** (1.0 / 6.0)
        )

    def _f_turb_peak(
        self,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float,
    ) -> float:
        r"""
        Peak frequency of the turbulent GW signal today, in Hz.

        Implements

        .. math::

            f_{\rm turb} \simeq 2.7\times 10^{-5}\,{\rm Hz}\,
                \frac{1}{v_w}
                \left(\frac{\beta}{H_*}\right)
                \left(\frac{T_*}{100~{\rm GeV}}\right)
                \left(\frac{g_*}{100}\right)^{1/6}.
        """
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if beta_over_H <= 0.0:
            raise ValueError("_f_turb_peak: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("_f_turb_peak: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("_f_turb_peak: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("_f_turb_peak: v_w must be positive.")

        # 2.7 × 10^{-2} mHz = 2.7 × 10^{-5} Hz
        prefactor_hz = 2.7e-5  # Hz

        return (
            prefactor_hz
            * (beta_over_H)
            * (T_star / 100.0)
            * (g_star / 100.0) ** (1.0 / 6.0)
            / v_w
        )

    def omega_turb_h2(
        self,
        f: npt.ArrayLike,
        *,
        alpha: float,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float = 1.0,
        kappa_turb: Optional[float] = None,
        epsilon: Optional[float] = None,
        shape: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
    ) -> npt.NDArray[np.float64]:
        r"""
        Present-day MHD-turbulence contribution :math:`h^2 \Omega_{\rm turb}(f)`.

        We implement the commonly used fit (e.g. Caprini et al.):

        .. math::

            h^2 \Omega_{\rm turb}(f)
            &= h^2 \Omega_{\rm turb}^{\rm peak} \, S_{\rm turb}(f), \\[0.5em]
            h^2 \Omega_{\rm turb}^{\rm peak}
            &= 3.35\times 10^{-4}\,
               \frac{v_w}{\beta/H_*}
               \left(\frac{\kappa_{\rm turb}\,\alpha}{1+\alpha}\right)^{3/2}
               \left(\frac{100}{g_*}\right)^{1/3}, \\[0.5em]
            S_{\rm turb}(f)
            &= \frac{(f/f_{\rm turb})^3}
                    {\bigl(1 + f/f_{\rm turb}\bigr)^{11/3}
                     \bigl(1 + 8\pi f/h_*\bigr)}.

        Here :math:`T_*` is the characteristic temperature of GW production
        (often :math:`T_n` or percolation), :math:`f_{\rm turb}` is the peak
        frequency given by :meth:`_f_turb_peak`, and :math:`h_*` is the
        redshifted Hubble frequency at the transition, given by
        :meth:`_h_star_Hz`.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz (scalar or array) at which to evaluate
            :math:`h^2 \Omega_{\rm turb}(f)`.
        alpha : float
            Strength parameter :math:`\alpha` at :math:`T_*`.
        beta_over_H : float
            Dimensionless ratio :math:`\beta/H_*` at :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV.
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at :math:`T_*`.
        v_w : float, optional
            Bubble wall velocity as a fraction of the speed of light.
            Default is ``1.0``.
        kappa_turb : float, optional
            Turbulent efficiency factor :math:`\kappa_{\rm turb}`. If provided,
            it is used directly.
        epsilon : float, optional
            If ``kappa_turb`` is not given, one can instead specify
            :math:`\epsilon` in the Ansatz
            :math:`\kappa_{\rm turb} = \epsilon \,\kappa_{\rm sw}`, where
            :math:`\kappa_{\rm sw}` is the usual sound-wave efficiency fit.
            If both ``kappa_turb`` and ``epsilon`` are ``None``, a mild
            default ``epsilon = 0.05`` is used.
        shape : callable, optional
            Optional custom shape function. If provided, it must have signature
            ``shape(f_array, f_peak) -> ndarray`` and return a dimensionless
            shape factor :math:`S(f)`. In that case, the user is responsible
            for any dependence on :math:`h_*`. If ``None``, the default
            :math:`S_{\rm turb}(f)` above is used.

        Returns
        -------
        ndarray
            Array with the same shape as ``f`` containing
            :math:`h^2 \Omega_{\rm turb}(f)`.

        Notes
        -----
        - The modeling of the turbulent GW signal is still under active
          investigation; the expressions implemented here should be viewed as
          phenomenological fits rather than first-principles predictions.
        - The parameter :math:`\kappa_{\rm turb}` is particularly uncertain.
          The ``epsilon`` parameter is provided to explore the commonly used
          Ansatz :math:`\kappa_{\rm turb} = \epsilon \kappa_{\rm sw}` with
          :math:`\epsilon` in the range :math:`0\text{--}1`.
        """
        f_arr = np.asarray(f, dtype=float)
        alpha = float(alpha)
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if alpha <= 0.0:
            raise ValueError("omega_turb_h2: alpha must be positive.")
        if beta_over_H <= 0.0:
            raise ValueError("omega_turb_h2: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("omega_turb_h2: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("omega_turb_h2: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("omega_turb_h2: v_w must be positive.")

        # ------------------------------------------------------------------
        # Turbulent efficiency κ_turb
        # ------------------------------------------------------------------
        if kappa_turb is None:
            # If the user did not provide epsilon, adopt a mild default 5%.
            if epsilon is None:
                epsilon = 0.05
            epsilon = float(epsilon)
            if epsilon < 0.0 or epsilon > 1.0:
                raise ValueError(
                    "omega_turb_h2: epsilon should lie in [0, 1] if provided."
                )

            # Standard sound-wave efficiency fit κ_sw(α)
            kappa_sw_eff = alpha / (0.73 + 0.83 * np.sqrt(alpha) + alpha)

            kappa_turb = epsilon * kappa_sw_eff

        kappa_turb = float(kappa_turb)
        if kappa_turb < 0.0:
            raise ValueError("omega_turb_h2: kappa_turb must be non-negative.")

        # ------------------------------------------------------------------
        # Peak frequency and h_* scale
        # ------------------------------------------------------------------
        f_peak = self._f_turb_peak(
            beta_over_H=beta_over_H,
            T_star=T_star,
            g_star=g_star,
            v_w=v_w,
        )
        h_star = self._h_star_Hz(T_star=T_star, g_star=g_star)

        # Shape function S_turb(f)
        if shape is None:
            x = f_arr / f_peak
            S = x**3 / ((1.0 + x) ** (11.0 / 3.0) * (1.0 + 8.0 * np.pi * f_arr / h_star))
        else:
            S = np.asarray(shape(f_arr, f_peak), dtype=float)

        # Peak amplitude
        amp_peak = (
            3.35e-4
            * (v_w / beta_over_H)
            * (kappa_turb * alpha / (1.0 + alpha)) ** 1.5
            * (100.0 / g_star) ** (1.0 / 3.0)
        )

        return amp_peak * S

    # ------------------------------------------------------------------
    # Gravitational-wave spectrum: bubble collisions (envelope approx.)
    # ------------------------------------------------------------------
    def _f_coll_peak(
        self,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float,
    ) -> float:
        r"""
        Peak frequency of the scalar-field (bubble-collision) GW signal,
        in Hz, within the envelope approximation.

        We use the standard fit

        .. math::

            f_{\rm env} \simeq 16.5\times 10^{-6}\,{\rm Hz}\;
                \frac{0.62}{1.8 - 0.1 v_w + v_w^2}
                \left(\frac{\beta}{H_*}\right)
                \left(\frac{T_*}{100~{\rm GeV}}\right)
                \left(\frac{g_*}{100}\right)^{1/6},

        where :math:`T_*` is the reference temperature for GW production
        (often :math:`T_n` or :math:`T_{\rm perc}`).

        Parameters
        ----------
        beta_over_H : float
            Dimensionless ratio :math:`\beta/H_*` at :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV.
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at :math:`T_*`.
        v_w : float
            Bubble wall velocity as a fraction of the speed of light.

        Returns
        -------
        float
            Peak frequency :math:`f_{\rm env}` in Hz.
        """
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if beta_over_H <= 0.0:
            raise ValueError("_f_coll_peak: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("_f_coll_peak: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("_f_coll_peak: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("_f_coll_peak: v_w must be positive.")

        prefactor_hz = 16.5e-6  # Hz

        velocity_factor = 0.62 / (1.8 - 0.1 * v_w + v_w**2)

        return (
            prefactor_hz
            * velocity_factor
            * beta_over_H
            * (T_star / 100.0)
            * (g_star / 100.0) ** (1.0 / 6.0)
        )

    def omega_coll_h2(
        self,
        f: npt.ArrayLike,
        *,
        alpha: float,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float = 1.0,
        shape: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
        kappa_coll: Optional[float] = None,
        delta_factor: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        r"""
        Present-day bubble-collision contribution :math:`h^2 \Omega_{\rm coll}(f)`.

        This implements the envelope approximation for the scalar-field
        contribution to the GW background:

        .. math::

            h^2 \Omega_{\rm coll}(f)
            &= h^2 \Omega_{\rm coll}^{\rm peak} \, S_{\rm env}(f), \\[0.3em]
            h^2 \Omega_{\rm coll}^{\rm peak}
            &= 1.67\times 10^{-5}\,
               \Delta(v_w)\,
               \left(\frac{H_*}{\beta}\right)^2
               \left(\frac{\kappa_{\rm coll}\,\alpha}{1+\alpha}\right)^2
               \left(\frac{100}{g_*}\right)^{1/3}, \\[0.3em]
            \Delta(v_w)
            &= \frac{0.11 v_w^3}{0.42 + v_w^2}, \\[0.3em]
            S_{\rm env}(f)
            &= \frac{3.8 (f/f_{\rm env})^{2.8}}
                     {1 + 2.8 (f/f_{\rm env})^{3.8}}.

        The peak frequency :math:`f_{\rm env}` is given by :meth:`_f_coll_peak`.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz (scalar or array) at which to evaluate
            :math:`h^2 \Omega_{\rm coll}(f)`.
        alpha : float
            Strength parameter :math:`\alpha` of the transition at :math:`T_*`.
        beta_over_H : float
            Dimensionless parameter :math:`\beta/H_*` evaluated at the
            chosen reference temperature :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV (e.g. :math:`T_n` or
            percolation temperature).
        g_star : float
            Effective number of relativistic degrees of freedom :math:`g_*`
            at :math:`T_*`.
        v_w : float, optional
            Bubble wall velocity as a fraction of the speed of light.
            Default is ``1.0`` (ultra-relativistic walls).
        shape : callable, optional
            Optional custom shape function. If provided, it must have
            signature ``shape(f_array, f_peak) -> ndarray`` and return
            a dimensionless shape factor :math:`S(f)`. If ``None``, the
            standard envelope shape :math:`S_{\rm env}(f)` above is used.
        kappa_coll : float, optional
            Efficiency factor :math:`\kappa_{\rm coll}` for the fraction of
            released vacuum energy that goes into scalar-field gradients
            (bubble walls). If ``None``, it is set to zero by default
            (negligible collisions), which is appropriate for non-runaway
            walls.
        delta_factor : float, optional
            Optional override for :math:`\Delta(v_w)`. If ``None``, the
            standard :math:`\Delta(v_w) = 0.11 v_w^3 / (0.42 + v_w^2)` is used.

        Returns
        -------
        ndarray
            Array with the same shape as ``f`` containing
            :math:`h^2 \Omega_{\rm coll}(f)`.

        Notes
        -----
        - For non-runaway walls in a thermal plasma, it is common to set
          ``kappa_coll = 0`` and neglect this contribution relative to
          sound waves and turbulence.
        - For runaway walls, you may supply a non-zero ``kappa_coll``
          based on a separate model (e.g. from hydrodynamics with an
          :math:`\alpha_\infty` parameter).
        """
        f_arr = np.asarray(f, dtype=float)
        alpha = float(alpha)
        beta_over_H = float(beta_over_H)
        T_star = float(T_star)
        g_star = float(g_star)
        v_w = float(v_w)

        if alpha <= 0.0:
            raise ValueError("omega_coll_h2: alpha must be positive.")
        if beta_over_H <= 0.0:
            raise ValueError("omega_coll_h2: beta_over_H must be positive.")
        if T_star <= 0.0:
            raise ValueError("omega_coll_h2: T_star must be positive (in GeV).")
        if g_star <= 0.0:
            raise ValueError("omega_coll_h2: g_star must be positive.")
        if v_w <= 0.0:
            raise ValueError("omega_coll_h2: v_w must be positive.")

        # Default kappa_coll: assume negligible collisions unless specified.
        if kappa_coll is None:
            kappa_coll = 0.0
        kappa_coll = float(kappa_coll)
        if kappa_coll < 0.0:
            raise ValueError("omega_coll_h2: kappa_coll cannot be negative.")

        # Velocity-dependent prefactor Delta(v_w)
        if delta_factor is None:
            delta_factor = 0.11 * v_w**3 / (0.42 + v_w**2)
        delta_factor = float(delta_factor)
        if delta_factor < 0.0:
            raise ValueError("omega_coll_h2: delta_factor cannot be negative.")

        # Peak frequency
        f_peak = self._f_coll_peak(
            beta_over_H=beta_over_H,
            T_star=T_star,
            g_star=g_star,
            v_w=v_w,
        )

        # Shape function S_env(f)
        if shape is None:
            x = f_arr / f_peak
            S = 3.8 * x**2.8 / (1.0 + 2.8 * x**3.8)
        else:
            S = np.asarray(shape(f_arr, f_peak), dtype=float)

        # Peak amplitude:
        # (H*/beta)^2 = 1 / (beta_over_H)^2
        amp_peak = (
            1.67e-5
            * delta_factor
            * (1.0 / beta_over_H**2)
            * (kappa_coll * alpha / (1.0 + alpha)) ** 2
            * (100.0 / g_star) ** (1.0 / 3.0)
        )

        return amp_peak * S

    # ------------------------------------------------------------------
    # Gravitational-wave spectrum: combined (sound + turbulence + bubbles)
    # ------------------------------------------------------------------
    def omega_total_h2(
        self,
        f: npt.ArrayLike,
        *,
        alpha: float,
        beta_over_H: float,
        T_star: float,
        g_star: float,
        v_w: float = 1.0,
        include_sw: bool = True,
        include_turb: bool = True,
        include_coll: bool = True,
        # Optional shape functions
        shape_sw: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
        shape_turb: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
        shape_coll: Optional[
            Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]]
        ] = None,
        # Extra physics knobs
        y_sup_sw: Optional[float] = None,
        kappa_sw: Optional[float] = None,
        kappa_turb: Optional[float] = None,
        epsilon_turb: Optional[float] = None,
        kappa_coll: Optional[float] = None,
        delta_factor_coll: Optional[float] = None,
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""
        Combined GW spectrum

        .. math::

            h^2 \Omega_{\rm tot}(f)
            = h^2 \Omega_{\rm sw}(f)
            + h^2 \Omega_{\rm turb}(f)
            + h^2 \Omega_{\rm coll}(f),

        returned together with its individual components.

        This is a convenience wrapper around :meth:`omega_sw_h2`,
        :meth:`omega_turb_h2` and :meth:`omega_coll_h2`. It evaluates
        each contribution at the same frequency array and returns a
        dictionary with four entries:

        - ``"sw"``: sound waves;
        - ``"turb"``: MHD turbulence;
        - ``"coll"``: bubble collisions (scalar field, envelope approx.);
        - ``"total"``: sum of all *included* components.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz (scalar or array) at which to evaluate
            the spectra.
        alpha : float
            Strength parameter :math:`\alpha` at the chosen reference
            temperature :math:`T_*`.
        beta_over_H : float
            Dimensionless parameter :math:`\beta/H_*` at :math:`T_*`.
        T_star : float
            Reference temperature :math:`T_*` in GeV (e.g. :math:`T_n`
            or percolation temperature).
        g_star : float
            Effective number of relativistic degrees of freedom
            :math:`g_*` at :math:`T_*`.
        v_w : float, optional
            Bubble wall velocity (in units of :math:`c`). Default is 1.
        include_sw, include_turb, include_coll : bool, optional
            Switches to enable/disable sound-wave, turbulence and
            bubble-collision contributions individually.
        shape_sw, shape_turb, shape_coll : callable, optional
            Custom shape functions for each component. If ``None``,
            the built-in analytic fits are used (see the individual
            methods for details).
        y_sup_sw : float, optional
            Sound-wave suppression factor :math:`Y_{\rm sup} \le 1`.
            If ``None``, it is estimated internally from the fluid
            RMS velocity (see :meth:`omega_sw_h2`).
        kappa_sw : float, optional
            Efficiency factor :math:`\kappa_{\rm sw}`. If ``None``,
            the standard fit :math:`\kappa_{\rm sw}(\alpha)` is used.
        kappa_turb : float, optional
            Turbulence efficiency :math:`\kappa_{\rm turb}`. If
            ``None``, the turbulence routine may build it from a
            fraction of :math:`\kappa_{\rm sw}` (depending on its
            implementation).
        epsilon_turb : float, optional
            Optional parameter controlling how :math:`\kappa_{\rm turb}`
            is related to :math:`\kappa_{\rm sw}` (e.g. via
            :math:`\kappa_{\rm turb} = \epsilon\,\kappa_{\rm sw}`).
        kappa_coll : float, optional
            Bubble-collision efficiency :math:`\kappa_{\rm coll}`. If
            ``None``, defaults to zero (negligible scalar-field
            contribution), appropriate for non-runaway walls.
        delta_factor_coll : float, optional
            Optional override for the velocity-dependent factor
            :math:`\Delta(v_w)` in the collision amplitude. If ``None``,
            the standard :math:`0.11 v_w^3/(0.42 + v_w^2)` is used.

        Returns
        -------
        dict of str -> ndarray
            Dictionary with four entries:

            - ``"sw"``: :math:`h^2 \Omega_{\rm sw}(f)`;
            - ``"turb"``: :math:`h^2 \Omega_{\rm turb}(f)`;
            - ``"coll"``: :math:`h^2 \Omega_{\rm coll}(f)`;
            - ``"total"``: sum of all active components.

        Notes
        -----
        - This method does not attempt to infer :math:`T_*`,
          :math:`\alpha` or :math:`\beta/H_*`; you should pass them
          based on your preferred prescription (e.g. evaluated at
          :math:`T_n` or at percolation).
        - Use the ``include_*`` flags to quickly compare, for example,
          a "sound-waves only" scenario against the full
          sound + turbulence + collisions signal.
        """
        f_arr = np.asarray(f, dtype=float)

        # Start with zeros for all components
        omega_sw = np.zeros_like(f_arr, dtype=float)
        omega_turb = np.zeros_like(f_arr, dtype=float)
        omega_coll = np.zeros_like(f_arr, dtype=float)

        # Sound waves
        if include_sw:
            omega_sw = self.omega_sw_h2(
                f_arr,
                alpha=alpha,
                beta_over_H=beta_over_H,
                T_star=T_star,
                g_star=g_star,
                v_w=v_w,
                shape=shape_sw,
                y_sup=y_sup_sw,
                kappa_sw=kappa_sw,
            )

        # Turbulence
        if include_turb:
            turb_kwargs: dict[str, Any] = dict(
                alpha=alpha,
                beta_over_H=beta_over_H,
                T_star=T_star,
                g_star=g_star,
                v_w=v_w,
                shape=shape_turb,
            )
            if kappa_turb is not None:
                turb_kwargs["kappa_turb"] = kappa_turb
            if epsilon_turb is not None:
                turb_kwargs["epsilon"] = epsilon_turb

            omega_turb = self.omega_turb_h2(f_arr, **turb_kwargs)

        # Bubble collisions
        if include_coll:
            coll_kwargs: dict[str, Any] = dict(
                alpha=alpha,
                beta_over_H=beta_over_H,
                T_star=T_star,
                g_star=g_star,
                v_w=v_w,
                shape=shape_coll,
            )
            if kappa_coll is not None:
                coll_kwargs["kappa_coll"] = kappa_coll
            if delta_factor_coll is not None:
                coll_kwargs["delta_factor"] = delta_factor_coll

            omega_coll = self.omega_coll_h2(f_arr, **coll_kwargs)

        omega_total = omega_sw + omega_turb + omega_coll

        return {
            "sw": omega_sw,
            "turb": omega_turb,
            "coll": omega_coll,
            "total": omega_total,
        }
