# New version of tunneling1D

"""
tunneling1D
===========

Tools to compute one-dimensional tunneling (bounce / instanton) solutions for a
single scalar field. The primary workflow follows the classic
overshoot/undershoot method to find the bubble profile and its Euclidean action
in arbitrary spatial dimension (controlled by the friction coefficient `alpha`).

This module exposes two main classes:

- `SingleFieldInstanton`: radius-dependent friction (∝ α/r), the standard O(α+1)
  bounce used in vacuum decay calculations. Provides `findProfile()` and
  `findAction()` plus utilities for initialization and post-processing.

- `WallWithConstFriction`: variant with *constant* friction term, useful as a
  rough proxy for plasma drag when estimating bubble-wall shapes (no action).

Not yet implemented: Coleman–De Luccia (CDL) instanton with gravity.

Notes
-----
- The API remains compatible with the legacy version while adopting clearer
  structure and numerics. Tests and docs are organized per functional block.
"""

# --- Basic imports (extend here as we modernize) --------------------------------

import numpy as np
from collections import namedtuple
from scipy import optimize, special, interpolate
from scipy.integrate import simpson
from typing import Callable, Optional, Union
import warnings

# Internal helpers
from .helper_functions import rkqs, IntegrationError, clamp_val, cubicInterpFunction, monotonic_indices

__all__ = ["PotentialError","SingleFieldInstanton"]

# --- Errors ---------------------------------------------------------------------
class PotentialError(Exception):
    """
    Raised when the potential lacks the expected features for tunneling.

    Convention (legacy-compatible):
    The exception message may be a tuple where the second item is one of
    ('no barrier', 'stable, not metastable').
    """
    pass


# === SingleFieldInstanton =======================================================
# (Implementation follows in SF-1…SF-6 blocks)

class SingleFieldInstanton:
    """
    Compute properties of a single–field instanton via the classic
    overshoot/undershoot method. Users primarily call :meth:`findProfile` and
    :meth:`findAction`.

    Notes
    -----
    • Thin-wall acceleration: when the minima are nearly degenerate, we start
      the integration close to the wall using a quadratic local solution, so
      the search converges quickly even for very thin walls.

    Parameters
    ----------
    phi_absMin : float
        Field value of the stable (true) vacuum.
    phi_metaMin : float
        Field value of the metastable (false) vacuum.
    V : Callable[[float], float]
        Potential function `V(phi)`. Should accept NumPy arrays (vectorized) or
        at least standard Python floats.
    dV, d2V : callable, optional
        User-supplied first/second derivatives. If provided, they override the
        finite-difference implementations (:meth:`dV`, :meth:`d2V`).
    phi_eps : float, optional
        *Relative* finite-difference step size. The absolute step is computed
        as `phi_eps * abs(phi_metaMin - phi_absMin)`. (Default: 1e-3)
    alpha : int or float, optional
        Friction coefficient in the ODE (equals spatial dimension). Default 2.
    phi_bar : float, optional
        Field value at the barrier edge (solution to V(phi_bar)=V(phi_metaMin)).
        If None, it is found by :meth:`findBarrierLocation`.
    rscale : float, optional
        Characteristic radial scale. If None, computed by :meth:`findRScale`.

    Extended options (backward compatible)
    --------------------------------------
    fd_order : {2, 4}, optional
        Order of the central finite differences used by the builtin :meth:`dV`
        and :meth:`d2V` (ignored if user passed `dV`/`d2V`). Default 4.
    fd_eps_min : float, optional
        Absolute lower bound for the finite-difference step. If None, a safe
        automatic floor is used (≈ sqrt(machine epsilon) × scale).
    validate : bool, optional
        If True (default), perform basic sanity checks on the potential and
        inputs and emit helpful errors/warnings.

    Raises
    ------
    PotentialError
        If `V(phi_metaMin) <= V(phi_absMin)` (no metastability) or the barrier
        is missing/ill-defined during scale detection.
    """

    # --- SF-1: ctor & potential interface -------------------------------------
    def __init__(
        self,
        phi_absMin: float,
        phi_metaMin: float,
        V: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        dV: Optional[Callable] = None,
        d2V: Optional[Callable] = None,
        phi_eps: float = 1e-3,
        alpha: Union[int, float] = 2,
        phi_bar: Optional[float] = None,
        rscale: Optional[float] = None,
        *,
        fd_order: int = 4,
        fd_eps_min: Optional[float] = None,
        validate: bool = True,
    ):
        # Store user inputs
        self.phi_absMin = float(phi_absMin)
        self.phi_metaMin = float(phi_metaMin)
        self.V = V  # callable, may be vectorized or scalar-only

        # Cached values & convenient deltas
        self.V_abs = float(V(self.phi_absMin))
        self.V_meta = float(V(self.phi_metaMin))
        self.delta_phi = self.phi_metaMin - self.phi_absMin
        self.abs_delta_phi = abs(self.delta_phi)

        # Basic validations (metastability, finiteness, alpha)
        if validate:
            if not np.isfinite(self.V_abs) or not np.isfinite(self.V_meta):
                raise PotentialError(
                    "Potential returned a non-finite value at a vacuum point."
                )
            if self.V_meta <= self.V_abs:
                raise PotentialError(
                    "V(phi_metaMin) <= V(phi_absMin); tunneling cannot occur.",
                    "stable, not metastable",
                )
            if not np.isfinite(alpha) or alpha < 0:
                raise PotentialError("`alpha` must be a non-negative number.")

        # Finite-difference configuration
        self._fd_order = 4 if fd_order not in (2, 4) else fd_order
        # Compute absolute FD step; prevent the legacy zero-step pitfall
        if self.abs_delta_phi > 0:
            base_scale = self.abs_delta_phi
        else:
            # Degenerate field locations: fall back to |phi| scale or unity
            base_scale = max(1.0, abs(self.phi_absMin) + abs(self.phi_metaMin))
        auto_floor = np.sqrt(np.finfo(float).eps) * base_scale
        floor = auto_floor if fd_eps_min is None else float(fd_eps_min)
        self.phi_eps = max(float(phi_eps) * base_scale, floor)

        # Optional user-derivative overrides (kept legacy behavior)
        if dV is not None:
            self.dV = dV
        if d2V is not None:
            self.d2V = d2V

        # Barrier location (accept given value but optionally sanity-check it)
        if phi_bar is None:
            self.phi_bar = self.findBarrierLocation()
        else:
            self.phi_bar = float(phi_bar)
            if validate:
                # Warn (don’t fail) if supplied value seems inconsistent
                Vb = float(V(self.phi_bar))
                if not np.isfinite(Vb):
                    warnings.warn(
                        "phi_bar provided but V(phi_bar) is non-finite; "
                        "subsequent computations may fail.",
                        RuntimeWarning,
                    )
                elif abs(Vb - self.V_meta) > max(1e-10, 1e-10 * max(1.0, abs(self.V_meta))):
                    warnings.warn(
                        "phi_bar provided does not satisfy V(phi_bar)≈V(phi_metaMin). "
                        "Continuing with the user-provided value.",
                        RuntimeWarning,
                    )

        # Characteristic scale
        self.rscale = float(self.findRScale()) if rscale is None else float(rscale)

        # Other parameters
        self.alpha = float(alpha)

    # --- SF-1: builtin finite-difference derivatives --------------------------
    def dV(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Finite-difference approximation to :math:`dV/d\phi`.

        This method supports scalars or array-like `phi` (broadcasted). If the
        user supplied a custom `dV` in the constructor, that callable overrides
        this implementation.

        Scheme
        ------
        • Order 4 (default):
          ``(V(phi-2h) - 8V(phi-h) + 8V(phi+h) - V(phi+2h)) / (12h)``

        • Order 2:
          ``(V(phi+h) - V(phi-h)) / (2h)``

        where ``h = self.phi_eps`` (an absolute step).

        Parameters
        ----------
        phi : float or array_like
            Evaluation point(s).

        Returns
        -------
        float or np.ndarray
            Approximation to :math:`V'(\phi)`.
        """
        V = self.V
        h = self.phi_eps
        if self._fd_order == 4:
            return (V(phi - 2 * h) - 8 * V(phi - h) + 8 * V(phi + h) - V(phi + 2 * h)) / (12.0 * h)
        # 2nd-order fallback
        return (V(phi + h) - V(phi - h)) / (2.0 * h)

    def dV_from_absMin(self, delta_phi: float) -> float:
        r"""
        High-accuracy :math:`dV/d\phi` at ``phi = phi_absMin + delta_phi``.

        Near the TRUE minimum, floating-point cancellation can degrade the FD
        derivative. We therefore blend the finite-difference value with the
        linearized Taylor estimate
        :math:`V'(\phi) \approx V''(\phi_{\rm absMin}) (\phi-\phi_{\rm absMin})`.

        Blending
        --------
        We use a smooth weight
        ``w = exp(-(delta_phi / self.phi_eps)**2)``, and return
        ``w * (d2V * delta_phi) + (1 - w) * dV(phi)``.

        Parameters
        ----------
        delta_phi : float
            Offset from the absolute minimum.

        Returns
        -------
        float
            Blended derivative estimate at ``phi_absMin + delta_phi``.
        """
        phi = self.phi_absMin + delta_phi
        dV_fd = float(self.dV(phi))  # may be user-supplied dV
        # If our derivative is overridden, d2V might be too; both are supported.
        if self.phi_eps > 0.0:
            dV_lin = float(self.d2V(phi)) * float(delta_phi)
            w = np.exp(- (float(delta_phi) / self.phi_eps) ** 2)
            return w * dV_lin + (1.0 - w) * dV_fd
        return dV_fd

    def d2V(self, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Finite-difference approximation to :math:`d^2V/d\phi^2`.

        Scheme
        ------
        • Order 4 (default):
          ``(-V(phi-2h) + 16V(phi-h) - 30V(phi) + 16V(phi+h) - V(phi+2h)) / (12h^2)``

        • Order 2:
          ``(V(phi+h) - 2V(phi) + V(phi-h)) / h^2``

        Parameters
        ----------
        phi : float or array_like
            Evaluation point(s).

        Returns
        -------
        float or np.ndarray
            Approximation to :math:`V''(\phi)`.
        """
        V = self.V
        h = self.phi_eps
        if self._fd_order == 4:
            return (-V(phi - 2 * h) + 16 * V(phi - h) - 30 * V(phi)
                    + 16 * V(phi + h) - V(phi + 2 * h)) / (12.0 * h * h)
        # 2nd-order fallback
        return (V(phi + h) - 2 * V(phi) + V(phi - h)) / (h * h)

    # --- SF-2: Barriers and Scales -------------------------------------
    def findBarrierLocation(self) -> float:
        R"""
        Locate the **edge of the barrier** between the metastable and absolute minima.

        We define ``phi_bar`` as the point **between** ``phi_metaMin`` and ``phi_absMin``
        where the potential crosses the false vacuum level:
        :math:`V(\phi_{\rm bar}) = V(\phi_{\rm metaMin})`, on the *downhill* side of
        the barrier (moving from the metastable minimum toward the absolute minimum).

        Implementation notes
        --------------------
        - We first locate the barrier **top** (argmax of :math:`V`) within the open
          interval between the two minima using a bounded 1D search. This is robust
          even when the potential is not strictly monotonic away from the top.
        - We then **root-find** :math:`G(\phi) \equiv V(\phi)-V(\phi_{\rm metaMin})`
          on the interval between the top and the absolute minimum; this bracket
          has opposite signs under metastability and guarantees a clean crossing.
        - Results (``phi_bar``, ``phi_top``, heights, etc.) are cached to
          ``self._barrier_info`` for inspection and reuse by other methods.

        Returns
        -------
        float
            ``phi_bar`` such that :math:`V(\phi_{\rm bar}) = V(\phi_{\rm metaMin})`.

        Raises
        ------
        PotentialError
            If no barrier top is found inside the interval, or the barrier height
            is non-positive (i.e. no barrier), with reason code ``"no barrier"``.
        """
        # Basic interval & scales
        a, b = (self.phi_metaMin, self.phi_absMin)
        left, right = (a, b) if a < b else (b, a)
        dphi = abs(self.phi_metaMin - self.phi_absMin)
        if dphi == 0:
            raise PotentialError(
                "phi_metaMin and phi_absMin coincide; barrier is ill-defined.",
                "no barrier"
            )

        V = self.V
        V_meta = V(self.phi_metaMin)
        V_abs = V(self.phi_absMin)
        if not (V_meta > V_abs):
            # This should already be caught in __init__, but keep it defensive.
            raise PotentialError(
                "Expected V(phi_metaMin) > V(phi_absMin); metastability violated.",
                "stable, not metastable"
            )

        # 1) Find the barrier top robustly in (left, right).
        # Use a bounded scalar maximization of V, i.e. minimization of -V.
        xtol = max(1e-12 * dphi, np.finfo(float).eps**0.5 * dphi)
        res = optimize.minimize_scalar(
            lambda x: -V(x),
            bounds=(left, right),
            method="bounded",
            options={"xatol": xtol}
        )
        phi_top = float(res.x)

        # Sanity: top must lie strictly within the interval.
        if not (left < phi_top < right):
            raise PotentialError(
                "Barrier top not found inside (phi_metaMin, phi_absMin). "
                "Assume no barrier.",
                "no barrier"
            )

        Vtop = V(phi_top) - V_meta
        if not (Vtop > 0.0):
            # No rise above the false vacuum level → no barrier.
            raise PotentialError(
                "Barrier height above the false vacuum is non-positive.",
                "no barrier"
            )

        # 2) Find phi_bar where G(phi) = V(phi) - V_meta crosses zero on the
        # downhill side of the barrier (from phi_top to the absolute minimum).
        def G(x: float) -> float:
            return V(x) - V_meta

        # Determine which side is the absolute minimum w.r.t. phi_top.
        # We always choose the segment [phi_top, absMin_side] that contains the
        # downhill crossing.
        if self.phi_absMin > self.phi_metaMin:
            # absMin is to the right; bracket [phi_top, absMin]
            x_lo, x_hi = phi_top, self.phi_absMin
        else:
            # absMin is to the left; bracket [absMin, phi_top] but keep (lo, hi) ordered
            x_lo, x_hi = self.phi_absMin, phi_top

        # Ensure a clean sign change; G(phi_top) > 0 by construction, G(abs) < 0.
        G_lo, G_hi = G(x_lo), G(x_hi)
        if not (np.sign(G_lo) * np.sign(G_hi) <= 0.0):
            # Very rare numeric corner: reinforce bracket by a tiny inward nudge.
            epsx = 1e-12 * dphi
            x_lo2 = x_lo + np.sign(x_hi - x_lo) * max(epsx, xtol)
            G_lo2 = G(x_lo2)
            if np.sign(G_lo2) * np.sign(G_hi) > 0.0:
                # As a last resort, scan a coarse grid to detect the first sign change.
                grid = np.linspace(x_lo, x_hi, 256)
                Gg = np.array([G(x) for x in grid])
                idx = np.where(np.sign(Gg[:-1]) * np.sign(Gg[1:]) <= 0.0)[0]
                if idx.size == 0:
                    raise PotentialError(
                        "Could not bracket the barrier edge where V=V(phi_metaMin).",
                        "no barrier"
                    )
                x_lo, x_hi = grid[idx[0]], grid[idx[0] + 1]

        # Robust root with Brent.
        phi_bar = float(optimize.brentq(G, x_lo, x_hi, xtol=xtol, rtol=1e-12, maxiter=200))

        # Cache useful diagnostics
        self._barrier_info = {
            "phi_bar": phi_bar,
            "phi_top": phi_top,
            "V_top_minus_Vmeta": Vtop,
            "V_meta": V_meta,
            "V_abs": V_abs,
            "interval": (left, right),
        }
        return phi_bar

    def findRScale(self) -> float:
        R"""
        Estimate a **characteristic radial scale** for the instanton.

        Physics & rationale
        -------------------
        Near the barrier **top** the Euclidean EoM linearizes to
        :math:`\phi'' + (\alpha/r)\phi' \simeq V''(\phi_{\rm top}) (\phi - \phi_{\rm top})`.
        A naive estimate would be :math:`r_{\rm curv} \sim 1/\sqrt{|V''(\phi_{\rm top})|}`.
        However, for **flat-topped** barriers :math:`V''(\phi_{\rm top}) \to 0`, making this
        estimate blow up even when tunneling is well-defined.

        We therefore use a **cubic-model** surrogate (legacy-compatible and robust):
        fit a cubic that has a maximum at the barrier top and a minimum at the false
        vacuum, which yields the scale

        .. math::
            r_{\rm cubic} = \frac{|\phi_{\rm top} - \phi_{\rm metaMin}|}
                                    {\sqrt{6 [V(\phi_{\rm top}) - V(\phi_{\rm metaMin})] }}.

        This stays finite on flat tops and tracks the small-oscillation period scale
        up to an :math:`\mathcal{O}(1)` factor.

        Implementation notes
        --------------------
        - Reuses/derives the barrier top from :meth:`findBarrierLocation`.
        - Optionally computes the curvature-based scale (diagnostic), but **returns
          the cubic scale** to remain fully backward compatible with cosmoTransitions.
        - Stores diagnostics in ``self._scale_info`` for introspection.

        Returns
        -------
        float
            The characteristic scale (cubic model), used elsewhere to set
            integration step sizes and wall extent.

        Raises
        ------
        PotentialError
            If the barrier does not exist or has non-positive height, with reason
            code ``"no barrier"``.
        """
        # Ensure barrier info exists (also validates the barrier)
        try:
            phi_bar = getattr(self, "phi_bar", None)
            if phi_bar is None:
                phi_bar = self.findBarrierLocation()
        except PotentialError:
            # Propagate with the same message
            raise

        # Either use the cached top or recompute to be safe.
        if not hasattr(self, "_barrier_info") or "phi_top" not in self._barrier_info:
            # Recreate minimal info via a lightweight call
            _ = self.findBarrierLocation()

        info = getattr(self, "_barrier_info", {})
        phi_top = info.get("phi_top", None)
        if phi_top is None:
            # Fallback: recompute locally
            left, right = sorted((self.phi_metaMin, self.phi_absMin))
            dphi = abs(self.phi_metaMin - self.phi_absMin)
            xtol = max(1e-12 * dphi, np.finfo(float).eps**0.5 * dphi)
            res = optimize.minimize_scalar(
                lambda x: -self.V(x),
                bounds=(left, right),
                method="bounded",
                options={"xatol": xtol}
            )
            phi_top = float(res.x)

        V_meta = self.V(self.phi_metaMin)
        Vtop = self.V(phi_top) - V_meta
        if not (Vtop > 0.0):
            raise PotentialError(
                "Barrier height above the false vacuum is non-positive.",
                "no barrier"
            )

        # Legacy-compatible cubic scale (robust for flat tops)
        xtop = phi_top - self.phi_metaMin
        rscale_cubic = abs(xtop) / np.sqrt(6.0 * Vtop)

        # Optional diagnostic: curvature-based scale near the top
        try:
            d2V_top = float(self.d2V(phi_top))
            rscale_curv = (1.0 / np.sqrt(-d2V_top)) if d2V_top < 0.0 else np.inf
        except Exception:
            d2V_top, rscale_curv = np.nan, np.inf

        # Cache diagnostics for users
        self._scale_info = {
            "phi_top": phi_top,
            "V_top_minus_Vmeta": Vtop,
            "xtop": xtop,
            "rscale_cubic": rscale_cubic,
            "rscale_curv": rscale_curv,
            "d2V_top": d2V_top,
        }

        return rscale_cubic

    # -----------------------------------------------------------------------
    # Lot SF-3 — Quadratic local solution (exactSolution) & Initial conditions
    # ------------------------------------------------------------------------
    _exactSolution_rval = namedtuple("exactSolution_rval", "phi dphi")

    def exactSolution(self, r: float, phi0: float, dV: float, d2V: float):
        r"""
        Regular solution of the EOM at radius r assuming a *local quadratic*
        potential around ``phi0``:

            phi'' + (alpha/r) phi' = V'(phi0) + V''(phi0) * (phi - phi0)

        Let ``nu = (alpha-1)/2``. With ``beta = sqrt(|d2V|)`` and ``t = beta*r``:

        * If ``d2V > 0``:
            phi(r) - phi0 = (dV/d2V) * [ Γ(nu+1) * (t/2)^(-nu) I_nu(t) - 1 ]

        * If ``d2V < 0`` (write ``beta = sqrt(-d2V)``):
            replace ``I_nu -> J_nu``.

        * If **d2V == 0** (flat curvature): the EOM reduces to a constant drive,
          and the *exact regular* solution is

            phi(r) = phi0 + dV * r^2 / (2*(alpha+1)),     phi'(r) = dV * r / (alpha+1).

        Numerical strategy
        ------------------
        * Return exactly ``(phi0, 0.0)`` for ``r == 0``.
        * Use the closed form above for ``d2V == 0`` (robust and overflow-free).
        * For small arguments (``t = beta*r`` below a cutoff), evaluate via a short
          even-power series that is well-conditioned and manifestly regular.
        * Otherwise, use the Bessel/modified-Bessel forms and suppress harmless
          overflow warnings (the combinations we form are finite).

        Parameters
        ----------
        r : float
            Radius (>= 0). At r=0 the regular solution has phi'(0)=0.
        phi0 : float
            Field value at r\approx 0 about which the quadratic expansion is taken.
        dV : float
            V'(phi0).
        d2V : float
            V''(phi0).

        Returns
        -------
        exactSolution_rval
            Named tuple ``(phi, dphi)`` evaluated at r.

        Notes
        -----
        * Regularity enforces phi'(0)=0 for any alpha >= 0.
        * The small-t series keeps terms up to O(t^6), which is plenty for
          t ≲ 1e-2 in double precision.
        """
        # Input hygiene
        if not np.isfinite(r) or r < 0:
            raise ValueError("exactSolution: 'r' must be finite and >= 0.")
        if not (np.isfinite(phi0) and np.isfinite(dV) and np.isfinite(d2V)):
            raise ValueError("exactSolution: phi0, dV, and d2V must be finite.")

        # r = 0 → regular boundary condition
        if r == 0.0:
            return self._exactSolution_rval(phi0, 0.0)

        # Trivial "flat" curvature: exact closed form, avoids any Bessel work
        if d2V == 0.0:
            denom = (self.alpha + 1.0)
            # alpha >= 0 in physical use; denom>0. Keep formula general nonetheless.
            phi = phi0 + (dV * r * r) / (2.0 * denom)
            dphi = (dV * r) / denom
            return self._exactSolution_rval(float(phi), float(dphi))

        # Common definitions
        nu = 0.5 * (self.alpha - 1.0)
        beta = float(np.sqrt(abs(d2V)))
        t = beta * r

        # If the local slope is zero, we still need the structure for dphi; keep path.
        if dV == 0.0:
            # In all branches, the solution collapses to phi(r)=phi0, phi'(r)=0.
            return self._exactSolution_rval(phi0, 0.0)

        # Robust small-argument expansion (even powers of t). Coefficients:
        # c_k = Γ(nu+1) / [ k! Γ(k+nu+1) ] for k >= 1, so that
        # Γ(nu+1)*(t/2)^(-nu) I_nu(t) - 1 ≈ Σ_{k=1..K} c_k (t/2)^{2k}
        # Derivative follows analytically.
        def small_t_series(sign: float):
            # sign = +1 for d2V>0 (I_nu), -1 for d2V<0 (J_nu) because J-series alternates
            g = special.gamma
            tau = 0.5 * t
            # Accumulate up to k=3 (t^6), which is ample for t ≲ 1e-2
            phi_acc = 0.0
            dphi_acc = 0.0
            for k in (1, 2, 3):
                ck = g(nu + 1.0) / (float(special.factorial(k)) * g(k + nu + 1.0))
                term = ck * (tau ** (2 * k))
                # For J_nu the even-power series alternates: I_nu → (+), J_nu → (+,-,+,...)
                term *= (sign ** k)
                phi_acc += term
                # d/d r of (tau^{2k}) = (2k) * tau^{2k-1} * (dtau/dr); dtau/dr = beta/2
                dphi_acc += (2 * k) * (tau ** (2 * k - 1)) * (beta / 2.0) * (sign ** k)
            # Multiply by dV/d2V and add phi0; derivative picks the same prefactor.
            pref = (dV / d2V)
            phi = phi0 + pref * phi_acc
            dphi = pref * dphi_acc
            return float(phi), float(dphi)

        # Choose evaluation path
        small_cut = 1e-5  # conservative; keeps series well within FP accuracy
        if t <= small_cut:
            # d2V>0 → I_nu (sign=+1); d2V<0 → J_nu (sign alternation)
            sign = +1.0 if d2V > 0.0 else -1.0
            return self._exactSolution_rval(*small_t_series(sign))

        # Full expressions via Bessel/modified-Bessel for moderate/large t
        gamma = special.gamma
        if d2V > 0.0:
            # Modified Bessel case (I_nu)
            with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                iv = special.iv
                # phi
                core = gamma(nu + 1.0) * (0.5 * t) ** (-nu) * iv(nu, t) - 1.0
                phi = phi0 + (dV / d2V) * core
                # dphi: careful with the r in the denominator; we are not in small-t branch anymore
                term1 = -nu * ((0.5 * t) ** (-nu) / r) * iv(nu, t)
                term2 = (0.5 * t) ** (-nu) * 0.5 * beta * (iv(nu - 1.0, t) + iv(nu + 1.0, t))
                dphi = (gamma(nu + 1.0) * (dV / d2V)) * (term1 + term2)
            return self._exactSolution_rval(float(phi), float(dphi))
        else:
            # Ordinary Bessel case (J_nu)
            with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                jv = special.jv
                core = gamma(nu + 1.0) * (0.5 * t) ** (-nu) * jv(nu, t) - 1.0
                phi = phi0 + (dV / d2V) * core
                term1 = -nu * ((0.5 * t) ** (-nu) / r) * jv(nu, t)
                term2 = (0.5 * t) ** (-nu) * 0.5 * beta * (jv(nu - 1.0, t) - jv(nu + 1.0, t))
                dphi = (gamma(nu + 1.0) * (dV / d2V)) * (term1 + term2)
            return self._exactSolution_rval(float(phi), float(dphi))

    _initialConditions_rval = namedtuple("initialConditions_rval", "r0 phi dphi")

    def initialConditions(self, delta_phi0: float, rmin: float, delta_phi_cutoff: float):
        r"""
        Construct *regular* initial conditions away from the r=0 singular point.

        Strategy
        --------
        Let ``phi0 = phi_absMin + delta_phi0``. Use the local quadratic solution
        (``exactSolution``) to evaluate the field at ``rmin``. If that already
        satisfies the requested *offset* from the absolute minimum,

            |phi(r0) - phi_absMin|  >  |delta_phi_cutoff|,

        we start at r0 = rmin. Otherwise, we *increase* r geometrically until the
        condition is met, and then solve for the *exact* r0 with a 1D root find.

        Edge cases & safeguards
        -----------------------
        * If the field is initially moving *toward the wrong side* (sign mismatch
          between ``dphi(rmin)`` and ``delta_phi0``), we return r0=rmin rather than
          pushing r out (increasing r will not fix the direction).
        * All returns are **named tuples** for consistency.
        * If the geometric search fails to bracket the root (pathological V),
          a clear ``IntegrationError`` is raised.

        Parameters
        ----------
        delta_phi0 : float
            Offset at the center: ``phi(0) - phi_absMin``.
        rmin : float
            Minimal radius to start the integration (>= 0).
        delta_phi_cutoff : float
            Target offset at r0: ``phi(r0) - phi_absMin`` in magnitude.

        Returns
        -------
        initialConditions_rval
            Named tuple ``(r0, phi(r0), dphi(r0))``.
        """
        if rmin < 0 or not np.isfinite(rmin):
            raise ValueError("initialConditions: rmin must be finite and >= 0.")
        if not (np.isfinite(delta_phi0) and np.isfinite(delta_phi_cutoff)):
            raise ValueError("initialConditions: delta_phi0 and delta_phi_cutoff must be finite.")

        phi0 = self.phi_absMin + delta_phi0
        dV0 = self.dV_from_absMin(delta_phi0)
        d2V0 = self.d2V(phi0)

        # Evaluate at rmin via the regular local solution
        phi_r0, dphi_r0 = self.exactSolution(rmin, phi0, dV0, d2V0)

        # If rmin already meets the requested offset, start there
        if abs(phi_r0 - self.phi_absMin) > abs(delta_phi_cutoff):
            return self._initialConditions_rval(rmin, float(phi_r0), float(dphi_r0))

        # If the field is moving the "wrong" way, do not expand r0 further
        if np.sign(dphi_r0) != np.sign(delta_phi0) and dphi_r0 != 0.0 and delta_phi0 != 0.0:
            return self._initialConditions_rval(rmin, float(phi_r0), float(dphi_r0))

        # Geometric growth to bracket the solution where |phi - phi_absMin| crosses the cutoff
        r_left = rmin if rmin > 0.0 else np.finfo(float).eps
        r = max(r_left, rmin)
        max_tries = 60  # extremely generous; avoids silent infinite loops
        growth = 10.0  # legacy behavior
        for _ in range(max_tries):
            r_last = r
            r *= growth
            phi, _dphi = self.exactSolution(r, phi0, dV0, d2V0)
            if abs(phi - self.phi_absMin) > abs(delta_phi_cutoff):
                break
        else:
            # Failed to bracket — this suggests a pathological potential or parameters
            raise IntegrationError("initialConditions: failed to bracket r0 (no crossing found).")

        # Root for |phi(r) - phi_absMin| - |delta_phi_cutoff| = 0 on [r_last, r]
        def deltaPhiDiff(r_):
            p, _ = self.exactSolution(r_, phi0, dV0, d2V0)
            return abs(p - self.phi_absMin) - abs(delta_phi_cutoff)

        r0 = optimize.brentq(deltaPhiDiff, r_last, r, disp=False)
        phi_r0, dphi_r0 = self.exactSolution(r0, phi0, dV0, d2V0)
        return self._initialConditions_rval(float(r0), float(phi_r0), float(dphi_r0))

    # -------------------------------
    # Lot SF-4 — ODE core (EOM + RKQS driver + sampler)
    # -------------------------------
    @staticmethod
    def _normalize_tolerances(epsfrac, epsabs):
        ef_arr = np.atleast_1d(epsfrac).astype(float)
        ea_arr = np.atleast_1d(epsabs).astype(float)

        # Scalars for rkqs (conservative / strictest across components)
        ef_scalar = float(np.min(ef_arr))
        ea_scalar = float(np.min(ea_arr))

        # Per-component absolute thresholds for event/convergence checks
        if ea_arr.size == 1:
            eps_phi = eps_dphi = 3.0 * ea_arr[0]
        else:
            eps_phi  = 3.0 * ea_arr[0]
            eps_dphi = 3.0 * ea_arr[1]
        return ef_scalar, ea_scalar, eps_phi, eps_dphi

    def equationOfMotion(self, y, r):
        """
        Right-hand side (RHS) of the single-field bounce ODE.

        Solves the 2D first-order system for y = [phi, dphi]:

            dphi/dr   = y[1]
            d^2phi/dr^2 = dV/dphi(phi) - (alpha / r) * dphi

        Notes
        -----
        - The regular instanton solution satisfies dphi ~ O(r) near r → 0,
          so (alpha/r)*dphi stays finite. We nevertheless guard against r <= 0
          to avoid spurious divisions in any caller that might probe r=0.

        Parameters
        ----------
        y : array_like of shape (2,)
            Current state vector [phi(r), dphi(r)].
        r : float
            Current radius (r > 0 in all production paths).

        Returns
        -------
        np.ndarray shape (2,)
            RHS evaluated at (y, r).
        """
        # Ensure a small but finite radius for the friction term
        # (robust against accidental r=0 calls from external drivers).
        r_eff = r if r > 0.0 else 1e-30
        phi, dphi = float(y[0]), float(y[1])
        return np.array([dphi, self.dV(phi) - self.alpha * dphi / r_eff], dtype=float)

    _integrateProfile_rval = namedtuple("integrateProfile_rval", "r y convergence_type")

    def integrateProfile(self, r0, y0, dr0,
                         epsfrac, epsabs, drmin, rmax, *eqn_args):
        r"""
        Integrate the bubble-wall ODE until we (a) converge to the false minimum,
        (b) bracket an overshoot/undershoot and extrapolate, or (c) hit limits.

        Equation
        --------
        .. math::
            \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} = \frac{dV}{d\phi}.

        Stopping modes
        --------------
        - "converged": |phi - phi_metaMin| and |dphi| are both within tolerance.
        - "overshoot": phi crosses phi_metaMin within the step; we cubic-interpolate
          to the crossing point.
        - "undershoot": field turns around (dphi has the "wrong" sign); we
          cubic-interpolate to the turning point where dphi = 0.

        Parameters
        ----------
        r0 : float
            Starting radius.
        y0 : array_like, shape (2,)
            Starting state [phi(r0), dphi(r0)].
        dr0 : float
            Initial stepsize supplied to the adaptive RK driver.
        epsfrac, epsabs : array_like, shape (2,)
            Relative/absolute tolerances passed to :func:`helper_functions.rkqs`.
            They also set our convergence thresholds.
        drmin : float
            Minimum allowed step size; below this we abort with IntegrationError.
        rmax : float
            Maximum allowed integration span (absolute); if r exceeds r0 + rmax,
            we abort with IntegrationError.
        *eqn_args :
            Extra arguments forwarded to :meth:`equationOfMotion` (used by subclasses).

        Returns
        -------
        r : float
            Final radius (end of integration or extrapolated event location).
        y : np.ndarray, shape (2,)
            Final state [phi, dphi].
        convergence_type : {"converged", "overshoot", "undershoot"}
            Classification of the stopping condition.

        Raises
        ------
        IntegrationError
            If step control fails (dr < drmin), if we exceed rmax, or if the
            event extrapolation cannot be bracketed.
        """
        # Normalize inputs
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != (2,):
            raise ValueError("integrateProfile: y0 must have shape (2,) [phi, dphi].")
        if not (np.isfinite(r0) and np.isfinite(y0).all()):
            raise ValueError("integrateProfile: non-finite initial state.")

        # Local view of the ODE
        def dY(y, r, args=eqn_args):
            return self.equationOfMotion(y, r, *args)

        # Precompute tolerances for event tests
        ef_scalar, ea_scalar, eps_phi, eps_dphi = self._normalize_tolerances(epsfrac, epsabs)


        dydr0 = dY(y0, r0)
        # Direction flag: if phi starts essentially at phi_metaMin, use -sign(dphi)
        # so that "moving away" is labeled undershoot and "crossing" is overshoot.
        disp0 = y0[0] - self.phi_metaMin
        ysign = np.sign(disp0) if abs(disp0) > eps_phi else -np.sign(y0[1]) or 1.00

        r_limit = r0 + float(rmax)
        dr = float(dr0)

        # Integration loop
        while True:
            dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, ef_scalar, ea_scalar)
            r1 = r0 + dr
            y1 = y0 + dy
            dydr1 = dY(y1, r1)

            # Hard guards
            if r1 > r_limit:
                raise IntegrationError(
                    f"integrateProfile: exceeded rmax (r={r1:.6e} > {r_limit:.6e}).")
            if dr < drmin:
                raise IntegrationError(
                    f"integrateProfile: step underflow (dr={dr:.3e} < drmin={drmin:.3e}).")

            # Converged?
            if (abs(y1[0] - self.phi_metaMin) < eps_phi) and (abs(y1[1]) < eps_dphi):
                return self._integrateProfile_rval(r1, y1, "converged")

            # Event detection
            disp1 = (y1[0] - self.phi_metaMin)
            slope1 = y1[1]

            # Undershoot: slope keeps the field moving away from the target
            if slope1 * ysign > +eps_dphi:
                # Interpolate within [0,1] in normalized substep to find dphi=0
                f = cubicInterpFunction(y0, dr * dydr0, y1, dr * dydr1)
                g0, g1 = f(0.0)[1], f(1.0)[1]
                # Try to bracket; otherwise fall back to the minimum |dphi|
                try:
                    if g0 * g1 > 0.0:
                        raise ValueError("no bracket for dphi=0")
                    x = optimize.brentq(lambda x: f(x)[1], 0.0, 1.0)
                except Exception:
                    # Fallback: pick x that minimizes |dphi|
                    xs = np.linspace(0.0, 1.0, 33)
                    x = xs[np.argmin(np.abs([f(t)[1] for t in xs]))]
                r_evt = r0 + dr * x
                y_evt = f(x)
                return self._integrateProfile_rval(r_evt, y_evt, "undershoot")

            # Overshoot: we crossed phi = phi_metaMin within the step
            if disp1 * ysign < -eps_phi:
                f = cubicInterpFunction(y0, dr * dydr0, y1, dr * dydr1)
                h0, h1 = f(0.0)[0] - self.phi_metaMin, f(1.0)[0] - self.phi_metaMin
                try:
                    if h0 * h1 > 0.0:
                        raise ValueError("no bracket for phi crossing")
                    x = optimize.brentq(lambda x: f(x)[0] - self.phi_metaMin, 0.0, 1.0)
                except Exception:
                    # Fallback: pick x that minimizes |phi - phi_metaMin|
                    xs = np.linspace(0.0, 1.0, 33)
                    x = xs[np.argmin(np.abs([f(t)[0] - self.phi_metaMin for t in xs]))]
                r_evt = r0 + dr * x
                y_evt = f(x)
                return self._integrateProfile_rval(r_evt, y_evt, "overshoot")

            # Advance
            r0, y0, dydr0 = r1, y1, dydr1
            dr = drnext

    profile_rval = namedtuple("Profile1D", "R Phi dPhi Rerr")

    def integrateAndSaveProfile(self, R, y0, dr, epsfrac, epsabs, drmin,
                                *eqn_args):
        """
        Integrate the bubble profile and sample it at user-specified radio R.

        This is a thin wrapper around the adaptive driver used in
        `integrateProfile`, but here we *always* step through the whole R grid,
        filling with cubic Hermite interpolation between accepted RK steps.

        Parameters
        ----------
        R : array_like
            Monotonic array of radii at which to record [phi, dphi].
            The first element (R[0]) is the starting radius.
        y0 : array_like, shape (2,)
            Initial state [phi(R[0]), dphi(R[0])].
        dr : float
            Initial stepsize suggestion for the adaptive RK driver.
        epsfrac, epsabs : array_like, shape (2,)
            Relative and absolute tolerances (as in `integrateProfile`).
        drmin : float
            Minimum allowed step.
        *eqn_args :
            Extra arguments forwarded to :meth:`equationOfMotion`.

        Returns
        -------
        Profile1D
            Named tuple with fields:
            - R   : np.ndarray of sample radii
            - Phi : np.ndarray of \phi(R)
            - dPhi: np.ndarray of d\phi/dR at the same points
            - Rerr: first radius where `dr < drmin`, else None

        Notes
        -----
        - If a step would drop below `drmin`, we clamp it to `drmin`, record
          `Rerr` (first occurrence), and keep going so the output is still filled.
        """
        R = np.asarray(R, dtype=float)
        if R.ndim != 1 or len(R) < 2:
            raise ValueError("integrateAndSaveProfile: R must be 1D with at least 2 points.")
        N = len(R)
        r0 = R[0]
        y0 = np.asarray(y0, dtype=float)
        if y0.shape != (2,):
            raise ValueError("integrateAndSaveProfile: y0 must have shape (2,).")

        Yout = np.zeros((N, len(y0)))
        Yout[0] = y0

        def dY(y, r, args=eqn_args):
            return self.equationOfMotion(y, r, *args)

        dydr0 = dY(y0, r0)
        Rerr = None

        ef_scalar, ea_scalar, _, _ = self._normalize_tolerances(epsfrac, epsabs)

        i = 1
        while i < N:
            dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, ef_scalar, ea_scalar)

            # Apply the step, clamping if necessary
            if dr >= drmin:
                r1 = r0 + dr
                y1 = y0 + dy
            else:
                # Clamp and tag the first occurrence
                y1 = y0 + dy * (drmin / max(dr, 1e-300))
                dr = drnext = drmin
                r1 = r0 + dr
                if Rerr is None:
                    Rerr = r1

            dydr1 = dY(y1, r1)

            # Fill samples between r0 and r1
            if (r0 < R[i] <= r1):
                f = cubicInterpFunction(y0, dr * dydr0, y1, dr * dydr1)
                while (i < N) and (r0 < R[i] <= r1):
                    x = (R[i] - r0) / dr
                    Yout[i] = f(x)
                    i += 1

            # Advance
            r0, y0, dydr0 = r1, y1, dydr1
            dr = drnext

        rval = (R,) + tuple(Yout.T) + (Rerr,)
        return self.profile_rval(*rval)

    # -------------------------------
    # Lot SF-5 — Find Profile
    # -------------------------------

    def findProfile(self,
                    xguess: float = None,
                    xtol: float = 1e-4,
                    phitol: float = 1e-4,
                    thinCutoff: float = 0.01,
                    npoints: int = 500,
                    rmin: float = 1e-4,
                    rmax: float = 1e4,
                    max_interior_pts: int = None,
                    _MAX_ITERS = 200):
        r"""
        Search for the bounce profile by **shooting** on the initial value
        :math:`\phi(0)` using the classic **overshoot/undershoot** strategy.

        Strategy
        --------
        We parametrize the center value via a monotone "shooting" parameter

        .. math::
            \phi(0) \equiv \phi_{\rm absMin} +
            e^{-x}\,(\phi_{\rm metaMin}-\phi_{\rm absMin}) \,,

        so that small ``x`` places the field close to the false vacuum
        (high energy ⇒ likely **overshoot**), while large ``x`` puts it close to
        the true vacuum (low energy ⇒ likely **undershoot**). For each trial ``x``:

        1. Build practical starting conditions at :math:`r=r_0>0` using
           :meth:`initialConditions(\Delta\phi_0,r_{\min},\Delta\phi_{\rm cutoff})`.
        2. Integrate with :meth:`integrateProfile` until convergence or an event:
           **overshoot** (crossing :math:`\phi_{\rm metaMin}`) or **undershoot**
           (turning point :math:`\phi'=0`).
        3. Update the bracketing interval in ``x`` and bisect until the target
           tolerance is met.
        4. Reintegrate once more and sample densely with
           :meth:`integrateAndSaveProfile` to return the full profile.

        Parameters
        ----------
        xguess : float, optional
            Initial guess for the shooting parameter. If ``None``, choose a
            barrier-informed value so that the implied ``phi(0)`` is close to
            ``phi_bar``.
        xtol : float, optional
            Target half-width for the bracketing in ``x``. When
            ``xmax - xmin < xtol``, the search stops.
        phitol : float, optional
            Base fractional tolerance that sets both the **relative** (``epsfrac``)
            and **absolute** (``epsabs``) tolerances used in the adaptive integrator.
            Internally we scale absolute tolerances as
            ``[phitol*|Δφ|, phitol*|Δφ|/rscale]`` for ``[φ, φ']``.
        thinCutoff : float, optional
            Dimensionless cutoff for the starting surface,
            :math:`|\Delta\phi(r_0)| = \texttt{thinCutoff}\cdot |\phi_{\rm metaMin}-\phi_{\rm absMin}|`.
            Larger values push :math:`r_0` outward (useful for very thin walls).
        npoints : int
            Number of sample radii to return in the final profile (≥ 2).
        rmin : float
            Minimum starting radius **in units of** ``rscale``. Also sets the
            initial stepsize and the minimum admissible stepsize (``0.01*rmin``).
        rmax : float
            Maximum integration span **in units of** ``rscale``.
        max_interior_pts : int, optional
            If provided and > 0, fill the bubble **interior** (``0 ≤ r < r0``)
            with up to this many extra points using the local quadratic solution
            (see :meth:`exactSolution`). If ``None``, defaults to ``npoints//2``.
            If zero, no interior points are added.

        Returns
        -------
        profile : profile_rval
            Named tuple (default: ``Profile1D``) with fields
            ``R, Phi, dPhi, Rerr``. For thin walls, note that ``R[0]`` can be
            significantly greater than zero.

        Raises
        ------
        IntegrationError
            If the solver cannot bracket a usable solution (e.g. both trial
            trajectories converge to the same side), or if step-size underflow /
            excessive range is encountered during the search.

        Notes
        -----
        * ``epsfrac``/``epsabs`` are passed as *2-vectors* for ``[φ, φ']``; the
          driver normalizes them internally (strictest scalar for RK step control,
          per-component thresholds for event checks).
        * The search caps the number of trial integrations to avoid infinite loops
          on pathological potentials; if that cap is hit, a clear error is raised.
        """
        # ---- 0) Sanity on npoints
        npoints = max(int(npoints), 2)

        # ---- 1) Dimensionful radii/scales
        rmin = float(rmin) * self.rscale
        dr0 = rmin  # initial step guess
        drmin = 0.01 * rmin  # hard lower bound for adaptive steps
        rmax = float(rmax) * self.rscale  # absolute span limit

        # ---- 2) Integration tolerances
        delta_phi = self.phi_metaMin - self.phi_absMin
        epsfrac = np.array([phitol, phitol], dtype=float)
        epsabs = np.array([abs(delta_phi) * phitol,
                           abs(delta_phi) * phitol / max(self.rscale, 1e-14)], dtype=float)

        # thin-wall cutoff for the ICs (magnitude only)
        delta_phi_cutoff = abs(thinCutoff * delta_phi)

        # ---- 3) Initial guess/brackets in x
        if xguess is None:
            # Put φ(0) roughly at φ_bar by default
            x = -np.log(
                abs((self.phi_bar - self.phi_absMin) /
                    (self.phi_metaMin - self.phi_absMin))
            )
        else:
            x = float(xguess)

        xmin = float(xtol) * 10.0
        xmax = np.inf
        xincrease = 5.0  # geometric expansion when no upper bound exists


        # Convenience bundle fed to integrateProfile
        integration_args = (dr0, epsfrac, epsabs, drmin, rmax)

        # Keep last successful ICs and end state (for the final sampling pass)
        r0 = rf = None
        y0 = yf = None
        last_event = None

        # ---- 4) Bracket in x by overshoot/undershoot and bisect
        for _ in range(_MAX_ITERS):
            # Map x -> center offset and build ICs at r0>0
            delta_phi0 = np.exp(-x) * delta_phi
            try:
                r0_try, phi0, dphi0 = self.initialConditions(delta_phi0, rmin, delta_phi_cutoff)
            except Exception as err:
                # If IC construction fails, try nudging x toward the "safer" side
                if xmax is np.inf:
                    x *= xincrease
                else:
                    x = 0.5 * (xmin + xmax)
                continue

            # Guard: ICs must be finite
            if not (np.isfinite(r0_try) and np.isfinite(phi0) and np.isfinite(dphi0)):
                if xmax is np.inf:
                    x *= xincrease
                else:
                    x = 0.5 * (xmin + xmax)
                continue

            r0 = float(r0_try)
            y0 = np.array([phi0, dphi0], dtype=float)

            # Integrate until event/convergence
            rf, yf, ctype = self.integrateProfile(r0, y0, *integration_args)
            last_event = ctype

            # Classify and update bracket
            if ctype == "converged":
                break

            if ctype == "undershoot":
                # x too *large* (too close to true minimum) → increase xmin
                xmin = x
                x = x * xincrease if np.isinf(xmax) else 0.5 * (xmin + xmax)

            elif ctype == "overshoot":
                # x too *small* (too close to false minimum) → decrease xmax
                xmax = x
                x = 0.5 * (xmin + xmax)

            # Stopping by bracket width
            if (not np.isinf(xmax)) and (xmax - xmin < xtol):
                break
        else:
            raise IntegrationError(
                "findProfile: failed to bracket a solution within the iteration cap. "
                f"Last event='{last_event}', bracket=[{xmin:.6g}, {xmax:.6g}]."
            )

        # ---- 5) Final dense pass (sampled profile)
        # Ensure we have a finite span; if rf == r0 (very rare), pad by a tiny step
        if not np.isfinite(rf) or not np.isfinite(r0):
            raise IntegrationError("findProfile: non-finite integration bounds for final pass.")
        if rf <= r0:
            rf = r0 + max(1e-12, 1e-6 * self.rscale)

        R = np.linspace(r0, rf, npoints)
        profile = self.integrateAndSaveProfile(R, y0, dr0, epsfrac, epsabs, drmin)

        # ---- 6) (Optional) Fill interior points 0 ≤ r < r0 using the local solution
        if max_interior_pts is None:
            max_interior_pts = len(R) // 2

        if max_interior_pts and max_interior_pts > 0:
            dx0 = R[1] - R[0]
            if R[0] / dx0 <= max_interior_pts:
                n = int(np.ceil(R[0] / dx0))
                R_int = np.linspace(0.0, R[0], n + 1)[:-1]
            else:
                n = int(max_interior_pts)
                # Non-uniform interior spacing that meets R[0] and avoids clustering
                a = (R[0] / dx0 - n) * 2.0 / (n * (n + 1))
                N = np.arange(1, n + 1)[::-1]
                R_int = R[0] - dx0 * (N + 0.5 * a * N * (N + 1))
                R_int[0] = 0.0  # enforce exactly

            Phi_int = np.empty_like(R_int)
            dPhi_int = np.empty_like(R_int)

            # Center value implied by current x (delta_phi0 computed last)
            Phi_int[0] = self.phi_absMin + delta_phi0
            dPhi_int[0] = 0.0
            dV0 = self.dV_from_absMin(delta_phi0)
            d2V0 = self.d2V(Phi_int[0])

            for i in range(1, len(R_int)):
                Phi_int[i], dPhi_int[i] = self.exactSolution(R_int[i], Phi_int[0], dV0, d2V0)

            # Concatenate interior + integrated segments
            R_all = np.append(R_int, profile.R)
            Phi_all = np.append(Phi_int, profile.Phi)
            dPhi_all = np.append(dPhi_int, profile.dPhi)
            profile = self.profile_rval(R_all, Phi_all, dPhi_all, profile.Rerr)

        return profile

    # -------------------------------------------------
    # Lot SF-6 — Action and other importants parameters
    # -------------------------------------------------
    def findAction(self, profile):
        r"""
        Compute the Euclidean action for the bounce profile.

        Definition
        ----------
        For ``alpha = d-1`` (so d is the dimension of the radial integral),
        the action we compute is

            S = ∫ [ ½ (dφ/dr)^2 + ( V(φ) - V(φ_metaMin) ) ] * r^α dr * Ω_α

        where Ω_α is the area of the unit α-sphere:
            Ω_α = 2 π^{(α+1)/2} / Γ((α+1)/2).

        Notes
        -----
        - The profile usually starts at r = r0 > 0 in thin-wall cases.
          The gradient contribution from (0, r0) is negligible to leading order
          (regularity implies φ'(r) ~ O(r)), but the *potential* bulk term inside
          the bubble matters. We therefore add the interior volume term

              ΔS_interior = Vol_d(r0) * [ V(φ(r0)) - V(φ_metaMin) ] ,

          with Vol_d(R) = π^{d/2} R^d / Γ(d/2 + 1).
        - The returned value is a scalar float. For a detailed breakdown, use
          :meth:`actionBreakdown` introduced below.

        Parameters
        ----------
        profile : namedtuple
            Output of :meth:`findProfile`, with fields R, Phi, dPhi (and Rerr).

        Returns
        -------
        float
            Euclidean action S.
        """
        # Validate
        r = np.asarray(profile.R, dtype=float)
        phi = np.asarray(profile.Phi, dtype=float)
        dphi = np.asarray(profile.dPhi, dtype=float)
        if r.ndim != 1 or phi.shape != r.shape or dphi.shape != r.shape or r.size < 2:
            raise ValueError("findAction: malformed profile (R, Phi, dPhi must be 1D and same length ≥ 2).")

        # Geometry factors
        d = self.alpha + 1  # radial integration dimension
        omega = 2.0 * np.pi ** (0.5 * (self.alpha + 1)) / special.gamma(0.5 * (self.alpha + 1))
        weight = (r ** self.alpha) * omega

        # Action density (excluding interior bulk correction)
        dV = self.V(phi) - self.V(self.phi_metaMin)
        kin = 0.5 * dphi ** 2
        integrand = (kin + dV) * weight
        S_line = simpson(integrand, x=r)

        # Interior bulk (potential-only) correction from 0 to r0
        r0 = float(r[0])
        if r0 > 0.0:
            volume_d = (np.pi ** (0.5 * d)) * (r0 ** d) / special.gamma(0.5 * d + 1.0)
            S_interior = volume_d * (self.V(phi[0]) - self.V(self.phi_metaMin))
        else:
            S_interior = 0.0

        return float(S_line + S_interior)

    def evenlySpacedPhi(self, phi, dphi, npoints=100, k=1, fixAbs=True):
        """
        Resample (phi, dphi) on a uniformly-spaced phi-grid.

        This is a convenient post-processing step to analyze quantities as
        functions of the *field value* rather than the radius. Typical use is to
        take (Phi, dPhi) from :meth:`findProfile`.

        Parameters
        ----------
        phi, dphi : array_like
            1D arrays with the same length (usually `profile.Phi` and `profile.dPhi`).
        npoints : int, default 100
            Number of output samples.
        k : int, default 1
            Spline degree for `scipy.interpolate.splrep` (k=1 linear, k=3 cubic).
        fixAbs : bool, default True
            If True, ensure the resampled grid spans exactly
            [phi_absMin, phi_metaMin] by padding endpoints with (dphi=0). If False,
            use the provided endpoint of `phi` as the lower bound.

        Returns
        -------
        phi2, dphi2 : np.ndarray
            `phi2` is uniformly spaced; `dphi2` is the spline-evaluated derivative
            at those points.

        Notes
        -----
        - We first enforce monotonicity of `phi` along the trajectory using
          `helper_functions.monotonicIndices` to avoid small numerical wiggles.
        - Endpoints are padded with zero slope when `fixAbs=True`, which is correct
          for regular solutions asymptoting to the vacua.
        """
        phi = np.asarray(phi, dtype=float).ravel()
        dphi = np.asarray(dphi, dtype=float).ravel()
        if phi.size != dphi.size or phi.ndim != 1:
            raise ValueError("evenlySpacedPhi: phi and dphi must be 1D arrays of equal length.")

        # Optional endpoint padding to enforce [phi_absMin, phi_metaMin]
        if fixAbs:
            phi = np.append(self.phi_absMin, np.append(phi, self.phi_metaMin))
            dphi = np.append(0.0, np.append(dphi, 0.0))
        else:
            phi = np.append(phi, self.phi_metaMin)
            dphi = np.append(dphi, 0.0)

        # Enforce monotonicity of phi (remove tiny backtracks)
        idx = monotonic_indices(phi)
        phi_mono = phi[idx]
        dphi_mono = dphi[idx]

        # Build the spline in φ-space and evaluate on a uniform φ-grid
        tck = interpolate.splrep(phi_mono, dphi_mono, k=int(k))
        if fixAbs:
            phi2 = np.linspace(self.phi_absMin, self.phi_metaMin, int(npoints))
        else:
            phi2 = np.linspace(phi_mono[0], self.phi_metaMin, int(npoints))
        dphi2 = interpolate.splev(phi2, tck)

        return phi2, np.asarray(dphi2, dtype=float)

    # New functions over the legacy versions

    _ActionBreakdown = namedtuple("ActionBreakdown", "S_total S_kin S_pot S_interior r phi dphi density")

    def actionBreakdown(self, profile):
        """
        Detailed action diagnostics.

        Returns a namedtuple with:
          - S_total   : total action (same as `findAction`)
          - S_kin     : ∫ ½ (dφ/dr)^2 r^α Ω_α dr
          - S_pot     : ∫ [V(φ)-V(φ_meta)] r^α Ω_α dr
          - S_interior: potential-only interior bulk from [0, r0]
          - r, phi, dphi: arrays copied from `profile`
          - density   : dict with arrays:
                'kin' : ½ (dφ/dr)^2 * r^α Ω_α
                'pot' : (V(φ)-V(φ_meta)) * r^α Ω_α
                'tot' : sum of the two (line contribution only)

        Notes
        -----
        - The interior bulk term is not added to `density['tot']` (it lives at r<r0).
        """
        r = np.asarray(profile.R, dtype=float)
        phi = np.asarray(profile.Phi, dtype=float)
        dphi = np.asarray(profile.dPhi, dtype=float)

        d = self.alpha + 1
        omega = 2.0 * np.pi ** (0.5 * (self.alpha + 1)) / special.gamma(0.5 * (self.alpha + 1))
        w = (r ** self.alpha) * omega

        dV = self.V(phi) - self.V(self.phi_metaMin)
        kin = 0.5 * dphi ** 2
        dens_kin = kin * w
        dens_pot = dV * w
        dens_tot = dens_kin + dens_pot

        S_kin = simpson(dens_kin, x=r)
        S_pot = simpson(dens_pot, x=r)

        r0 = float(r[0])
        if r0 > 0.0:
            volume_d = (np.pi ** (0.5 * d)) * (r0 ** d) / special.gamma(0.5 * d + 1.0)
            S_interior = volume_d * (self.V(phi[0]) - self.V(self.phi_metaMin))
        else:
            S_interior = 0.0

        S_total = float(S_kin + S_pot + S_interior)

        density = {"kin": dens_kin, "pot": dens_pot,"int": S_interior , "tot": dens_tot}
        return self._ActionBreakdown(S_total, float(S_kin), float(S_pot), float(S_interior),
                                r, phi, dphi, density)

    _WallStats = namedtuple("WallStats", "r_peak r_mid r_lo r_hi thickness phi_lo phi_hi")

    def wallDiagnostics(self, profile, frac=(0.1, 0.9)):
        """
        Estimate wall location and thickness from the profile.

        Parameters
        ----------
        profile : namedtuple
            Output of :meth:`findProfile`.
        frac : tuple(float, float), default (0.1, 0.9)
            Fractions f_lo < f_hi defining φ levels
            φ(f) = φ_absMin + f * (φ_metaMin - φ_absMin).
            The thickness is | r(φ_hi) - r(φ_lo) |.

        Returns
        -------
        WallStats (namedtuple)
            - r_peak: radius where |dφ/dr| is maximal (center of the wall)
            - r_mid : radius where φ = (φ_absMin + φ_metaMin)/2
            - r_lo, r_hi: radii at the chosen fractional levels
            - thickness: |r_hi - r_lo|
            - phi_lo, phi_hi: the corresponding φ levels
        """
        r = np.asarray(profile.R, dtype=float)
        phi = np.asarray(profile.Phi, dtype=float)
        dphi = np.asarray(profile.dPhi, dtype=float)

        # Peak of |dφ/dr|
        i_peak = int(np.nanargmax(np.abs(dphi)))
        r_peak = float(r[i_peak])

        # Levels in φ
        phi_lo = float(self.phi_absMin + frac[0] * (self.phi_metaMin - self.phi_absMin))
        phi_hi = float(self.phi_absMin + frac[1] * (self.phi_metaMin - self.phi_absMin))
        phi_mid = 0.5 * (self.phi_absMin + self.phi_metaMin)

        # Enforce monotonic φ for robust inversion
        idx = monotonic_indices(phi)
        r_mono, phi_mono = r[idx], phi[idx]

        def _interp_r_at(phi_star):
            # map φ -> r via linear interpolation on the monotone branch
            return float(np.interp(phi_star, phi_mono, r_mono))

        r_lo = _interp_r_at(phi_lo)
        r_hi = _interp_r_at(phi_hi)
        r_mid = _interp_r_at(phi_mid)
        thickness = abs(r_hi - r_lo)

        return self._WallStats(r_peak=r_peak, r_mid=r_mid, r_lo=r_lo, r_hi=r_hi,
                          thickness=thickness, phi_lo=phi_lo, phi_hi=phi_hi)

    def betaEff(self, profile, method="rscale"):
        r"""
        Return a *proxy* for the nucleation rate timescale β (dimension of inverse length),
        useful for order-of-magnitude reasoning in the absence of a full thermal history.

        Parameters
        ----------
        profile : namedtuple
            Output of :meth:`findProfile`.
        method : {"rscale", "curvature", "wall"}, default "rscale"
            - "rscale"   : β_eff ≡ 1 / rscale  (robust, always defined)
            - "curvature": β_eff ≡ sqrt( |V''(φ_top)| )  at the barrier top
                           (needs a proper barrier; may coincide with 1/rscale up to O(1))
            - "wall"     : β_eff ≡ 1 / thickness, with thickness from :meth:`wallDiagnostics`.

        Returns
        -------
        float
            β_eff in the same units as inverse radius.

        Notes
        -----
        This is *not* the cosmological β ≡ -d(S3/T)/dt used for nucleation histories.
        Computing that requires the temperature dependence of the potential and the
        derivative of S3(T)/T with respect to time (or T). Here we provide geometry/
        curvature-based proxies that are often used as quick scales.
        """
        method = str(method).lower().strip()
        if method == "rscale":
            return 1.0 / float(self.rscale)

        if method == "wall":
            ws = self.wallDiagnostics(profile)
            return float(np.inf) if ws.thickness == 0.0 else 1.0 / ws.thickness

        if method == "curvature":
            # re-locate the barrier top between meta and bar and use |V''|^1/2
            x1 = min(self.phi_bar, self.phi_metaMin)
            x2 = max(self.phi_bar, self.phi_metaMin)
            phi_tol = abs(self.phi_bar - self.phi_metaMin) * 1e-8
            phi_top = optimize.fminbound(lambda x: -self.V(x), x1, x2, xtol=phi_tol)
            d2 = self.d2V(phi_top)
            return float(np.sqrt(abs(d2)))

        raise ValueError("betaEff: unknown method (use 'rscale', 'curvature', or 'wall').")
