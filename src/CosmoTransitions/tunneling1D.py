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
from .helper_functions import rkqs, IntegrationError, clamp_val, cubicInterpFunction

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
            Field value at r=0 about which the quadratic expansion is taken.
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
        small_cut = 1e-2  # conservative; keeps series well within FP accuracy
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
