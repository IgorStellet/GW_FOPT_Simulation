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

            self.phi_bar = None  #self.findBarrierLocation()
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
        #self.rscale = float(self.findRScale()) if rscale is None else float(rscale)

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

        Near the minimum, floating-point cancellation can degrade the FD
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
