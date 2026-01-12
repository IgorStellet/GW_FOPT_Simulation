"""
generic_potential
=================

This module provides **numerical-derivative utilities** for generic finite-temperature
effective potentials V(X, T).

Design goal (for the modernized CosmoTransitions project)
---------------------------------------------------------
In the legacy CosmoTransitions codebase, `generic_potential` bundled many concerns:
1-loop corrections, thermal functions, plotting, phase tracing, transition finding, etc.

In this modernized project, we keep this module focused and reusable:

- build numerical wrappers for:
  * V_offset(X, T)        : V(X,T) - V(X_ref,T)  (usually X_ref = 0)
  * gradV(X, T)           : ∂V/∂X_i
  * hessV(X, T)           : ∂²V/∂X_i∂X_j
  * dV_dT(X, T)           : ∂V/∂T
  * dgradV_dT(X, T)       : ∂/∂T [∂V/∂X_i]

These wrappers match what transitionFinder expects:
- single-point input X with shape (Ndim,) returns scalar (for V) or vectors/matrices
  of shape (Ndim,) and (Ndim, Ndim).

The user supplies the physics: a callable Vtot(X, T, *extra, **kwargs).
"""


from typing import Any, Callable, NamedTuple, Tuple, Dict, Optional
import contextlib
import io
import os
import sys
from collections.abc import Hashable, Mapping, Sequence
from functools import partial

import numpy as np
import numpy.typing as npt

from .helper_functions import gradientFunction, hessianFunction

ArrayLike = npt.ArrayLike
NDArrayF = npt.NDArray[np.float64]


class PotentialDerivatives(NamedTuple):
    """
    Container returned by :func:`build_finite_T_derivatives`.

    All callables accept:
        (X, T, *extra, **kwargs)

    where X is array_like with last axis of length Ndim (or a single point (Ndim,))
    and T is scalar or array_like broadcastable with X[..., 0].

    Returns
    -------
    V : callable
        Offset potential V(X,T) - V(X_ref,T).
    gradV : callable
        Gradient wrt fields, shape (..., Ndim) for point clouds, or (Ndim,) for a point.
    hessV : callable
        Hessian wrt fields, shape (..., Ndim, Ndim) or (Ndim, Ndim).
    dV_dT : callable
        Temperature derivative of the (un-offset) potential.
    dgradV_dT : callable
        Temperature derivative of the gradient.
    """
    V: Callable[..., NDArrayF | float]
    gradV: Callable[..., NDArrayF]
    hessV: Callable[..., NDArrayF]
    dV_dT: Callable[..., NDArrayF | float]
    dgradV_dT: Callable[..., NDArrayF]


def scalar_to_vector_potential_1d(
    V_scalar: Callable[..., ArrayLike],
) -> Callable[..., NDArrayF]:
    """
    Lift a *scalar-field* potential V(phi, T, ...) into vector form V(X, T, ...)
    with Ndim = 1, using phi = X[..., 0].

    This is convenient when your physics code naturally uses a scalar `phi`,
    but transitionFinder expects a vector `X`.

    Notes
    -----
    The returned function has signature:
        Vtot(X, T, *extra, **kwargs) -> ndarray

    where X may be:
        - scalar (interpreted as phi) [allowed],
        - shape (1,) single point,
        - shape (..., 1) point cloud.

    If you already write V as V(X, T), you do NOT need this wrapper.
    """
    def V_vec(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF:
        X_arr = np.asarray(X, dtype=float)
        # Allow phi passed directly as scalar/1D array for convenience:
        if X_arr.ndim == 0:
            phi = X_arr
        elif X_arr.ndim == 1:
            # Could be a single point (1,) OR a scan of phi values (N,)
            # We interpret:
            #   - (1,) as a single point in field space
            #   - (N,) as a scan, NOT a point in Ndim>1 space (since this is 1D)
            phi = X_arr if X_arr.shape[0] != 1 else X_arr[0]
        else:
            if X_arr.shape[-1] != 1:
                raise ValueError(
                    "scalar_to_vector_potential_1d: expected last axis length 1 for X, "
                    f"got X.shape={X_arr.shape}."
                )
            phi = X_arr[..., 0]

        out = V_scalar(phi, T, *extra, **kwargs)
        return np.asarray(out, dtype=float)

    return V_vec


def build_finite_T_derivatives(
    Vtot: Callable[..., ArrayLike],
    *,
    Ndim: int,
    x_eps: ArrayLike = 1e-3,
    T_eps: float = 1e-3,
    deriv_order: int = 4,
    X_ref: ArrayLike | None = None,
) -> PotentialDerivatives:
    """
    Construct derivative wrappers for a generic finite-temperature potential V(X, T).

    Parameters
    ----------
    Vtot
        Callable of the form:
            Vtot(X, T, *extra, **kwargs) -> array_like
        where X has last axis length Ndim and T is broadcastable with X[..., 0].
    Ndim
        Number of field dimensions.
    x_eps
        Field-space finite-difference step(s) used by gradientFunction/hessianFunction.
        Can be float or array_like of length Ndim.
    T_eps
        Temperature step used for finite differences in T.
    deriv_order
        2 or 4. Controls the finite-difference order in *field derivatives* and
        in the *T-derivative stencils* implemented here.
    X_ref
        Reference point for the offset potential. Default is the origin.

    Returns
    -------
    PotentialDerivatives
        (V_offset, gradV, hessV, dV_dT, dgradV_dT).

    Notes
    -----
    - `V` is returned **offset**: V(X,T) - V(X_ref,T). This is usually what you
      want for phase-transition work (free-energy differences).
    - `dV_dT` is computed for the **un-offset** potential Vtot. If you want the
      derivative of the offset potential, just subtract dV_dT(X_ref,T).
    """
    if int(Ndim) <= 0:
        raise ValueError(f"build_finite_T_derivatives: Ndim must be >= 1, got {Ndim}.")
    if deriv_order not in (2, 4):
        raise ValueError(
            f"build_finite_T_derivatives: deriv_order must be 2 or 4, got {deriv_order}."
        )
    if float(T_eps) <= 0.0:
        raise ValueError(f"build_finite_T_derivatives: T_eps must be > 0, got {T_eps}.")

    Ndim = int(Ndim)
    T_eps = float(T_eps)

    X_ref_arr = np.zeros((1, Ndim), dtype=float) if X_ref is None else np.asarray(X_ref, dtype=float).reshape(1, Ndim)

    # Low-level field-derivative engines
    grad_phi = gradientFunction(Vtot, eps=x_eps, Ndim=Ndim, order=deriv_order)
    hess_phi = hessianFunction(Vtot, eps=x_eps, Ndim=Ndim, order=deriv_order)

    def _normalize_X(X: ArrayLike) -> Tuple[NDArrayF, bool, bool]:
        """
        Normalize X to a point-cloud array with shape (..., Ndim).

        Returns
        -------
        X_pts : ndarray
            Shape (..., Ndim).
        is_single_point : bool
            True if the user provided a single field-space point.
        T_is_scalar_hint : bool
            True if X was given as a single point in a way that usually pairs with scalar T.
        """
        X_arr = np.asarray(X, dtype=float)

        # Scalar X only makes sense for Ndim==1
        if X_arr.ndim == 0:
            if Ndim != 1:
                raise ValueError(
                    "build_finite_T_derivatives: scalar X is only allowed when Ndim=1. "
                    f"Got Ndim={Ndim}."
                )
            return np.array([[float(X_arr)]], dtype=float), True, True

        # 1D input: either a single point (Ndim,) OR (for Ndim==1) a scan (N,)
        if X_arr.ndim == 1:
            if X_arr.shape[0] == Ndim:
                return X_arr.reshape(1, Ndim).astype(float), True, True
            if Ndim == 1:
                # Interpret as a scan of points along the 1D field direction
                return X_arr.reshape(-1, 1).astype(float), False, False
            raise ValueError(
                "build_finite_T_derivatives: ambiguous 1D X. Expected shape (Ndim,) "
                f"for a single point, got X.shape={X_arr.shape} with Ndim={Ndim}."
            )

        # ND input: must already have last axis Ndim
        if X_arr.shape[-1] != Ndim:
            raise ValueError(
                "build_finite_T_derivatives: expected last axis length Ndim. "
                f"Got X.shape={X_arr.shape}, Ndim={Ndim}."
            )
        return X_arr.astype(float), False, False

    def _T_is_scalar(T: ArrayLike) -> bool:
        Tarr = np.asarray(T)
        return Tarr.ndim == 0

    def V_offset(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF | float:
        X_pts, is_single, _ = _normalize_X(X)
        T_arr = np.asarray(T, dtype=float)

        V_val = np.asarray(Vtot(X_pts, T_arr, *extra, **kwargs), dtype=float)
        V0_val = np.asarray(Vtot(X_ref_arr, T_arr, *extra, **kwargs), dtype=float)
        out = V_val - V0_val

        # For a single point + scalar T, return a python float (nice for optimizers).
        if is_single and _T_is_scalar(T_arr):
            return float(np.asarray(out).reshape(-1)[0])
        return np.asarray(out, dtype=float)

    def gradV(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF:
        X_pts, is_single, _ = _normalize_X(X)
        T_arr = np.asarray(T, dtype=float)

        g = np.asarray(grad_phi(X_pts, T_arr, *extra, **kwargs), dtype=float)

        # If single point, remove the point axis while keeping any broadcasted T axes.
        if is_single:
            if g.shape[-1] != Ndim:
                raise RuntimeError(
                    f"gradV: unexpected gradient shape {g.shape}; expected last axis {Ndim}."
                )
            # Expect a singleton point axis right before the last axis
            if g.ndim >= 2 and g.shape[-2] == 1:
                g = g[..., 0, :]
            return np.asarray(g, dtype=float).reshape((-1, Ndim))[-1] if _T_is_scalar(T_arr) else np.asarray(g, dtype=float)

        return np.asarray(g, dtype=float)

    def hessV(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF:
        X_pts, is_single, _ = _normalize_X(X)
        T_arr = np.asarray(T, dtype=float)

        H = np.asarray(hess_phi(X_pts, T_arr, *extra, **kwargs), dtype=float)

        if H.shape[-2:] != (Ndim, Ndim):
            raise RuntimeError(
                f"hessV: unexpected Hessian shape {H.shape}; expected last axes {(Ndim, Ndim)}."
            )

        if is_single:
            # Remove singleton point axis (usually right before the last two axes)
            if H.ndim >= 3 and H.shape[-3] == 1:
                H = H[..., 0, :, :]
            return H.reshape(Ndim, Ndim) if _T_is_scalar(T_arr) else np.asarray(H, dtype=float)

        return np.asarray(H, dtype=float)

    def dV_dT(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF | float:
        X_pts, is_single, _ = _normalize_X(X)
        T0 = np.asarray(T, dtype=float)

        if deriv_order == 2:
            Vp = np.asarray(Vtot(X_pts, T0 + T_eps, *extra, **kwargs), dtype=float)
            Vm = np.asarray(Vtot(X_pts, T0 - T_eps, *extra, **kwargs), dtype=float)
            d = (Vp - Vm) / (2.0 * T_eps)
        else:
            Vp2 = np.asarray(Vtot(X_pts, T0 + 2.0 * T_eps, *extra, **kwargs), dtype=float)
            Vp1 = np.asarray(Vtot(X_pts, T0 + 1.0 * T_eps, *extra, **kwargs), dtype=float)
            Vm1 = np.asarray(Vtot(X_pts, T0 - 1.0 * T_eps, *extra, **kwargs), dtype=float)
            Vm2 = np.asarray(Vtot(X_pts, T0 - 2.0 * T_eps, *extra, **kwargs), dtype=float)
            d = (-Vp2 + 8.0 * Vp1 - 8.0 * Vm1 + Vm2) / (12.0 * T_eps)

        if is_single and _T_is_scalar(T0):
            return float(np.asarray(d).reshape(-1)[0])
        return np.asarray(d, dtype=float)

    def dgradV_dT(X: ArrayLike, T: ArrayLike, *extra: Any, **kwargs: Any) -> NDArrayF:
        X_pts, is_single, _ = _normalize_X(X)
        T0 = np.asarray(T, dtype=float)

        if deriv_order == 2:
            gp = np.asarray(grad_phi(X_pts, T0 + T_eps, *extra, **kwargs), dtype=float)
            gm = np.asarray(grad_phi(X_pts, T0 - T_eps, *extra, **kwargs), dtype=float)
            dg = (gp - gm) / (2.0 * T_eps)
        else:
            gp2 = np.asarray(grad_phi(X_pts, T0 + 2.0 * T_eps, *extra, **kwargs), dtype=float)
            gp1 = np.asarray(grad_phi(X_pts, T0 + 1.0 * T_eps, *extra, **kwargs), dtype=float)
            gm1 = np.asarray(grad_phi(X_pts, T0 - 1.0 * T_eps, *extra, **kwargs), dtype=float)
            gm2 = np.asarray(grad_phi(X_pts, T0 - 2.0 * T_eps, *extra, **kwargs), dtype=float)
            dg = (-gp2 + 8.0 * gp1 - 8.0 * gm1 + gm2) / (12.0 * T_eps)

        dg = np.asarray(dg, dtype=float)

        if is_single:
            if dg.shape[-1] != Ndim:
                raise RuntimeError(
                    f"dgradV_dT: unexpected shape {dg.shape}; expected last axis {Ndim}."
                )
            if dg.ndim >= 2 and dg.shape[-2] == 1:
                dg = dg[..., 0, :]
            return dg.reshape(Ndim,) if _T_is_scalar(T0) else np.asarray(dg, dtype=float)

        return np.asarray(dg, dtype=float)

    return PotentialDerivatives(
        V=V_offset,
        gradV=gradV,
        hessV=hessV,
        dV_dT=dV_dT,
        dgradV_dT=dgradV_dT,
    )

# =============================================================================
# Generic workflow utilities (reusable across different potentials/examples)
# =============================================================================

def ensure_dir(path: Optional[str]) -> Optional[str]:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path
        Directory path. If None or empty, no action is taken.

    Returns
    -------
    path
        The same path, for convenience.
    """
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def savefig(fig: object, save_dir: Optional[str], name: str, *, dpi: int = 160) -> None:
    """
    Save a matplotlib figure in a consistent way.

    Notes
    -----
    - This function does *not* import matplotlib; it just assumes `fig` exposes
      a `savefig(...)` method.
    """
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=dpi, bbox_inches="tight")


@contextlib.contextmanager
def tee_stdout(save_dir: Optional[str], filename: str = "showcase_log.txt"):
    """
    Context manager that duplicates stdout into a text file.

    Parameters
    ----------
    save_dir
        If provided, prints are written to save_dir/filename as well as stdout.
        If None, this is a no-op.
    filename
        Log filename under save_dir.

    Examples
    --------
    >>> with tee_stdout("results", "run.log"):
    ...     print("Hello")
    """
    if not save_dir:
        yield
        return

    os.makedirs(save_dir, exist_ok=True)

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s: str):
            for st in self.streams:
                st.write(s)
                st.flush()
            return len(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    log_path = os.path.join(save_dir, filename)
    with open(log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
        yield


def build_phi_grid(obj_or_phi_a, phi_b: Optional[float] = None, *, margin: float = 0.1, n: int = 800):
    """
    Build a 1D field grid around two endpoints (typically false/true vacua).

    Parameters
    ----------
    obj_or_phi_a
        Either:
        - an object with attributes `phi_metaMin` and `phi_absMin` (e.g. SingleFieldInstanton), OR
        - the first endpoint value (float).
    phi_b
        Second endpoint value (float). If None, `obj_or_phi_a` is interpreted as an object.
    margin
        Fractional margin added beyond [min, max], relative to the span.
    n
        Number of points.

    Returns
    -------
    grid : ndarray
        A 1D grid suitable for plotting V(phi) and related diagnostics.
    """
    if phi_b is None:
        inst = obj_or_phi_a
        lo = float(min(inst.phi_metaMin, inst.phi_absMin))
        hi = float(max(inst.phi_metaMin, inst.phi_absMin))
    else:
        lo = float(min(obj_or_phi_a, phi_b))
        hi = float(max(obj_or_phi_a, phi_b))

    span = hi - lo
    if span <= 0.0:
        span = 1.0
    return np.linspace(lo - margin * span, hi + margin * span, int(n))


def _extract_params_from_V(V_callable):
    """
    Best-effort extraction of common model parameters from functools.partial.

    Returns
    -------
    C, Lambda, T, finiteT : tuple
        Values if found in partial keywords, else None.
    """
    C = Lambda = T = finiteT = None
    try:
        if isinstance(V_callable, partial):
            kws = V_callable.keywords or {}
            C = kws.get("C", None)
            Lambda = kws.get("Lambda", None)
            T = kws.get("T", None)
            finiteT = kws.get("finiteT", None)
    except Exception:
        pass
    return C, Lambda, T, finiteT


def _build_phases_and_transitions(
    V_XT,
    dVdphi_XT,
    hessV_XT,
    dVdT_XT,
    dgradT_XT,
    *,
    T_min: float,
    T_max: float,
    phi_range: tuple[float, float] = (-3.0, 3.0),
    n_phi_scan: int = 200,
    n_T_seeds: int = 5,
    deltaX_target: float = 0.05,
    dtstart_frac: float = 1e-3,
    tjump_frac: float = 1e-3,
    forbidCrit=None,
    tunnelFromPhase_args: Optional[Dict[str, object]] = None,
    nuclCriterion=None,
    Tn_Ttol: float = 1e-3,
    Tn_maxiter: int = 80,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Generic transitionFinder pipeline: seed minima -> trace phases -> thermal history.

    This wraps the typical "Block A+B+C" workflow used in finite-T cosmological
    phase transition studies.

    Returns
    -------
    summary : dict
        Contains:
          - phases, start_phase_key, start_phase
          - critical_transitions, full_transitions, main_transition
          - plus the potential wrappers passed in, for downstream reuse.
    """
    # Local import avoids hard dependencies / circular imports at module import time
    from . import transitionFinder

    T_min = float(T_min)
    T_max = float(T_max)
    phi_low, phi_high = float(phi_range[0]), float(phi_range[1])

    if T_max <= T_min:
        raise ValueError(f"_build_phases_and_transitions: require T_max > T_min, got {T_max} <= {T_min}.")

    # ------------------------------------------------------------------
    # 1) Seed minima along a 1D line (x_low -> x_high) at several T values
    # ------------------------------------------------------------------
    x_low = np.array([phi_low], dtype=float)
    x_high = np.array([phi_high], dtype=float)

    seeds: list[tuple[np.ndarray, float]] = []
    T_refs = np.linspace(T_max, T_min, int(n_T_seeds))

    for T_ref in T_refs:
        T_ref = float(T_ref)

        def V_line(x, T_local=T_ref):
            x_arr = np.asarray(x, dtype=float)
            return V_XT(x_arr, T_local)

        minima = transitionFinder.findApproxLocalMin(
            V_line,
            x_low,
            x_high,
            args=(),
            n=int(n_phi_scan),
            edge=0.05,
        )
        if getattr(minima, "size", 0) == 0:
            continue

        for xm in minima:
            seeds.append((np.asarray(xm, dtype=float).reshape(-1), T_ref))

    if not seeds:
        seeds = [
            (np.zeros(1, dtype=float), T_max),
            (np.zeros(1, dtype=float), T_min),
        ]

    # remove near-duplicates
    unique_seeds: list[tuple[np.ndarray, float]] = []
    for x_seed, T_seed in seeds:
        keep = True
        for x_old, T_old in unique_seeds:
            if abs(T_seed - T_old) < 1e-6 and np.linalg.norm(x_seed - x_old) < 1e-3:
                keep = False
                break
        if keep:
            unique_seeds.append((x_seed, T_seed))

    if verbose:
        print(f"[build_phases] Seeds: {len(unique_seeds)} minima between T = {T_min:.3f} and {T_max:.3f}")

    # ------------------------------------------------------------------
    # 2) Trace phases
    # ------------------------------------------------------------------
    phases = transitionFinder.traceMultiMin(
        V_XT,
        dgradT_XT,
        hessV_XT,
        unique_seeds,
        tLow=T_min,
        tHigh=T_max,
        deltaX_target=float(deltaX_target),
        dtstart=float(dtstart_frac),
        tjump=float(tjump_frac),
        forbidCrit=forbidCrit,
    )

    transitionFinder.removeRedundantPhases(V_XT, phases)
    start_key = transitionFinder.getStartPhase(phases, V_XT)
    start_phase = phases[start_key]

    if verbose:
        print(f"[build_phases] Traced {len(phases)} phases; start phase key = {start_key!r}")

    # ------------------------------------------------------------------
    # 3) Critical temperatures + full transition history (including Tn)
    # ------------------------------------------------------------------
    crit_trans = transitionFinder.findCriticalTemperatures(phases, V_XT, start_high=False)
    if verbose:
        print(f"[build_phases] Found {len(crit_trans)} critical temperatures.")

    def _default_nucl(S: float, T: float) -> float:
        return S / (T + 1e-100) - 140.0

    tf_args: Dict[str, object] = dict(
        Ttol=float(Tn_Ttol),
        maxiter=int(Tn_maxiter),
        phitol=1e-6,
        overlapAngle=45.0,
        verbose=verbose,
        fullTunneling_params=None,
        nuclCriterion=nuclCriterion or _default_nucl,
    )
    if tunnelFromPhase_args:
        tf_args.update(tunnelFromPhase_args)

    full_trans = transitionFinder.findAllTransitions(
        phases,
        V_XT,
        dVdphi_XT,
        tunnelFromPhase_args=tf_args,
    )

    transitionFinder.addCritTempsForFullTransitions(phases, crit_trans, full_trans)

    main_transition = None
    for tdict in full_trans:
        if int(tdict.get("trantype", 0)) == 1:
            main_transition = tdict
            break

    if verbose:
        if main_transition is None:
            print("[build_phases] No first-order transitions found.")
        else:
            Tn = float(main_transition.get("Tnuc", np.nan))
            print(
                "[build_phases] Main FO transition: "
                f"high_phase={main_transition['high_phase']}, "
                f"low_phase={main_transition['low_phase']}, "
                f"Tn ≈ {Tn:.4g}"
            )

    return {
        "V_XT": V_XT,
        "dVdphi_XT": dVdphi_XT,
        "hessV_XT": hessV_XT,
        "dVdT_XT": dVdT_XT,
        "dgradT_XT": dgradT_XT,
        "T_min": T_min,
        "T_max": T_max,
        "phi_range": (phi_low, phi_high),
        "phases": phases,
        "start_phase_key": start_key,
        "start_phase": start_phase,
        "critical_transitions": crit_trans,
        "full_transitions": full_trans,
        "main_transition": main_transition,
    }


def _spinodal_data_for_phase(phase, hessV_XT, *, n_T_scan: int = 300) -> Dict[str, object]:
    """
    Sample the smallest Hessian eigenvalue along a Phase minimum, and find m^2(T)=0 crossings.

    Returns
    -------
    dict with keys: "T_grid", "m2", "T_spinodals"
    """
    from scipy import optimize  # local import

    T_min = float(phase.T[0])
    T_max = float(phase.T[-1])
    T_grid = np.linspace(T_min, T_max, int(n_T_scan))

    def m2_of_T(T: float) -> float:
        x = np.asarray(phase.valAt(T), dtype=float)
        H = np.asarray(hessV_XT(x, float(T)), dtype=float)
        if H.ndim == 0:
            return float(H)
        H = H.reshape(x.size, x.size)
        eigs = np.linalg.eigvalsh(H)
        return float(np.min(eigs))

    m2_vals = np.array([m2_of_T(float(T)) for T in T_grid], dtype=float)

    spinodals: list[float] = []
    for i in range(len(T_grid) - 1):
        m1 = m2_vals[i]
        m2 = m2_vals[i + 1]
        if not (np.isfinite(m1) and np.isfinite(m2)):
            continue
        if m1 == 0.0:
            spinodals.append(float(T_grid[i]))
        elif m1 * m2 < 0.0:
            T_left = float(T_grid[i])
            T_right = float(T_grid[i + 1])
            try:
                root = optimize.brentq(m2_of_T, T_left, T_right)
            except ValueError:
                root = 0.5 * (T_left + T_right)
            spinodals.append(float(root))

    if not spinodals:
        idx = int(np.argmin(np.abs(m2_vals)))
        spinodals = [float(T_grid[idx])]

    spinodals = sorted(spinodals)
    return {"T_grid": T_grid, "m2": m2_vals, "T_spinodals": spinodals}


def _closest_spinodal_to_T(T_target: float, T_spinodals: Sequence[float]) -> float | None:
    """Return the spinodal temperature closest to T_target (or None)."""
    if not T_spinodals:
        return None
    arr = np.asarray(T_spinodals, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(T_target))))
    return float(arr[idx])


def _build_gw_calculator_from_summary(transition_summary: Mapping[str, object]):
    """
    Best-effort GravitationalWaveCalculator construction from a transition summary dict.
    Returns None if insufficient data is available.
    """
    try:
        V_XT = transition_summary["V_XT"]
        dVdphi_XT = transition_summary["dVdphi_XT"]
        dVdT_XT = transition_summary["dVdT_XT"]
        phases = transition_summary["phases"]
        main_transition = transition_summary.get("main_transition", None)
    except Exception:
        return None

    if main_transition is None:
        return None

    try:
        high_key = main_transition["high_phase"]
        low_key = main_transition["low_phase"]
    except Exception:
        return None

    # Local import avoids module-level dependency
    from .gravitational_Waves import GravitationalWaveCalculator

    def V_for_gw(x: np.ndarray, T: float) -> float:
        return float(V_XT(np.asarray(x, dtype=float), float(T)))

    def dV_for_gw(x: np.ndarray, T: float) -> np.ndarray:
        return np.asarray(dVdphi_XT(np.asarray(x, dtype=float), float(T)), dtype=float)

    def dVdT_for_gw(x: np.ndarray, T: float) -> float:
        return float(dVdT_XT(np.asarray(x, dtype=float), float(T)))

    try:
        return GravitationalWaveCalculator(
            V=V_for_gw,
            dV=dV_for_gw,
            dVdT=dVdT_for_gw,
            phases=phases,
            high_phase_key=high_key,
            low_phase_key=low_key,
            fullTunneling_params=None,
        )
    except Exception:
        return None


def _compute_gw_scales_from_calculator(
    gw_calc,
    transition_summary: Mapping[str, object],
    *,
    g_star: float,
    v_w: float,
    dT_fraction: float = 0.001,
) -> Dict[str, float]:
    """
    Compute GW thermodynamic scales using GravitationalWaveCalculator.

    Returns a dict with:
      T_*, alpha, beta/H_*, R_*H_*, Gamma(T_*), peak frequencies, etc.
    """
    keyT = transition_summary.get("key_temperatures", {}) or {}
    T_star = float(getattr(keyT, "get", lambda *_: np.nan)("Tn", np.nan))

    if not np.isfinite(T_star):
        main_transition = transition_summary.get("main_transition", None)
        if isinstance(main_transition, Mapping):
            T_candidate = main_transition.get("Tnuc", main_transition.get("Tcrit"))
            if T_candidate is not None:
                T_star = float(T_candidate)

    if not np.isfinite(T_star):
        nan = float("nan")
        return {
            "gw_T_star_GeV": nan,
            "gw_alpha": nan,
            "gw_beta_over_H": nan,
            "gw_R_star_times_H": nan,
            "gw_nucleation_rate_GeV4": nan,
            "gw_dT_for_beta_GeV": nan,
            "gw_f_sw_peak_Hz": nan,
            "gw_f_turb_peak_Hz": nan,
            "gw_f_coll_peak_Hz": nan,
            "gw_g_star": float(g_star),
            "gw_v_w": float(v_w),
        }

    # Choose dT safely inside overlap window if available
    Tmin = float(getattr(gw_calc, "_T_min", T_star - 1.0))
    Tmax = float(getattr(gw_calc, "_T_max", T_star + 1.0))
    margin = min(T_star - Tmin, Tmax - T_star)
    if margin <= 0.0:
        dT = max(1e-3, dT_fraction * max(T_star, 1.0))
    else:
        dT_max = 0.001 * margin
        dT_pref = dT_fraction * max(T_star, 1.0)
        dT = min(dT_pref, dT_max)
        dT = max(dT, 1e-4)

    beta_over_H = float(gw_calc.beta_over_H(T_star, dT))
    alpha_val = float(gw_calc.alpha(T_star, g_star))
    R_star_H = float(gw_calc.bubble_radius_over_H(beta_over_H, v_w=v_w))
    Gamma_val = float(gw_calc.nucleation_rate(T_star))

    # Peak frequencies (best effort)
    try:
        f_sw_peak = float(gw_calc._f_sw_peak(beta_over_H=beta_over_H, T_star=T_star, g_star=g_star, v_w=v_w))
    except Exception:
        f_sw_peak = float("nan")
    try:
        f_turb_peak = float(gw_calc._f_turb_peak(beta_over_H=beta_over_H, T_star=T_star, g_star=g_star, v_w=v_w))
    except Exception:
        f_turb_peak = float("nan")
    try:
        f_coll_peak = float(gw_calc._f_coll_peak(beta_over_H=beta_over_H, T_star=T_star, g_star=g_star, v_w=v_w))
    except Exception:
        f_coll_peak = float("nan")

    return {
        "gw_T_star_GeV": float(T_star),
        "gw_alpha": float(alpha_val),
        "gw_beta_over_H": float(beta_over_H),
        "gw_R_star_times_H": float(R_star_H),
        "gw_nucleation_rate_GeV4": float(Gamma_val),
        "gw_dT_for_beta_GeV": float(dT),
        "gw_f_sw_peak_Hz": float(f_sw_peak),
        "gw_f_turb_peak_Hz": float(f_turb_peak),
        "gw_f_coll_peak_Hz": float(f_coll_peak),
        "gw_g_star": float(g_star),
        "gw_v_w": float(v_w),
    }


def compute_profile(
    inst,
    xguess: Optional[float] = None,
    phitol: float = 1e-5,
    thinCutoff: float = 0.01,
    npoints: int = 600,
    max_interior_pts=None,
    _MAX_ITERS: int = 200,
):
    """
    Generic wrapper around SingleFieldInstanton.findProfile.

    Notes
    -----
    This is kept intentionally generic: it only assumes `inst.findProfile(...)`
    exists and supports the keyword arguments used here.
    """
    return inst.findProfile(
        xguess=xguess,
        xtol=1e-4,
        phitol=phitol,
        thinCutoff=thinCutoff,
        npoints=npoints,
        rmin=1e-4,
        rmax=1e4,
        max_interior_pts=max_interior_pts,
        _MAX_ITERS=_MAX_ITERS,
    )


def gather_diagnostics(
    inst,
    profile,
    label: str = "",
    transition_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Collect a compact diagnostic dictionary for a bounce + (optional) transition summary.

    This is meant to be reusable across different potentials: it only assumes
    the `inst` object exposes the standard SingleFieldInstanton API.
    """
    r0info = getattr(inst, "_profile_info", {}) or {}
    sinfo = getattr(inst, "_scale_info", {}) or {}

    V_meta = float(inst.V(inst.phi_metaMin))
    V_true = float(inst.V(inst.phi_absMin))
    dV_true_meta = V_true - V_meta
    phi_top = float(sinfo.get("phi_top", np.nan))
    V_top = float(inst.V(phi_top)) if np.isfinite(phi_top) else np.nan

    br = inst.actionBreakdown(profile)

    C, Lambda, T, finiteT = _extract_params_from_V(inst.V)
    S_over_T = (float(br.S_total) / float(T)) if (T is not None and T > 0) else np.nan

    def _safe_beta(method: str) -> float:
        try:
            return float(inst.betaEff(profile, method=method))
        except Exception:
            return np.nan

    betas = {
        "beta_rscale": _safe_beta("rscale"),
        "beta_curvature": _safe_beta("curvature"),
        "beta_wall": _safe_beta("wall"),
    }

    try:
        ws = inst.wallDiagnostics(profile, frac=(0.1, 0.9))
        r_wall = float(ws.r_hi)
        thickness = float(ws.thickness)
    except Exception:
        r_wall, thickness = np.nan, np.nan

    base: Dict[str, object] = {
        "label": label,
        "C": (float(C) if C is not None else np.nan),
        "Lambda_GeV": (float(Lambda) if Lambda is not None else np.nan),
        "finiteT": (bool(finiteT) if finiteT is not None else None),
        "temperature_GeV": (float(T) if T is not None else np.nan),
        "phi_metaMin": float(inst.phi_metaMin),
        "phi_absMin": float(inst.phi_absMin),
        "phi_bar": float(getattr(inst, "phi_bar", np.nan)),
        "phi_top": phi_top,
        "V(phi_meta)": V_meta,
        "V(phi_true)": V_true,
        "V(phi_top)": V_top,
        "DeltaV_true_minus_meta": dV_true_meta,
        "r0": float(r0info.get("r0", np.nan)),
        "phi0": float(r0info.get("phi0", np.nan)),
        "dphi0": float(r0info.get("dphi0", np.nan)),
        "rscale_cubic": float(sinfo.get("rscale_cubic", np.nan)),
        "rscale_curv": float(sinfo.get("rscale_curv", np.nan)),
        "wall_r_hi": r_wall,
        "wall_thickness": thickness,
        "S_total": float(br.S_total),
        "S_kin": float(br.S_kin),
        "S_pot": float(br.S_pot),
        "S_interior": float(br.S_interior),
        "S3_over_T": float(S_over_T),
        **betas,
    }

    if transition_summary is not None:
        keyT = transition_summary.get("key_temperatures", {}) or {}
        Tn_tf = keyT.get("Tn", np.nan)
        Tc_tf = keyT.get("Tc", None)
        Thigh_sp = keyT.get("T_spinodal_high_phase", None)
        Tlow_sp = keyT.get("T_spinodal_low_phase", None)

        base.update(
            {
                "Tn_from_transitionFinder_GeV": float(Tn_tf) if np.isfinite(Tn_tf) else np.nan,
                "Tc_from_transitionFinder_GeV": float(Tc_tf) if (Tc_tf is not None) else np.nan,
                "T_spinodal_high_GeV": float(Thigh_sp) if (Thigh_sp is not None) else np.nan,
                "T_spinodal_low_GeV": float(Tlow_sp) if (Tlow_sp is not None) else np.nan,
            }
        )

        gw_scales = transition_summary.get("gw_scales", None)
        if gw_scales is None:
            gw_calc = _build_gw_calculator_from_summary(transition_summary)
            if gw_calc is not None:
                gw_scales = _compute_gw_scales_from_calculator(
                    gw_calc,
                    transition_summary,
                    g_star=106.75,
                    v_w=1.0,
                )
                transition_summary["gw_scales"] = gw_scales

        if gw_scales:
            base.update(
                {
                    "gw_T_star_GeV": float(gw_scales.get("gw_T_star_GeV", np.nan)),
                    "gw_alpha": float(gw_scales.get("gw_alpha", np.nan)),
                    "gw_beta_over_H": float(gw_scales.get("gw_beta_over_H", np.nan)),
                    "gw_R_star_times_H": float(gw_scales.get("gw_R_star_times_H", np.nan)),
                    "gw_Gamma_Tstar_GeV4": float(gw_scales.get("gw_nucleation_rate_GeV4", np.nan)),
                    "gw_dT_for_beta_GeV": float(gw_scales.get("gw_dT_for_beta_GeV", np.nan)),
                    "gw_f_sw_peak_Hz": float(gw_scales.get("gw_f_sw_peak_Hz", np.nan)),
                    "gw_f_turb_peak_Hz": float(gw_scales.get("gw_f_turb_peak_Hz", np.nan)),
                    "gw_f_coll_peak_Hz": float(gw_scales.get("gw_f_coll_peak_Hz", np.nan)),
                    "gw_g_star": float(gw_scales.get("gw_g_star", np.nan)),
                    "gw_v_w": float(gw_scales.get("gw_v_w", np.nan)),
                    "omega_sw_peak": float(transition_summary.get("omega_sw_peak", np.nan)),
                    "omega_turb_peak": float(transition_summary.get("omega_turb_peak", np.nan)),
                    "omega_coll_peak": float(transition_summary.get("omega_coll_peak", np.nan)),
                    "omega_tot_peak": float(transition_summary.get("omega_tot_peak", np.nan)),
                }
            )

    return base


def save_diagnostics_summary(
    di: Dict[str, object],
    save_dir: Optional[str],
    basename: str = "diagnostics_summary",
    fmt: str = "json",
) -> None:
    """
    Save a diagnostics dictionary to disk.

    Parameters
    ----------
    di
        Diagnostics dictionary.
    save_dir
        Output directory. If None, no file is written.
    basename
        Filename without extension.
    fmt
        "json" | "csv" | "txt"
    """
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)

    fmt = fmt.lower().strip()
    path = os.path.join(save_dir, f"{basename}.{fmt}")

    if fmt == "json":
        import json  # local import
        with open(path, "w", encoding="utf-8") as f:
            json.dump(di, f, indent=2)
        return

    if fmt == "csv":
        with open(path, "w", encoding="utf-8") as f:
            f.write("key,value\n")
            for k, v in di.items():
                f.write(f"{k},{v}\n")
        return

    if fmt == "txt":
        pad = max(len(str(k)) for k in di.keys()) if di else 0
        with open(path, "w", encoding="utf-8") as f:
            for k, v in di.items():
                f.write(f"{str(k).ljust(pad)} : {v}\n")
        return

    raise ValueError("save_diagnostics_summary: fmt must be one of: 'json', 'csv', 'txt'")
