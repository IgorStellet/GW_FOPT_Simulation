# New version of transitionFinder


"""
transitionFinder
================

Tools to study finite-temperature cosmological phase transitions.

This module provides:

- low-level routines to **trace temperature–dependent minima** of a potential
  V(φ, T), most notably :func:`traceMinimum`;
- a lightweight :class:`Phase` container that stores a single minimum as a
  function of T and exposes a spline-based interpolation interface.

Higher-level utilities (phase structure, transitions, nucleation temperatures)
build on top of these primitives.

In contrast, :mod:`.pathDeformation` is meant for finding the tunneling
solution at a fixed temperature, and :mod:`.tunneling1D` contains the
SingleFieldInstanton O(3)/O(4) solvers.

Historically this file supported multi-field configurations; in the current
project we focus primarily on **one scalar order parameter**, but the core
tracing routines remain fully vector-valued.
"""


from typing import NamedTuple
from typing import Any, Callable, Dict, Hashable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, List


import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, interpolate, optimize

#from . import pathDeformation
from . import tunneling1D

# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container for traceMinimum
# ---------------------------------------------------------------------------

class _traceMinimum_rval(NamedTuple):
    """
    Result of :func:`traceMinimum`.

    Parameters
    ----------
    X :
        Array of shape ``(n_steps, n_fields)`` containing the locations of the
        minimum at each traced temperature.
    T :
        Array of shape ``(n_steps,)`` with the corresponding temperature values.
    dXdT :
        Array of shape ``(n_steps, n_fields)`` with ``dx_min/dT`` at each step.
    overX :
        Location reached when the algorithm judges that the phase has
        disappeared / become singular (typically close to a saddle).
    overT :
        Temperature at which ``overX`` was recorded.
    """
    X: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    dXdT: npt.NDArray[np.float64]
    overX: npt.NDArray[np.float64]
    overT: float


TraceMinimumResult = _traceMinimum_rval  # nicer public alias


# ---------------------------------------------------------------------------
# Core: trace a minimum as T changes
# ---------------------------------------------------------------------------

def traceMinimum(
    f: Callable[[npt.NDArray[np.float64], float], float],
    d2f_dxdt: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    d2f_dx2: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    x0: npt.ArrayLike,
    t0: float,
    tstop: float,
    dtstart: float,
    deltaX_target: float,
    dtabsMax: float = 20.0,
    dtfracMax: float = 0.25,
    dtmin: float = 1e-3,
    deltaX_tol: float = 1.2,
    minratio: float = 1e-2,
) -> _traceMinimum_rval:
    """
    Trace a temperature–dependent minimum ``x_min(T)`` of a scalar function
    ``f(x, T)``.

    Starting from ``(x0, t0)``, this routine integrates the implicit ODE for
    the minimum,
        d/dT [∂f/∂x] = 0  ⇒  H · dx/dT = -∂/∂T (∂f/∂x) ,
    where ``H = ∂²f/∂x²`` is the Hessian, while repeatedly re-minimizing
    along the way to stay on the true minimum.

    The integration stops when:
    - the stepsize in T falls below a minimum threshold, or
    - the Hessian becomes effectively singular / develops a negative mode
      according to ``minratio``.

    Parameters
    ----------
    f :
        Objective function ``f(x, T)`` to be minimized with respect to ``x``.
        Must accept a 1D NumPy array and a scalar temperature, and return a
        scalar float.
    d2f_dxdt :
        Function returning ``∂/∂T (∂f/∂x)`` as a 1D array of length ``Ndim``.
        Signature: ``d2f_dxdt(x, T) -> ndarray[(Ndim,)]``.
    d2f_dx2 :
        Function returning the Hessian matrix ``∂²f/∂x²`` evaluated at
        ``(x, T)``, as a 2D array with shape ``(Ndim, Ndim)``.
    x0 :
        Initial position of the minimum at ``T = t0``. Will be converted to a
        1D float array. Even for a single scalar field, this should be an
        array-like of length 1.
    t0 :
        Initial temperature.
    tstop :
        Target temperature. The trace proceeds from ``t0`` toward ``tstop``
        with initial stepsize ``dtstart`` (sign matters).
    dtstart :
        Initial guess for the T-stepsize. The effective absolute and minimal
        stepsizes are scaled from this value.
    deltaX_target :
        Target displacement in x between successive accepted steps. The
        algorithm adapts the stepsize in T to keep the estimated error near
        this scale.
    dtabsMax :
        Sets the maximum absolute T-step as
        ``max(|dtstart| * dtabsMax, |T| * dtfracMax)``.
    dtfracMax :
        See above; fractional upper bound on the step relative to |T|.
    dtmin :
        Minimal allowed |ΔT|, expressed **relative to |dtstart|**. If the step
        size drops below ``|dtstart| * dtmin``, the trace is terminated.
    deltaX_tol :
        Relative tolerance on the displacement in x. A step is accepted if the
        estimated error is below
        ``deltaX_tol * deltaX_target``.
    minratio :
        Eigenvalue-ratio criterion used to decide when the Hessian has
        effectively developed a zero / negative mode. Internally, it is
        rescaled by the ratio of smallest to largest eigenvalue at the
        starting point; smaller values make the criterion less strict.

    Returns
    -------
    result : TraceMinimumResult
        Named tuple with fields ``(X, T, dXdT, overX, overT)``:

        * ``X`` – array of minima along the path;
        * ``T`` – corresponding temperatures;
        * ``dXdT`` – derivative of the minimum with respect to T;
        * ``overX``, ``overT`` – location and temperature where the algorithm
          last saw a saddle / disappearance of the phase.

    Notes
    -----
    In the original CosmoTransitions implementation, finite-difference
    machinery to approximate derivatives from ``f`` itself lived directly in
    this function. In the modernized version, derivative construction is
    delegated to generic utilities (for example,
    :class:`helper_functions.gradientFunction` and
    :class:`helper_functions.hessianFunction`), and ``traceMinimum`` assumes
    that ``d2f_dxdt`` and ``d2f_dx2`` are provided explicitly.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    if x.ndim != 1:
        msg = f"x0 must be 1D; got shape {x.shape!r}"
        raise ValueError(msg)

    Ndim = x.size

    # Initial Hessian and eigenvalue ratio threshold
    M0 = np.asarray(d2f_dx2(x, t0), dtype=float)
    if M0.shape != (Ndim, Ndim):
        msg = f"d2f_dx2(x0, t0) must have shape ({Ndim}, {Ndim}); got {M0.shape}"
        raise ValueError(msg)

    eig0 = linalg.eigvalsh(M0)
    if np.all(eig0 == 0):
        raise ValueError("Initial Hessian is exactly singular at (x0, t0).")

    # Rescale minratio by the condition at the start point
    minratio = float(minratio) * float(
        np.min(np.abs(eig0)) / np.max(np.abs(eig0))
    )

    def dxmindt(x_now: npt.NDArray[np.float64], t_now: float):
        """
        Solve H · dx/dT = -∂/∂T (∂f/∂x).

        Returns
        -------
        dxdt, is_negative_mode
            dxdt is None if the system is judged singular; is_negative_mode
            is True if a negative/flat direction is detected via eigenvalues.
        """
        M = np.asarray(d2f_dx2(x_now, t_now), dtype=float)
        # crude singularity test; keeps legacy behaviour
        if np.abs(linalg.det(M)) < (1e-3 * np.max(np.abs(M))) ** Ndim:
            return None, False

        b = -np.asarray(d2f_dxdt(x_now, t_now), dtype=float).reshape(Ndim)
        eigs = linalg.eigvalsh(M)
        try:
            dxdt_local = linalg.solve(M, b, overwrite_a=False, overwrite_b=False)
            isneg = (eigs <= 0).any() or (
                np.min(eigs) / np.max(eigs) < minratio
            )
        except linalg.LinAlgError:
            dxdt_local = None
            isneg = False
        return dxdt_local, isneg

    # Minimization tolerance in x
    xeps = float(deltaX_target) * 1e-2

    def fmin(x_guess: npt.NDArray[np.float64], t_now: float) -> npt.NDArray[np.float64]:
        """Refine the minimum of f(x, t_now) starting from x_guess."""
        x_guess = np.asarray(x_guess, dtype=float).reshape(Ndim)

        res = optimize.fmin(
            f,
            x_guess,
            args=(t_now,),
            xtol=xeps,
            ftol=np.inf,
            disp=False,
        )
        return np.asarray(res, dtype=float).reshape(Ndim)

    # Scale dt-related parameters by |dtstart|
    tscale = abs(float(dtstart))
    if tscale == 0.0:
        raise ValueError("dtstart must be non-zero.")

    dtabsMax = float(dtabsMax) * tscale
    dtmin = float(dtmin) * tscale
    deltaX_tol_abs = float(deltaX_tol) * float(deltaX_target)

    # Initial state
    t = float(t0)
    dt = float(dtstart)
    dxdt, negeig = dxmindt(x, t)

    X_list = [x.copy()]
    T_list = [t]
    dXdT_list = [np.zeros_like(x) if dxdt is None else dxdt.copy()]
    overX = x.copy()
    overT = t

    log.debug("traceMinimum: start at T=%g, dtstart=%g", t, dt)

    # Main stepping loop
    while dxdt is not None:
        # Candidate next temperature and position predicted by ODE
        tnext = t + dt
        x_pred = x + dxdt * dt
        xnext = fmin(x_pred, tnext)
        dxdt_next, negeig = dxmindt(xnext, tnext)

        if dxdt_next is None or negeig:
            # We hit a saddle / boundary of the minimum basin. Back off.
            dt *= 0.5
            overX, overT = xnext.copy(), float(tnext)
        else:
            # Estimate error by comparing forward/backward predictions
            err1 = np.linalg.norm(x + dxdt * dt - xnext)
            err2 = np.linalg.norm(xnext - dxdt_next * dt - x)
            xerr = max(err1, err2)

            if xerr < deltaX_tol_abs:
                # Accept the step
                T_list.append(float(tnext))
                X_list.append(xnext.copy())
                dXdT_list.append(dxdt_next.copy())

                if overT is None:
                    # Only adapt dt if the last step was not problematic
                    dt *= float(deltaX_target) / (xerr + 1e-100)

                x, t, dxdt = xnext, float(tnext), dxdt_next
                overX, overT = x.copy(), float(t)
            else:
                # Too aggressive: try a smaller step.
                dt *= 0.5
                overX, overT = xnext.copy(), float(tnext)

        # --- dt sanity checks -------------------------------------------------
        if abs(dt) < dtmin:
            log.debug(
                "traceMinimum: stopping because |dt| < dtmin (|dt|=%g, dtmin=%g)",
                abs(dt),
                dtmin,
            )
            break

        if (dt > 0 and t >= tstop) or (dt < 0 and t <= tstop):
            # Force a last step exactly to tstop
            dt = tstop - t
            x = fmin(x + (dxdt if dxdt is not None else 0.0) * dt, tstop)
            dxdt, negeig = dxmindt(x, tstop)
            t = float(tstop)
            X_list[-1], T_list[-1], dXdT_list[-1] = x.copy(), t, (
                np.zeros_like(x) if dxdt is None else dxdt.copy()
            )
            break

        # Clamp overly large steps
        dtmax = max(abs(t) * float(dtfracMax), dtabsMax)
        if abs(dt) > dtmax:
            dt = np.sign(dt) * dtmax

    if overT is None:
        overX, overT = X_list[-1].copy(), float(T_list[-1])

    X_arr = np.asarray(X_list, dtype=float)
    T_arr = np.asarray(T_list, dtype=float)
    dXdT_arr = np.asarray(dXdT_list, dtype=float)

    return _traceMinimum_rval(
        X=X_arr,
        T=T_arr,
        dXdT=dXdT_arr,
        overX=overX,
        overT=float(overT),
    )


# ---------------------------------------------------------------------------
# Phase container
# ---------------------------------------------------------------------------

class Phase:
    """
    Represent a single temperature–dependent minimum (a "phase").

    A :class:`Phase` stores the traced minimum ``x_min(T)`` and its first
    derivative with respect to T, together with a spline representation used
    for interpolation. It also keeps track of second–order connections to
    neighbouring phases via ``low_trans`` / ``high_trans``.

    Parameters
    ----------
    key :
        Hashable identifier for the phase (often an integer).
    X :
        Array of minima along the path, shape ``(n_T, n_fields)``.
    T :
        Temperature array corresponding to ``X``, shape ``(n_T,)``.
    dXdT :
        Array of derivatives ``dx_min/dT`` along the path, same shape as ``X``.
    """

    def __init__(
        self,
        key,
        X: npt.NDArray[np.float64],
        T: npt.NDArray[np.float64],
        dXdT: npt.NDArray[np.float64],
    ) -> None:
        self.key = key

        # Sort by temperature to guard against occasional backwards steps.
        T = np.asarray(T, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        dXdT = np.asarray(dXdT, dtype=float)

        if X.shape != dXdT.shape:
            msg = (
                f"X and dXdT must have the same shape; "
                f"got {X.shape} and {dXdT.shape}"
            )
            raise ValueError(msg)

        if X.shape[0] != T.size:
            msg = (
                f"First dimension of X/dXdT must match T.size; "
                f"got X.shape={X.shape}, T.size={T.size}"
            )
            raise ValueError(msg)

        order = np.argsort(T)
        self.T = T[order]
        self.X = X[order]
        self.dXdT = dXdT[order]

        # Build a spline in T → X space.
        # splprep expects (n_dim, n_T) input.
        k = 3 if self.T.size > 3 else 1
        tck, _ = interpolate.splprep(self.X[order].T, u=self.T, s=0.0, k=k)
        self.tck = tck

        # Sets of neighbouring phases connected by (approx.) second–order
        # transitions.
        self.low_trans: set = set()
        self.high_trans: set = set()

    # ------------------------------------------------------------------ #
    # Spline interface
    # ------------------------------------------------------------------ #

    def valAt(self, T, deriv: int = 0) -> npt.NDArray[np.float64]:
        """
        Evaluate the phase minimum (or its T-derivatives) at a given temperature.

        Parameters
        ----------
        T :
            Scalar or array of temperatures at which to evaluate the phase.
        deriv :
            Order of the derivative with respect to T. If zero (default), this
            returns the minimum ``x_min(T)``. For cubic splines, derivatives up
            to order 3 are available. For linear splines, only the first
            derivative is defined.

        Returns
        -------
        ndarray
            Array with shape ``(n_T, n_fields)`` if ``T`` is array-like, or
            ``(n_fields,)`` if ``T`` is scalar.
        """
        if deriv < 0:
            raise ValueError("deriv must be non-negative.")
        T_arr = np.asanyarray(T)
        y = interpolate.splev(T_arr, self.tck, der=deriv)
        # splev returns a list-of-arrays for multi-D data → stack & transpose
        arr = np.asanyarray(y).T
        return arr

    # ------------------------------------------------------------------ #
    # Phase linkage (second-order transitions)
    # ------------------------------------------------------------------ #

    def addLinkFrom(self, other_phase: "Phase") -> None:
        """
        Register a link from ``other_phase`` to this phase.

        This method also attempts to detect second-order transitions by
        comparing the T-ranges of the two phases:

        - if this phase exists only at **lower** temperatures than
          ``other_phase``, we add ``other_phase.key`` to ``low_trans``;
        - if this phase exists only at **higher** temperatures, we add the
          reverse mapping in ``high_trans``.

        Parameters
        ----------
        other_phase :
            Phase instance to be linked to this one.
        """
        if np.min(self.T) >= np.max(other_phase.T):
            self.low_trans.add(other_phase.key)
            other_phase.high_trans.add(self.key)
        if np.max(self.T) <= np.min(other_phase.T):
            self.high_trans.add(other_phase.key)
            other_phase.low_trans.add(self.key)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        popts = np.get_printoptions()
        try:
            np.set_printoptions(formatter={"float": lambda x: f"{x:0.4g}"})
            if self.X.shape[0] > 1:
                Xstr = f"[{self.X[0]}, ..., {self.X[-1]}]"
            else:
                Xstr = f"[{self.X[0]}]"
            if self.T.size > 1:
                Tstr = f"[{self.T[0]:0.4g}, ..., {self.T[-1]:0.4g}]"
            else:
                Tstr = f"[{self.T[0]:0.4g}]"
            if self.dXdT.shape[0] > 1:
                dstr = f"[{self.dXdT[0]}, ..., {self.dXdT[-1]}]"
            else:
                dstr = f"[{self.dXdT[0]}]"
            return f"Phase(key={self.key!r}, X={Xstr}, T={Tstr}, dXdT={dstr})"
        finally:
            np.set_printoptions(**popts)


def traceMultiMin(
    f: Callable[[np.ndarray, float], float],
    d2f_dxdt: Callable[[np.ndarray, float], np.ndarray],
    d2f_dx2: Callable[[np.ndarray, float], np.ndarray],
    points: Sequence[Tuple[np.ndarray, float]],
    tLow: float,
    tHigh: float,
    deltaX_target: float,
    dtstart: float = 1e-3,
    tjump: float = 1e-3,
    forbidCrit: Optional[Callable[[np.ndarray], bool]] = None,
    single_trace_args: Optional[Mapping[str, object]] = None,
    local_min_args: Optional[Mapping[str, object]] = None,
) -> Dict[Hashable, Phase]:
    """
    Trace multiple temperature–dependent minima ``xmin(T)`` of ``f(x, T)``.

    This routine orchestrates a set of 1D traces (via :func:`traceMinimum`)
    starting from a collection of seed points. Whenever a minimum disappears,
    it looks for new nearby minima (using :func:`findApproxLocalMin`) and
    launches new traces from those points. Each continuous branch of minima is
    stored as a :class:`Phase` object.

    Parameters
    ----------
    f :
        Scalar function ``f(x, T)`` to be minimized. The first argument must
        be a 1D array (or an array of shape ``(N, ndim)`` if vectorized), and
        the second a scalar temperature.
    d2f_dxdt :
        Function returning the derivative of the gradient with respect to
        temperature, evaluated at ``(x, T)``. Shape should match ``x``.
    d2f_dx2 :
        Hessian matrix of ``f(x, T)`` with respect to ``x``, evaluated at
        ``(x, T)``. For an ``ndim``–dimensional field, this should have
        shape ``(ndim, ndim)``.
    points :
        Initial seeds for tracing, as a sequence of ``(x, T)`` pairs, where
        each ``x`` is a 1D array (even for a single field).
    tLow, tHigh :
        Minimum and maximum temperatures over which phases are traced.
    deltaX_target :
        Target step size in field space. Passed to :func:`traceMinimum` and
        used to define tolerances in the local minimizations.
    dtstart :
        Initial step size as a fraction of ``(tHigh - tLow)``. The actual
        step size used internally is ``dtstart * (tHigh - tLow)``.
    tjump :
        Temperature offset between the end of one trace and the starting
        point for searches of new phases. Also given as a fraction of
        ``(tHigh - tLow)``.
    forbidCrit :
        Optional predicate ``forbidCrit(x) -> bool``. If True, phases whose
        endpoints contain such points are discarded. This is useful, e.g., to
        forbid minima in unphysical regions of field space.
    single_trace_args :
        Optional dictionary of keyword arguments forwarded to
        :func:`traceMinimum` (e.g. ``dtabsMax``, ``dtfracMax``).
    local_min_args :
        Optional dictionary of keyword arguments forwarded to
        :func:`findApproxLocalMin`, *except* for the ``args`` argument, which
        is always set internally to pass the appropriate temperature.

    Returns
    -------
    phases :
        Dictionary mapping phase keys (ints 0, 1, 2, ...) to :class:`Phase`
        instances. Each phase encodes a continuous branch of minima
        ``X(T)`` between some temperatures within ``[tLow, tHigh]``.
    """
    if tHigh <= tLow:
        raise ValueError("traceMultiMin requires tHigh > tLow.")
    if deltaX_target <= 0.0:
        raise ValueError("deltaX_target must be positive.")

    single_trace_kwargs = dict(single_trace_args or {})
    # We manage 'args' ourselves when calling findApproxLocalMin
    local_min_kwargs = dict(local_min_args or {})
    local_min_kwargs.pop("args", None)

    # High-accuracy local minimizer around a seed point
    xeps = deltaX_target * 1e-2

    def fmin(x: np.ndarray, t: float) -> np.ndarray:
        """
        High-accuracy minimization of f(x, t) starting from x.

        We nudge the starting point slightly by xeps to help avoid
        pathologies in flat regions.
        """
        x = np.asarray(x, dtype=float)

        # optimize.fmin returns a 1D ndarray at the minimum
        xmin = optimize.fmin(
            f,
            x + xeps,
            args=(t,),
            xtol=xeps * 1e-3,
            ftol=np.inf,
            disp=False,
        )
        return np.asarray(xmin, dtype=float)

    # Rescale dtstart and tjump to absolute temperatures
    dt_scale = tHigh - tLow
    dtstart_abs = dtstart * dt_scale
    tjump_abs = tjump * dt_scale

    phases: Dict[Hashable, Phase] = {}
    # Each item: [T_current, dt_current, x_current, linked_from_key]
    next_points: list[list[object]] = []

    # Initialize the queue of seeds
    for x_seed, t_seed in points:
        x_seed = np.asarray(x_seed, dtype=float)
        next_points.append([float(t_seed), dtstart_abs, fmin(x_seed, t_seed), None])

    while next_points:
        t1, dt1, x1, linked_from = next_points.pop()
        t1 = float(t1)
        dt1 = float(dt1)
        x1 = fmin(x1, t1)  # Make sure we start exactly at a local minimum

        # 1. Check if this seed is outside the temperature bounds
        if t1 < tLow or (t1 == tLow and dt1 < 0.0):
            continue
        if t1 > tHigh or (t1 == tHigh and dt1 > 0.0):
            continue

        # 2. Optionally forbid seeds in certain regions of field space
        if forbidCrit is not None and forbidCrit(x1):
            continue

        # 3. Check redundancy with already traced phases
        is_redundant = False
        for key, phase in phases.items():
            # Skip phases that do not cover this temperature at all
            Tmin_phase = min(phase.T[0], phase.T[-1])
            Tmax_phase = max(phase.T[0], phase.T[-1])
            if t1 < Tmin_phase or t1 > Tmax_phase:
                continue

            # Evaluate the known phase at t1, re-minimize, and compare
            x_phase = fmin(phase.valAt(t1), t1)
            if np.linalg.norm(x_phase - x1) < 2.0 * deltaX_target:
                # Already covered by this phase; adjust linkage if needed
                if linked_from is not None and linked_from != key:
                    phase.addLinkFrom(phases[linked_from])
                is_redundant = True
                break

        if is_redundant:
            continue

        # 4. The seed is not redundant → trace this phase
        print(f"Tracing phase starting at x = {x1}, T = {t1}")
        phase_key: Hashable = len(phases)
        old_num_points = len(next_points)

        X_down = T_down = dXdT_down = None
        X_up = T_up = dXdT_up = None

        # ---- 4a. Trace downwards in temperature ----
        if t1 > tLow:
            print("  Tracing minimum down in T")
            down_trace = traceMinimum(
                f=f,
                d2f_dxdt=d2f_dxdt,
                d2f_dx2=d2f_dx2,
                x0=x1,
                t0=t1,
                tstop=tLow,
                dtstart=-abs(dt1),
                deltaX_target=deltaX_target,
                **single_trace_kwargs,
            )
            X_down = down_trace.X
            T_down = down_trace.T
            dXdT_down = down_trace.dXdT
            nX_down = down_trace.overX
            nT_down = down_trace.overT

            # Seed follow-up searches slightly below the last temperature
            t2 = nT_down - tjump_abs
            dt2 = 0.1 * tjump_abs
            x2 = fmin(nX_down, t2)
            next_points.append([t2, dt2, x2, phase_key])

            # Look for intermediate minima between the last point and the new seed
            if np.linalg.norm(X_down[-1] - x2) > deltaX_target:
                for point in findApproxLocalMin(
                    f,
                    X_down[-1],
                    x2,
                    args=(t2,),
                    **local_min_kwargs,
                ):
                    next_points.append([t2, dt2, fmin(point, t2), phase_key])

            # We want T to be increasing, so reverse the arrays
            X_down = X_down[::-1]
            T_down = T_down[::-1]
            dXdT_down = dXdT_down[::-1]

        # ---- 4b. Trace upwards in temperature ----
        if t1 < tHigh:
            print("  Tracing minimum up in T")
            up_trace = traceMinimum(
                f=f,
                d2f_dxdt=d2f_dxdt,
                d2f_dx2=d2f_dx2,
                x0=x1,
                t0=t1,
                tstop=tHigh,
                dtstart=+abs(dt1),
                deltaX_target=deltaX_target,
                **single_trace_kwargs,
            )
            X_up = up_trace.X
            T_up = up_trace.T
            dXdT_up = up_trace.dXdT
            nX_up = up_trace.overX
            nT_up = up_trace.overT

            # Seed follow-up searches slightly above the last temperature
            t2 = nT_up + tjump_abs
            dt2 = 0.1 * tjump_abs
            x2 = fmin(nX_up, t2)
            next_points.append([t2, dt2, x2, phase_key])

            # Look for intermediate minima between the last point and the new seed
            if np.linalg.norm(X_up[-1] - x2) > deltaX_target:
                for point in findApproxLocalMin(
                    f,
                    X_up[-1],
                    x2,
                    args=(t2,),
                    **local_min_kwargs,
                ):
                    next_points.append([t2, dt2, fmin(point, t2), phase_key])

        # ---- 4c. Join down/up pieces into a single branch ----
        if X_down is None:  # only up
            X, T, dXdT = X_up, T_up, dXdT_up
        elif X_up is None:  # only down
            X, T, dXdT = X_down, T_down, dXdT_down
        else:
            # Avoid duplicating the pivot point (t1)
            X = np.append(X_down, X_up[1:], axis=0)
            T = np.append(T_down, T_up[1:], axis=0)
            dXdT = np.append(dXdT_down, dXdT_up[1:], axis=0)

        # 5. Apply the forbidCrit filter to the branch endpoints, if requested
        if forbidCrit is not None and (forbidCrit(X[0]) or forbidCrit(X[-1])):
            # Forbidden phase: discard it and any descendants seeded from it
            next_points = next_points[:old_num_points]
        elif len(X) > 1:
            # Valid phase: construct Phase and link it
            new_phase = Phase(phase_key, X, T, dXdT)
            if linked_from is not None:
                new_phase.addLinkFrom(phases[linked_from])
            phases[phase_key] = new_phase
        else:
            # Only a single point: treat as dead end
            next_points = next_points[:old_num_points]

    return phases

def findApproxLocalMin(
    f: Callable[..., np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    args: Tuple[object, ...] = (),
    n: int = 100,
    edge: float = 0.05,
) -> np.ndarray:
    """
    Find approximate local minima along the straight line between two points.

    This is used as a safety net when jumping between phases: we want to make
    sure we do not skip an intermediate minimum/phase along the straight segment
    from x1 to x2 at fixed auxiliary parameters (typically T).

    The routine samples f(x) on a uniform grid along the line between x1 and x2,
    discards a small fraction of the points near the edges (controlled by
    ``edge``), and then finds local minima in the resulting 1D profile.

    Notes
    -----
    - The function first *attempts* to call f in a vectorized way,
      ``f(x_grid, *args)``, where ``x_grid`` has shape (n, ndim). If the output
      does not match this shape along axis 0 or if the call fails, it falls back
      to a safe point-by-point evaluation.
    - This makes the helper robust even if the user's potential is not fully
      vectorized in the field variables.

    Parameters
    ----------
    f : callable
        Function ``f(x, *args)`` to be probed along the line segment.
        It should return a scalar for each point x. A vectorized implementation
        (accepting an array of shape (n, ndim)) is recommended but not required.
    x1, x2 : array_like
        Endpoints of the line segment. They must have the same shape and
        represent field configurations in the same field space.
    args : tuple, optional
        Extra arguments to pass to ``f`` (e.g. the temperature T).
    n : int, optional
        Number of sample points along the line (including interior points only,
        after applying the ``edge`` trimming).
    edge : float, optional
        Fraction of the interval near each endpoint that is excluded from the
        search. If ``edge == 0``, the search may include points arbitrarily
        close to ``x1`` and ``x2``. If ``edge == 0.5``, the search degenerates
        to a single central point.

    Returns
    -------
    minima : ndarray, shape (k, ndim)
        Array of approximate minima along the line segment. Each row is a point
        in field space. If no internal minima are found, this will have shape
        (0, ndim).
    """
    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)

    if x1_arr.shape != x2_arr.shape:
        raise ValueError(
            "findApproxLocalMin: x1 and x2 must have the same shape. "
            f"Got x1.shape={x1_arr.shape}, x2.shape={x2_arr.shape}."
        )

    # Represent the endpoints as 1D vectors in field space
    x1_vec = x1_arr.reshape(1, -1)
    x2_vec = x2_arr.reshape(1, -1)
    ndim = x1_vec.shape[1]

    if not (0.0 <= edge < 0.5):
        raise ValueError(
            f"findApproxLocalMin: 'edge' must satisfy 0 <= edge < 0.5, got {edge}."
        )
    if n < 3:
        # With fewer than 3 points we cannot define a local minimum
        return np.empty((0, ndim), dtype=float)

    # Build grid along the line between x1 and x2, trimming the edges
    t_grid = np.linspace(edge, 1.0 - edge, n).reshape(n, 1)  # (n, 1)
    x_grid = x1_vec + t_grid * (x2_vec - x1_vec)             # (n, ndim)

    # --------------------------------------------------------------
    # Try vectorized evaluation first; if that fails or shapes do not
    # match, fall back to a point-by-point loop.
    # --------------------------------------------------------------
    def _evaluate_scalar_grid() -> np.ndarray:
        """Fallback: evaluate f row-by-row along x_grid."""
        values = []
        for k in range(x_grid.shape[0]):
            val = f(x_grid[k], *args)
            values.append(val)
        y_arr = np.asarray(values, dtype=float).ravel()
        if y_arr.shape[0] != x_grid.shape[0]:
            raise ValueError(
                "findApproxLocalMin: after fallback scalar evaluation, "
                "the number of function values does not match the number "
                f"of sample points: y.shape={y_arr.shape}, "
                f"x_grid points={x_grid.shape[0]}."
            )
        return y_arr

    try:
        y_raw = f(x_grid, *args)
        y_arr = np.asarray(y_raw, dtype=float)
        if y_arr.ndim == 0 or y_arr.shape[0] != x_grid.shape[0]:
            # Shape mismatch or scalar: use fallback
            y = _evaluate_scalar_grid()
        else:
            # Collapse any extra dimensions beyond the first into a single value
            if y_arr.ndim > 1:
                y = y_arr.reshape(y_arr.shape[0], -1)[:, 0]
            else:
                y = y_arr
    except Exception:
        # If the vectorized call fails for any reason, fall back
        y = _evaluate_scalar_grid()

    # Now y is a 1D array with y.shape[0] == x_grid.shape[0]
    if y.shape[0] < 3:
        return np.empty((0, ndim), dtype=float)

    # Internal points: indices 1..n-2
    # Local min: y[k] < y[k-1] and y[k] < y[k+1]
    is_min = (y[2:] > y[1:-1]) & (y[:-2] > y[1:-1])
    minima = x_grid[1:-1][is_min]

    # Ensure a consistent 2D shape, even if zero minima
    if minima.size == 0:
        minima = np.empty((0, ndim), dtype=float)

    return minima


def _removeRedundantPhase(
    phases: MutableMapping[Hashable, Phase],
    removed_phase: Phase,
    redundant_with_phase: Phase,
) -> None:
    """
    Internal helper to remove a redundant phase and reconnect its neighbors.

    Parameters
    ----------
    phases :
        Dictionary of phases (modified in-place).
    removed_phase :
        Phase instance to be removed.
    redundant_with_phase :
        Phase instance that survives and absorbs the connections of
        ``removed_phase``.
    """
    # Redirect low-temperature links
    for key in removed_phase.low_trans:
        if key == redundant_with_phase.key:
            continue
        p = phases[key]
        p.high_trans.discard(removed_phase.key)
        redundant_with_phase.addLinkFrom(p)

    # Redirect high-temperature links
    for key in removed_phase.high_trans:
        if key == redundant_with_phase.key:
            continue
        p = phases[key]
        p.low_trans.discard(removed_phase.key)
        redundant_with_phase.addLinkFrom(p)

    # Finally, drop the redundant phase from the dictionary
    del phases[removed_phase.key]

def removeRedundantPhases(
    f: Callable[[np.ndarray, float], float],
    phases: MutableMapping[Hashable, Phase],
    xeps: float = 1e-5,
    diftol: float = 1e-2,
) -> None:
    """
    Remove redundant phases from a dictionary produced by :func:`traceMultiMin`.

    Although :func:`traceMultiMin` attempts to trace each phase only once,
    there are situations where the same physical phase is represented by
    two different :class:`Phase` objects. This function merges such
    duplicates, modifying ``phases`` in-place.

    Two phases are considered redundant if they coincide (within ``diftol``)
    at both ends of their overlapping temperature range.

    Parameters
    ----------
    f :
        The same function ``f(x, T)`` that was passed to
        :func:`traceMultiMin`. It is used here to polish minima with a
        local minimization.
    phases :
        Dictionary of :class:`Phase` objects to be cleaned. Modified in-place.
    xeps :
        Tolerance in the local minimization used to refine minima for
        comparison.
    diftol :
        Maximum separation in field space for two minima to be considered
        coincident.

    Returns
    -------
    None
        The input dictionary ``phases`` is updated in-place.

    Notes
    -----
    - If two phases are merged, the new phase key is a string combining
      the original keys (e.g. ``"1_3"``).
    - This function favors clarity over raw speed. It is unlikely to be a
      bottleneck compared to tunneling and phase tracing.
    """
    def fmin(x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xmin = optimize.fmin(
            f,
            x,
            args=(t,),
            xtol=xeps,
            ftol=np.inf,
            disp=False,
        )
        return np.asarray(xmin, dtype=float)

    has_redundant_phase = True

    while has_redundant_phase:
        has_redundant_phase = False
        keys = list(phases.keys())

        for i in keys:
            if i not in phases:
                continue
            for j in keys:
                if j == i or j not in phases:
                    continue

                phase1 = phases[i]
                phase2 = phases[j]

                # Overlapping temperature window
                tmax = min(phase1.T[-1], phase2.T[-1])
                tmin = max(phase1.T[0], phase2.T[0])
                if tmin > tmax:
                    # No overlap in temperature
                    continue

                # Compare at tmax
                if tmax == phase1.T[-1]:
                    x1 = phase1.X[-1]
                else:
                    x1 = fmin(phase1.valAt(tmax), tmax)

                if tmax == phase2.T[-1]:
                    x2 = phase2.X[-1]
                else:
                    x2 = fmin(phase2.valAt(tmax), tmax)

                dif = np.linalg.norm(x1 - x2)
                same_at_tmax = dif < diftol

                # Compare at tmin
                if tmin == phase1.T[0]:
                    x1 = phase1.X[0]
                else:
                    x1 = fmin(phase1.valAt(tmin), tmin)

                if tmin == phase2.T[0]:
                    x2 = phase2.X[0]
                else:
                    x2 = fmin(phase2.valAt(tmin), tmin)

                dif = np.linalg.norm(x1 - x2)
                same_at_tmin = dif < diftol

                if same_at_tmin and same_at_tmax:
                    # Phases are redundant → merge them
                    has_redundant_phase = True

                    p_low = phase1 if phase1.T[0] < phase2.T[0] else phase2
                    p_high = phase1 if phase1.T[-1] > phase2.T[-1] else phase2

                    if p_low is p_high:
                        # Completely overlapping: keep one, drop the other
                        p_reject = phase1 if p_low is phase2 else phase2
                        _removeRedundantPhase(phases, p_reject, p_low)
                    else:
                        # Stitch together low- and high-T segments
                        mask_low = p_low.T <= tmax
                        T_low = p_low.T[mask_low]
                        X_low = p_low.X[mask_low]
                        dXdT_low = p_low.dXdT[mask_low]

                        mask_high = p_high.T > tmax
                        T_high = p_high.T[mask_high]
                        X_high = p_high.X[mask_high]
                        dXdT_high = p_high.dXdT[mask_high]

                        T = np.append(T_low, T_high, axis=0)
                        X = np.append(X_low, X_high, axis=0)
                        dXdT = np.append(dXdT_low, dXdT_high, axis=0)

                        new_key: Hashable = f"{p_low.key}_{p_high.key}"
                        new_phase = Phase(new_key, X, T, dXdT)
                        phases[new_key] = new_phase

                        _removeRedundantPhase(phases, p_low, new_phase)
                        _removeRedundantPhase(phases, p_high, new_phase)

                    break

                elif same_at_tmin or same_at_tmax:
                    # More subtle case: overlap only at one end ⇒ not handled
                    raise NotImplementedError(
                        "Two phases coincide at one end of their overlapping "
                        "temperature range but not the other. Implementing "
                        "a robust splitting/merging strategy for this case "
                        "is non-trivial, so an explicit error is raised."
                    )

            if has_redundant_phase:
                # Restart the search after any modification of 'phases'
                break

def getStartPhase(
    phases: Mapping[Hashable, Phase],
    V: Optional[Callable[[np.ndarray, float], float]] = None,
) -> Hashable:
    """
    Return the key of the high-temperature starting phase.

    The starting phase is defined as the one that extends to the highest
    temperature ``T_max``. If more than one phase shares the same
    ``T_max``, and a potential ``V(x, T)`` is provided, the energetically
    favored phase at that temperature is selected.

    Parameters
    ----------
    phases :
        Dictionary of :class:`Phase` instances (e.g. the output of
        :func:`traceMultiMin`).
    V :
        Optional potential function ``V(x, T)``. Only required if multiple
        phases share the same maximum temperature. It must accept a field
        configuration (1D array) and a scalar temperature.

    Returns
    -------
    key :
        Key of the selected starting phase. This key is guaranteed to be
        present in ``phases``.

    Raises
    ------
    ValueError
        If ``phases`` is empty.
    """
    if not phases:
        raise ValueError("getStartPhase requires a non-empty 'phases' mapping.")

    start_phase_candidates: list[Hashable] = []
    Tmax: Optional[float] = None

    # 1. Find all phases with the largest Tmax
    for key, phase in phases.items():
        phase_Tmax = phase.T[-1]
        if Tmax is None or phase_Tmax > Tmax:
            Tmax = phase_Tmax
            start_phase_candidates = [key]
        elif phase_Tmax == Tmax:
            start_phase_candidates.append(key)

    # 2. If unique or no potential is provided, just return the candidate
    if len(start_phase_candidates) == 1 or V is None:
        return start_phase_candidates[0]

    # 3. Resolve degeneracy using the potential at high temperature
    assert Tmax is not None  # for type checkers
    Vmin: Optional[float] = None
    start_phase: Optional[Hashable] = None

    for key in start_phase_candidates:
        phase = phases[key]
        V_val = float(V(phase.X[-1], phase.T[-1]))
        if Vmin is None or V_val < Vmin:
            Vmin = V_val
            start_phase = key

    assert start_phase is not None and start_phase in phases
    return start_phase

# ---------------------------------------------------------------------------
# Block B – tunneling core: bounce solving and nucleation temperature
# ---------------------------------------------------------------------------


def _solve_bounce(
    x_high: np.ndarray,
    x_low: np.ndarray,
    V_fixed: Callable[[np.ndarray], float],
    dV_fixed: Callable[[np.ndarray], np.ndarray],
    T: float,
    fullTunneling_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Any], float, int]:
    """
    Unified backend to compute a bounce between two minima at fixed temperature T.

    Parameters
    ----------
    x_high : array_like
        Field value(s) of the metastable (false) vacuum.
    x_low : array_like
        Field value(s) of the more stable (true) vacuum.
    V_fixed : callable
        Potential at fixed T, V_fi~xed(x) = V(x, T).
    dV_fixed : callable
        Gradient at fixed T, dV_fixed(x) = dV(x, T).
    T : float
        Temperature (passed only for bookkeeping / callbacks).
    fullTunneling_params : dict, optional
        Extra keyword arguments forwarded to the underlying backend
        (pathDeformation if available).

    Returns
    -------
    instanton : object or None
        Backend-dependent object describing the instanton, or None if the
        transition is effectively second order or absent.
    action : float
        Euclidean action S. May be 0.0 (no barrier) or +inf (stable).
    trantype : int
        1 for first-order (bounce found), 0 otherwise.
    """
    x_high = np.atleast_1d(np.asarray(x_high, dtype=float))
    x_low = np.atleast_1d(np.asarray(x_low, dtype=float))

    if x_high.shape != x_low.shape:
        raise ValueError(
            "_solve_bounce: x_high and x_low must have the same shape; "
            f"got {x_high.shape} vs {x_low.shape}."
        )

    ndim = x_high.size
    if fullTunneling_params is None:
        fullTunneling_params = {}

    # ------------------------------------------------------------------
    # 1) Preferred backend: pathDeformation (multi-field capable)
    # ------------------------------------------------------------------
    try:
        from . import pathDeformation as _pd  # type: ignore[import]
        has_path_deformation = True
    except Exception:
        has_path_deformation = False

    if has_path_deformation:
        # Keep legacy behaviour: pathDeformation.fullTunneling([x_low, x_high], ...)
        try:
            tobj = _pd.fullTunneling(
                [x_low, x_high],
                V_fixed,
                dV_fixed,
                callback_data=T,
                **fullTunneling_params,
            )
            action = float(tobj.action)
            return tobj, action, 1
        except Exception as err:
            # Interpret tunneling-related PotentialError; re-raise others.
            try:
                from . import tunneling1D as _t1d  # type: ignore[import]
                PotentialError = _t1d.PotentialError
            except Exception:
                PotentialError = Exception  # best-effort fallback

            if isinstance(err, PotentialError):
                reason = err.args[1] if len(err.args) > 1 else None
                if reason == "no barrier":
                    return None, 0.0, 0
                if reason == "stable, not metastable":
                    return None, np.inf, 0
            # Not a recognised PotentialError → propagate
            raise

    # ------------------------------------------------------------------
    # 2) Fallback backend: tunneling1D.SingleFieldInstanton (1D only)
    # ------------------------------------------------------------------
    from . import tunneling1D as _t1d  # type: ignore[import]

    if ndim != 1:
        raise RuntimeError(
            "_solve_bounce: pathDeformation is unavailable but field dimension "
            f"is {ndim} > 1. The tunneling1D backend only supports single-field "
            "potentials."
        )

    phi_meta = float(x_high[0])
    phi_abs = float(x_low[0])

    def V1D(phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """1D wrapper around V_fixed(x) with x ∈ R."""
        phi_arr = np.asarray(phi, dtype=float)
        if phi_arr.ndim == 0:
            return float(V_fixed(np.array([phi_arr], dtype=float)))
        flat = phi_arr.ravel()
        vals = [V_fixed(np.array([v], dtype=float)) for v in flat]
        return np.asarray(vals, dtype=float).reshape(phi_arr.shape)

    try:
        inst = _t1d.SingleFieldInstanton(
            phi_absMin=phi_abs,
            phi_metaMin=phi_meta,
            V=V1D,
        )
        profile = inst.findProfile()
        action = inst.findAction(profile)
        return inst, float(action), 1
    except _t1d.PotentialError as err:
        reason = err.args[1] if len(err.args) > 1 else None
        if reason == "no barrier":
            return None, 0.0, 0
        if reason == "stable, not metastable":
            return None, np.inf, 0
        # Unexpected tunneling error → propagate
        raise

def _tunnelFromPhaseAtT(
    T: Union[float, np.ndarray],
    phases: Mapping[Hashable, "Phase"],
    start_phase: "Phase",
    V: Callable[[np.ndarray, float], float],
    dV: Callable[[np.ndarray, float], np.ndarray],
    phitol: float,
    overlapAngle: float,
    nuclCriterion: Callable[[float, float], float],
    fullTunneling_params: Optional[Mapping[str, Any]],
    verbose: bool,
    outdict: MutableMapping[float, Dict[str, Any]],
) -> float:
    """
    Internal helper: evaluate the nucleation criterion at temperature ``T``.

    This function:
    1. Finds all energetically allowed target minima (phases with lower
       free energy than ``start_phase`` at ``T``).
    2. Optionally prunes targets that point in nearly the same direction
       in field space (``overlapAngle``).
    3. For each remaining candidate, attempts to solve the bounce using
       :func:`_solve_bounce`.
    4. Caches the best (lowest action) transition in ``outdict[T]``.
    5. Returns ``nuclCriterion(S_min, T)`` for the lowest-action solution.

    It is designed to be used as a 1D scalar function in root-finding
    (`scipy.optimize.brentq`) and minimization (`scipy.optimize.fmin`).

    Parameters
    ----------
    T : float or array_like
        Temperature at which the tunneling is evaluated. If an array is
        passed (as `optimize.fmin` does), only the first element is used.
    phases : mapping
        Mapping from phase key to :class:`Phase`. Typically the output of
        :func:`traceMultiMin`.
    start_phase : Phase
        Metastable phase from which tunneling is attempted.
    V, dV : callable
        Potential and its gradient, with signatures
        ``V(x, T) -> float`` and ``dV(x, T) -> ndarray``.
    phitol : float
        Tolerance used in the 1D minimizations for locating minima at
        fixed T.
    overlapAngle : float
        Maximum angle (in degrees) allowed between directions in field
        space towards different target phases. If two targets are closer
        than this angle, only the closer one (in field space) is kept.
        Set to zero to keep all targets.
    nuclCriterion : callable
        Function of the action and temperature, ``nuclCriterion(S, T)``.
        Should return 0 at the desired nucleation condition, >0 for
        insufficient tunneling, <0 for too fast tunneling.
    fullTunneling_params : mapping or None
        Extra keyword arguments passed down to :func:`pathDeformation.fullTunneling`.
    verbose : bool
        If True, prints information about each tunneling attempt.
    outdict : dict-like
        Mutable mapping used as a cache. For each temperature ``T``,
        stores the best transition dictionary under key ``T``. The
        dictionary has keys ``'Tnuc'``, ``'low_vev'``, ``'high_vev'``,
        ``'low_phase'``, ``'high_phase'``, ``'action'``, ``'instanton'``,
        and ``'trantype'``.

    Returns
    -------
    float
        The value of ``nuclCriterion(S_min, T)`` for the lowest-action
        tunneling solution at that temperature. If no acceptable target
        phases exist or no bounce is found, this will typically be
        positive (e.g. ``+∞`` in terms of action).
    """
    T_arr = np.asarray(T, dtype=float)
    T_val = float(T_arr.ravel()[0])

    # Cache check
    if T_val in outdict:
        best = outdict[T_val]
        return float(nuclCriterion(float(best["action"]), T_val))

    # Local 1D minimizer at fixed T
    def fmin_min(x0: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float)
        xmin = optimize.fmin(
            V,
            x0,
            args=(T_val,),
            xtol=phitol,
            ftol=np.inf,
            disp=False,
        )
        return np.asarray(xmin, dtype=float)

    # High-T (metastable) minimum
    x0_guess = start_phase.valAt(T_val)
    x0 = fmin_min(x0_guess)
    V0 = float(V(x0, T_val))

    # Collect candidate low-T minima (target phases)
    tunnel_list: list[Dict[str, Any]] = []
    for key, phase in phases.items():
        if key == start_phase.key:
            continue
        # Only consider phases that exist at this T
        if phase.T[0] > T_val or phase.T[-1] < T_val:
            continue
        x1_guess = phase.valAt(T_val)
        x1 = fmin_min(x1_guess)
        V1 = float(V(x1, T_val))
        # Require target phase to be energetically lower
        if V1 >= V0:
            continue
        tdict: Dict[str, Any] = dict(
            low_vev=x1,
            high_vev=x0,
            Tnuc=T_val,
            low_phase=phase.key,
            high_phase=start_phase.key,
        )
        tunnel_list.append(tdict)

    # Optional pruning of nearly overlapping target directions
    if overlapAngle > 0.0 and len(tunnel_list) > 1:
        cos_overlap = float(np.cos(np.deg2rad(overlapAngle)))
        excluded_indices: list[int] = []
        for i in range(1, len(tunnel_list)):
            for j in range(i):
                xi = np.asarray(tunnel_list[i]["low_vev"], dtype=float)
                xj = np.asarray(tunnel_list[j]["low_vev"], dtype=float)
                dx_i = xi - x0
                dx_j = xj - x0
                xi2 = float(np.dot(dx_i, dx_i))
                xj2 = float(np.dot(dx_j, dx_j))
                if xi2 == 0.0 or xj2 == 0.0:
                    # Degenerate direction; skip overlap test.
                    continue
                dotij = float(np.dot(dx_j, dx_i))
                if dotij >= (xi2 * xj2) ** 0.5 * cos_overlap:
                    # Directions are too aligned; keep the shorter vector.
                    excluded_indices.append(i if xi2 > xj2 else j)
        for idx in sorted(set(excluded_indices), reverse=True):
            del tunnel_list[idx]

    # Wrap V and dV for fixed T
    def V_fixed(x: np.ndarray, T=T_val, V=V) -> float:
        return float(V(np.asarray(x, dtype=float), T))

    def dV_fixed(x: np.ndarray, T=T_val, dV=dV) -> np.ndarray:
        return np.asarray(dV(np.asarray(x, dtype=float), T), dtype=float)

    # Try tunneling for each candidate and keep the one with lowest action
    lowest_action = float("inf")
    lowest_tdict: Dict[str, Any] = dict(action=float("inf"), trantype=0)

    for tdict in tunnel_list:
        x1 = tdict["low_vev"]
        if verbose:
            print(
                "Tunneling from phase %s to phase %s at T=%0.7g"
                % (tdict["high_phase"], tdict["low_phase"], T_val)
            )
            print("  high_vev =", tdict["high_vev"])
            print("  low_vev  =", tdict["low_vev"])

        instanton, action, trantype = _solve_bounce(
            x_high=x0,
            x_low=x1,
            V_fixed=V_fixed,
            dV_fixed=dV_fixed,
            T=T_val,
            fullTunneling_params=fullTunneling_params,
        )

        tdict["instanton"] = instanton
        tdict["action"] = action
        tdict["trantype"] = trantype

        if action <= lowest_action:
            lowest_action = action
            lowest_tdict = tdict

    # Cache result (even if no acceptable tunneling was found)
    outdict[T_val] = lowest_tdict
    return float(nuclCriterion(lowest_action, T_val))


def _potentialDiffForPhase(
    T: float,
    start_phase: "Phase",
    other_phases: Sequence["Phase"],
    V: Callable[[np.ndarray, float], float],
) -> float:
    """
    Maximum free-energy difference between `start_phase` and other phases.

    Parameters
    ----------
    T : float
        Temperature at which the comparison is made.
    start_phase : Phase
        Reference phase.
    other_phases : sequence of Phase
        Other phases to compare against.
    V : callable
        Potential function ``V(x, T) -> float``.

    Returns
    -------
    float
        The minimum value of ``V(other) - V(start_phase)`` over all
        `other_phases`. Hence:
        - If the return value is positive, `start_phase` is energetically
          preferred at that T (locally stable).
        - If the return value is negative, some other phase is preferred
          (start_phase unstable).
    """
    T_val = float(T)
    x0 = np.asarray(start_phase.valAt(T_val), dtype=float)
    V0 = float(V(x0, T_val))

    delta_V = float("inf")
    for phase in other_phases:
        x1 = np.asarray(phase.valAt(T_val), dtype=float)
        V1 = float(V(x1, T_val))
        diff = V1 - V0
        if diff < delta_V:
            delta_V = diff
    return delta_V


def _maxTCritForPhase(
    phases: Mapping[Hashable, "Phase"],
    start_phase: "Phase",
    V: Callable[[np.ndarray, float], float],
    Ttol: float,
) -> float:
    """
    Find the maximum temperature at which `start_phase` is degenerate with
    one of the other phases.

    In practice this finds a temperature `Tcrit` where `start_phase` and
    some other phase have the same free energy (within the given tolerance).

    Parameters
    ----------
    phases : mapping
        All phases returned by :func:`traceMultiMin`.
    start_phase : Phase
        Phase for which we want the maximum critical temperature.
    V : callable
        Potential function ``V(x, T) -> float``.
    Ttol : float
        Tolerance passed to `scipy.optimize.brentq`.

    Returns
    -------
    float
        Maximum critical temperature at which `start_phase` is degenerate
        with another phase. If there are no other phases, returns the
        lowest temperature of `start_phase`.
    """
    other_phases = [phase for phase in phases.values() if phase.key != start_phase.key]
    if not other_phases:
        # No other phases: nothing to be degenerate with.
        return float(start_phase.T[0])

    Tmin = min(phase.T[0] for phase in other_phases)
    Tmax = max(phase.T[-1] for phase in other_phases)
    Tmin = max(Tmin, start_phase.T[0])
    Tmax = min(Tmax, start_phase.T[-1])
    Tmin = float(Tmin)
    Tmax = float(Tmax)

    DV_Tmin = _potentialDiffForPhase(Tmin, start_phase, other_phases, V)
    DV_Tmax = _potentialDiffForPhase(Tmax, start_phase, other_phases, V)

    if DV_Tmin >= 0.0:
        # start_phase is stable at Tmin.
        return Tmin
    if DV_Tmax <= 0.0:
        # start_phase is already unstable at Tmax.
        return Tmax

    root = optimize.brentq(
        _potentialDiffForPhase,
        Tmin,
        Tmax,
        args=(start_phase, other_phases, V),
        xtol=Ttol,
        maxiter=200,
        disp=False,
    )
    return float(root)


def tunnelFromPhase(
    phases: Mapping[Hashable, "Phase"],
    start_phase: "Phase",
    V: Callable[[np.ndarray, float], float],
    dV: Callable[[np.ndarray, float], np.ndarray],
    Tmax: float,
    Ttol: float = 1e-3,
    maxiter: int = 100,
    phitol: float = 1e-8,
    overlapAngle: float = 45.0,
    nuclCriterion: Callable[[float, float], float] = lambda S, T: S / (T + 1e-100) - 140.0,
    verbose: bool = True,
    fullTunneling_params: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find the instanton and nucleation temperature for tunneling from a
    metastable phase.

    Parameters
    ----------
    phases : mapping
        Output from :func:`traceMultiMin`. Keys are arbitrary hashables,
        values are :class:`Phase` objects.
    start_phase : Phase
        Metastable phase from which tunneling occurs.
    V, dV : callable
        Potential and gradient, with signatures
        ``V(x, T) -> float`` and ``dV(x, T) -> ndarray``.
    Tmax : float
        Highest temperature at which to search for a nucleation solution.
        The effective upper bound will be min(`Tmax`, highest T among all
        phases).
    Ttol : float, optional
        Absolute tolerance on T for the root-finding step that locates
        the nucleation temperature.
    maxiter : int, optional
        Maximum number of function evaluations in the root find / minimization.
    phitol : float, optional
        Tolerance used in the local minimizations for locating minima at
        fixed T.
    overlapAngle : float, optional
        Angle (in degrees) used to prune nearly overlapping target phases
        (see `_tunnelFromPhaseAtT`). Set to zero to disable pruning.
    nuclCriterion : callable, optional
        Nucleation criterion function. It should satisfy:

        * ``nuclCriterion(S, T) > 0`` for too small tunneling rate,
        * ``nuclCriterion(S, T) < 0`` for too large tunneling rate,
        * ``nuclCriterion(S, T) = 0`` at the desired nucleation condition.

        The default corresponds to the classic condition ``S(T)/T ≈ 140``.
    verbose : bool, optional
        If True, prints information about each tunneling attempt.
    fullTunneling_params : mapping or None, optional
        Extra keyword arguments forwarded to
        :func:`pathDeformation.fullTunneling`.

    Returns
    -------
    dict or None
        A dictionary describing the tunneling solution at the nucleation
        temperature, or None if no tunneling solution is found. For a
        first-order transition, the dictionary contains:

        - ``'Tnuc'``: nucleation temperature.
        - ``'low_vev'``, ``'high_vev'``: field values of low-T (true)
          and high-T (false) minima.
        - ``'low_phase'``, ``'high_phase'``: keys of the corresponding
          phases.
        - ``'action'``: Euclidean action of the instanton.
        - ``'instanton'``: object returned by
          :func:`pathDeformation.fullTunneling`.
        - ``'trantype'``: 1 for first-order transitions.

        For transitions that turn out to be second-order or for which no
        suitable bounce is found, returns None.
    """
    # Copy tunneling params to avoid mutating caller's dict
    params: Dict[str, Any] = {}
    if fullTunneling_params is not None:
        params.update(fullTunneling_params)

    outdict: Dict[float, Dict[str, Any]] = {}
    args = (
        phases,
        start_phase,
        V,
        dV,
        phitol,
        overlapAngle,
        nuclCriterion,
        params,
        verbose,
        outdict,
    )

    Tmin = float(start_phase.T[0])
    T_highest_other = Tmin
    for phase in phases.values():
        T_highest_other = max(T_highest_other, float(phase.T[-1]))

    Tmax_eff = min(float(Tmax), T_highest_other)
    if Tmax_eff < Tmin:
        raise ValueError(
            f"Tmax ({Tmax_eff}) is smaller than Tmin ({Tmin}); "
            "cannot search for tunneling in this range."
        )

    # First attempt: directly bracket and find a root of nuclCriterion(S(T), T)
    try:
        Tnuc = float(
            optimize.brentq(_tunnelFromPhaseAtT,Tmin, Tmax_eff, args=args, xtol=Ttol,
                maxiter=maxiter, disp=False,) )
    except ValueError as err:
        # Only handle the "same sign" case; re-raise everything else.
        if str(err) != "f(a) and f(b) must have different signs":
            raise

        # Ensure endpoints have been evaluated and cached.
        _tunnelFromPhaseAtT(Tmax_eff, *args)
        _tunnelFromPhaseAtT(Tmin, *args)

        # If even at Tmax the nucleation criterion is > 0, tunneling is too
        # suppressed by Tmax; check whether there is any chance at low T.
        if nuclCriterion(outdict[Tmax_eff]["action"], Tmax_eff) > 0.0:
            if nuclCriterion(outdict[Tmin]["action"], Tmin) < 0.0:
                # Tunneling *may* be possible somewhere in (Tmin, Tmax_eff).
                # Narrow the search to the region where the false vacuum is
                # at least metastable (up to the last critical temperature).
                Tmax_crit = _maxTCritForPhase(phases, start_phase, V, Ttol)

                def abort_fmin(
                    T_arr: np.ndarray,
                    outdict: Dict[float, Dict[str, Any]] = outdict,
                    nc: Callable[[float, float], float] = nuclCriterion,
                ) -> None:
                    T_val = float(np.asarray(T_arr, dtype=float).ravel()[0])
                    if T_val in outdict and nc(outdict[T_val]["action"], T_val) <= 0.0:
                        # As soon as we cross nuclCriterion <= 0, stop the minimization
                        raise StopIteration(T_val)

                try:
                    Tmin_opt = float(
                        optimize.fmin(
                            _tunnelFromPhaseAtT,
                            0.5 * (Tmin + Tmax_crit),
                            args=args,
                            xtol=Ttol * 10.0,
                            ftol=1.0,
                            maxiter=maxiter,
                            disp=0,
                            callback=abort_fmin,
                        )[0]
                    )
                except StopIteration as stop_err:
                    Tmin_opt = float(stop_err.args[0])

                # Check if at Tmin_opt the criterion is still too positive;
                # if yes, no tunneling solution.
                if nuclCriterion(outdict[Tmin_opt]["action"], Tmin_opt) > 0.0:
                    return None

                # Final bracketing between Tmin_opt and Tmax_crit
                Tnuc = float(
                    optimize.brentq( _tunnelFromPhaseAtT, Tmin_opt, Tmax_crit, args=args, xtol=Ttol, maxiter=maxiter,
                    disp=False, ))
            else:
                # Even at Tmin the nucleation criterion is not negative:
                # tunneling never becomes efficient in [Tmin, Tmax_eff].
                return None
        else:
            # Tunneling is already efficient at Tmax_eff: nucleation happens
            # "right away" at the upper end of the range.
            Tnuc = Tmax_eff

    # Extract the best transition at Tnuc
    if Tnuc not in outdict:
        # Ensure the cache is filled at Tnuc
        _tunnelFromPhaseAtT(Tnuc, *args)

    rdict = outdict[Tnuc]
    return rdict if rdict.get("trantype", 0) > 0 else None


# ---------------------------------------------------------------------------
# Block C – Transition history: from phase structure to full thermal history
# ---------------------------------------------------------------------------

def secondOrderTrans(high_phase, low_phase, Tstr: str = "Tnuc") -> Dict[str, Any]:
    """
    Assemble a dictionary describing a **second-order** phase transition.

    This is a lightweight helper used by :func:`findAllTransitions` and
    :func:`findCriticalTemperatures` when the transition proceeds without a
    tunneling barrier (no bounce / instanton).

    Parameters
    ----------
    high_phase : Phase
        Phase from which the system transitions as temperature decreases
        (the high-T branch in the history).
    low_phase : Phase
        Phase to which the system transitions as temperature decreases
        (the low-T branch).
    Tstr : str, optional
        Key name under which the characteristic temperature is stored in
        the returned dictionary. For nucleation histories this is usually
        ``"Tnuc"``, while for purely critical-temperature scans it is
        convenient to use ``"Tcrit"``.

    Returns
    -------
    dict
        A transition dictionary with the following keys:

        - ``Tstr`` (e.g. ``"Tnuc"`` or ``"Tcrit"``) :
          characteristic temperature, here taken as the simple midpoint

          .. math::
              T = \\tfrac{1}{2} (T^{\\text{high}}_{\\max} + T^{\\text{low}}_{\\max}).

        - ``low_vev``, ``high_vev`` :
          VEVs of the fields at the transition. For genuinely second-order
          transitions these coincide, so we take the first point of
          ``high_phase.X`` as representative.

        - ``low_phase``, ``high_phase`` :
          Keys identifying the low-T and high-T phases, respectively.

        - ``action`` :
          Set to ``0.0`` for second-order transitions (no tunneling action).

        - ``instanton`` :
          Always ``None`` here (no bounce solution).

        - ``trantype`` :
          Integer flag ``2`` to indicate a second-order transition.
    """
    T_high = float(high_phase.T[0])
    T_low  = float(low_phase.T[-1])
    Tchar  = 0.5 * (T_high + T_low)

    rdict: Dict[str, Any] = {
        Tstr: Tchar,
        "low_vev": high_phase.X[0],
        "high_vev": high_phase.X[0],
        "low_phase": low_phase.key,
        "high_phase": high_phase.key,
        "action": 0.0,
        "instanton": None,
        "trantype": 2,
    }
    return rdict


def findAllTransitions(
    phases: Mapping[Hashable, "Phase"],
    V: Callable[[np.ndarray, float], float],
    dV: Callable[[np.ndarray, float], np.ndarray],
    tunnelFromPhase_args: Dict[str, Any] = {},
) -> List[Dict[str, Any]]:
    """
    Build the **full phase transition history** starting from the high-T phase.

    This function iteratively applies :func:`tunnelFromPhase` (Block B) and
    :func:`secondOrderTrans` (above) to reconstruct a single **cooling path**

    .. math::
        \\text{Phase}_{\\text{high}} \\to \\text{Phase}_1 \\to \\text{Phase}_2 \\to \\dots

    in order of decreasing temperature.

    At each step it tries, in order:

    1. A **first-order** transition out of the current phase via
       :func:`tunnelFromPhase`. If successful, the returned transition
       dictionary is appended with ``trantype = 1`` and the algorithm
       continues from the resulting low-T phase at the corresponding
       nucleation temperature ``Tnuc``.
    2. If no first-order path is found but the current phase has entries
       in ``phase.low_trans``, it picks the first such target that still
       exists in the phase dictionary and constructs a **second-order**
       transition via :func:`secondOrderTrans`.
    3. If neither route is available, the history terminates.

    Notes
    -----
    - Only *one* outgoing transition per phase is kept. If there are
      multiple degenerate possibilities (e.g. due to symmetries), only
      the first compatible one is used.
    - The input mapping ``phases`` is **not** mutated; a shallow working
      copy is created internally and pruned as transitions are traversed.

    Parameters
    ----------
    phases : mapping
        Output from :func:`traceMultiMin`. Keys are arbitrary hashables,
        values are :class:`Phase` instances.
    V : callable
        Potential function with signature ``V(x, T) -> float``.
    dV : callable
        Gradient of the potential, ``dV(x, T) -> ndarray``.
    tunnelFromPhase_args : dict, optional
        Extra keyword arguments forwarded verbatim to
        :func:`tunnelFromPhase`, e.g.

        - ``Ttol``, ``maxiter``,
        - ``phitol``, ``overlapAngle``,
        - ``nuclCriterion``, ``verbose``,
        - ``fullTunneling_params``.

    Returns
    -------
    list of dict
        A list of transition dictionaries in **descending temperature**
        order along the chosen thermal history. Each entry is either:

        - a first-order transition dictionary as returned by
          :func:`tunnelFromPhase` (with ``trantype = 1``), or
        - a second-order transition dictionary from
          :func:`secondOrderTrans` (with ``trantype = 2``).
    """
    if not phases:
        return []

    # Working copy to avoid mutating the caller's mapping
    phases_work: Dict[Hashable, Phase] = dict(phases)

    # Identify the high-T starting phase
    start_key = getStartPhase(phases_work, V)
    start_phase = phases_work[start_key]
    Tmax = float(start_phase.T[-1])

    # Defensive copy of tunneling kwargs (do not mutate caller's dict)
    tf_args: Dict[str, Any] = dict(tunnelFromPhase_args)

    transitions: List[Dict[str, Any]] = []

    while start_phase is not None:
        # Once we've "left" a phase, we do not consider tunneling back into it.
        phases_work.pop(start_phase.key, None)

        # Try first-order tunneling from this phase
        trans = tunnelFromPhase(
            phases_work,
            start_phase,
            V,
            dV,
            Tmax,
            **tf_args,
        )

        if trans is None:
            # No first-order transition; try any second-order links encoded
            # in low_trans.
            low_targets = getattr(start_phase, "low_trans", []) or []

            if not low_targets:
                # No lower phases attached → end of the thermal history
                start_phase = None
                break

            # Pick the first candidate that is still present
            low_key: Optional[Hashable] = None
            for key in low_targets:
                if key in phases_work:
                    low_key = key
                    break

            if low_key is None:
                # All listed descendants have already been consumed in
                # previous steps; nothing left to transition to.
                start_phase = None
                break

            low_phase = phases_work[low_key]
            transitions.append(secondOrderTrans(start_phase, low_phase))
            start_phase = low_phase
            Tmax = float(low_phase.T[-1])
        else:
            # Genuine first-order transition found
            transitions.append(trans)
            low_key = trans["low_phase"]

            if low_key not in phases_work:
                # In well-formed histories this should not happen, but we
                # fail gracefully rather than raising.
                start_phase = None
                break

            start_phase = phases_work[low_key]
            Tmax = float(trans["Tnuc"])

    return transitions


def findCriticalTemperatures(
    phases: Mapping[Hashable, "Phase"],
    V: Callable[[np.ndarray, float], float],
    start_high: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find all temperatures ``Tcrit`` where any two phases are **degenerate**.

    For each ordered pair of phases (phase1 → phase2) with overlapping
    temperature ranges, this routine:

    1. Constructs the free-energy difference

       .. math::
           \\Delta V(T) = V_1(T) - V_2(T)
                         = V(\\phi_1(T), T) - V(\\phi_2(T), T),

       where :math:`\\phi_i(T) = \\text{phase}_i.\\text{valAt}(T)`.

    2. Checks the signs of :math:`\\Delta V` at the ends of the overlap
       interval. If there is a sign change, a critical temperature exists
       between them.
    3. Uses :func:`scipy.optimize.brentq` to locate the root ``Tcrit``.
    4. Assembles a transition dictionary with

       - ``Tcrit`` ,
       - VEVs ``high_vev``, ``low_vev``,
       - phase keys ``high_phase``, ``low_phase``,
       - ``trantype = 1`` (first-order) by construction.

    If two phases have **no overlap** in T but there is a second-order link
    (``phase2.key in phase1.low_trans``), a synthetic second-order
    transition is added via :func:`secondOrderTrans` with ``Tstr="Tcrit"``.

    Parameters
    ----------
    phases : mapping
        Output from :func:`traceMultiMin`. Keys are identifiers, values
        are :class:`Phase` instances.
    V : callable
        Potential function ``V(x, T) -> float``.
    start_high : bool, optional
        If False (default), return *all* critical temperatures between
        any pair of phases, sorted in decreasing ``Tcrit``. If True,
        one would like to keep only those that can be reached starting
        from the high-T phase; this mode is currently **not implemented**
        and will raise :class:`NotImplementedError`.

    Returns
    -------
    list of dict
        A list of transition dictionaries. For first-order degeneracies,
        entries have keys:

        - ``"Tcrit"``,
        - ``"high_vev"``, ``"low_vev"``,
        - ``"high_phase"``, ``"low_phase"``,
        - ``"trantype" = 1``.

        For purely second-order degeneracies (no overlap in T but a
        ``low_trans`` link), entries are generated by
        :func:`secondOrderTrans` with ``Tstr="Tcrit"`` and ``trantype = 2``.
    """
    transitions: List[Dict[str, Any]] = []

    for i_key, phase1 in phases.items():
        for j_key, phase2 in phases.items():
            if i_key == j_key:
                continue

            # Overlap in temperature range
            tmin = max(float(phase1.T[0]), float(phase2.T[0]))
            tmax = min(float(phase1.T[-1]), float(phase2.T[-1]))

            if tmin >= tmax:
                # No overlap → possible second-order transition if the
                # phase-graph encodes phase2 as a descendant of phase1.
                if getattr(phase1, "low_trans", None) and phase2.key in phase1.low_trans:
                    transitions.append(secondOrderTrans(phase1, phase2, "Tcrit"))
                continue

            def DV(T: float) -> float:
                # Free-energy difference between the two branches
                phi1 = phase1.valAt(T)
                phi2 = phase2.valAt(T)
                return float(V(phi1, T) - V(phi2, T))

            DV_tmin = DV(tmin)
            DV_tmax = DV(tmax)

            # For a meaningful Tcrit with phase1 as the *higher* branch,
            # we require:
            #   DV(tmin) >= 0   and   DV(tmax) <= 0,
            # so that there is a sign change somewhere in [tmin, tmax].
            if DV_tmin < 0.0:
                # phase1 already lower at the cold end: no crossing in the
                # direction phase1 → phase2
                continue
            if DV_tmax > 0.0:
                # phase1 still higher even at the hot end: no degeneracy
                continue

            Tcrit = float(optimize.brentq(DV, tmin, tmax, disp=False))

            tdict: Dict[str, Any] = {
                "Tcrit": Tcrit,
                "high_vev": phase1.valAt(Tcrit),
                "high_phase": phase1.key,
                "low_vev": phase2.valAt(Tcrit),
                "low_phase": phase2.key,
                "trantype": 1,
            }
            transitions.append(tdict)

    if not start_high:
        # Sort in decreasing Tcrit (hottest critical transitions first)
        return sorted(transitions, key=lambda x: float(x["Tcrit"]), reverse=True)

    # Placeholder for a future refinement that would prune to a single
    # high-T–reachable history; for now we keep the original behaviour.
    _ = getStartPhase(phases, V)
    raise NotImplementedError("start_high=True not yet supported")


def addCritTempsForFullTransitions(
    phases: Mapping[Hashable, "Phase"],
    crit_trans: Sequence[Dict[str, Any]],
    full_trans: Sequence[Dict[str, Any]],
) -> None:
    """
    Attach critical-temperature information to a list of **supercooled**
    transitions.

    This routine takes:

    - ``crit_trans`` : a list of degeneracy transitions (from
      :func:`findCriticalTemperatures`), typically labelled by ``"Tcrit"``.
    - ``full_trans`` : a list of nucleation transitions (from
      :func:`findAllTransitions` / :func:`tunnelFromPhase`), typically
      labelled by ``"Tnuc"``.

    For each element ``tdict`` in ``full_trans`` it tries to identify the
    **corresponding** critical-temperature transition:

    - It builds, for each phase, a list of "parents" in the critical-
      temperature graph, by walking from **low T to high T** along the
      high→low edges in ``crit_trans``.
    - For a given nucleation transition with phases
      ``high_phase`` → ``low_phase``, it compares the ancestry chains of
      both and prunes out common parents.
    - It then searches ``crit_trans`` (from low T to high T) for the
      first transition whose phases are compatible with the pruned
      ancestry chains and whose ``Tcrit`` is **not below** ``Tnuc``.
    - If found, it stores this dictionary under the key
      ``tdict["crit_trans"]``. Otherwise, it sets
      ``tdict["crit_trans"] = None``.

    Parameters
    ----------
    phases : mapping
        Phase mapping used to interpret the graph structure.
    crit_trans : sequence of dict
        Critical-temperature transitions, typically the output of
        :func:`findCriticalTemperatures`, assumed to be sorted in
        decreasing ``Tcrit``.
    full_trans : sequence of dict
        Full (supercooled) transitions, typically the output of
        :func:`findAllTransitions`. Each dict is **modified in place**
        by adding a ``"crit_trans"`` key.

    Returns
    -------
    None
        The function operates by side-effect on the entries of
        ``full_trans``.
    """
    # ------------------------------------------------------------------
    # 1. Build ancestry lists ("parents") for each phase in the critical
    #    transition graph, scanning from low T to high T.
    # ------------------------------------------------------------------
    parents_dict: Dict[Hashable, List[Hashable]] = {}

    # We interpret crit_trans[::-1] as running from low to high T.
    crit_low_to_high = list(crit_trans)[::-1]

    for key in phases.keys():
        parents: List[Hashable] = [key]
        for tcdict in crit_low_to_high:
            high = tcdict["high_phase"]
            low = tcdict["low_phase"]
            if low in parents and high not in parents:
                parents.append(high)
        parents_dict[key] = parents

    # ------------------------------------------------------------------
    # 2. For each full (supercooled) transition, find the matching
    #    critical transition and attach it.
    # ------------------------------------------------------------------
    for tdict in full_trans:
        low_phase_key = tdict["low_phase"]
        high_phase_key = tdict["high_phase"]

        low_parents = list(parents_dict.get(low_phase_key, []))
        high_parents = list(parents_dict.get(high_phase_key, []))

        # Identify common ancestors in the critical graph. We then prune
        # them out so that we focus on the "differential" part of the
        # ancestry between the two phases.
        common_parents = set(low_parents).intersection(high_parents)

        for p in common_parents:
            # Remove p and anything *above* it in the low_parents chain
            try:
                k_low = low_parents.index(p)
                low_parents = low_parents[:k_low]
            except ValueError:
                pass

            # Remove anything *below* p in the high_parents chain, but
            # keep p itself at the top.
            try:
                k_high = high_parents.index(p)
                high_parents = high_parents[: k_high + 1]
            except ValueError:
                pass

        # Now search critical transitions from low T to high T, matching
        # both ancestry and the requirement Tcrit >= Tnuc.
        Tnuc = float(tdict["Tnuc"])
        attached: Optional[Dict[str, Any]] = None

        for tcdict in crit_low_to_high:
            Tcrit = float(tcdict["Tcrit"])
            if Tcrit < Tnuc:
                # Critical point lies below the nucleation temperature:
                # not the right match for this supercooled transition.
                continue

            if (
                tcdict["low_phase"] in low_parents
                and tcdict["high_phase"] in high_parents
            ):
                attached = tcdict
                break

        tdict["crit_trans"] = attached


# ---------------------------------------------------------------------------
# Block D – Observables: thermodynamic and GW–friendly diagnostics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _partial_dVdT(
    V: Callable[[np.ndarray, float], float],
    x: np.ndarray,
    T: float,
    *,
    h_rel: float = 1e-3,
    h_abs: float = 1e-3,
) -> float:
    """
    Finite-difference estimate of the *partial* derivative ∂V/∂T at fixed field x.

    Parameters
    ----------
    V : callable
        Potential V(x, T) returning a scalar free-energy density.
    x : ndarray
        Field value (same shape used elsewhere, typically (N,) or (1,)).
    T : float
        Temperature at which to evaluate ∂V/∂T.
    h_rel : float, optional
        Relative step size in temperature (fraction of max(|T|, 1)).
    h_abs : float, optional
        Absolute minimum step in T.

    Returns
    -------
    float
        Finite-difference estimate of ∂V/∂T at fixed x.
    """
    T = float(T)
    x_arr = np.asarray(x, dtype=float)
    dT = max(h_abs, h_rel * max(abs(T), 1.0))

    Vp = float(V(x_arr, T + dT))
    Vm = float(V(x_arr, T - dT))
    return (Vp - Vm) / (2.0 * dT)


def hubble_rad(
    T: float,
    *,
    g_star: float = 106.75,
    M_pl: float = 1.2209e19,
) -> float:
    r"""
    Hubble rate in a radiation-dominated Universe, in natural units.

    Uses the standard approximation

        H(T) ≃ 1.66 * sqrt(g_*) * T^2 / M_pl,

    where g_* is the effective number of relativistic degrees of freedom.

    Parameters
    ----------
    T : float
        Temperature at which to evaluate H(T).
    g_star : float, optional
        Effective number of relativistic degrees of freedom g_*.
    M_pl : float, optional
        (Non-reduced) Planck mass, default 1.2209e19 GeV.

    Returns
    -------
    float
        H(T) in the same units as T^2 / M_pl (typically GeV).
    """
    T = float(T)
    return 1.66 * np.sqrt(float(g_star)) * T * T / float(M_pl)


# ---------------------------------------------------------------------------
# Core observable builder for a single transition
# ---------------------------------------------------------------------------

def thermalObservablesForTransition(
    tdict: Mapping[str, Any],
    V: Callable[[np.ndarray, float], float],
    *,
    dVdT: Optional[Callable[[np.ndarray, float], float]] = None,
    T_key: str = "Tnuc",
    g_star: float = 106.75,
    T_eps_rel: float = 1e-3,
    T_eps_abs: float = 1e-3,
    beta_from_geometry: bool = True,
    beta_geom_method: str = "rscale",
    M_pl: float = 1.2209e19,
) -> Dict[str, Any]:
    r"""
    Compute thermodynamic observables for a single phase transition.

    This function takes a transition dictionary (as returned by
    :func:`tunnelFromPhase` or :func:`findAllTransitions`) and the underlying
    potential :math:`V(x, T)` and returns a dictionary with quantities such as

    - :math:`S(T_*)` and :math:`S(T_*)/T_*`,
    - free-energy difference :math:`\Delta V`,
    - energy-density difference :math:`\Delta \rho` (latent heat),
    - strength parameter :math:`\alpha \equiv \Delta \rho / \rho_{\rm rad}`,
    - approximate :math:`\beta/H_*` from instanton geometry when available.

    Parameters
    ----------
    tdict : mapping
        Transition dictionary. Must contain at least:

        - ``Tnuc`` or another temperature key given by ``T_key``.
        - ``high_vev``, ``low_vev``: field values of high-T and low-T phases.
        - ``high_phase``, ``low_phase``: phase identifiers.
        - ``action``: Euclidean action S(T_*).
        - ``instanton``: backend-dependent instanton object (may be None).

        If ``tdict`` also contains a key ``"crit_trans"`` with a nested
        dictionary that has a ``"Tcrit"`` entry, this is used to compute
        the supercooling :math:`\Delta T = T_{\rm crit} - T_*`.
    V : callable
        Potential ``V(x, T) -> float``.
    dVdT : callable, optional
        Partial derivative ∂V/∂T at fixed x, with signature
        ``dVdT(x, T) -> float``. If omitted, a finite-difference estimate
        is used via :func:`_partial_dVdT`.
    T_key : {"Tnuc", "Tcrit", ...}, optional
        Name of the temperature key in ``tdict`` to be used as the evaluation
        temperature :math:`T_*`. Typically "Tnuc" (default) or "Tcrit".
    g_star : float, optional
        Effective number of relativistic degrees of freedom g_* at T_*.
    T_eps_rel, T_eps_abs : float, optional
        Relative and absolute scales for the finite-difference step in T
        used when `dVdT` is not provided.
    beta_from_geometry : bool, optional
        If True, attempt to use geometric information from the instanton
        to estimate a length-scale :math:`\beta_{\rm eff}` via the method
        :meth:`tunneling1D.SingleFieldInstanton.betaEff`. If not available,
        the corresponding entries are set to NaN.
    beta_geom_method : {"rscale", "curvature", "wall"}, optional
        Method passed to ``instanton.betaEff(profile, method=...)`` when
        geometry-based beta is requested.
    M_pl : float, optional
        Planck mass used to convert :math:`\beta_{\rm eff}` into
        :math:`\beta/H_*` via :func:`hubble_rad`.

    Returns
    -------
    dict
        Dictionary with the following fields (keys):

        - ``"T_star"`` : float – the temperature used for evaluation (T_*).
        - ``"T_ref_key"`` : str – the key used ("Tnuc", "Tcrit", ...).
        - ``"Tcrit"`` : float or NaN – critical temperature if available.
        - ``"DeltaT"`` : float or NaN – Tcrit − T_*, supercooling measure.
        - ``"S"`` : float – Euclidean action S(T_*).
        - ``"S_over_T"`` : float – S(T_*) / T_*.
        - ``"deltaV"`` : float – free-energy difference
          :math:`\Delta V = V_{\rm high} - V_{\rm low}` at T_*.
        - ``"rho_high"`` : float – energy density of the high-T phase,
          :math:`\rho_{\rm high} = V_{\rm high} - T_* (\partial V/\partial T)_{\rm high}`.
        - ``"rho_low"`` : float – same for low-T phase.
        - ``"delta_rho"`` : float – energy-density difference
          :math:`\Delta \rho = \rho_{\rm high} - \rho_{\rm low}` (latent heat).
        - ``"latent_heat"`` : float – alias for ``delta_rho``.
        - ``"rho_rad"`` : float – radiation energy density
          :math:`\rho_{\rm rad} = (\pi^2/30) g_* T_*^4`.
        - ``"alpha_strength"`` : float – transition strength parameter
          :math:`\alpha = \Delta \rho / \rho_{\rm rad}`.
        - ``"beta_eff"`` : float – geometric proxy for β (inverse length),
          or NaN if not available.
        - ``"beta_over_H_eff"`` : float – :math:`\beta_{\rm eff} / H(T_*)` or NaN.
        - ``"beta_method"`` : str – description of how β was obtained.

    Notes
    -----
    * The potential :math:`V(\phi, T)` is interpreted as a *free-energy density*.
      For each phase,

      .. math::

          \rho(\phi, T) = V(\phi, T) - T \left( \frac{\partial V}{\partial T} \right)_\phi

      is used as the energy density (no total derivative along the phase
      trajectory; the derivative is taken at fixed field value).

    * ``delta_rho`` is defined as :math:`\rho_{\rm high} - \rho_{\rm low}`:
      a positive value corresponds to *released* vacuum energy when the
      system tunnels from the high-T to the low-T phase.
    """
    # ------------------------------------------------------------
    # 1) Choose evaluation temperature T_star
    # ------------------------------------------------------------
    if T_key not in tdict:
        raise KeyError(
            f"thermalObservablesForTransition: T_key='{T_key}' not found in transition dict."
        )

    T_star = float(tdict[T_key])
    if T_star <= 0.0:
        raise ValueError(
            f"thermalObservablesForTransition: T_star={T_star:.6g} must be positive."
        )

    # Optional critical temperature for supercooling
    Tcrit = np.nan
    if "crit_trans" in tdict and tdict["crit_trans"] is not None:
        ct = tdict["crit_trans"]
        if isinstance(ct, Mapping) and "Tcrit" in ct:
            Tcrit = float(ct["Tcrit"])

    DeltaT = Tcrit - T_star if np.isfinite(Tcrit) else np.nan

    # ------------------------------------------------------------
    # 2) Extract minima and potential values at T_star
    # ------------------------------------------------------------
    x_high = np.asarray(tdict["high_vev"], dtype=float)
    x_low = np.asarray(tdict["low_vev"], dtype=float)

    V_high = float(V(x_high, T_star))
    V_low = float(V(x_low, T_star))

    # Free-energy difference: high minus low (energy released when going high → low)
    deltaV = V_high - V_low

    # ------------------------------------------------------------
    # 3) Partial derivatives ∂V/∂T at fixed field values
    # ------------------------------------------------------------
    if dVdT is not None:
        dVdT_high = float(dVdT(x_high, T_star))
        dVdT_low = float(dVdT(x_low, T_star))
    else:
        dVdT_high = _partial_dVdT(V, x_high, T_star, h_rel=T_eps_rel, h_abs=T_eps_abs)
        dVdT_low = _partial_dVdT(V, x_low, T_star, h_rel=T_eps_rel, h_abs=T_eps_abs)

    # Energy densities: rho = V - T * (∂V/∂T)_φ
    rho_high = V_high - T_star * dVdT_high
    rho_low = V_low - T_star * dVdT_low

    # Latent heat / energy density difference (released energy)
    delta_rho = rho_high - rho_low

    # Radiation bath energy density
    rho_rad = (np.pi**2 / 30.0) * float(g_star) * (T_star**4)

    # Strength parameter α = Δρ / ρ_rad
    alpha_strength = delta_rho / rho_rad if rho_rad != 0.0 else np.nan

    # ------------------------------------------------------------
    # 4) Action and S/T
    # ------------------------------------------------------------
    S = float(tdict.get("action", np.nan))
    S_over_T = S / T_star if T_star != 0.0 else np.nan

    # ------------------------------------------------------------
    # 5) Geometric β proxies (SingleFieldInstanton if available)
    # ------------------------------------------------------------
    beta_eff = np.nan
    beta_over_H_eff = np.nan
    beta_method_used = "none"

    if beta_from_geometry:
        inst = tdict.get("instanton", None)
        if inst is not None and hasattr(inst, "findProfile") and hasattr(inst, "betaEff"):
            try:
                profile = inst.findProfile()
                beta_eff_val = float(inst.betaEff(profile, method=str(beta_geom_method)))
                if np.isfinite(beta_eff_val) and beta_eff_val > 0.0:
                    H_star = hubble_rad(T_star, g_star=g_star, M_pl=M_pl)
                    beta_eff = beta_eff_val
                    beta_over_H_eff = beta_eff_val / H_star if H_star > 0.0 else np.nan
                    beta_method_used = f"geometry:{beta_geom_method}"
            except Exception:
                # If anything goes wrong, fall back to NaN but keep code robust.
                beta_eff = np.nan
                beta_over_H_eff = np.nan
                beta_method_used = f"geometry:{beta_geom_method}:failed"

    # ------------------------------------------------------------
    # 6) Assemble observables dictionary
    # ------------------------------------------------------------
    obs = dict(
        T_star=T_star,
        T_ref_key=str(T_key),
        Tcrit=Tcrit,
        DeltaT=DeltaT,
        S=S,
        S_over_T=S_over_T,
        deltaV=deltaV,
        rho_high=rho_high,
        rho_low=rho_low,
        delta_rho=delta_rho,
        latent_heat=delta_rho,
        rho_rad=rho_rad,
        alpha_strength=alpha_strength,
        beta_eff=beta_eff,
        beta_over_H_eff=beta_over_H_eff,
        beta_method=beta_method_used,
    )
    return obs


# ---------------------------------------------------------------------------
# Helper: attach observables to an entire transition history
# ---------------------------------------------------------------------------

def addObservablesToTransitions(
    transitions: Sequence[Dict[str, Any]],
    V: Callable[[np.ndarray, float], float],
    *,
    dVdT: Optional[Callable[[np.ndarray, float], float]] = None,
    T_key: str = "Tnuc",
    g_star: float = 106.75,
    T_eps_rel: float = 1e-3,
    T_eps_abs: float = 1e-3,
    beta_from_geometry: bool = True,
    beta_geom_method: str = "rscale",
    M_pl: float = 1.2209e19,
) -> None:
    """
    Enrich a list of transition dictionaries with thermodynamic observables.

    For each transition dictionary in `transitions`, this function calls
    :func:`thermalObservablesForTransition` and stores the resulting
    observables under the key ``"obs"`` in-place.

    Parameters
    ----------
    transitions : sequence of dict
        List as returned by :func:`findAllTransitions`. Each element is
        modified in-place to include an ``"obs"`` entry.
    V : callable
        Potential ``V(x, T) -> float``.
    dVdT : callable, optional
        Partial derivative ∂V/∂T at fixed x. If omitted, a finite-difference
        estimate is used.
    T_key : str, optional
        Temperature key to use as T_* (typically "Tnuc").
    g_star : float, optional
        Effective number of relativistic degrees of freedom at the transition.
    T_eps_rel, T_eps_abs : float, optional
        Relative/absolute step sizes for T finite differences when dVdT is None.
    beta_from_geometry : bool, optional
        If True, attempt to estimate β/H from the instanton geometry.
    beta_geom_method : {"rscale", "curvature", "wall"}, optional
        Method passed to ``instanton.betaEff``.
    M_pl : float, optional
        Planck mass used in :func:`hubble_rad`.
    """
    for tdict in transitions:
        tdict["obs"] = thermalObservablesForTransition(
            tdict,
            V,
            dVdT=dVdT,
            T_key=T_key,
            g_star=g_star,
            T_eps_rel=T_eps_rel,
            T_eps_abs=T_eps_abs,
            beta_from_geometry=beta_from_geometry,
            beta_geom_method=beta_geom_method,
            M_pl=M_pl,
        )
