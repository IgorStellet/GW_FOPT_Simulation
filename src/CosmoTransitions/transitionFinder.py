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
from typing import Callable, Dict, Hashable, Mapping, MutableMapping, Optional, Sequence, Tuple



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
