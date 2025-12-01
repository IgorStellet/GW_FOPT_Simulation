# -----------------------------------------------------------------------------
# Complete, end-to-end demo for SingleFieldInstanton & Transition_Finder
# -----------------------------------------------------------------------------
# What this script provides
# ------------------------
# Ten cohesive figures/examples (A....) that take you from the potential geometry, to
# initial conditions, to the final bounce profile and physically meaningful params
# Reproducing also article potential: Cosmological phase transitions from the functional
# measure (another example test ).

# -----------------------------------------------------------------------------
import os
import math
from typing import Tuple, Optional, Callable, NamedTuple, Dict, Any, Sequence, Mapping, Hashable
import json, sys, io, contextlib
from functools import partial
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy import interpolate, integrate, optimize


# Import the modernized class
from CosmoTransitions import SingleFieldInstanton
from CosmoTransitions import deriv14, gradientFunction, hessianFunction
from CosmoTransitions import Jb, Jf


np.set_printoptions(precision=6, suppress=True)

# -----------------------------------------------------------------------------
# Potentials used at the paper
# -----------------------------------------------------------------------------

# --- Masses and vev in GeV ---
_VEW = 246.0
_MH  = 125.0
_MW  = 80.36
_MZ  = 91.19
_MT  = 173.1

# --- Degeneracies (+bosons , -fermions ) ---
_nW, _nZ, _nt = 6.0, 3.0, 12.0


# --- Couplings g_i = m_i / v and Constants ---

_gW, _gZ, _gt = _MW/_VEW, _MZ/_VEW, _MT/_VEW
_64pi2 = 64.0 * np.pi**2
_8pi2  = 8.0  * np.pi**2


def V_paper(phi: np.ndarray | float, T: np.ndarray | float | None=None , C: float = 3.0, Lambda: float = 1000.0,
            finiteT: bool =False,
            include_daisy: bool= True,
            real_cont: bool= False) -> np.ndarray:
    """
    Zero-temperature effective potential with modified functional measure.
    optional finite-temperature corrections (your Eq. 21).

    Parameters
    ----------
    phi : array_like
        Higgs radial field ϕ (GeV).
    C : float
        Measure-deformation parameter (C=0 recovers SM one-loop).
    Lambda : float
        Sensitivity scale of the modified measure (GeV).

    Notes
    -----
    * Zero-T one-loop piece matches your correction: φ^4 * (ln(φ^2/v^2) - 3/2) only.
    * Thermal integrals use your finiteT module: FT.Jb(x), FT.Jf(x) with x = m/T.
    * Daisy term: simplified EW longitudinal resummation consistent with the paper’s
      shorthand '3 m_L^3' → 2×W_L + 1×Z_L.

    Returns
    -------
    V : ndarray
        V(ϕ, 0) in GeV^4. We choose the renormalization conditions at ϕ=v
        exactly as in the paper.
    """
    phi = np.asarray(phi, dtype=float)
    v   = _VEW

    # Tree
    V_tree = (_MH**2)/(8.0*v**2) * (phi**2 - v**2)**2

    # One-loop CW-like piece (W, Z, top); M_i(ϕ)=g_i ϕ and n_i as below.

    # Use safe log for ϕ=0 (limit ϕ^4 log(ϕ^2) -> 0).
    with np.errstate(divide='ignore', invalid='ignore'):
        log_phi = np.where(phi != 0.0, np.log((phi**2)/(v**2)), 0.0)
    common_bracket = (phi**4)*(log_phi - 1.5) + 2.0*(phi**2)*(v**2)
    pref = 1/_64pi2
    V_loops = pref * (
        _nW*(_gW**4) + _nZ*(_gZ**4) - _nt*(_gt**4)
    ) * common_bracket

    # Measure contribution, Eq. (3.4): t = C ϕ^2 / Λ^2, t0 = C v^2 / Λ^2
    t  = C * (phi**2) / (Lambda**2)
    t0 = C * (v**2)   / (Lambda**2)
    one_minus_t0 = 1.0 - t0
    denom = (one_minus_t0**2)

    # Guard the log for t -> 1 (ϕ -> Λ/√C): mask beyond the branch cut
    if real_cont:
        log_term = np.log(np.abs(1.0-t))
    else:
        valid = (t < 1.0) | (C <= 0.0)
        log_term = np.empty_like(t)
        log_term[valid]   = np.log(1.0 - t[valid])
        log_term[~valid]  = np.nan  # outside the domain, as in the paper’s discussion


    poly = ((1.0 - 2.0*t0) * t + 0.5 * t**2) / denom
    V_meas = - (Lambda**4)/_8pi2 * (log_term + poly)

    V0 = V_tree + V_loops + V_meas
    V0 = np.real_if_close(V0)
    if not finiteT:
        return V0

    # -------------------------------
    # Finite-T corrections (Eq. 21)
    # -------------------------------
    if T is None:
        raise ValueError("finiteT=True requires a temperature T>0 (in GeV).")

    # thermal arguments x = m/T with m_i(φ) = g_i * |φ|
    absphi = np.abs(phi)

    with np.errstate(divide="ignore", invalid="ignore"):
        xW = (_gW * absphi) / T
        xZ = (_gZ * absphi) / T
        xt = (_gt * absphi) / T


    # bosons positive, fermions negative (as in your ΔV expression)
    DV_b = (T**4)/(2.0*np.pi**2) * (_nW*Jb(xW, approx="exact") + _nZ*Jb(xZ,approx="exact"))
    DV_f = (T**4)/(2.0*np.pi**2) * (_nt*Jf(xt,approx="exact"))

    DV_b = np.real_if_close(DV_b)
    DV_f = np.real_if_close(DV_f)

    DV_daisy = 0.0
    if include_daisy:
        # paper’s effective g^2 (dimensionless)
        g2 = 4*(_MW**2+_MZ**2) /(3*(_VEW**2))

        # build m_L^2
        mL2_phi = 0.25 * g2 *(absphi**2)
        mL2_T = mL2_phi +(11.0/6.0) * g2 *(T**2)

        # guard tiny negatives from roundoff
        mL_phi = np.sqrt(np.maximum(mL2_phi, 0.0))
        mL_T = np.sqrt(np.maximum(mL2_T, 0.0))

        # 3 longitudinal modes (2 W_L + 1 Z_L compressed into g^2)
        DV_daisy = -(T/(12.0*np.pi)) * 3.0 * (mL_T**3 - mL_phi**3)
        DV_daisy = np.real_if_close(DV_daisy)

    V_tot = V0 + DV_b + DV_f + DV_daisy
    V_tot = np.real_if_close(V_tot)

    return V_tot

# ------------------------------------------------------------
# Numerical derivatives of V_paper with respect to φ and T
# ------------------------------------------------------------
PotentialFn = Callable[..., np.ndarray]


class PotentialDerivatives(NamedTuple):
    """
    Container for all finite-T potential derivative wrappers.

    All callables have signatures like

        f(phi, T, *extra) -> ndarray

    where:
    - phi : array_like, shape (..., Ndim)
    - T   : scalar or array_like, broadcastable with phi
    """
    V: Callable[..., np.ndarray]          # offset: V(phi,T) - V(0,T)
    gradV: Callable[..., np.ndarray]      # ∂V/∂phi
    hessV: Callable[..., np.ndarray]      # ∂²V/∂phi_i ∂phi_j
    dV_dT: Callable[..., np.ndarray]      # ∂V/∂T
    dgradV_dT: Callable[..., np.ndarray]  # ∂/∂T (∂V/∂phi)

def build_finite_T_derivatives(
    Vtot: PotentialFn,
    Ndim: int,
    x_eps: ArrayLike,
    T_eps: float,
    deriv_order: int = 4,
) -> PotentialDerivatives:
    """
    Construct high-level derivative wrappers for a finite-T potential.

    Parameters
    ----------
    Vtot : callable
        Base potential Vtot(phi, T, *extra). It must accept an array of points
        `phi` with shape (..., Ndim) and a temperature T (scalar or array-like)
        broadcastable to phi[..., 0].
    Ndim : int
        Number of scalar fields (dimension of phi-space).
    x_eps : float or array_like
        Finite-difference step(s) in field space, passed to gradientFunction/
        hessianFunction.
    T_eps : float
        Finite-difference step in temperature for d/dT stencils.
    deriv_order : {2, 4}, optional
        Accuracy order for the φ-derivatives (2 or 4). The same order is used
        in the d/dT stencils.

    Returns
    -------
    PotentialDerivatives
        NamedTuple with callables (V, gradV, hessV, dV_dT, dgradV_dT).

    Notes
    -----
    * For a single point x0 (shape (Ndim,)) and scalar T, the returned functions
      have shapes:
         V(x0,T)              → scalar
         gradV(x0,T)          → (Ndim,)
         hessV(x0,T)          → (Ndim, Ndim)
         dV_dT(x0,T)          → scalar
         dgradV_dT(x0,T)      → (Ndim,)
      o que é exatamente o que `traceMinimum/traceMultiMin` esperam.
    """
    # --- low-level φ-derivatives from helper_functions -----------------------
    grad_phi = gradientFunction(
        Vtot, eps=x_eps, Ndim=Ndim, order=deriv_order
    )
    hess_phi = hessianFunction(
        Vtot, eps=x_eps, Ndim=Ndim, order=deriv_order
    )

    # Normaliza φ para o formato que gradientFunction/hessianFunction esperam
    # e marca se é um único ponto em espaço de campos.
    def _normalize_phi(phi: ArrayLike) -> tuple[np.ndarray, bool]:
        phi_arr = np.asarray(phi, dtype=float)
        # caso escalar: vira (1, Ndim)
        if phi_arr.ndim == 0:
            phi_pts = np.tile(phi_arr, (1, Ndim))
            return phi_pts, True
        # caso 1D com exatamente Ndim componentes: interpreta como um único ponto
        if phi_arr.ndim == 1 and phi_arr.shape[0] == Ndim:
            phi_pts = phi_arr.reshape(1, Ndim)
            return phi_pts, True
        # caso geral: assume que o último eixo já é o de dimensão Ndim
        return phi_arr, False

    # --- 1) V with offset V(0, T) = 0 ---------------------------------------
    def V_offset(phi: ArrayLike, T: ArrayLike, *extra) -> np.ndarray:
        """
        Effective potential with T-dependent constant subtracted such that
        V(0, T) = 0 for all T.
        """
        phi_pts, is_single = _normalize_phi(phi)
        T_arr = np.asarray(T, dtype=float)

        # ponto de referência φ=0
        phi0 = np.zeros((1, Ndim), dtype=float)

        V_val = Vtot(phi_pts, T_arr, *extra)
        V0_val = Vtot(phi0, T_arr, *extra)
        V_shift = V_val - V0_val  # broadcast adequado

        V_shift = np.real_if_close(V_shift)
        if is_single:
            # colapsa para escalar
            return float(np.asarray(V_shift, dtype=float).reshape(-1)[0])
        return np.asarray(V_shift, dtype=float)

    # --- 2) ∇_phi V  --------------------------------------------------------
    def gradV(phi: ArrayLike, T: ArrayLike, *extra) -> np.ndarray:
        """
        Gradient in field space: ∂V/∂phi_i.

        Shapes:
            phi single point (Ndim,) + scalar T  →  (Ndim,)
            Caso geral                          →  (..., Ndim)
        """
        phi_pts, is_single = _normalize_phi(phi)
        T_arr = np.asarray(T, dtype=float)
        g = grad_phi(phi_pts, T_arr, *extra)
        g = np.asarray(g, dtype=float)

        if is_single:
            # grad_phi retorna algo tipo (1, Ndim) → pegamos o ponto 0
            if g.ndim >= 2:
                return g[0]
            # fallback paranoico
            return g.reshape(-1)[:Ndim]
        return g

    # --- 3) Hessian in field space -----------------------------------------
    def hessV(phi: ArrayLike, T: ArrayLike, *extra) -> np.ndarray:
        """
        Hessian in field space: ∂²V/∂phi_i ∂phi_j.

        Shapes:
            phi single point (Ndim,) + scalar T  →  (Ndim, Ndim)
            Caso geral                          →  (..., Ndim, Ndim)
        """
        phi_pts, is_single = _normalize_phi(phi)
        T_arr = np.asarray(T, dtype=float)
        H = hess_phi(phi_pts, T_arr, *extra)
        H = np.asarray(H, dtype=float)

        if is_single:
            # gradientFunction/hessianFunction normalmente devolvem (1,Ndim,Ndim)
            if H.ndim == 3:
                H = H[0]
            # garante (Ndim, Ndim) mesmo que venha como escalar ou vetor
            return H.reshape(Ndim, Ndim)
        return H

    # --- 4) ∂V/∂T  ----------------------------------------------------------
    def dV_dT(phi: ArrayLike, T: ArrayLike, *extra) -> np.ndarray:
        """
        Numerical derivative of the potential with respect to temperature.

        Uses the same finite-difference order in T as in φ (2 or 4).
        """
        phi_pts, is_single = _normalize_phi(phi)
        T0 = np.asarray(T, dtype=float)

        if deriv_order == 2:
            Vp = Vtot(phi_pts, T0 + T_eps, *extra)
            Vm = Vtot(phi_pts, T0 - T_eps, *extra)
            d = (Vp - Vm) / (2.0 * T_eps)
        else:
            Vp2 = Vtot(phi_pts, T0 + 2.0 * T_eps, *extra)
            Vp1 = Vtot(phi_pts, T0 + 1.0 * T_eps, *extra)
            Vm1 = Vtot(phi_pts, T0 - 1.0 * T_eps, *extra)
            Vm2 = Vtot(phi_pts, T0 - 2.0 * T_eps, *extra)
            d = (-Vp2 + 8.0 * Vp1 - 8.0 * Vm1 + Vm2) / (12.0 * T_eps)

        d = np.asarray(d, dtype=float)
        if is_single:
            return float(d.reshape(-1)[0])
        return d

    # --- 5) ∂/∂T (∇_phi V)  ------------------------------------------------
    def dgradV_dT(phi: ArrayLike, T: ArrayLike, *extra) -> np.ndarray:
        """
        Derivative of the gradient with respect to T:
            ∂/∂T [∇_phi V(phi, T)].

        Shapes:
            phi single point (Ndim,) + scalar T  →  (Ndim,)
            Caso geral                          →  (..., Ndim)
        """
        phi_pts, is_single = _normalize_phi(phi)
        T0 = np.asarray(T, dtype=float)

        if deriv_order == 2:
            gp = grad_phi(phi_pts, T0 + T_eps, *extra)
            gm = grad_phi(phi_pts, T0 - T_eps, *extra)
            dg = (gp - gm) / (2.0 * T_eps)
        else:
            gp2 = grad_phi(phi_pts, T0 + 2.0 * T_eps, *extra)
            gp1 = grad_phi(phi_pts, T0 + 1.0 * T_eps, *extra)
            gm1 = grad_phi(phi_pts, T0 - 1.0 * T_eps, *extra)
            gm2 = grad_phi(phi_pts, T0 - 2.0 * T_eps, *extra)
            dg = (-gp2 + 8.0 * gp1 - 8.0 * gm1 + gm2) / (12.0 * T_eps)

        dg = np.real_if_close(dg)
        dg = np.asarray(dg, dtype=float)
        if is_single:
            # normalmente (1, Ndim)
            if dg.ndim >= 2:
                return dg[0]
            return dg.reshape(-1)[:Ndim]
        return dg

    return PotentialDerivatives(
        V=V_offset,
        gradV=gradV,
        hessV=hessV,
        dV_dT=dV_dT,
        dgradV_dT=dgradV_dT,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def make_inst( V: Callable,
    case: str = "paper",
    alpha: int = 2,
    phi_abs: float = _VEW,
    phi_meta: float = 0.0,
    C: float = 3.7,
    Lambda: float = 1000.0,
    finiteT: bool = True,
    include_daisy: bool = True,
) -> Tuple[SingleFieldInstanton, str]:
    """
    Construct a SingleFieldInstanton with the standard vacua:
      phi_absMin = 1.0 (true/stable), phi_metaMin = 0.0 (false/metastable).
    """
    case = case.lower().strip()
    if case == "paper":
        V, label = V(
            C=C,
            Lambda=Lambda,
            finiteT=finiteT,
            include_daisy=include_daisy,
        ), "Glauber_Paper_"
        phi_abs, phi_meta = phi_abs, phi_meta
    else:
        raise ValueError("Unknown case.")
    inst = SingleFieldInstanton(
        phi_absMin=phi_abs,
        phi_metaMin=phi_meta,
        V=V,
        alpha=alpha,
        phi_eps=1e-3,
    )
    return inst, label


def ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path

def build_phi_grid(inst: SingleFieldInstanton, margin: float = 0.1, n: int = 800):
    lo = min(inst.phi_metaMin, inst.phi_absMin)
    hi = max(inst.phi_metaMin, inst.phi_absMin)
    span = hi - lo
    return np.linspace(lo - margin*span, hi + margin*span, n)

def _extract_params_from_V(V_callable):
    """Best-effort extraction of {C, Lambda, T, finiteT} from functools.partial."""
    C = Lambda = T = finiteT = None
    try:
        if isinstance(V_callable, partial):
            kws = V_callable.keywords or {}
            C       = kws.get("C", None)
            Lambda  = kws.get("Lambda", None)
            T       = kws.get("T", None)
            finiteT = kws.get("finiteT", None)
    except Exception:
        pass
    return C, Lambda, T, finiteT


def savefig(fig: plt.Figure, save_dir: Optional[str], name: str):
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=160, bbox_inches="tight")

@contextlib.contextmanager
def tee_stdout(save_dir: Optional[str], filename: str = "showcase_log.txt"):
    """
    If save_dir is provided, duplicate all `print` output to save_dir/filename.
    Otherwise, behave as a no-op.
    """
    if not save_dir:
        yield
        return
    os.makedirs(save_dir, exist_ok=True)

    class _Tee(io.TextIOBase):
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s); st.flush()
        def flush(self):
            for st in self.streams: st.flush()

    log_path = os.path.join(save_dir, filename)
    with open(log_path, "w", encoding="utf-8") as f, \
         contextlib.redirect_stdout(_Tee(sys.stdout, f)):
        yield


# ============================================================================
# High-level glue: phases + transitions from transitionFinder
# ============================================================================

def _build_phases_and_transitions(
    V_XT,
    dVdphi_XT,
    hessV_XT,
    dVdT_XT,
    dgradT_XT,
    *,
    T_min: float,
    T_max: float,
    phi_range: Tuple[float, float] = (-3.0, 3.0),
    n_phi_scan: int = 200,
    n_T_seeds: int = 5,
    deltaX_target: float = 0.05,
    dtstart_frac: float = 1e-3,
    tjump_frac: float = 1e-3,
    forbidCrit=None,
    tunnelFromPhase_args: Dict[str, Any] | None = None,
    nuclCriterion: Callable[[float, float], float] | None = None,
    Tn_Ttol: float = 1e-3,
    Tn_maxiter: int = 80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build phases + transitions for a given finite-T potential V(phi, T).

    This wraps Block A+B+C of transitionFinder:

    - find seed minima along a line in field space at a few temperatures;
    - trace all phases with traceMultiMin;
    - remove redundant phases;
    - find critical temperatures and the full thermal history.

    It returns a summary dict with everything the examples K–M need.
    """
    from CosmoTransitions import transitionFinder  # local import inside example

    T_min = float(T_min)
    T_max = float(T_max)
    phi_low, phi_high = phi_range

    if T_max <= T_min:
        raise ValueError(
            f"_build_phases_and_transitions: require T_max > T_min, "
            f"got {T_max} <= {T_min}"
        )

    # ------------------------------------------------------------------
    # 1. Seed minima along a simple 1D line in field space
    # ------------------------------------------------------------------
    x_low = np.array([phi_low], dtype=float)
    x_high = np.array([phi_high], dtype=float)

    seeds: list[Tuple[np.ndarray, float]] = []
    T_refs = np.linspace(T_max, T_min, n_T_seeds)

    for T_ref in T_refs:
        T_ref = float(T_ref)

        def V_line(x, T_local=T_ref):
            # Restrict to a line, but keep x as ndarray for generality
            x_arr = np.asarray(x, dtype=float)
            return V_XT(x_arr, T_local)

        minima = transitionFinder.findApproxLocalMin(
            V_line,
            x_low,
            x_high,
            args=(),
            n=n_phi_scan,
            edge=0.05,
        )
        if minima.size == 0:
            continue

        for xm in minima:
            seeds.append((np.asarray(xm, dtype=float).reshape(-1), T_ref))

    if not seeds:
        # Defensive fallback: start at phi = 0 at both ends of T
        seeds = [
            (np.zeros(1, dtype=float), T_max),
            (np.zeros(1, dtype=float), T_min),
        ]

    # Remove trivial duplicates
    unique_seeds: list[Tuple[np.ndarray, float]] = []
    for x_seed, T_seed in seeds:
        keep = True
        for x_old, T_old in unique_seeds:
            if abs(T_seed - T_old) < 1e-6 and np.linalg.norm(x_seed - x_old) < 1e-3:
                keep = False
                break
        if keep:
            unique_seeds.append((x_seed, T_seed))

    if verbose:
        print(
            f"[build_phases] Seeds: {len(unique_seeds)} minima between "
            f"T = {T_min:.3f} and {T_max:.3f}"
        )

    # ------------------------------------------------------------------
    # 2. Trace all phases with traceMultiMin
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
        print(f"[build_phases] Traced {len(phases)} phases; "
              f"start phase key = {start_key!r}")

    # ------------------------------------------------------------------
    # 3. Critical temperatures and full thermal history
    # ------------------------------------------------------------------
    crit_trans = transitionFinder.findCriticalTemperatures(
        phases,
        V_XT,
        start_high=False,
    )
    if verbose:
        print(f"[build_phases] Found {len(crit_trans)} critical temperatures.")

    def _default_nucl(S: float, T: float) -> float:
        # S(T)/T ≈ 140
        return S / (T + 1e-100) - 140.0

    tf_args: Dict[str, Any] = dict(
        Ttol=float(Tn_Ttol),
        maxiter=int(Tn_maxiter),
        phitol=1e-6,
        overlapAngle=45.0,
        verbose=verbose,
        fullTunneling_params=None,
        # Default nucleation criterion S(T)/T ≈ 140
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

    transitionFinder.addCritTempsForFullTransitions(
        phases,
        crit_trans,
        full_trans,
    )

    # First genuine first-order transition, if any
    main_transition = None
    for tdict in full_trans:
        if int(tdict.get("trantype", 0)) == 1:
            main_transition = tdict
            break

    if verbose:
        if main_transition is None:
            print(
                "[build_phases] No first-order transitions found "
                "(history may be purely second-order)."
            )
        else:
            Tn = float(main_transition.get("Tnuc", np.nan))
            print(
                "[build_phases] Main FO transition: "
                f"high_phase={main_transition['high_phase']}, "
                f"low_phase={main_transition['low_phase']}, "
                f"Tn ≈ {Tn:.4g}"
            )

    summary: Dict[str, Any] = {
        # Store potential + derivatives for later reuse
        "V_XT": V_XT,
        "dVdphi_XT": dVdphi_XT,
        "hessV_XT": hessV_XT,
        "dVdT_XT": dVdT_XT,
        "dgradT_XT": dgradT_XT,
        # Metadata
        "T_min": T_min,
        "T_max": T_max,
        "phi_range": (float(phi_low), float(phi_high)),
        # Phase structure and transitions
        "phases": phases,
        "start_phase_key": start_key,
        "start_phase": start_phase,
        "critical_transitions": crit_trans,
        "full_transitions": full_trans,
        "main_transition": main_transition,
    }
    return summary


# ============================================================================
# Spinodals: m^2(T) along a phase and spinodal temperatures
# ============================================================================

def _spinodal_data_for_phase(
    phase,
    hessV_XT,
    *,
    n_T_scan: int = 300,
) -> Dict[str, Any]:
    """
    For a given Phase, sample the smallest eigenvalue of the Hessian along the
    minimum x_min(T) and find approximate spinodal temperatures m^2(T) ≈ 0.
    """
    from scipy import optimize

    T_min = float(phase.T[0])
    T_max = float(phase.T[-1])
    T_grid = np.linspace(T_min, T_max, n_T_scan)

    def m2_of_T(T: float) -> float:
        x = np.asarray(phase.valAt(T), dtype=float)
        H = np.asarray(hessV_XT(x, float(T)), dtype=float)
        if H.ndim == 0:
            return float(H)
        H = H.reshape(x.size, x.size)
        eigs = np.linalg.eigvalsh(H)
        return float(np.min(eigs))

    m2_vals = np.array([m2_of_T(float(T)) for T in T_grid])

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
        # Fallback: the point where |m²| is minimal
        idx = int(np.argmin(np.abs(m2_vals)))
        spinodals = [float(T_grid[idx])]

    spinodals = sorted(spinodals)
    return {
        "T_grid": T_grid,
        "m2": m2_vals,
        "T_spinodals": spinodals,
    }


def _closest_spinodal_to_T(
    T_target: float,
    T_spinodals: Sequence[float],
) -> float | None:
    """Return the spinodal temperature closest to T_target (or None)."""
    if not T_spinodals:
        return None
    arr = np.asarray(T_spinodals, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(T_target))))
    return float(arr[idx])

#####################################################
# ZERO TEMPERATURE PLOTS
#####################################################
# -----------------------------------------------------------------------------
# Example A
# -----------------------------------------------------------------------------

def example_A_fig1_multiC(Cs=( -5.0, 0.0, 3.65, 3.75, 3.83, 4.14, 5.0 ),
                          Lambda=1000.0, phi_max=300.0,
                          save_dir: Optional[str] = None, tag: str = ""):
    """
    Fig.1-like comparison: plot [V(φ,0)-V(0,0)] for several C at fixed Λ.
    Highlights C=0 and C=3.7. Draws V=0 line.
    """
    ϕ_axis = np.linspace(0.0, phi_max, 1400)
    V0_by_C = {}

    fig = plt.figure(figsize=(7.6, 4.8))
    for C in Cs:
        cutoff = (Lambda/np.sqrt(C))*0.995 if C > 0 else phi_max
        ϕ = ϕ_axis[ϕ_axis <= cutoff]
        V0 = V_paper(0.0, C=C, Lambda=Lambda, finiteT=False)
        Vn = V_paper(ϕ,  C=C, Lambda=Lambda, finiteT=False) - V0
        V0_by_C[C] = V0

        style = dict(lw=2, zorder=3) if (C==0.0 or abs(C-3.7)<1e-12) else dict(lw=1.6, alpha=0.9, zorder=2)
        ls = "--" if C==0.0 or C==4.14 else "-"
        plt.plot(ϕ, Vn, ls=ls, label=f"C={C:g}", **style)

    plt.axhline(0.0, color="#444", lw=1.0, ls="--", label="V=0")
    plt.axvline(_VEW, color="#888", lw=1.0, ls=":", label="ϕ = v")
    plt.xlabel("ϕ  [GeV]"); plt.ylabel("V(ϕ,T=0) − V(0,T=0)  [GeV⁴]")
    plt.title(f"Zero-T potential, Λ = {Lambda:.0f} GeV (Fig.1-like)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout(); plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figA{suffix}")

# -----------------------------------------------------------------------------
# Example B
# -----------------------------------------------------------------------------

def example_B_paper_potential_with_inset(C=3, Lambda=1000.0,
                                         save_dir: Optional[str] = None, tag: str = ""):
    """
    Plot V_paper(φ) - V(0) up to φ≈650 GeV, showing both sides of the log singularity
    via real_cont=True (ln|1-t|). Inset: zoom up to φ=300 GeV, tick numbers only.
    """
    # Domain limits
    phi_cap = (Lambda/np.sqrt(C)) if C > 0 else np.inf
    phi_max_main  = min(650.0, np.inf if not np.isfinite(phi_cap) else 4*phi_cap)  # allow showing both sides
    phi_max_inset = min(300.0, phi_max_main)

    φ = np.linspace(0.0, phi_max_main, 4000)
    V_main = V_paper(φ, C=C, Lambda=Lambda, finiteT=False, real_cont=True)
    V0     = V_paper(0.0, C=C, Lambda=Lambda, finiteT=False, real_cont=True)
    V_shift = V_main - V0

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(φ, V_shift, lw=2.2, label=rf"$C={C}$, $\Lambda={Lambda:.0f}$ GeV")
    ax.axhline(0.0, color="#888", lw=1.0, ls="--", label="V=0")

    if np.isfinite(phi_cap):
        ax.axvline(phi_cap, color="#aa0000", lw=1.2, ls="--", label=r"$\phi_\star=\Lambda/\sqrt{C}$")

    ax.set_xlim(0.0, phi_max_main)
    ax.set_xlabel(r"$\phi$ [GeV]"); ax.set_ylabel(r"$V(\phi)-V(0)$ [GeV$^4$]")
    ax.set_title("Paper potential (real-part continuation) — Fig.1 style")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # Inset
    axins = ax.inset_axes([0.06, 0.3, 0.42, 0.42])
    mask = (φ <= phi_max_inset)
    axins.plot(φ[mask], V_shift[mask], lw=1.6)
    y0, y1 = np.nanmin(V_shift[mask]), np.nanmax(V_shift[mask])
    pad = 0.05 * (y1 - y0 + 1e-30)
    axins.set_xlim(0.0, phi_max_inset)
    axins.set_ylim(y0 - pad, y1 + pad)
    axins.grid(True, alpha=0.2)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="#666", lw=1.0, alpha=0.85)

    plt.tight_layout(); plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figB{suffix}")



#####################################################
# FINITE TEMPERATURE PLOTS
#####################################################

# ============================================================================
# Example C – key scales: T_n, T_c, spinodals
# ============================================================================

def example_C_transition_summary(
    V_XT,
    dVdphi_XT,
    hessV_XT,
    dVdT_XT,
    dgradT_XT,
    *,
    T_min: float = 0.0,
    T_max: float = 200.0,
    phi_range: Tuple[float, float] = (-3.0, 3.0),
    n_T_seeds: int = 5,
    nuclCriterion: Callable[[float, float], float] | None = None,
    Tn_Ttol: float = 1e-3,
    Tn_maxiter: int = 80,
    save_dir: str = "figs",
    tag: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Example C (finite T): thermal key scales from transitionFinder.

    Steps:
      1. Build phases + transitions via _build_phases_and_transitions.
      2. Identify the "main" first-order transition.
      3. Compute:
         - T_n : nucleation temperature (S/T ≈ 140, by default),
         - T_c : corresponding critical temperature (degeneracy ΔV = 0),
         - spinodals of the high- and low-phase Hessians (m²(T) ≈ 0).
      4. Save a 1D plot with characteristic temperatures, coloured cold→hot.
    """
    summary = _build_phases_and_transitions(
        V_XT,
        dVdphi_XT,
        hessV_XT,
        dVdT_XT,
        dgradT_XT,
        T_min=T_min,
        T_max=T_max,
        phi_range=phi_range,
        n_T_seeds=n_T_seeds,
        nuclCriterion=nuclCriterion,
        Tn_Ttol=Tn_Ttol,
        Tn_maxiter=Tn_maxiter,
        verbose=verbose,
    )

    phases = summary["phases"]
    main_transition = summary["main_transition"]

    if main_transition is None:
        print("\n[Example C] No first-order transition found – nothing to summarize.")
        return summary

    high_key = main_transition["high_phase"]
    low_key = main_transition["low_phase"]
    high_phase = phases[high_key]
    low_phase = phases[low_key]

    # Nucleation temperature
    T_n = float(main_transition.get("Tnuc", np.nan))

    # Critical temperature: prefer the one attached by addCritTempsForFullTransitions
    T_c = None
    if main_transition.get("crit_trans") is not None:
        T_c = float(main_transition["crit_trans"]["Tcrit"])
    else:
        # Fallback: search in the list of critical transitions
        for tcdict in summary["critical_transitions"]:
            if (
                tcdict["high_phase"] == high_key
                and tcdict["low_phase"] == low_key
            ):
                T_c = float(tcdict["Tcrit"])
                break

    # Spinodals of both phases
    spin_high = _spinodal_data_for_phase(high_phase, hessV_XT)
    spin_low = _spinodal_data_for_phase(low_phase, hessV_XT)

    T_spin_high = _closest_spinodal_to_T(T_n, spin_high["T_spinodals"])
    T_spin_low = _closest_spinodal_to_T(T_n, spin_low["T_spinodals"])

    summary["spinodal_high_phase"] = spin_high
    summary["spinodal_low_phase"] = spin_low

    # ---- Action at nucleation / observables from transition dict ----
    S_n = np.nan
    S_over_Tn = np.nan

    obs = main_transition.get("obs") if isinstance(main_transition, Mapping) else None
    if obs:
        try:
            if "S" in obs:
                S_n = float(obs["S"])
            if "S_over_T" in obs:
                S_over_Tn = float(obs["S_over_T"])
        except Exception:
            pass

    if np.isnan(S_n):
        for key in ("S3", "S_E", "S_action", "action"):
            if key in main_transition:
                try:
                    S_n = float(main_transition[key])
                    break
                except Exception:
                    pass

    if np.isnan(S_over_Tn):
        for key in ("S3T", "S_E/T", "S_over_T", "S_over_Tn"):
            if key in main_transition:
                try:
                    S_over_Tn = float(main_transition[key])
                    break
                except Exception:
                    pass

    summary["key_temperatures"] = dict(
        Tn=float(T_n),
        Tc=None if T_c is None else float(T_c),
        T_spinodal_high_phase=T_spin_high,
        T_spinodal_low_phase=T_spin_low,
        S_at_Tn=S_n,
        S_over_Tn=S_over_Tn,
    )

    # ------------------------------------------------------------------
    # Print a compact table of scales
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Example C – key thermal scales from transitionFinder")
    print("=" * 72)
    print(f"  High-T starting phase key : {summary['start_phase_key']!r}")
    print(f"  Main transition           : high_phase={high_key}, low_phase={low_key}")
    print("")
    print(f"  T_n  (nucleation)         = {T_n:10.4g}")

    if not np.isnan(S_n):
        print(f"  S(T_n)                    = {S_n:10.4g}")
    if not np.isnan(S_over_Tn):
        print(f"  S(T_n)/T_n               ≈ {S_over_Tn:10.4g}")
    if T_c is not None:
        print(f"  T_c  (degeneracy)         = {T_c:10.4g}")
    else:
        print("  T_c  (degeneracy)         =   (not found / not applicable)")

    if T_spin_high is not None:
        print(f"  T_sp^(high phase)         = {T_spin_high:10.4g}")
    else:
        print("  T_sp^(high phase)         =   (not determined)")

    if T_spin_low is not None:
        print(f"  T_sp^(low phase)          = {T_spin_low:10.4g}")
    else:
        print("  T_sp^(low phase)          =   (not determined)")

    if (
        T_spin_high is not None
        and T_spin_low is not None
        and T_c is not None
        and np.isfinite(T_n)
    ):
        Tmin_interval = min(T_spin_high, T_spin_low)
        Tmax_interval = max(T_spin_high, T_spin_low)
        inside = (T_n > Tmin_interval) and (T_n < Tmax_interval)
        status = "YES" if inside else "NO"
        print("")
        print(
            f"  Check: is T_n between the two spinodals?  "
            f"[{Tmin_interval:0.4g}, {Tmax_interval:0.4g}]  →  {status}"
        )

    # ------------------------------------------------------------------
    # 1D plot: fundo colorido frio→quente + linhas verticais
    # ------------------------------------------------------------------
    T_plot_min = min(float(high_phase.T[0]), float(low_phase.T[0]))
    T_plot_max = max(float(high_phase.T[-1]), float(low_phase.T[-1]))

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.set_title("Example C – key thermal scales", fontsize=11)
    ax.set_xlabel(r"$T$")
    ax.set_yticks([])
    ax.set_xlim(T_plot_min, T_plot_max)
    ax.set_ylim(0.0, 1.0)

    # Fundo com gradiente de cor: frio (T_min) → quente (T_max)
    n_col = 256
    band = np.tile(np.linspace(0.0, 1.0, n_col), (2, 1))
    ax.imshow(
        band,
        extent=(T_plot_min, T_plot_max, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        cmap="plasma",
        alpha=0.6,
        zorder=0,
    )

    # Linhas verticais para as escalas características
    if T_c is not None:
        ax.axvline(T_c, linestyle="--", linewidth=1.4, color="k", label=r"$T_c$", zorder=2)
    if np.isfinite(T_n):
        ax.axvline(T_n, linestyle="-.", linewidth=1.4, color="w", label=r"$T_n$", zorder=3)

    if T_spin_high is not None:
        ax.axvline(
            T_spin_high,
            linestyle=":",
            linewidth=1.1,
            color="#222222",
            label=r"$T_{\rm sp}^{\rm (high)}$",
            zorder=2,
        )
    if T_spin_low is not None:
        ax.axvline(
            T_spin_low,
            linestyle=":",
            linewidth=1.1,
            color="#444444",
            label=r"$T_{\rm sp}^{\rm (low)}$",
            zorder=2,
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="upper right")

    fig.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figC{suffix}")

    print(f"[Example C] Saved figure: {tag}")
    return summary


# ============================================================================
# Example D – φ_min(T) and m²(T) around the main transition
# ============================================================================
def example_D_phi_min_and_mass2_vs_T(
    summary: Dict[str, Any],
    *,
    n_T_plot: int = 300,
    save_dir: str = "figs",
    tag: str = "",
) -> Dict[str, Any]:
    """
    Example D (finite T):

    - Usa o summary do Example C (phases, main transition, key scales).
    - Plota φ_min(T) e m²(T) na região de overlap das duas fases.
    - Adiciona uma segunda figura com o range completo de T de cada fase.
    """
    phases = summary["phases"]
    main_transition = summary["main_transition"]
    hessV_XT = summary["hessV_XT"]
    key_T = summary.get("key_temperatures", {})

    suffix = f"_{tag}" if tag else ""

    if main_transition is None:
        print(
            "\n[Example D] No main transition in summary – "
            "run example_C_transition_summary first."
        )
        return summary

    high_key = main_transition["high_phase"]
    low_key = main_transition["low_phase"]
    high_phase = phases[high_key]
    low_phase = phases[low_key]

    T_n = float(key_T.get("Tn", np.nan))
    T_c = key_T.get("Tc", None)
    if T_c is not None:
        T_c = float(T_c)

    T_spin_high = key_T.get("T_spinodal_high_phase", None)
    T_spin_low = key_T.get("T_spinodal_low_phase", None)
    if T_spin_high is not None:
        T_spin_high = float(T_spin_high)
    if T_spin_low is not None:
        T_spin_low = float(T_spin_low)

    # Overlap region where both minima exist
    Tmin_overlap = max(float(high_phase.T[0]), float(low_phase.T[0]))
    Tmax_overlap = min(float(high_phase.T[-1]), float(low_phase.T[-1]))
    T_vals = np.linspace(Tmin_overlap, Tmax_overlap, n_T_plot)

    phi_high = np.array(
        [np.asarray(high_phase.valAt(T), dtype=float).ravel()[0] for T in T_vals]
    )
    phi_low = np.array(
        [np.asarray(low_phase.valAt(T), dtype=float).ravel()[0] for T in T_vals]
    )

    def m2_on_phase(phase):
        vals = []
        for T in T_vals:
            x = np.asarray(phase.valAt(T), dtype=float)
            H = np.asarray(hessV_XT(x, float(T)), dtype=float)
            if H.ndim == 0:
                vals.append(float(H))
            else:
                H = H.reshape(x.size, x.size)
                eigs = np.linalg.eigvalsh(H)
                vals.append(float(np.min(eigs)))
        return np.array(vals)

    m2_high = m2_on_phase(high_phase)
    m2_low = m2_on_phase(low_phase)

    print("\n" + "=" * 72)
    print("Example D – φ_min(T) and m²(T) for the main transition")
    print("=" * 72)
    print(f"  Overlap region in T: [{Tmin_overlap:0.4g}, {Tmax_overlap:0.4g}]")
    print(f"  T_n (nucleation)   = {T_n:0.4g}")
    if T_c is not None:
        print(f"  T_c (degeneracy)   = {T_c:0.4g}")
    if T_spin_high is not None:
        print(f"  T_sp^(high phase)  = {T_spin_high:0.4g}")
    if T_spin_low is not None:
        print(f"  T_sp^(low phase)   = {T_spin_low:0.4g}")

    # ------------------------------------------------------------------
    # Duas figuras: (i) overlap, (ii) full range
    # ------------------------------------------------------------------

    # (i) Overlap – duas colunas (φ_min e m²)
    fig, (ax_phi, ax_m2) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)

    ax_phi.plot(T_vals, phi_high, label=r"high-T minimum")
    ax_phi.plot(T_vals, phi_low, label=r"low-T minimum")
    ax_phi.set_ylabel(r"$\phi_{\min}(T)$")
    ax_phi.set_title("Example D – minima and curvature vs temperature", fontsize=11)
    ax_phi.legend(fontsize=8, loc="best")

    ax_m2.plot(T_vals, m2_high, label=r"$m^2_{\rm high}(T)$")
    ax_m2.plot(T_vals, m2_low, label=r"$m^2_{\rm low}(T)$")
    ax_m2.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_m2.set_ylabel(r"$m^2(T)$")
    ax_m2.set_xlabel(r"$T$")
    ax_m2.legend(fontsize=8, loc="best")

    for ax in (ax_phi, ax_m2):
        if np.isfinite(T_n):
            ax.axvline(T_n, linestyle="-.", linewidth=1.0)
        if T_c is not None:
            ax.axvline(T_c, linestyle="--", linewidth=1.0)
        if T_spin_high is not None:
            ax.axvline(T_spin_high, linestyle=":", linewidth=1.0)
        if T_spin_low is not None:
            ax.axvline(T_spin_low, linestyle=":", linewidth=1.0)

    fig.tight_layout()
    plt.show()
    savefig(fig, save_dir, f"figD{suffix}")

    print(f"[Example D] Saved figure (overlap): {tag}")

    # (ii) Full range: φ_min(T) em todos os pontos de cada fase
    T_high_full = np.asarray(high_phase.T, dtype=float)
    phi_high_full = np.array(
        [np.asarray(high_phase.valAt(T), dtype=float).ravel()[0] for T in T_high_full]
    )
    T_low_full = np.asarray(low_phase.T, dtype=float)
    phi_low_full = np.array(
        [np.asarray(low_phase.valAt(T), dtype=float).ravel()[0] for T in T_low_full]
    )

    fig_full, ax_full = plt.subplots(figsize=(6.5, 3.4))
    ax_full.plot(T_high_full, phi_high_full, label="high-T phase φ_min(T)")
    ax_full.plot(T_low_full, phi_low_full, label="low-T phase φ_min(T)")
    ax_full.set_xlabel(r"$T$")
    ax_full.set_ylabel(r"$\phi_{\min}(T)$")
    ax_full.set_title("Example D – minima over full temperature range", fontsize=11)

    for x in (T_n if np.isfinite(T_n) else None, T_c, T_spin_high, T_spin_low):
        if x is None:
            continue
        ax_full.axvline(x, linestyle=":", linewidth=0.9)

    ax_full.legend(fontsize=8, loc="best")
    ax_full.grid(True, alpha=0.3)
    fig_full.tight_layout()
    plt.show()
    savefig(fig_full, save_dir, f"figD_full{suffix}")

    print(f"[Example D] Saved figure (full range): {tag}")

    summary["example_D"] = dict(
        T_vals=T_vals,
        phi_high=phi_high,
        phi_low=phi_low,
        m2_high=m2_high,
        m2_low=m2_low,
        T_high_full=T_high_full,
        phi_high_full=phi_high_full,
        T_low_full=T_low_full,
        phi_low_full=phi_low_full,
    )
    return summary



# ============================================================================
# Example E – ΔV(T) between the two minima
# ============================================================================

def example_E_deltaV_vs_T(
    summary: Dict[str, Any],
    *,
    n_T_plot: int = 300,
    save_dir: str = "figs",
    tag: str = "",
) -> Dict[str, Any]:
    """
    Example E (finite T):

    - For the same two minima as in Example L, compute ΔV(T) = V_high − V_low.
    - Focus on the interval between T_n and T_c (clipped to the overlap).
    - Check explicitly: ΔV(T_c) ≈ 0 and ΔV(T_n) > 0 (metastability).
    """
    V_XT = summary["V_XT"]
    phases = summary["phases"]
    main_transition = summary["main_transition"]
    key_T = summary.get("key_temperatures", {})

    if main_transition is None:
        print(
            "\n[Example E] No main transition in summary – "
            "run example_E_transition_summary first."
        )
        return summary

    high_key = main_transition["high_phase"]
    low_key = main_transition["low_phase"]
    high_phase = phases[high_key]
    low_phase = phases[low_key]

    T_n = float(key_T.get("Tn", np.nan))
    T_c = key_T.get("Tc", None)
    if T_c is not None:
        T_c = float(T_c)

    # Overlap where both minima exist
    Tmin_overlap = max(float(high_phase.T[0]), float(low_phase.T[0]))
    Tmax_overlap = min(float(high_phase.T[-1]), float(low_phase.T[-1]))

    if T_c is None or not np.isfinite(T_n):
        # If one of the scales is missing, just use the whole overlap
        T_lo = Tmin_overlap
        T_hi = Tmax_overlap
    else:
        T_lo = max(min(T_n, T_c), Tmin_overlap)
        T_hi = min(max(T_n, T_c), Tmax_overlap)

    T_vals = np.linspace(T_lo, T_hi, n_T_plot)

    def V_on_phase(phase, T):
        x = np.asarray(phase.valAt(T), dtype=float)
        return float(V_XT(x, float(T)))

    deltaV = np.array(
        [V_on_phase(high_phase, T) - V_on_phase(low_phase, T) for T in T_vals]
    )

    # Checks at T_n and T_c
    deltaV_Tn = np.nan
    deltaV_Tc = np.nan
    if np.isfinite(T_n) and (T_lo <= T_n <= T_hi):
        deltaV_Tn = V_on_phase(high_phase, T_n) - V_on_phase(low_phase, T_n)
    if (T_c is not None) and (T_lo <= T_c <= T_hi):
        deltaV_Tc = V_on_phase(high_phase, T_c) - V_on_phase(low_phase, T_c)

    print("\n" + "=" * 72)
    print("Example E – ΔV(T) between the two minima")
    print("=" * 72)
    print(f"  Plot interval in T: [{T_lo:0.4g}, {T_hi:0.4g}]")
    print(f"  T_n (nucleation)  = {T_n:0.4g}")
    if T_c is not None:
        print(f"  T_c (degeneracy)  = {T_c:0.4g}")
    if np.isfinite(deltaV_Tn):
        print(
            f"  ΔV(T_n)           = {deltaV_Tn:0.4g}  "
            f"(should be > 0 for metastability)"
        )
    if np.isfinite(deltaV_Tc):
        print(
            f"  ΔV(T_c)           = {deltaV_Tc:0.4g}  "
            f"(should be ≈ 0 at degeneracy)"
        )

    # Figure
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(T_vals, deltaV)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\Delta V(T) = V_{\rm high} - V_{\rm low}$")
    ax.set_title("Example E – Free-energy difference between minima", fontsize=11)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    if np.isfinite(T_n) and (T_lo <= T_n <= T_hi):
        ax.axvline(T_n, linestyle="-.", linewidth=1.0)
    if (T_c is not None) and (T_lo <= T_c <= T_hi):
        ax.axvline(T_c, linestyle="--", linewidth=1.0)


    fig.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figE{suffix}")
    print(f"[Example E] Saved figure: {tag}")

    summary["example_E"] = dict(
        T_vals=T_vals,
        deltaV=deltaV,
        deltaV_Tn=deltaV_Tn,
        deltaV_Tc=deltaV_Tc,
    )
    return summary


# ============================================================================
# Example F –: schematic phase-history map
# ============================================================================

def example_F_phase_history_map(
    summary: Dict[str, Any],
    *,
    save_dir: str = "figs",
    tag: str = "",
) -> Dict[str, Any]:
    """
    Example F (optional, finite T):

    Build a compact 1D "phase map" of the thermal history:

      - x-axis: temperature T (shown naturally as cooling: high → low);
      - colored horizontal bands: which Phase is realized in each T-interval;
      - vertical lines: transition temperatures from the full history.
    """
    phases = summary["phases"]
    start_phase = summary["start_phase"]
    full_trans = summary["full_transitions"]

    if not full_trans:
        print("\n[Example F] No transitions in the history – nothing to plot.")
        return summary

    # Build segments T ∈ [T_low, T_high] labeled by phase key
    segments = []
    current_phase_key = start_phase.key
    T_top = float(start_phase.T[-1])  # hottest T covered by start phase

    for tdict in full_trans:
        T_trans = float(tdict.get("Tnuc", tdict.get("Tcrit")))
        segments.append(
            dict(
                T_high=T_top,
                T_low=T_trans,
                phase_key=current_phase_key,
            )
        )
        current_phase_key = tdict["low_phase"]
        T_top = T_trans

    last_phase = phases[current_phase_key]
    T_bottom = float(last_phase.T[0])
    segments.append(
        dict(
            T_high=T_top,
            T_low=T_bottom,
            phase_key=current_phase_key,
        )
    )

    print("\n" + "=" * 72)
    print("Example F – schematic phase history along cooling")
    print("=" * 72)
    for seg in segments:
        print(
            f"  Phase {seg['phase_key']} :  T ∈ "
            f"[{seg['T_low']:0.4g}, {seg['T_high']:0.4g}]"
        )

    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.set_xlabel(r"$T$")
    ax.set_yticks([])
    ax.set_title("Example F – thermal phase history (schematic)", fontsize=11)

    # Colored bands
    for seg in segments:
        T_high = seg["T_high"]
        T_low = seg["T_low"]
        key = seg["phase_key"]
        ax.fill_between(
            [T_low, T_high],
            0.0,
            1.0,
            alpha=0.25,
            label=f"Phase {key}",
        )

    # Vertical lines for transitions
    for tdict in full_trans:
        T_trans = float(tdict.get("Tnuc", tdict.get("Tcrit")))
        ax.axvline(T_trans, linestyle="--", linewidth=1.0)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lbl in zip(handles, labels):
        if lbl not in unique:
            unique[lbl] = h
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper right")

    # Show cooling: high T on the left, low T on the right
    ax.set_xlim(float(start_phase.T[-1]), float(last_phase.T[0]))

    fig.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figF{suffix}")

    print(f"[Example F ] Saved figure: {tag}")
    summary["example_F"] = dict(segments=segments)
    return summary



# -----------------------------------------------------------------------------
# Example G
# -----------------------------------------------------------------------------
def example_G_T_scan(
    summary: Dict[str, Any],
    *,
    C: float = 3.7,
    Lambda: float = 1000.0,
    phi_max: float = 300.0,
    save_dir: Optional[str] = None,
    shift_ref: str = "V0",
    tag: str = "",
) -> Dict[str, Any]:
    """
    Example G (finite T):

    Plot V(φ,T) − V_ref(T) for four temperatures:
      - one bellow T_nuc,
      - T_nuc,
      - T_c,
      - above T_c.
    """
    assert shift_ref in ("V0", "Vv"), "shift_ref must be 'V0' or 'Vv'."
    v = _VEW
    eps = 0.97

    key_T = summary.get("key_temperatures", {}) or {}
    T_n = float(key_T.get("Tn", np.nan))
    Tc_val = key_T.get("Tc", None)
    T_c = float(Tc_val) if (Tc_val is not None) else np.nan

    T_list: list[float] = []

    if np.isfinite(T_n):
        if np.isfinite(T_c) and T_c != T_n:
            dT = max(0.05 * abs(T_c - T_n), 0.5)
        else:
            dT = max(0.1 * T_n, 1.0)

        T_below = max(T_n - 15*dT, 0.1)
        if np.isfinite(T_c):
            T_above = T_c + 15*dT
            T_list = [T_below, T_n, T_c, T_above]
        else:
            T_list = [T_below, T_n, T_n + dT, T_n + 2.0 * dT]
    else:
        # fallback razoável
        T_list = [60.0, 80.0, 100.0, 120.0]

    # ordena e remove duplicados
    T_list = sorted({float(T) for T in T_list if T > 0.0})

    if C > 0:
        phi_cap = float(Lambda) / np.sqrt(C)     # branch point
        phi_max_eff = min(phi_max, eps * phi_cap)  # stay on physical side
    else:
        phi_cap = np.inf
        phi_max_eff = phi_max

    if phi_max_eff < 50.0:
        print(f"[G] Warning: φ-domain limited to {phi_max_eff:.1f} GeV by the measure branch (C={C}, Λ={Lambda}).")

    ϕ = np.linspace(0.0, phi_max_eff, 2200)

    # --- helper: quick broken-minimum locator at this T (grid search) ---
    def _phi_true_at_T(T: float) -> float:
        grid = np.linspace(0.0, phi_max_eff, 3001)
        Vg = V_paper(grid, C=C, Lambda=Lambda, finiteT=True, T=T)
        idx = np.nanargmin(Vg)
        return float(grid[idx])

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_list)))

    print("\n[G] Temperature sweep diagnostics (T_below, T_n, T_c, T_above)")
    print("    T [GeV]   phi_true(T) [GeV]   V(phi_true,T)-V(0,T) [GeV^4]   V(v,T)-V(0,T) [GeV^4]")
    print("    --------  -------------------  ------------------------------  ----------------------")

    for T, col in zip(T_list, colors):
        VϕT = V_paper(ϕ, C=C, Lambda=Lambda, finiteT=True, T=T)
        V0T = float(V_paper(0.0, C=C, Lambda=Lambda, finiteT=True, T=T))
        VvT = float(V_paper(v,    C=C, Lambda=Lambda, finiteT=True, T=T))

        if shift_ref == "V0":
            Vshift = VϕT - V0T
        else:  # 'Vv'
            Vshift = VϕT - VvT

        lw, ls, z = (2.2, "-", 3) if T in (min(T_list), max(T_list)) else (1.8, "-", 2)
        ax.plot(ϕ, Vshift, color=col, lw=lw, ls=ls, label=f"T = {T:g} GeV", zorder=z)

        phi_true_T = _phi_true_at_T(T)
        V_true_T   = float(V_paper(phi_true_T, C=C, Lambda=Lambda, finiteT=True, T=T))
        print(f"    {T:7.1f}  {phi_true_T:19.3f}  {V_true_T - V0T:30.3e}  {VvT - V0T:22.3e}")

    ax.axhline(0.0, color="#444", lw=1.0, ls="--", label="shift reference")
    ax.axvline(v,   color="#888", lw=1.0, ls=":",  label="ϕ = v")
    if np.isfinite(phi_cap):
        ax.axvline(phi_cap, color="#aa0000", lw=1.0, ls="--", alpha=0.7, label=r"$\phi_\star=\Lambda/\sqrt{C}$")

    ylabel = r"$V(\phi,T)-V(0,T)$" if shift_ref == "V0" else r"$V(\phi,T)-V(v,T)$"
    ax.set_xlim(0.0, phi_max_eff)
    ax.set_xlabel(r"$\phi$ [GeV]"); ax.set_ylabel(ylabel + r"  [GeV$^4$]")
    ax.set_title(fr"Temperature sweep at $C={C}$, $\Lambda={Lambda:.0f}$ GeV")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.show()

    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figG{suffix}")

    summary["example_G"] = dict(T_list=T_list)
    return summary


#####################################################
# BOUNCE PLOTS - PLOTS IN NUCLEATION TEMPERATURES
#####################################################

# -----------------------------------------------------------------------------
# Example H
# -----------------------------------------------------------------------------
def example_H_potential_geometry(inst: SingleFieldInstanton,
                                 profile,
                                 save_dir: Optional[str] = None, tag: str = ""):
    """
    H) Plot V(φ) with markers at phi_meta, phi_abs(true), phi_bar, phi_top.
       Also plot the *inverted* potential -V(φ) with vertical markers and a
       single red dot at φ0 = profile.Phi[0].
       Print rscale (cubic & curvature) and ΔV diagnostics.
    """
    r0info = getattr(inst, "_profile_info", None) or {}
    r0 = r0info.get("r0", np.nan)
    phi0 = r0info.get("phi0", np.nan)


    sinfo = getattr(inst, "_scale_info", {}) or {}
    rscale_cubic = sinfo.get("rscale_cubic", np.nan)
    rscale_curv  = sinfo.get("rscale_curv", np.inf)
    phi_top = sinfo.get("phi_top", None)

    # Values & deltas
    V_meta = inst.V(inst.phi_metaMin)
    V_abs  = inst.V(inst.phi_absMin)
    V_top  = inst.V(phi_top)
    dV_true_meta = V_abs - V_meta
    dV_top = sinfo.get("V_top_minus_Vmeta", None)

    # Console summary
    print("\n[A] Potential geometry & scales")
    print(f"  phi_meta = {inst.phi_metaMin:.9f}, V(phi_meta) = {V_meta:.9e}")
    print(f"  phi_abs  = {inst.phi_absMin:.9f}, V(phi_abs)  = {V_abs :.9e}")
    print(f"  phi_bar  = {inst.phi_bar:.9f}, V(phi_bar)  = {inst.V(inst.phi_bar):.9e}")
    print(f"  phi_top  = {phi_top:.9f}, V(phi_top)  = {V_top :.9e}")
    print(f"  phi_0    = {phi0:.9f}, V(phi_0)    = {inst.V(phi0):.9e}, r0 = {r0:.6e}")
    print(f"  ΔV_true-meta = {dV_true_meta:.9e}")
    print(f"  ΔV_top -meta = {dV_top:.9e}")
    print(f"  rscale_cubic = {rscale_cubic:.9e}")
    if math.isfinite(rscale_curv):
        print(f"  rscale_curv  = {rscale_curv :.9e}   (from |V''(phi_top)|)")
    else:
        print( "  rscale_curv  = ∞ (flat top)")

    # Colors
    c_meta, c_abs, c_bar, c_top, c_0 = "#d62728", "#2ca02c", "#d62728", "#ff7f0e", "#e377c2"

    # azul: #1f77b4
    # laranja: #ff7f0e

    # Left: V(φ) with markers
    phi_grid = build_phi_grid(inst, margin=0.10, n=900)
    V_grid = inst.V(phi_grid)

    fig1, ax1 = plt.subplots(figsize=(7.8, 4.5))
    ax1.plot(phi_grid, V_grid, lw=2.2, color="#444444", label="V(φ)")
    ax1.scatter([inst.phi_metaMin], [V_meta], color=c_meta, s=40, label="φ_meta")
    ax1.scatter([inst.phi_absMin ], [V_abs ], color=c_abs , s=40, label="φ_true")
    ax1.scatter([inst.phi_bar    ], [inst.V(inst.phi_bar)], color=c_bar, s=40, label="φ_bar")
    ax1.scatter([phi_top         ], [V_top], color=c_top, s=40, label="φ_top")

    # Horizontal line at V(phi_meta); vertical lines at the markers
    ax1.axhline(V_meta, lw=1.0, ls="--", color=c_meta, alpha=0.8)
    for x, col in [(inst.phi_metaMin, c_meta), (inst.phi_absMin, c_abs),
                   (inst.phi_bar, c_bar), (phi_top, c_top)]:
        ax1.axvline(x, lw=1.0, ls=":", color=col, alpha=0.9)

    ax1.set_xlabel("φ"); ax1.set_ylabel(r"V($\phi$, $T_n$)")
    ax1.set_title("Potential with barrier markers")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", ncol=2)
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    savefig(fig1, save_dir, f"figH1{suffix}")

    # Right: -V(φ) with vertical lines and a red dot at φ0
    fig2, ax2 = plt.subplots(figsize=(7.8, 4.5))
    ax2.plot(phi_grid, -V_grid, lw=2.2, color="#444444", label="-V(φ)")
    for x, col in [(inst.phi_metaMin, c_meta), (inst.phi_absMin, c_abs),
                   (inst.phi_bar, c_bar), (phi_top, c_top)]:
        ax2.axvline(x, lw=1.0, ls=":", color=col, alpha=0.9)

    # φ0 marker + arrow towards downhill direction in V(φ)
    V0 = inst.V(phi0)
    dV0 = inst.dV(phi0)
    span = (max(inst.phi_absMin, inst.phi_metaMin) - min(inst.phi_absMin, inst.phi_metaMin))
    dphi_arrow = 0.06 * span * (np.sign(dV0) if dV0 != 0 else -1.0)  # move opposite to +∂V
    phi1 = np.clip(phi0 + dphi_arrow, phi_grid.min(), phi_grid.max())
    V1   = inst.V(phi1)
    ax2.scatter([phi0], [-V0], color=c_0, s=48, zorder=5, label="φ0")
    ax2.annotate(
        "", xy=(phi1, -V1), xytext=(phi0, -V0),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=c_0)
    )

    ax2.scatter([inst.phi_metaMin], [-V_meta], color=c_meta, s=40, label="φ_meta")
    ax2.scatter([inst.phi_absMin ], [-V_abs ], color=c_abs , s=40, label="φ_true")
    ax2.scatter([inst.phi_bar    ], [-inst.V(inst.phi_bar)], color=c_bar, s=40, label="φ_bar")
    ax2.scatter([phi_top         ], [-V_top], color=c_top, s=40, label="φ_top")
    ax2.set_xlabel("φ"); ax2.set_ylabel(r"-V(φ, $T_n$)")
    ax2.set_title("Inverted potential with φ0")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    savefig(fig2, save_dir, f"figH2{suffix}")

# -----------------------------------------------------------------------------
# Example I
# -----------------------------------------------------------------------------
def example_I_local_quadratic_at_phi0(inst: SingleFieldInstanton,
                                      profile,
                                      save_dir: Optional[str] = None, tag: str = ""):
    """
    I) Print initial choice (r0, φ0, φ'(r0), V(φ0)) and overlay the potential
       with its quadratic Taylor expansion around φ0.
    """
    r0info = getattr(inst, "_profile_info", None) or {}
    r0 = r0info.get("r0", np.nan)
    phi0 = r0info.get("phi0", np.nan)
    dphi0 = r0info.get("dphi0", np.nan)

    V0   = inst.V(phi0)
    dV0  = inst.dV(phi0)
    d2V0 = inst.d2V(phi0)

    print("\n[B] Initial local data at r0")
    print(f"  r0      = {r0:.9e}")
    print(f"  φ(r0)   = {phi0:.9f}")
    print(f"  φ'(r0)  = {dphi0:.9e}  (should be ~ 0 by regularity)")
    print(f"  V(φ0)   = {V0:.9e}")
    print(f"  dV(φ0)  = {dV0:.9e}")
    print(f"  d2V(φ0) = {d2V0:.9e}")

    # ---- Small-r expansion point at (near) the origin ----
    r = np.asarray(profile.R)
    # Take φ0 at the first stored radius (interior already filled → r[0] ≈ 0)
    phi0 = float(profile.Phi[0])
    dV0  = inst.dV(phi0)
    d2V0 = inst.d2V(phi0)

    # Choose a small-r window: min(0.2*rscale, 0.5*Rmax)
    sinfo = getattr(inst, "_scale_info", {}) or {}
    rscale_cubic = sinfo.get("rscale_cubic", np.nan)
    rscale_curv  = sinfo.get("rscale_curv",  np.nan)
    rscale = rscale_cubic if np.isfinite(rscale_cubic) else rscale_curv
    Rmax   = float(r[-1])
    rmax_small = 0.2 * rscale if np.isfinite(rscale) else 0.05 * Rmax
    rmax_small = max(10.0 * (r[1] - r[0]), min(rmax_small, 0.5 * Rmax))  # robust lower/upper bounds

    r_small = np.linspace(0.0, rmax_small, 220)
    phi_small  = np.empty_like(r_small)
    dphi_small = np.empty_like(r_small)

    # Evaluate exact small-r solution around φ0 at r≈0
    for i, ri in enumerate(r_small):
        sol = inst.exactSolution(ri, phi0, dV0, d2V0)   # <- assumes exactSolution exists
        phi_small[i], dphi_small[i] = float(sol.phi), float(sol.dphi)
    print(np.max(dphi_small))
    # ---- Plot: φ(r)−φ0 and φ′(r) ----
    fig = plt.figure(figsize=(8.0,4.6))
    plt.plot(r_small, phi_small - phi0, label=r"$\phi(r)-\phi_0$")
    plt.plot(r_small, dphi_small,       label=r"$\phi'(r)$")
    plt.title(r"Near $r\!\approx\!0$ — exact small-$r$ solution")
    plt.xlabel("r"); plt.ylabel("value")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    plt.tight_layout(); plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figI{suffix}")


# -----------------------------------------------------------------------------
# Example J
# -----------------------------------------------------------------------------
def example_J_inverted_path(inst: SingleFieldInstanton,
                            profile,
                            save_dir: Optional[str] = None, tag: str = ""):
    """
    J) Inverted potential -V(φ) with a highlighted path from φ0 to φ_meta.
       Add three arrows (start/middle/end) to emphasize the trajectory direction.
       Right side points unchanged.
    """
    r0info = getattr(inst, "_profile_info", None) or {}
    phi0 = r0info.get("phi0", np.nan)


    phi_grid = build_phi_grid(inst, margin=0.12, n=1200)
    V_grid = inst.V(phi_grid)

    sinfo = getattr(inst, "_scale_info", {}) or {}
    phi_top = sinfo.get("phi_top", None)
    V_meta = inst.V(inst.phi_metaMin)
    V_abs  = inst.V(inst.phi_absMin)
    V_top  = inst.V(phi_top)

    # LEFT: -V with a magenta path from φ0 to φ_meta + direction arrows
    fig1, ax1 = plt.subplots(figsize=(7.8, 4.5))
    ax1.plot(phi_grid, -V_grid, lw=2.2, color="#1f5fb4", label="-V(φ)")

    # path segment
    ph_a, ph_b = sorted([phi0, inst.phi_metaMin])
    mask = (phi_grid >= ph_a) & (phi_grid <= ph_b)
    ax1.plot(phi_grid[mask], -V_grid[mask], lw=3.0, color="#e377c2", label="path φ0→φ_meta")
    ax1.scatter([phi0], [-inst.V(phi0)], color="#2ca02c", s=50, label="φ0")

    # three arrowheads along the path
    span = (max(inst.phi_absMin, inst.phi_metaMin) - min(inst.phi_absMin, inst.phi_metaMin))
    direction = np.sign(inst.phi_metaMin - phi0)  # +1 if moving to larger φ, else -1
    steps = np.linspace(phi0, (inst.phi_metaMin+phi_top)/2, 3)
    dphi = 0.04 * span * direction
    for ph in steps:
        p0 = (ph, -inst.V(ph))
        p1_phi = np.clip(ph + dphi, phi_grid.min(), phi_grid.max())
        p1 = (p1_phi, -inst.V(p1_phi))
        ax1.annotate("", xy=p1, xytext=p0,
                     arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#2ca02c"))

    # reference markers
    ax1.scatter([inst.phi_metaMin], [-V_meta], color="black", s=40, label="φ_meta")
    ax1.scatter([inst.phi_absMin ], [-V_abs ], color="black", s=40, label="φ_true")
    ax1.scatter([inst.phi_bar    ], [-inst.V(inst.phi_bar)], color="black", s=40, label="φ_bar")
    ax1.scatter([phi_top         ], [-V_top], color="black", s=40, label="φ_top")

    ax1.set_xlabel("φ"); ax1.set_ylabel(r"-V(φ, $T_n$)")
    ax1.set_title("Inverted potential: trajectory with start/middle/end arrows")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    plt.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig1, save_dir, f"figJ{suffix}")

# -----------------------------------------------------------------------------
# Example K
# -----------------------------------------------------------------------------
def example_K_phi_of_r(inst: SingleFieldInstanton,
                       profile,
                       save_dir: Optional[str] = None, tag: str = ""):
    """
    D) Plot φ(r) highlighting the starting point (r0, φ0); shade the interior
       region r ∈ [0, r0] to distinguish bubble interior vs exterior.
    """
    r = np.asarray(profile.R); phi = np.asarray(profile.Phi)
    r0info = getattr(inst, "_profile_info", None) or {}
    r0 = r0info.get("r0", np.nan)
    phi0 = r0info.get("phi0", np.nan)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    # plot de r_0 founded
    ax.scatter([r0], [phi0], color="#e377c2", s=50, zorder=5, label="(r0, φ0)")

    # Shade the interior (0..r0)
    ax.axvspan(0.0, r0, color="#2ca02c", alpha=0.10, label="interior (shaded)")

    ax.plot(r, phi, lw=2.2, color="#444444", label="φ(r)")
    ax.axhline(inst.phi_metaMin, ls="--", lw=1.0, color="#d62728", label="φ_meta")
    ax.axhline(inst.phi_absMin , ls="--", lw=1.0, color="#2ca02c", label="φ_true")
    ax.set_xlabel("r"); ax.set_ylabel("φ(r)")
    ax.set_title("Bounce profile in radius")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figK{suffix}")

# -----------------------------------------------------------------------------
# Example L
# -----------------------------------------------------------------------------
def example_L_spherical_maps(inst: SingleFieldInstanton,
                             profile,
                             save_dir: Optional[str] = None, tag: str = ""):
    """
    L) Visualize the spherical profile φ(r) in 2D and 3D at t=0.
       - Cartesian slice (x,y) colored by φ(√(x^2+y^2)), colorbar labeled with φ_true / φ_meta.
       - A small radial tick at r = rscale to indicate the interior/exterior separation scale.
       - 3D surface of φ(x,y) at t=0.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

    r = np.asarray(profile.R); phi = np.asarray(profile.Phi); dphi = np.asarray(profile.dPhi)

    # Interpolant φ(r) with sensible fill outside the tabulated range
    chs = interpolate.CubicHermiteSpline(
        r, phi, dphi, extrapolate=True)

    # Helper that clamps outside to the correct vacua
    def phi_of_r(Rad):
        Rad = np.asarray(Rad, dtype=float)
        val = chs(np.clip(Rad, r[0], r[-1]))
        # fill interior (r<r[0]) with φ(r[0]) and exterior (r>r[-1]) with φ_meta
        val = np.where(Rad < r[0],  phi[0], val)
        val = np.where(Rad > r[-1], inst.phi_metaMin, val)
        return val

    # rscale: pick cubic if finite, else curvature
    sinfo = getattr(inst, "_scale_info", {}) or {}
    rscale_cubic = sinfo.get("rscale_cubic", np.nan)
    rscale_curv  = sinfo.get("rscale_curv", np.inf)
    rscale = rscale_curv if np.isfinite(rscale_curv) else rscale_cubic

    # --- wall location near the false-vacuum side ---
    ws = inst.wallDiagnostics(profile, frac=(0.1, 0.9))  # r_lo ~ true side; r_hi ~ false side
    r_wall = float(ws.r_hi)
    thickness = float(ws.thickness)

    print("\n[E] t=0 visualization")
    print(f"  rscale ≈ {rscale:.6e}   (expected interior–exterior separation scale)")
    print(f"  thickness = {thickness:.3e} (thickness found by wall status)")
    print("  This is the instantaneous t=0 slice of the nucleated bubble.")



    # Cartesian slice
    Rmax = float(r[-1])
    pad  = 0.10 * Rmax
    Nx = Ny = 400
    x = np.linspace(-Rmax-pad, Rmax+pad, Nx)
    y = np.linspace(-Rmax-pad, Rmax+pad, Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    RAD = np.hypot(X, Y)
    PHI = phi_of_r(RAD)

    vmin = min(inst.phi_metaMin, inst.phi_absMin)
    vmax = max(inst.phi_metaMin, inst.phi_absMin)

    fig1, ax1 = plt.subplots(figsize=(6.6, 6.0))
    im = ax1.imshow(PHI, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                    vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_title(r"Bounce profile at t=z=0 (Cartesian slice: $\phi(\rho)$ ")
    cb = plt.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label(fr"$\phi$  (φ_true={inst.phi_absMin:.2f} ;  φ_meta={inst.phi_metaMin:.2f})")

    # small radial "bar"  to indicate rscale
    if np.isfinite(r_wall):
        ax1.plot([0.0, 0.0], [r_wall-rscale, r_wall], color="w", lw=2.0, solid_capstyle="butt")
        ax1.text(0, r_wall+0.5*rscale, "thickness", color="w", ha="center", va="bottom", fontsize=9)
                        #+0.1*(r.max()-r.min())
    plt.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig1, save_dir, f"figL1{suffix}")

    # --- 3D surface at t=0 ---
    fig2 = plt.figure(figsize=(7.2, 6.0))
    ax2 = fig2.add_subplot(111, projection="3d")
    # Downsample for 3D performance if needed
    step = max(1, int(Nx/220))
    ax2.plot_surface(X[::step, ::step], Y[::step, ::step], PHI[::step, ::step],
                     linewidth=0, antialiased=True, cmap="viridis")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel(r"\phi")
    ax2.set_title(r"3D view of \phi(x,y) at t=z=0")
    plt.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig2, save_dir, f"figL2{suffix}")

# -----------------------------------------------------------------------------
# Example M
# -----------------------------------------------------------------------------
def example_M_ode_terms(inst: SingleFieldInstanton,
                        profile,
                        save_dir: Optional[str] = None, tag: str = ""):
    """
    M) Plot the contributions to the ODE:
         φ''(r)           (acceleration)
         (α/r) φ'(r)      (friction)
         V'(φ(r))         (force)
       and also overlay V(φ(r)) for reference (secondary axis).
    """
    r = np.asarray(profile.R); phi = np.asarray(profile.Phi); dphi = np.asarray(profile.dPhi)

    phi0 = phi[0]

    # Second derivative φ'' via centered finite difference (with end corrections)
    d2phi = deriv14(dphi,r)

    friction = inst.alpha * dphi / np.maximum(r, 1e-30)  # (α/r) φ'
    force = inst.dV(phi)                                  # V'(φ)

    # --- potential levels and drop ---
    V_meta = float(inst.V(inst.phi_metaMin))
    V_0 = float(inst.V(phi0))
    dV_drop = V_0 - V_meta

    print("\n[F] False → True vacuum potential drop")
    print(f"  V_0 = {V_0:.3e}")
    print(f"  V_meta = {V_meta:.3e}")
    print(f"  ΔV = V_0 - V_meta = {dV_drop:.3e}   (|ΔV| = {abs(dV_drop):.3e})")

    # Plot ODE terms vs r
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(r, d2phi, lw=2.0, label="φ''(r)")
    ax.plot(r, friction, lw=2.0, label="(α/r) φ'(r)")
    ax.plot(r, force, lw=2.0, label="V'(φ(r))")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)

    # Second y-axis for V(φ(r))
    ax2 = ax.twinx()
    ax2.plot(r, inst.V(phi), lw=1.8, ls="--", color="#444444", label="V(φ(r))")
    ax.set_xlabel("r")
    ax.set_ylabel("ODE terms")
    ax2.set_ylabel("V(φ(r))")
    ax.set_title("ODE term decomposition along the profile")
    ax.grid(True, alpha=0.3)

    # --- place ΔV bar at the wall (false-vacuum side) ---
    ws = inst.wallDiagnostics(profile, frac=(0.1, 0.9))
    r_mark = (3*float(ws.r_hi) +r[-1])/4  # center of the wall
    y0, y1 = (V_0, V_meta)
    y_lo, y_hi = (min(y0, y1), max(y0, y1))
    ax2.vlines(r_mark, y_lo, y_hi, color="tab:purple", lw=3.0, alpha=0.9)
    ax2.scatter([r_mark, r_mark], [y0, y1], s=18, color="tab:purple", zorder=5)
    # annotate ΔV next to the bar
    xpad = 0.02 * (r[-1] - r[0])
    ax2.text(r_mark + xpad, 0.5*(y_lo + y_hi),
             f"ΔV = {dV_drop:.1e}",
             color="tab:purple", va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))

    # Build a unified legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figM{suffix}")

# -----------------------------------------------------------------------------
# Example N
# -----------------------------------------------------------------------------
def example_N_action_and_beta(inst: SingleFieldInstanton,
                              profile,
                              save_dir: Optional[str] = None,
                              beta_methods=("rscale", "curvature", "wall"), tag: str = ""):
    """
    N) Action diagnostics:
       - Print S_total and its breakdown (kin, pot, interior).
       - Print β_eff proxies.
       - Plot action *density* versus r and the *cumulative* action S(<r).
       - Also print temperature T (if present in V) and S3/T.

    """
    br = inst.actionBreakdown(profile)

    # Try to read (C, Λ, T) from the V partial used by this instanton
    C, Lambda, T, finiteT = _extract_params_from_V(inst.V)
    S_over_T = (float(br.S_total) / float(T)) if (T is not None and T > 0) else np.nan

    # --- prints ---
    print("\n[G] Action and β proxies")
    print(f"  S_total     = {br.S_total:.6e}")
    print(f"   S_kin      = {br.S_kin:.6e}")
    print(f"   S_pot      = {br.S_pot:.6e}")
    print(f"   S_interior = {br.S_interior:.6e}")
    print(f"  (check) S_kin + S_pot + S_interior = {br.S_kin + br.S_pot + br.S_interior:.6e}")

    if T is not None:
        print(f"  T           = {T:.6g} GeV")
        print(fr"  S3/T        = {S_over_T:.6e} (should be $\approx$140)")
    else:
        print("  T           = (not provided in V)")
        print("  S3/T        = NaN (no T)")

    betas = {}
    for m in beta_methods:
        try:
            betas[m] = float(inst.betaEff(profile, method=m))
        except Exception:
            betas[m] = np.nan
        print(f"  beta_{m:9s}= {betas[m]:.6e}")

    # --- data for plots ---
    r = np.asarray(br.r)
    dens_kin = np.asarray(br.density["kin"])
    dens_pot = np.asarray(br.density["pot"])
    dens_tot = np.asarray(br.density["tot"])

    # cumulative action: S(<r) = S_interior + ∫_r0^r (dens_tot) dr
    S_line_cum = integrate.cumulative_trapezoid(dens_tot, r, initial=0.0)
    S_cum = S_line_cum + br.S_interior

    # --- figure: densities + cumulative action ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=False)

    # left: densities
    ax1.plot(r, dens_kin, lw=1.9, label="kinetic density")
    ax1.plot(r, dens_pot, lw=1.9, ls="--", label="potential density")
    ax1.plot(r, dens_tot, lw=2.2, label="total density", alpha=0.9)
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax1.set_xlabel("r"); ax1.set_ylabel("action density")
    ax1.set_title("Action density vs r")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # right: cumulative action
    ax2.plot(r, S_cum, lw=2.2, color="#1f77b4",
             label=r"$S(<r)=S_{\mathrm{int}}+\int \mathrm{dens}_{\mathrm{tot}}\,dr$")
    ax2.axhline(br.S_total, color="#ff7f0e", lw=1.6, ls="--",
                label=f"S_total = {br.S_total:.3e}")
    ax2.scatter([r[-1]], [br.S_total], color="#ff7f0e", zorder=5)
    # opcional: marcar r0
    r0 = float(r[0])
    ax2.axvline(r0, color="#888", lw=1.0, ls=":", alpha=0.8)
    ax2.text(r0, ax2.get_ylim()[0], " r0", va="bottom", ha="left", color="#666")

    ax2.set_xlabel("r"); ax2.set_ylabel("action")
    ax2.set_title("Cumulative action")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout(); plt.show()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figN{suffix}")



def gather_diagnostics(
    inst: SingleFieldInstanton,
    profile,
    label: str = "",
    transition_summary: Dict[str, Any] | None = None,
) -> dict:
    r0info = getattr(inst, "_profile_info", {}) or {}
    sinfo  = getattr(inst, "_scale_info", {}) or {}

    # basic geometry
    V_meta = float(inst.V(inst.phi_metaMin))
    V_true = float(inst.V(inst.phi_absMin))
    dV_true_meta = V_true - V_meta
    phi_top = float(sinfo.get("phi_top", np.nan))
    V_top  = float(inst.V(phi_top)) if np.isfinite(phi_top) else np.nan

    # actions
    br = inst.actionBreakdown(profile)

    # try reading parameters from V
    C, Lambda, T, finiteT = _extract_params_from_V(inst.V)
    S_over_T = (float(br.S_total) / float(T)) if (T is not None and T > 0) else np.nan

    # betas
    def _safe_beta(method: str) -> float:
        try:
            return float(inst.betaEff(profile, method=method))
        except Exception:
            return np.nan

    betas = {
        "beta_rscale":   _safe_beta("rscale"),
        "beta_curvature":_safe_beta("curvature"),
        "beta_wall":     _safe_beta("wall"),
    }

    # wall
    try:
        ws = inst.wallDiagnostics(profile, frac=(0.1, 0.9))
        r_wall = float(ws.r_hi); thickness = float(ws.thickness)
    except Exception:
        r_wall, thickness = np.nan, np.nan

    base = {
        "label": label,
        # potential params (if available)
        "C": (float(C) if C is not None else np.nan),
        "Lambda_GeV": (float(Lambda) if Lambda is not None else np.nan),
        "finiteT": bool(finiteT) if finiteT is not None else None,
        "temperature_GeV": (float(T) if T is not None else np.nan),

        # geometry
        "phi_metaMin": float(inst.phi_metaMin),
        "phi_absMin":  float(inst.phi_absMin),
        "phi_bar":     float(getattr(inst, "phi_bar", np.nan)),
        "phi_top":     phi_top,
        "V(phi_meta)": V_meta,
        "V(phi_true)": V_true,
        "V(phi_top)":  V_top,
        "DeltaV_true_minus_meta": dV_true_meta,

        # r0 & scales
        "r0": float(r0info.get("r0", np.nan)),
        "phi0": float(r0info.get("phi0", np.nan)),
        "dphi0": float(r0info.get("dphi0", np.nan)),
        "rscale_cubic": float(sinfo.get("rscale_cubic", np.nan)),
        "rscale_curv":  float(sinfo.get("rscale_curv",  np.nan)),
        "wall_r_hi": r_wall,
        "wall_thickness": thickness,

        # actions
        "S_total": float(br.S_total),
        "S_kin":   float(br.S_kin),
        "S_pot":   float(br.S_pot),
        "S_interior": float(br.S_interior),
        "S3_over_T": float(S_over_T),

        # betas da geometria do bounce
        **betas,
    }

    # ---- Extra: dados do transitionFinder (Block C + D) ----
    if transition_summary is not None:
        extra: Dict[str, float] = {}
        keyT = transition_summary.get("key_temperatures", {}) or {}

        # Important temperatures found by transitionFinder
        Tn_tf = keyT.get("Tn", np.nan)
        Tc_tf = keyT.get("Tc", None)
        Thigh_sp = keyT.get("T_spinodal_high_phase", None)
        Tlow_sp  = keyT.get("T_spinodal_low_phase", None)

        extra["Tn_from_transitionFinder_GeV"] = float(Tn_tf) if np.isfinite(Tn_tf) else np.nan
        extra["Tc_from_transitionFinder_GeV"] = float(Tc_tf) if (Tc_tf is not None) else np.nan
        extra["T_spinodal_high_GeV"] = float(Thigh_sp) if (Thigh_sp is not None) else np.nan
        extra["T_spinodal_low_GeV"]  = float(Tlow_sp)  if (Tlow_sp  is not None) else np.nan

        base.update(extra)

    return base



def save_diagnostics_summary(
    di: dict,
    save_dir: Optional[str],
    basename: str = "diagnostics_summary",
    fmt: str = "json",   # choose: "json" | "csv" | "txt"
):
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    fmt = fmt.lower()
    path = os.path.join(save_dir, f"{basename}.{fmt}")

    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(di, f, indent=2)
    elif fmt == "csv":
        with open(path, "w", encoding="utf-8") as f:
            f.write("key,value\n")
            for k, v in di.items():
                f.write(f"{k},{v}\n")
    elif fmt == "txt":
        pad = max(len(k) for k in di.keys())
        with open(path, "w", encoding="utf-8") as f:
            for k, v in di.items():
                f.write(f"{k.ljust(pad)} : {v}\n")
    else:
        raise ValueError("fmt must be one of: 'json', 'csv', 'txt'")



# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def compute_profile(inst: SingleFieldInstanton,
                    xguess: Optional[float] = None,
                    phitol: float = 1e-5,
                    thinCutoff: float = 0.01,
                    npoints: int = 600,
                    max_interior_pts =  None,
                    _MAX_ITERS=200
                    ) -> object:
    """
    Run the overshoot/undershoot solver to obtain the profile.
    Returns the profile namedtuple (R, Phi, dPhi, Rerr).
    """
    profile = inst.findProfile(
        xguess=xguess, xtol=1e-4, phitol=phitol,
        thinCutoff=thinCutoff, npoints=npoints,
        rmin=1e-4, rmax=1e4, max_interior_pts=max_interior_pts,
        _MAX_ITERS= _MAX_ITERS
    )
    return profile

def run_all(case: str = "paper",
            xguess: Optional[float] = None,
            phitol: float = 1e-5,
            save_dir: Optional[str] = None,
            phi_scan_range: Optional[Tuple[float, float]] = None,
            thinCutoff: float = 0.01,
            C: float = 3.7,
            Lambda: float = 1000.0,
            finiteT: bool = True,
            include_daisy: bool = True,
            npoints: int = 800,
            T_min: float = 5.0,
            T_max: float = 200.0,
            n_T_seeds: int = 2,
            nuclCriterion: Callable[[float, float], float] | None = None,
            Tn_Ttol: float = 1e-3,
            Tn_maxiter: int = 80, ):
    """
    Execute all examples A... in sequence for the chosen case ("thin", "thick" or "mine").

    Parameters
    ----------
    case : {"thin","thick", "mine"}
        Which benchmark potential to use.
    xguess : float or None
        Optional initial guess for the internal shooting parameter used by findProfile.
    phitol : float
        Fractional tolerance for integration in findProfile (smaller = tighter).
    save_dir : str or None
        If provided, figures are saved under this folder.
    """
    save_dir = ensure_dir(save_dir)
    tag = case

    def V_paper_XT(X: ArrayLike,
                   T: ArrayLike,
                   C: float = C,
                   Lambda: float = Lambda,
                   finiteT: bool = finiteT,
                   include_daisy: bool = include_daisy) -> np.ndarray:
        """
        Wrapper for V_paper in the (X, T) convention used by transitionFinder.

        Parameters
        ----------
        X : array_like, shape (..., 1)
            Field-space point(s).
        T : scalar or array_like
            Temperature(s) in GeV (broadcastable with X[..., 0]).
        """
        X = np.asarray(X, dtype=float)
        phi = X[..., 0]  # 1D field space
        return V_paper(phi, T=T, C=C, Lambda=Lambda,
                       finiteT=finiteT, include_daisy=include_daisy,
                       real_cont=False)

    # --- Finite-T derivative infrastructure for transitionFinder (1D) ---
    derivs = build_finite_T_derivatives(
        Vtot= V_paper_XT,
        Ndim=1,
        x_eps=1e-3,  # step in field space
        T_eps=1e-2,  # step in temperature
        deriv_order=4,
    )
    V_XT      = derivs.V          # V(X,T) - V(0,T)
    gradV_XT  = derivs.gradV      # ∂V/∂X
    hessV_XT  = derivs.hessV      # ∂²V/∂X_i ∂X_j
    dVdT_XT   = derivs.dV_dT      # ∂V/∂T
    dgradT_XT = derivs.dgradV_dT  # ∂/∂T (∂V/∂X)


    print(f"=== Running complete showcase on: {tag} potential ===")

    ############################ Zero Temperature #########################################
    # A): multi-C Fig.1-like comparison (highlights C=0 and your default C)
    example_A_fig1_multiC(Cs=(-5.0, 0.0, 3.65, 3.75, 3.83, 4.14, 5.0),  Lambda=Lambda, phi_max=300.0, save_dir=save_dir,tag=tag,)

    # B): paper-like potential with inset & real continuation (fig2)
    example_B_paper_potential_with_inset(C=3, Lambda=Lambda,save_dir=save_dir, tag=tag,)


    ########################### Finite Temperature #####################################
    if phi_scan_range is None:
        phi_low_scan, phi_high_scan = 0.0, _VEW
    else:
        phi_low_scan, phi_high_scan = phi_scan_range

    summary = example_C_transition_summary(
        V_XT=V_XT,
        dVdphi_XT= gradV_XT,
        hessV_XT=hessV_XT,
        dVdT_XT=dVdT_XT,
        dgradT_XT=dgradT_XT,
        T_min= T_min,
        T_max= T_max,
        phi_range=(phi_low_scan, phi_high_scan),
        n_T_seeds=n_T_seeds,
        nuclCriterion=nuclCriterion,
        Tn_Ttol=Tn_Ttol,
        Tn_maxiter=Tn_maxiter,
        save_dir=save_dir,
        tag=tag,
    )

    # D, E, F using the same o summary
    summary = example_D_phi_min_and_mass2_vs_T(summary, save_dir=save_dir, tag=tag)
    summary = example_E_deltaV_vs_T(summary, save_dir=save_dir, tag=tag)
    summary = example_F_phase_history_map(summary, save_dir=save_dir, tag=tag)

    # G): Temperature sweep focusing on (T_below, T_n, T_c, T_above)
    summary = example_G_T_scan(summary,C=C, Lambda=Lambda, phi_max=300.0,
                               save_dir=save_dir, shift_ref="V0", tag=tag,)


    ########################## Bounce Solution #########################################

    # ------------------------------------------------------------------
    # Extract T_n, phi_meta(T_n) and phi_true(T_n) of transitionFinder
    # ------------------------------------------------------------------
    main_transition = summary.get("main_transition")
    phases = summary.get("phases", {})
    key_T = summary.get("key_temperatures", {}) or {}

    if main_transition is None:
        raise RuntimeError(
            "run_all: no first-order transition found in transitionFinder summary. "
            "Cannot define T_n. Check C, (T_min, T_max) or the nucleation criterion."
        )

    if "Tn" not in key_T:
        raise RuntimeError(
            "run_all: key_temperatures does not contain 'Tn'. "
            "Make sure example_C_transition_summary ran successfully."
        )

    T_n = float(key_T["Tn"])
    if not np.isfinite(T_n):
        raise RuntimeError(
            f"run_all: T_n is NaN or non-finite (T_n = {T_n}). "
            "This usually means the nucleation condition S(T)/T ≈ 140 was never satisfied."
        )

    high_phase = phases[main_transition["high_phase"]]
    low_phase  = phases[main_transition["low_phase"]]

    # high_phase = metastable fase (bigger free energy )
    phi_meta_Tn = float(np.asarray(high_phase.valAt(T_n), dtype=float).ravel()[0])
    phi_true_Tn = float(np.asarray(low_phase.valAt(T_n),  dtype=float).ravel()[0])

    T_bounce = T_n

    print("\n=== Bounce setup (from transitionFinder) ===")
    print(f"  T_n (from transitionFinder) = {T_n:.6g} GeV")
    print(f"  phi_meta(T_n) ≈ {phi_meta_Tn:.6g}")
    print(f"  phi_true(T_n) ≈ {phi_true_Tn:.6g}")

    def make_V_phi_only(C=C,
                        Lambda=Lambda,
                        T: float = T_bounce,
                        finiteT=finiteT,
                        include_daisy=include_daisy):
        """
        Factory: returns a phi-only potential V(φ) with T fixed,
        suitable for SingleFieldInstanton.
        """
        return partial(
            V_paper,
            C=C, Lambda=Lambda,
            finiteT=finiteT, T=T,
            include_daisy=include_daisy,
            real_cont=False,
        )

    inst, label = make_inst(V=make_V_phi_only,
                            case=case,
                            phi_abs=phi_true_Tn,
                            phi_meta=phi_meta_Tn,
                            C=C,
                            Lambda=Lambda,
                            finiteT=finiteT,
                            include_daisy=include_daisy,
                            )
    # Solve once and reuse the profile for all examples
    profile = compute_profile(inst, xguess=xguess,phitol=phitol,thinCutoff=thinCutoff,npoints=npoints,)

     #H) Potential geometry & inverted view with φ0
    example_H_potential_geometry(inst, profile, save_dir=save_dir, tag=tag,)

    # I) Local quadratic at φ0
    example_I_local_quadratic_at_phi0(inst, profile, save_dir=save_dir, tag=tag,)

    # J) Inverted potential with path, and V(φ) with points
    example_J_inverted_path(inst, profile, save_dir=save_dir, tag=tag,)

    # K) φ(r) with interior shading and markers
    example_K_phi_of_r(inst, profile, save_dir=save_dir, tag=tag,)

    # L) 2D spherical visualizations (Cartesian & polar)
    example_L_spherical_maps(inst, profile, save_dir=save_dir, tag=tag,)

    # M) ODE terms decomposition along the profile
    example_M_ode_terms(inst, profile, save_dir=save_dir, tag=tag,)

    # N) Action and β proxies (print β; only action is plotted)
    example_N_action_and_beta(inst, profile, save_dir=save_dir, tag=tag,)




    # consolidated table
    di = gather_diagnostics(inst, profile, label=label, transition_summary=summary)
    save_diagnostics_summary(di, save_dir, basename=f"diagnostics_summary_C_{C}", fmt="json")

    print("=== Showcase complete. ===")
    if save_dir:
        print(f"Figures saved under: {os.path.abspath(save_dir)}")


# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------
"""

if __name__ == "__main__":
    C = 3.83
    #run paper potential
    run_all(
        case="paper",
        C=C,
        Lambda=1000.0,
        finiteT=True,
        include_daisy=True,
        xguess=None,
        phitol=1e-5,
        npoints=800,
        thinCutoff=0.0001,
        phi_scan_range = None,
        T_min=5.0,
        T_max=200.0,
        n_T_seeds=2,
        nuclCriterion=None,
        Tn_Ttol= 1e-3,
        Tn_maxiter= 80,
        save_dir=f"results_C_{C}",        # "results"
    )



C_list = [3.65, 3.75, 3.83]

for C in C_list:
    run_all(
        case="paper",
        C=C,
        Lambda=1000.0,
        finiteT=True,
        include_daisy=True,
        xguess=None,
        phitol=1e-5,
        npoints=800,
        thinCutoff=0.0001,
        phi_scan_range = None,
        T_min=5.0,
        T_max=200.0,
        n_T_seeds=2,
        nuclCriterion=None,
        Tn_Ttol= 1e-3,
        Tn_maxiter= 100,
        save_dir=f"results_C_{C}",        # "results"
    )
"""