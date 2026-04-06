"""
Effective_Potential_OPT_cleaned.py
==================================

Cleaned end-to-end finite-temperature showcase for comparing two effective
potentials built from the same underlying model parameters:

1. Daisy-resummed SM-like finite-temperature potential.
2. OPT on-shell effective potential imported from ``OPT.py``.

Design goals
------------
- keep a single high-level entry point ``run_all(...)``;
- expose the physically relevant inputs directly at the top level;
- make the Daisy/OPT switch explicit through ``potential_kind`` instead of the
  old overloaded ``include_daisy`` boolean;
- keep the same tree-level sector for both methods by deriving the OPT
  parameters internally from the same common masses/vev;
- define the working potential used by transitionFinder and the bounce solver
  as a function of ``phi`` and ``T`` only, while the OPT backend solves the gap
  equation internally for every requested ``(phi, T)``.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from CosmoTransitions import SingleFieldInstanton
from CosmoTransitions import deriv14
from CosmoTransitions import Jb, Jf
from CosmoTransitions.gravitational_Waves import (
        lisa_sensitivity_s_pis,
        decigo_sensitivity_s_pis,
        bbo_sensitivity_s_pis,
    )
from CosmoTransitions.generic_potential import (
        build_finite_T_derivatives,
        ensure_dir,
        savefig,
        build_phi_grid,
        _extract_params_from_V,
        _build_phases_and_transitions,
        _spinodal_data_for_phase,
        _closest_spinodal_to_T,
        _build_gw_calculator_from_summary,
        _compute_gw_scales_from_calculator,
        compute_profile,
        gather_diagnostics,
        save_diagnostics_summary,
    )

from CosmoTransitions.OPT import (
        OPTModelParams,
        ThermalOptions,
        SolverOptions as OPTSolverOptions,
        veff_on_shell,
    )


np.set_printoptions(precision=6, suppress=True)

PotentialKind = Literal["daisy", "opt"]


# -----------------------------------------------------------------------------
# Shared model inputs
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CommonModelParameters:
    """
    Common physical input shared by the Daisy and OPT descriptions.

    Parameters
    ----------
    vev, mh, mw, mz, mt
        Vacuum expectation value and particle masses in GeV used to define the
        tree-level sector and the SM-like thermal/Daisy terms.
    mu
        Chemical potential entering the common tree-level quadratic term.
    opt_renorm_scale
        Renormalization scale passed to the OPT module. If omitted, ``vev`` is
        used.
    """

    vev: float = 246.0
    mh: float = 80
    mw: float = 80.36
    mz: float = 91.19
    mt: float = 173.1
    mu: float = 0.5
    opt_renorm_scale: float | None = None

    @property
    def opt_m2(self) -> float:
        # Match V_tree = (mh^2 / 8 v^2) (phi^2 - v^2)^2
        return -0.5 * self.mh**2

    @property
    def opt_lam(self) -> float:
        return 3.0 * self.mh**2 / self.vev**2

    @property
    def opt_M(self) -> float:
        return float(self.vev if self.opt_renorm_scale is None else self.opt_renorm_scale)

    @property
    def gW(self) -> float:
        return self.mw / self.vev

    @property
    def gZ(self) -> float:
        return self.mz / self.vev

    @property
    def gt(self) -> float:
        return self.mt / self.vev


@dataclass(frozen=True)
class PotentialComparisonConfig:
    """Controls for the comparison-only plots (Examples A and B)."""

    phi_max: float = 300.0
    n_phi: int = 1201
    inset_phi_max: float = 80.0
    compare_methods: bool = True


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
_64pi2 = 64.0 * np.pi**2
_VEW = 246.0


# -----------------------------------------------------------------------------
# Common parameter mapping between Daisy and OPT
# -----------------------------------------------------------------------------
def build_opt_model_params(common: CommonModelParameters) -> OPTModelParams:
    """
    Map the common Higgs-like input to the OPT tree-level convention.

    Both descriptions therefore share the same tree-level potential written as

        V_tree(phi) = 1/2 (m^2 - mu^2) phi^2 + lambda/24 phi^4 + const.
    """
    return OPTModelParams(
        m2=float(common.opt_m2),
        lam=float(common.opt_lam),
        M=float(common.opt_M),
    )


# -----------------------------------------------------------------------------
# Daisy potential built from the same common parameters
# -----------------------------------------------------------------------------
def _daisy_tree_potential(phi: np.ndarray, common: CommonModelParameters) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    return (common.mh**2) / (8.0 * common.vev**2) * (phi**2 - common.vev**2)**2

def _cw_like_zeroT_loops(phi: np.ndarray, common: CommonModelParameters) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_phi = np.where(phi != 0.0, np.log((phi**2) / (common.vev**2)), 0.0)

    pref = 1.0 / _64pi2
    coeff = 6.0 * common.gW**4 + 3.0 * common.gZ**4 - 12.0 * common.gt**4
    bracket = phi**4 * (log_phi - 1.5) + 2.0 * phi**2 * common.vev**2
    return np.real_if_close(pref * coeff * bracket)


def V_daisy_sm_like(
    phi: np.ndarray | float,
    T: np.ndarray | float | None = None,
    *,
    finiteT: bool = True,
    common: CommonModelParameters,
    include_daisy_resummation: bool = True,
) -> np.ndarray:
    """
    Daisy-resummed SM-like benchmark built from the same common inputs used by
    the OPT mapping.

    Important
    ---------
    The chemical potential enters the common tree sector through

        1/2 (m^2 - mu^2) phi^2,

    so the Daisy and OPT branches are compared with the same quadratic and
    quartic terms. The thermal approximation is what changes, not the chosen
    benchmark parameters.
    """
    phi = np.asarray(phi, dtype=float)
    V0 = np.real_if_close(_daisy_tree_potential(phi, common) + _cw_like_zeroT_loops(phi, common))

    if not finiteT:
        return V0
    if T is None:
        raise ValueError("V_daisy_sm_like: finiteT=True requires T.")

    T_arr = np.asarray(T, dtype=float)
    phi, T_arr = np.broadcast_arrays(phi, T_arr)

    absphi = np.abs(phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        xW = (common.gW * absphi) / T_arr
        xZ = (common.gZ * absphi) / T_arr
        xt = (common.gt * absphi) / T_arr

    DV_b = (T_arr**4) / (2.0 * np.pi**2) * (
        6.0 * Jb(xW, approx="exact") + 3.0 * Jb(xZ, approx="exact")
    )
    DV_f = (T_arr**4) / (2.0 * np.pi**2) * (12.0 * Jf(xt, approx="exact"))

    DV_daisy = 0.0
    if include_daisy_resummation:
        g2 = 4.0 * (common.mw**2 + common.mz**2) / (3.0 * common.vev**2)
        mL2_phi = 0.25 * g2 * absphi**2
        mL2_T = mL2_phi + (11.0 / 6.0) * g2 * T_arr**2
        mL_phi = np.sqrt(np.maximum(mL2_phi, 0.0))
        mL_T = np.sqrt(np.maximum(mL2_T, 0.0))
        DV_daisy = -(T_arr / (12.0 * np.pi)) * 3.0 * (mL_T**3 - mL_phi**3)

    return np.real_if_close(V0 + DV_b + DV_f + DV_daisy)


# -----------------------------------------------------------------------------
# OPT backend imported from OPT.py and wrapped as V(phi, T)
# -----------------------------------------------------------------------------
class OPTOnShellBackend:
    """
    Thin wrapper around ``OPT.py`` that exposes a cached on-shell potential as
    a function of ``(phi, T)`` only.
    """

    def __init__(
        self,
        common: CommonModelParameters,
        thermal: ThermalOptions,
        solver: OPTSolverOptions,
        *,
        cache_round_phi: int = 10,
        cache_round_T: int = 10,
    ) -> None:
        self.common = common
        self.mu = float(common.mu)
        self.params = build_opt_model_params(common)
        self.thermal = thermal
        self.solver = solver
        self.cache_round_phi = int(cache_round_phi)
        self.cache_round_T = int(cache_round_T)
        self._cache: dict[tuple[float, float], float] = {}

    def _key(self, phi: float, T: float) -> tuple[float, float]:
        return (
            round(float(phi), self.cache_round_phi),
            round(float(T), self.cache_round_T),
        )

    def scalar(self, phi: float, T: float) -> float:
        phi = float(phi)
        T = float(T)
        if T <= 0.0:
            raise ValueError(
                "OPTOnShellBackend works with the finite-temperature on-shell OPT "
                f"potential and therefore requires T > 0. Got T={T}."
            )
        key = self._key(phi, T)
        if key not in self._cache:
            veff, _ = veff_on_shell(
                phi=phi,
                T=T,
                mu=self.mu,
                params=self.params,
                thermal=self.thermal,
                solver=self.solver,
            )
            self._cache[key] = float(veff)
        return self._cache[key]

    def __call__(
        self,
        phi: np.ndarray | float,
        T: np.ndarray | float | None = None,
        *,
        finiteT: bool = True,
    ) -> np.ndarray:
        if not finiteT:
            raise ValueError(
                "The imported OPT module in this workflow is used as a finite-temperature "
                "on-shell potential. Use finiteT=True."
            )
        if T is None:
            raise ValueError("OPTOnShellBackend: finiteT=True requires T.")

        phi_arr = np.asarray(phi, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        phi_arr, T_arr = np.broadcast_arrays(phi_arr, T_arr)

        out = np.empty_like(phi_arr, dtype=float)
        for idx in np.ndindex(phi_arr.shape):
            out[idx] = self.scalar(float(phi_arr[idx]), float(T_arr[idx]))
        return out


class DaisyBackend:
    def __init__(self, common: CommonModelParameters, *, include_daisy_resummation: bool = True) -> None:
        self.common = common
        self.include_daisy_resummation = bool(include_daisy_resummation)

    def __call__(
        self,
        phi: np.ndarray | float,
        T: np.ndarray | float | None = None,
        *,
        finiteT: bool = True,
    ) -> np.ndarray:
        return V_daisy_sm_like(
            phi,
            T,
            finiteT=finiteT,
            common=self.common,
            include_daisy_resummation=self.include_daisy_resummation,
        )


def build_potential_backend(
    potential_kind: PotentialKind,
    common: CommonModelParameters,
    *,
    opt_thermal: ThermalOptions,
    opt_solver: OPTSolverOptions,
    include_daisy_resummation: bool = True,
) -> Callable:
    if potential_kind == "daisy":
        return DaisyBackend(common, include_daisy_resummation=include_daisy_resummation)
    if potential_kind == "opt":
        return OPTOnShellBackend(common, opt_thermal, opt_solver)
    raise ValueError(f"Unknown potential_kind={potential_kind!r}. Use 'daisy' or 'opt'.")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def make_inst(
    V_phi_only: Callable[[np.ndarray | float], np.ndarray],
    *,
    alpha: int = 2,
    phi_abs: float,
    phi_meta: float,
) -> tuple[SingleFieldInstanton, str]:
    inst = SingleFieldInstanton(
        phi_absMin=float(phi_abs),
        phi_metaMin=float(phi_meta),
        V=V_phi_only,
        alpha=alpha,
        phi_eps=1e-3,
    )
    return inst, "effective_potential_"


def _deltaV_curve(VphiT: Callable, phi_grid: np.ndarray, T: float) -> np.ndarray:
    vals = np.asarray(VphiT(phi_grid, T=T, finiteT=True), dtype=float)
    v0 = float(np.asarray(VphiT(0.0, T=T, finiteT=True)))
    return vals - v0


def _grid_minimum(phi_grid: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmin(values))
    return float(phi_grid[idx]), float(values[idx])


def _characteristic_temperatures_for_comparison(summary: Mapping[str, Any]) -> list[float]:
    key_T = summary.get("key_temperatures", {}) or {}
    Tn = key_T.get("Tn", np.nan)
    Tc = key_T.get("Tc", np.nan)

    temps: list[float] = []
    if np.isfinite(Tn):
        temps.append(max(0.2, 0.95 * float(Tn)))
        temps.append(float(Tn))
    if np.isfinite(Tc):
        temps.append(float(Tc))
        temps.append(1.05 * float(Tc))
    elif np.isfinite(Tn):
        temps.append(1.10 * float(Tn))

    if not temps:
        return [20.0, 60.0, 100.0]

    out = []
    for T in temps:
        if T > 0.0 and all(abs(T - old) > 1e-10 for old in out):
            out.append(float(T))
    return out


# -----------------------------------------------------------------------------
# Comparison plots replacing the legacy examples A and B
# -----------------------------------------------------------------------------
def example_A_compare_daisy_and_opt_at_key_temperatures(
    summary: Mapping[str, Any],
    *,
    common: CommonModelParameters,
    opt_thermal: ThermalOptions,
    opt_solver: OPTSolverOptions,
    comparison: PotentialComparisonConfig,
    save_dir: Optional[str] = None,
    tag: str = "",
) -> None:
    """
    Example A.

    Compare ``Delta V(phi, T) = V(phi, T) - V(0, T)`` for Daisy and OPT at a
    small set of characteristic temperatures extracted from the currently traced
    transition. This replaces the old legacy-potential figure.
    """
    if not comparison.compare_methods:
        return

    daisy = build_potential_backend(
        "daisy", common, opt_thermal=opt_thermal, opt_solver=opt_solver, include_daisy_resummation=True
    )
    opt = build_potential_backend(
        "opt", common, opt_thermal=opt_thermal, opt_solver=opt_solver, include_daisy_resummation=True
    )

    T_list = _characteristic_temperatures_for_comparison(summary)
    phi_grid = np.linspace(-comparison.phi_max, comparison.phi_max, int(comparison.n_phi))

    fig, axes = plt.subplots(1, len(T_list), figsize=(5.4 * len(T_list), 4.6), squeeze=False)
    axes = axes.ravel()

    for ax, T in zip(axes, T_list):
        d_daisy = _deltaV_curve(daisy, phi_grid, T)
        d_opt = _deltaV_curve(opt, phi_grid, T)
        phi_daisy, v_daisy = _grid_minimum(phi_grid, d_daisy)
        phi_opt, v_opt = _grid_minimum(phi_grid, d_opt)

        ax.plot(phi_grid, d_daisy, lw=2.0, label="Daisy")
        ax.plot(phi_grid, d_opt, lw=2.0, ls="--", label="OPT")
        ax.scatter([phi_daisy], [v_daisy], s=36, zorder=5)
        ax.scatter([phi_opt], [v_opt], s=36, zorder=5)
        ax.axhline(0.0, color="#666", lw=1.0, ls=":")
        ax.set_title(rf"$T={T:.3g}$")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\Delta V(\phi,T)$")
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False)
    fig.suptitle("Example A – Daisy vs OPT at characteristic temperatures", y=1.02)
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figA{suffix}")
    plt.close(fig)


def example_B_compare_daisy_and_opt_with_inset(
    summary: Mapping[str, Any],
    *,
    common: CommonModelParameters,
    opt_thermal: ThermalOptions,
    opt_solver: OPTSolverOptions,
    comparison: PotentialComparisonConfig,
    save_dir: Optional[str] = None,
    tag: str = "",
) -> None:
    """
    Example B.

    Side-by-side Daisy/OPT comparison at a representative near-transition
    temperature with a zoomed inset around the origin/minima region.
    """
    if not comparison.compare_methods:
        return

    daisy = build_potential_backend(
        "daisy", common, opt_thermal=opt_thermal, opt_solver=opt_solver, include_daisy_resummation=True
    )
    opt = build_potential_backend(
        "opt", common, opt_thermal=opt_thermal, opt_solver=opt_solver, include_daisy_resummation=True
    )

    key_T = summary.get("key_temperatures", {}) or {}
    T_ref = float(key_T.get("Tn", np.nan))
    if not np.isfinite(T_ref):
        Tc = key_T.get("Tc", np.nan)
        T_ref = float(Tc) if np.isfinite(Tc) else 60.0

    phi_grid = np.linspace(-comparison.phi_max, comparison.phi_max, int(comparison.n_phi))
    curves = {
        "Daisy": _deltaV_curve(daisy, phi_grid, T_ref),
        "OPT": _deltaV_curve(opt, phi_grid, T_ref),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for ax, (label, values) in zip(axes, curves.items()):
        phi_star, V_star = _grid_minimum(phi_grid, values)
        ax.plot(phi_grid, values, lw=2.2)
        ax.scatter([phi_star], [V_star], s=42, zorder=5, label=f"grid minimum ({phi_star:.3g})")
        ax.axhline(0.0, color="#666", lw=1.0, ls=":")
        ax.set_title(f"{label} at T={T_ref:.4g}")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\Delta V(\phi,T)$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")

        axins = ax.inset_axes([0.08, 0.52, 0.38, 0.38])
        mask = np.abs(phi_grid) <= comparison.inset_phi_max
        x_zoom = phi_grid[mask]
        y_zoom = values[mask]
        axins.plot(x_zoom, y_zoom, lw=1.5)
        y0, y1 = float(np.nanmin(y_zoom)), float(np.nanmax(y_zoom))
        pad = 0.05 * max(1e-12, y1 - y0)
        axins.set_xlim(-comparison.inset_phi_max, comparison.inset_phi_max)
        axins.set_ylim(y0 - pad, y1 + pad)
        axins.grid(True, alpha=0.2)
        axins.tick_params(labelsize=8)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="#666", lw=1.0, alpha=0.85)

    fig.suptitle("Example B – Daisy/OPT near-transition potential comparison", y=1.02)
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figB{suffix}")
    plt.close(fig)

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
    deltaX_target: float = 0.05,
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
        deltaX_target= deltaX_target,
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
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figC{suffix}")
    plt.close()

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
    savefig(fig, save_dir, f"figD{suffix}")
    plt.close()

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
    savefig(fig_full, save_dir, f"figD_full{suffix}")
    plt.close()

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
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figE{suffix}")
    plt.close()
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
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figF{suffix}")
    plt.close()

    print(f"[Example F ] Saved figure: {tag}")
    summary["example_F"] = dict(segments=segments)
    return summary



# -----------------------------------------------------------------------------
# Example G
# -----------------------------------------------------------------------------
def example_G_T_scan(
    summary: Dict[str, Any],
    *,
    phi_max: float = 300.0,
    VphiT: Callable,
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
    T_spin_high = key_T.get("T_spinodal_high_phase", None)
    T_spin_low = key_T.get("T_spinodal_low_phase", None)

    T_list: list[float] = []

    if np.isfinite(T_n):
        T_list = [0.1, T_spin_high, T_n, T_c, T_spin_low]

    else:
        # fallback
        T_list = [60.0, 80.0, 100.0, 120.0]

    # sort
    T_list = sorted({float(T) for T in T_list if T > 0.0})

    phi_cap = np.inf
    phi_max_eff = phi_max

    ϕ = np.linspace(0.0, phi_max_eff, 2200)

    # --- helper: quick broken-minimum locator at this T (grid search) ---
    def _phi_true_at_T(T: float) -> float:
        grid = np.linspace(0.0, phi_max_eff, 3001)
        Vg = VphiT(grid, T=T, finiteT=True)  # include_daisy is already baked in VphiT closure
        idx = np.nanargmin(Vg)
        return float(grid[idx])

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_list)))

    print("\n[G] Temperature sweep diagnostics (0,T_spin_hight, T_n, T_c, T_spin_low)")
    print("    T [GeV]   phi_true(T) [GeV]   V(phi_true,T)-V(0,T) [GeV^4]   V(v,T)-V(0,T) [GeV^4]")
    print("    --------  -------------------  ------------------------------  ----------------------")

    for T, col in zip(T_list, colors):
        VϕT = VphiT(ϕ, T=T, finiteT=True)
        V0T = float(VphiT(0.0, T=T, finiteT=True))
        VvT = float(VphiT(v, T=T, finiteT=True))

        if shift_ref == "V0":
            Vshift = VϕT - V0T
        else:  # 'Vv'
            Vshift = VϕT - VvT

        lw, ls, z = (2.2, "-", 3) if T in (min(T_list), max(T_list)) else (1.8, "-", 2)
        ax.plot(ϕ, Vshift, color=col, lw=lw, ls=ls, label=f"T = {T:g} GeV", zorder=z)

        phi_true_T = _phi_true_at_T(T)
        V_true_T = float(VphiT(phi_true_T, T=T, finiteT=True))
        print(f"    {T:7.1f}  {phi_true_T:19.3f}  {V_true_T - V0T:30.3e}  {VvT - V0T:22.3e}")

    ax.axhline(0.0, color="#444", lw=1.0, ls="--", label="shift reference")
    ax.axvline(v,   color="#888", lw=1.0, ls=":",  label="ϕ = v")

    ylabel = r"$V(\phi,T)-V(0,T)$" if shift_ref == "V0" else r"$V(\phi,T)-V(v,T)$"
    ax.set_xlim(0.0, phi_max_eff)
    ax.set_xlabel(r"$\phi$ [GeV]"); ax.set_ylabel(ylabel + r"  [GeV$^4$]")
    ax.set_title(fr"Temperature sweep at $....algo$ GeV")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout();

    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figG{suffix}")
    plt.close()
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
    print("\n[H] Potential geometry & scales")
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
    savefig(fig2, save_dir, f"figH2{suffix}")
    plt.close()

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

    print("\n[I] Initial local data at r0")
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
    plt.tight_layout();
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figI{suffix}")
    plt.close()


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
    suffix = f"_{tag}" if tag else ""
    savefig(fig1, save_dir, f"figJ{suffix}")
    plt.close()

# -----------------------------------------------------------------------------
# Example K
# -----------------------------------------------------------------------------
def example_K_phi_of_r(inst: SingleFieldInstanton,
                       profile,
                       save_dir: Optional[str] = None, tag: str = ""):
    """
    K) Plot φ(r) highlighting the starting point (r0, φ0); shade the interior
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
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figK{suffix}")
    plt.close()

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

    print("\n[L] t=0 visualization")
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
    suffix = f"_{tag}" if tag else ""
    savefig(fig1, save_dir, f"figL1{suffix}")
    plt.close()

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
    suffix = f"_{tag}" if tag else ""
    savefig(fig2, save_dir, f"figL2{suffix}")
    plt.close()

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

    print("\n[M] False → True vacuum potential drop")
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
    ax.legend(lines1+lines2, labels1+labels2, loc="lower right", ncol=2)
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figM{suffix}")
    plt.close()

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
    print("\n[N] Action and β proxies")
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

    plt.tight_layout();
    suffix = f"_{tag}" if tag else ""
    savefig(fig, save_dir, f"figN{suffix}")
    plt.close()


# -----------------------------------------------------------------------------
# Example O – Gravitational-wave spectrum at T_* ≈ T_n
# -----------------------------------------------------------------------------

def example_O_gravitational_wave_spectrum(
    summary: Dict[str, Any],
    *,
    g_star: float = 106.75,
    v_w: float = 1.0,
    dT_fraction: float = 0.02,
    n_freq: int = 400,
    save_dir: str = "figs",
    tag: str = "",
) -> Dict[str, Any]:
    """
    Example O (finite T + GWs):

    - Use GravitationalWaveCalculator to extract:
        T_* (taken as T_n), α(T_*), β/H_*, R_* H_*, Γ(T_*),
        and peak frequencies for sound waves / turbulence / collisions.
    - Build the GW spectrum h² Ω(f) over f ∈ [10^{-4}, 10^{5}] Hz.
    - Make two log–log plots:
        (1) components (sw, turb, coll) + total;
        (2) total only with vertical lines at the peak frequencies.
    - Cache the scales and spectra back into `summary` under keys:
        "gw_scales", "gw_frequency_Hz", "gw_spectra_h2".
    """
    suffix = f"_{tag}" if tag else ""

    gw_calc = _build_gw_calculator_from_summary(summary)
    if gw_calc is None:
        print("\n[Example O] Could not build GravitationalWaveCalculator "
              "from the transition summary – skipping GW spectrum.")
        return summary

    # ------------------------------------------------------------------
    # Thermodynamic / geometric GW scales at T_* (≈ T_n)
    # ------------------------------------------------------------------
    gw_scales = _compute_gw_scales_from_calculator(
        gw_calc,
        summary,
        g_star=g_star,
        v_w=v_w,
        dT_fraction=dT_fraction,
    )
    summary["gw_scales"] = gw_scales  # cache for later use (e.g. diagnostics)

    T_star = gw_scales["gw_T_star_GeV"]
    alpha_val = gw_scales["gw_alpha"]
    beta_over_H = gw_scales["gw_beta_over_H"]
    R_star_H = gw_scales["gw_R_star_times_H"]
    Gamma_val = gw_scales["gw_nucleation_rate_GeV4"]
    dT_used = gw_scales["gw_dT_for_beta_GeV"]
    f_sw_peak = gw_scales["gw_f_sw_peak_Hz"]
    f_turb_peak = gw_scales["gw_f_turb_peak_Hz"]
    f_coll_peak = gw_scales["gw_f_coll_peak_Hz"]

    print("\n" + "=" * 72)
    print("Example O – GW scales from GravitationalWaveCalculator")
    print("=" * 72)
    print(f"  T_* (reference)         = {T_star:10.4g}  GeV")
    print(f"  g_*                     = {g_star:10.4g}")
    print(f"  v_w                     = {v_w:10.4g}")
    print("")
    print(f"  α(T_*)                  = {alpha_val:10.4g}")
    print(f"  β/H_*                   = {beta_over_H:10.4g}")
    print(f"  R_* H_*                 = {R_star_H:10.4g}")
    print(f"  Γ(T_*)                  = {Gamma_val:10.4e}  [GeV^4]")
    print(f"  ΔT used for β/H_*       = {dT_used:10.4g}  GeV")
    print("")
    print(f"  f_sw^peak               = {f_sw_peak:10.4g}  Hz")
    print(f"  f_turb^peak             = {f_turb_peak:10.4g}  Hz")
    print(f"  f_coll^peak             = {f_coll_peak:10.4g}  Hz")

    # ------------------------------------------------------------------
    # GW spectrum on a log-spaced frequency grid
    # ------------------------------------------------------------------
    f_min, f_max = 1e-3, 1e5  # mHz
    f_arr = np.logspace(np.log10(f_min), np.log10(f_max), int(n_freq))

    spectra = gw_calc.omega_total_h2(
        f_arr,
        alpha=alpha_val,
        beta_over_H=beta_over_H,
        T_star=T_star,
        g_star=g_star,
        v_w=v_w,
        include_sw=True,
        include_turb=True,
        include_coll=True,
        epsilon_turb = 1
    )

    omega_sw = spectra["sw"]
    omega_turb = spectra["turb"]
    omega_coll = spectra["coll"]
    omega_tot = spectra["total"]

    print(f"  omega_sw_peak             = {np.max(omega_sw):10.4g} ")
    print(f"  omega_turb_peak             = {np.max(omega_turb):10.4g} ")
    print(f"  omega_coll_peak             = {np.max(omega_coll):10.4g} ")
    print(f"  omega_tot_peak             = {np.max(omega_tot):10.4g} ")

    summary["gw_frequency_Hz"] = f_arr
    summary["gw_spectra_h2"] = spectra

    summary["omega_sw_peak"] = np.max(omega_sw)
    summary["omega_turb_peak"] = np.max(omega_turb)
    summary["omega_coll_peak"] = np.max(omega_coll)
    summary["omega_tot_peak"] = np.max(omega_tot)

    # Helper to avoid log(0)
    def _has_signal(arr: np.ndarray) -> bool:
        return np.any(arr > 0.0)

    # Detector PIS curves (s-channel), f in mHz
    omega_LISA = lisa_sensitivity_s_pis(f_arr )
    omega_DECIGO = decigo_sensitivity_s_pis(f_arr)
    omega_BBO = bbo_sensitivity_s_pis(f_arr)

    # ------------------------------------------------------------------
    # Figure O1: components + total (log–log)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    if _has_signal(omega_sw):
        ax1.plot(f_arr, omega_sw, lw=1.8, label=r"sound waves")
    if _has_signal(omega_turb):
        ax1.plot(f_arr, omega_turb, lw=1.8, label=r"MHD turbulence")
    if _has_signal(omega_coll):
        ax1.plot(f_arr, omega_coll, lw=1.8, label=r"bubble collisions")
    if _has_signal(omega_tot):
        ax1.plot(f_arr, omega_tot, lw=2.2, label=r"total", linestyle="--")

    ax1.set_xlabel(r"$f$  [mHz]")
    ax1.set_ylabel(r"$h^2 \Omega_{\rm GW}(f)$")
    ax1.set_title("Example O – GW spectrum components (log–log)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    fig1.tight_layout()
    savefig(fig1, save_dir, f"figO1{suffix}")
    plt.close()

    # ------------------------------------------------------------------
    # Figure O2: total only + vertical lines at peak frequencies
    # ------------------------------------------------------------------
    spectra2 = gw_calc.omega_total_h2(
        f_arr,
        alpha=alpha_val,
        beta_over_H=beta_over_H,
        T_star=T_star,
        g_star=g_star,
        v_w=v_w,
        include_sw=True,
        include_turb=True,
        include_coll=True,
        epsilon_turb = 0
    )
    omega_tot2 = spectra2["total"]
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    #  LISA red, DECIGO orange, BBO yellow
    ax2.plot(f_arr, omega_LISA, color="red",    lw=1.8, label="LISA")
    ax2.plot(f_arr, omega_DECIGO, color="orange", lw=1.8, label="DECIGO")
    ax2.plot(f_arr, omega_BBO, color="yellow",  lw=1.8, label="BBO")

    if _has_signal(omega_tot2):
        ax2.plot(f_arr, omega_tot2, lw=2.2,color="purple", label=r"total, $\epsilon_{\rm turb}=0$")

    if _has_signal(omega_tot):
        ax2.plot(f_arr, omega_tot, lw=2.2, linestyle="--", color="purple", label=r"total, $\epsilon_{\rm turb}=1$",)

    # fill between in purple (only where both > 0)
    mask = (omega_tot > 0.0) & (omega_tot2 > 0.0)

    ax2.fill_between( f_arr[mask],omega_tot[mask], omega_tot2[mask], color="purple", alpha=0.15,)

    # Mark peaks with vertical dashed lines (no labels)
    for f_peak in (f_sw_peak, f_turb_peak, f_coll_peak):
        if np.isfinite(f_peak) and f_min < f_peak < f_max:
            ax2.axvline(f_peak, linestyle=":", linewidth=1.2)

    ax2.set_xlabel(r"$f$  [mHz]")
    ax2.set_ylabel(r"$h^2 \Omega_{\rm GW}(f)$")
    ax2.set_title("Example O – total GW spectrum (log–log)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    fig2.tight_layout()
    savefig(fig2, save_dir, f"figO2{suffix}")
    plt.close()
    print(f"[Example O] Saved figures figO1{suffix}.png and figO2{suffix}.png")
    return summary




def _normalize_potential_kind(
    potential_kind: str | None,
    include_daisy: bool | None,
) -> PotentialKind:
    if potential_kind is not None:
        kind = potential_kind.lower().strip()
        if kind not in {"daisy", "opt"}:
            raise ValueError(
                f"Unknown potential_kind={potential_kind!r}. Use 'daisy' or 'opt'."
            )
        return kind  # type: ignore[return-value]

    if include_daisy is None:
        return "opt"
    return "daisy" if include_daisy else "opt"



def run_all(
    *,
    potential_kind: Literal["daisy", "opt"] | None = "opt",
    include_daisy: bool | None = None,
    xguess: Optional[float] = None,
    phitol: float = 1e-5,
    save_dir: Optional[str] = None,
    phi_scan_range: Optional[Tuple[float, float]] = None,
    thinCutoff: float = 0.01,
    deltaX_target: float = 0.05,
    finiteT: bool = True,
    npoints: int = 800,
    T_min: float = 5.0,
    T_max: float = 200.0,
    n_T_seeds: int = 2,
    nuclCriterion: Callable[[float, float], float] | None = None,
    Tn_Ttol: float = 1e-3,
    Tn_maxiter: int = 80,
    # shared model input
    vev: float = 246.0,
    mh: float = 50.0,
    mw: float = 80.36,
    mz: float = 91.19,
    mt: float = 173.1,
    mu: float = 0.5,
    opt_M: float | None = None,
    # Daisy options
    include_daisy_resummation: bool = True,
    # imported OPT.py options
    opt_mode_mu0: Literal["auto", "notebook_highT", "ct_backend"] = "auto",
    opt_mode_muneq0: Literal["notebook_highT"] = "notebook_highT",
    opt_ct_mu0_approx: Literal["exact", "high"] = "exact",
    opt_ct_mu0_n: int = 20,
    opt_root_tol: float = 1e-12,
    opt_max_iter: int = 300,
    opt_continuation: bool = True,
    # comparison plots A/B
    compare_methods: bool = True,
    compare_phi_max: float = 300.0,
    compare_n_phi: int = 1201,
    compare_inset_phi_max: float = 80.0,
) -> Dict[str, Any]:
    """
    Execute the cleaned finite-temperature showcase for one selected backend.

    Parameters
    ----------
    potential_kind
        Explicit choice between the Daisy benchmark and the imported OPT
        effective potential.
    include_daisy
        Backward-compatible alias of the legacy interface. If provided, it is
        translated internally as
            True  -> potential_kind='daisy'
            False -> potential_kind='opt'.
    vev, mh, mw, mz, mt, mu, opt_M
        Common physical inputs from which both Daisy and OPT are built.
    opt_*
        Numerical/thermal controls forwarded to the imported ``OPT.py`` module.
    compare_*
        Controls for the new Examples A and B, which directly compare Daisy and
        OPT using the same common parameter point.
    """
    if not finiteT:
        raise ValueError(
            "This showcase is organized around finite-temperature transitionFinder, "
            "bounce, and GW diagnostics. Use finiteT=True."
        )

    kind = _normalize_potential_kind(potential_kind, include_daisy)
    save_dir = ensure_dir(save_dir)

    common = CommonModelParameters(
        vev=vev,
        mh=mh,
        mw=mw,
        mz=mz,
        mt=mt,
        mu=mu,
        opt_renorm_scale=opt_M,
    )
    opt_thermal = ThermalOptions(
        mode_mu0=opt_mode_mu0,
        mode_muneq0=opt_mode_muneq0,
        ct_mu0_approx=opt_ct_mu0_approx,
        ct_mu0_n=opt_ct_mu0_n,
    )
    opt_solver = OPTSolverOptions(
        root_tol=opt_root_tol,
        max_iter=opt_max_iter,
        continuation=opt_continuation,
    )
    comparison = PotentialComparisonConfig(
        phi_max=compare_phi_max,
        n_phi=compare_n_phi,
        inset_phi_max=compare_inset_phi_max,
        compare_methods=compare_methods,
    )

    backend = build_potential_backend(
        kind,
        common,
        opt_thermal=opt_thermal,
        opt_solver=opt_solver,
        include_daisy_resummation=include_daisy_resummation,
    )

    def V_model(phi, T=None, finiteT: bool = True):
        return backend(phi, T=T, finiteT=finiteT)

    def V_model_XT(X, T, finiteT: bool = True):
        X = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return V_model(phi, T=T, finiteT=finiteT)

    derivs = build_finite_T_derivatives(
        Vtot=V_model_XT,
        Ndim=1,
        x_eps=1e-3,
        T_eps=1e-2,
        deriv_order=4,
    )
    V_XT = derivs.V
    gradV_XT = derivs.gradV
    hessV_XT = derivs.hessV
    dVdT_XT = derivs.dV_dT
    dgradT_XT = derivs.dgradV_dT

    tag = f"{kind}_mu{common.mu:g}"
    print(f"=== Running cleaned showcase with backend: {kind.upper()} ===")
    print(
        f"Shared parameters: v={common.vev}, mh={common.mh}, mw={common.mw}, "
        f"mz={common.mz}, mt={common.mt}, mu={common.mu}, M_OPT={common.opt_M}"
    )

    if phi_scan_range is None:
        phi_low_scan, phi_high_scan = 0.0, common.vev
    else:
        phi_low_scan, phi_high_scan = phi_scan_range

    summary = example_C_transition_summary(
        V_XT=V_XT,
        dVdphi_XT=gradV_XT,
        hessV_XT=hessV_XT,
        dVdT_XT=dVdT_XT,
        dgradT_XT=dgradT_XT,
        deltaX_target=deltaX_target,
        T_min=T_min,
        T_max=T_max,
        phi_range=(phi_low_scan, phi_high_scan),
        n_T_seeds=n_T_seeds,
        nuclCriterion=nuclCriterion,
        Tn_Ttol=Tn_Ttol,
        Tn_maxiter=Tn_maxiter,
        save_dir=save_dir,
        tag=tag,
    )

    summary = example_D_phi_min_and_mass2_vs_T(summary, save_dir=save_dir, tag=tag)
    summary = example_E_deltaV_vs_T(summary, save_dir=save_dir, tag=tag)
    summary = example_F_phase_history_map(summary, save_dir=save_dir, tag=tag)
    summary = example_G_T_scan(
        summary,
        VphiT=V_model,
        phi_max=max(compare_phi_max, common.vev),
        save_dir=save_dir,
        shift_ref="V0",
        tag=tag,
    )


    #example_A_compare_daisy_and_opt_at_key_temperatures(
    #    summary,
    #    common=common,
    #    opt_thermal=opt_thermal,
    #    opt_solver=opt_solver,
    #    comparison=comparison,
    #    save_dir=save_dir,
    #    tag=tag,
    #)
    #example_B_compare_daisy_and_opt_with_inset(
    #    summary,
    #    common=common,
    #    opt_thermal=opt_thermal,
    #    opt_solver=opt_solver,
    #    comparison=comparison,
    #    save_dir=save_dir,
    #    tag=tag,
    #)

    main_transition = summary.get("main_transition")
    phases = summary.get("phases", {})
    key_T = summary.get("key_temperatures", {}) or {}

    if main_transition is None:
        raise RuntimeError(
            "run_all: no first-order transition found in the transition summary. "
            "Cannot define the bounce setup."
        )
    if "Tn" not in key_T:
        raise RuntimeError(
            "run_all: key_temperatures does not contain 'Tn'. "
            "Make sure Example C completed successfully."
        )

    T_n = float(key_T["Tn"])
    if not np.isfinite(T_n):
        raise RuntimeError(
            f"run_all: T_n is non-finite (T_n={T_n}). The nucleation criterion "
            "was likely never satisfied."
        )

    high_phase = phases[main_transition["high_phase"]]
    low_phase = phases[main_transition["low_phase"]]
    phi_meta_Tn = float(np.asarray(high_phase.valAt(T_n), dtype=float).ravel()[0])
    phi_true_Tn = float(np.asarray(low_phase.valAt(T_n), dtype=float).ravel()[0])

    print("\n=== Bounce setup (from transitionFinder) ===")
    print(f"  backend                 = {kind}")
    print(f"  T_n (from transitionFinder) = {T_n:.6g} GeV")
    print(f"  phi_meta(T_n) ≈ {phi_meta_Tn:.6g}")
    print(f"  phi_true(T_n) ≈ {phi_true_Tn:.6g}")

    def make_V_phi_only(T: float = T_n):
        return partial(V_model, T=T, finiteT=True)

    inst, label = make_inst(
        V_phi_only=make_V_phi_only(T_n),
        phi_abs=phi_true_Tn,
        phi_meta=phi_meta_Tn,
    )
    profile = compute_profile(
        inst,
        xguess=xguess,
        phitol=phitol,
        thinCutoff=thinCutoff,
        npoints=npoints,
    )

    example_H_potential_geometry(inst, profile, save_dir=save_dir, tag=tag)
    example_I_local_quadratic_at_phi0(inst, profile, save_dir=save_dir, tag=tag)
    example_J_inverted_path(inst, profile, save_dir=save_dir, tag=tag)
    example_K_phi_of_r(inst, profile, save_dir=save_dir, tag=tag)
    example_L_spherical_maps(inst, profile, save_dir=save_dir, tag=tag)
    example_M_ode_terms(inst, profile, save_dir=save_dir, tag=tag)
    example_N_action_and_beta(inst, profile, save_dir=save_dir, tag=tag)

    summary = example_O_gravitational_wave_spectrum(
        summary,
        g_star=106.75,
        v_w=1.0,
        save_dir=save_dir,
        tag=tag,
    )

    di = gather_diagnostics(inst, profile, label=f"{kind}_{label}", transition_summary=summary)
    save_diagnostics_summary(di, save_dir, basename=f"diagnostics_summary_{tag}", fmt="json")

    print("=== Cleaned showcase complete. ===")
    if save_dir:
        print(f"Figures saved under: {os.path.abspath(save_dir)}")

    return summary


if __name__ == "__main__":
    run_all(
        potential_kind="opt",
        finiteT=True,
        mu=0,
        xguess=None,
        phitol=1e-5,
        npoints=800,
        thinCutoff=1e-4,
        phi_scan_range=None,
        T_min=5.0,
        T_max=90.0,
        deltaX_target=0.05,
        n_T_seeds=2,
        nuclCriterion=None,
        Tn_Ttol=1e-3,
        Tn_maxiter=80,
        save_dir="results_cleaned_daisy_opt_50",
    )
