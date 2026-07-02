# gw_multiC_from_diagnostics.py
# (only the sensitivity part is changed; everything else kept consistent, but translated to English)

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from CosmoTransitions import gw_omega_total_h2


C_LIST_DEFAULT: Sequence[float] = (3.65, 3.75, 3.83)

# Frequency grid (NOTE: mHz, not Hz)
F_MIN = 1e-3   # mHz
F_MAX = 1e5    # mHz
N_FREQ = 800

Y_MIN = 1e-23
Y_MAX = 1e-9


@dataclass
class GWParams:
    """Effective parameters used to build the GW spectrum."""
    alpha: float
    beta_over_H: float
    T_star: float
    g_star: float
    v_w: float
    C_value: float
    Lambda: float


# ---------------------------------------------------------------------------
# I/O helpers: read diagnostics_summary JSON files
# ---------------------------------------------------------------------------

def load_diagnostics_for_C(C: float, base_results_dir: str = ".") -> Dict[str, Any]:
    """
    Load diagnostics_summary_C_<C>.json for a given C.

    Expected path:
        <base_results_dir>/results_C_<C>/diagnostics_summary_C_<C>.json
    """
    folder = os.path.join(base_results_dir, f"results_C_{C}")
    fname = os.path.join(folder, f"diagnostics_summary_C_{C}.json")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Could not find diagnostics file for C={C}: {fname}")
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gw_params(di: Dict[str, Any], C_value: float) -> GWParams:
    """
    Extract (alpha, beta/H, T*, g*, v_w) from the diagnostics dictionary.

    Uses the fields written by your gather_diagnostics routine.
    """
    alpha = float(di.get("gw_alpha", np.nan))
    beta_over_H = float(di.get("gw_beta_over_H", np.nan))

    T_star = float(di.get("gw_T_star_GeV", np.nan))
    if not np.isfinite(T_star):
        T_star = float(di.get("temperature_GeV", np.nan))

    g_star = float(di.get("gw_g_star", np.nan))
    if not np.isfinite(g_star):
        g_star = 106.75

    v_w = float(di.get("gw_v_w", np.nan))
    if not np.isfinite(v_w) or v_w <= 0.0:
        v_w = 1.0

    if not (np.isfinite(alpha) and np.isfinite(beta_over_H) and np.isfinite(T_star)):
        raise RuntimeError(
            f"Invalid GW parameters for C={C_value}: "
            f"alpha={alpha}, beta_over_H={beta_over_H}, T_star={T_star}"
        )

    Lambda = float(di.get("Lambda_GeV", np.nan))

    return GWParams(
        alpha=alpha,
        beta_over_H=beta_over_H,
        T_star=T_star,
        g_star=g_star,
        v_w=v_w,
        C_value=C_value,
        Lambda=Lambda,
    )


# ---------------------------------------------------------------------------
# Detector sensitivity curves (PIS, s-channel) from arXiv:2002.04615
# Frequency convention: f is in mHz, so x_s = f/(1 mHz) = f numerically.
# ---------------------------------------------------------------------------

def _pisc_poly_sum(f_mHz: np.ndarray, terms: list[tuple[float, float]], scale: float) -> np.ndarray:
    """
    Utility: compute  scale * Σ_i c_i * x^{p_i}, where x = f_mHz.

    Parameters
    ----------
    f_mHz : ndarray
        Frequency array in mHz.
    terms : list of (coefficient, power)
        Polynomial-like list in x = f_mHz.
    scale : float
        Overall multiplicative scale (e.g. 1e-14 in the paper).

    Returns
    -------
    ndarray
        PIS curve h^2 Omega_sens,PIS(f) for the chosen detector/channel.
    """
    x = np.asarray(f_mHz, dtype=float)  # x_s = f_s / (1 mHz)
    out = np.zeros_like(x, dtype=float)
    for c, p in terms:
        out += c * x**p
    return scale * out


def lisa_sensitivity_s_pis(f_mHz: np.ndarray) -> np.ndarray:
    """
    LISA PIS (s-channel) sensitivity curve: Eq. (3.18) of arXiv:2002.04615.

    Returns h^2 Omega_sens,PIS(f) as a function of f in mHz.
    """
    terms = [
        (3.58e-3, -4.0),
        (3.26e-1, -3.0),
        (1.20e0, -2.0),
        (2.48e0, -1.0),
        (2.85e-1, +1.0),
        (1.81e-2, +2.0),
        (1.50e-3, +3.0),
    ]
    return _pisc_poly_sum(f_mHz, terms, scale=1e-14)


def decigo_sensitivity_s_pis(f_mHz: np.ndarray) -> np.ndarray:
    """
    DECIGO PIS (s-channel) sensitivity curve: Eq. (3.24) of arXiv:2002.04615.

    Returns h^2 Omega_sens,PIS(f) as a function of f in mHz.
    """
    terms = [
        (3.82e-1, -4.0),
        (2.26e0, -1.5),
        (1.10e-3,  0.0),
        (2.56e-6, +1.0),
        (2.91e-8, +2.0),
        (7.54e-12, +3.0),
    ]
    return _pisc_poly_sum(f_mHz, terms, scale=1e-14)


def bbo_sensitivity_s_pis(f_mHz: np.ndarray) -> np.ndarray:
    """
    BBO PIS (s-channel) sensitivity curve: Eq. (3.30) of arXiv:2002.04615.

    Returns h^2 Omega_sens,PIS(f) as a function of f in mHz.
    """
    terms = [
        (1.77e-1, -4.0),
        (1.06e0, -1.5),
        (1.35e-4,  0.0),
        (2.23e-6, +1.0),
        (1.29e-9, +2.0),
        (2.99e-12, +3.0),
    ]
    return _pisc_poly_sum(f_mHz, terms, scale=1e-14)


# ---------------------------------------------------------------------------
# Main plot: multi-C + LISA/BBO/DECIGO in one figure
# ---------------------------------------------------------------------------

def plot_multiC_spectra_from_diagnostics(
    C_list: Sequence[float] = C_LIST_DEFAULT,
    base_results_dir: str = ".",
    save_dir: str | None = "figs_multiC",
    filename: str = "fig_GW_multiC",
) -> None:
    """
    Single figure with:
      - multi-C h^2 Omega_tot(f) bands (filled between eps_turb=0 and eps_turb=1);
      - PIS sensitivity curves (s-channel) for LISA, DECIGO, BBO from 2002.04615.

    Visual convention:
      - solid line: epsilon_turb = 0
      - dashed line: epsilon_turb = 1
    Frequency convention:
      - f is in mHz everywhere (both signal and sensitivity curves).
    """
    f = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ)  # mHz

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Color map by C value:
    # C=3.83 -> purple, C=3.75 -> blue, C=3.65 -> green
    color_map = {
        3.83: "#9467bd",
        3.75: "#1f77b4",
        3.65: "#2ca02c",
    }

    Lambda_last = np.nan

    for idx, C in enumerate(C_list):
        di = load_diagnostics_for_C(C, base_results_dir=base_results_dir)
        params = extract_gw_params(di, C_value=C)

        alpha_val = params.alpha
        beta_over_H = params.beta_over_H
        T_star = params.T_star
        g_star = params.g_star
        v_w = params.v_w
        Lambda_last = params.Lambda

        spectra_eps0 = gw_omega_total_h2(
            f=f,
            alpha=alpha_val,
            beta_over_H=beta_over_H,
            T_star=T_star,
            g_star=g_star,
            v_w=v_w,
            include_sw=True,
            include_turb=True,
            include_coll=True,
            epsilon_turb=0.0,
        )
        spectra_eps1 = gw_omega_total_h2(
            f=f,
            alpha=alpha_val,
            beta_over_H=beta_over_H,
            T_star=T_star,
            g_star=g_star,
            v_w=v_w,
            include_sw=True,
            include_turb=True,
            include_coll=True,
            epsilon_turb=1.0,
        )

        omega0 = spectra_eps0["total"]  # solid
        omega1 = spectra_eps1["total"]  # dashed

        color = color_map.get(round(C, 2), f"C{idx}")

        # Solid curve (eps=0): keep label
        ax.plot(
            f,
            omega0,
            color=color,
            lw=1.4,
            label=rf"$h^2\Omega_{{\rm tot}}(f),\, C={C:g}$",
        )
        # Dashed curve (eps=1): no label
        ax.plot(
            f,
            omega1,
            color=color,
            lw=1.2,
            ls="--",
        )

        # Filled band
        ax.fill_between(
            f,
            np.minimum(omega0, omega1),
            np.maximum(omega0, omega1),
            color=color,
            alpha=0.18,
        )

        print(
            f"[C={C:g}] alpha={params.alpha:.3g}, "
            f"beta/H*={params.beta_over_H:.3g}, "
            f"T*={params.T_star:.3g} GeV"
        )

    # Detector PIS curves (s-channel), f in mHz
    omega_LISA = lisa_sensitivity_s_pis(f)
    omega_DECIGO = decigo_sensitivity_s_pis(f)
    omega_BBO = bbo_sensitivity_s_pis(f)

    # Colors requested: LISA red, DECIGO orange, BBO yellow
    ax.plot(f, omega_LISA, color="red",    lw=1.8, label="LISA")
    ax.plot(f, omega_DECIGO, color="orange", lw=1.8, label="DECIGO")
    ax.plot(f, omega_BBO, color="yellow",  lw=1.8, label="BBO")

    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_xlabel(r"$f$  [mHz]")
    ax.set_ylabel(r"$h^2 \Omega_{\rm GW}(f)$")
    ax.set_title(fr"Multi-$C$ GW spectra + LISA/DECIGO/BBO (PIS, s-channel)  ($\Lambda={Lambda_last:g}$ GeV)")
    ax.grid(True, which="both", alpha=0.3)

    # Legend: keep only detectors + solid curves (already true: dashed have no label)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lbl in zip(handles, labels):
        if lbl not in unique:
            unique[lbl] = h
    ax.legend(unique.values(), unique.keys(), fontsize=8, loc="center right", ncol=2)

    plt.tight_layout()
    plt.show()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"{filename}.png")
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        print(f"[plot_multiC] Saved figure to: {outpath}")


if __name__ == "__main__":
    plot_multiC_spectra_from_diagnostics(
        C_list=C_LIST_DEFAULT,
        base_results_dir=".",
        save_dir="results",
        filename="fig_GW_multiC_LISA_DECIGO_BBO_PIS_s_channel_mHz",
    )
