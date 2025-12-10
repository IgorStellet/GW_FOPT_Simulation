# gw_multiC_from_diagnostics.py

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
from CosmoTransitions import gw_omega_total_h2


C_LIST_DEFAULT: Sequence[float] = (3.65, 3.75, 3.83)

# Grade de frequências e limites dos eixos
F_MIN = 1e-3   # mHz
F_MAX = 1e5    # mHz
N_FREQ = 800

Y_MIN = 1e-23
Y_MAX = 1e-9


@dataclass
class GWParams:
    """Parâmetros efetivos para o espectro de ondas gravitacionais."""
    alpha: float
    beta_over_H: float
    T_star: float
    g_star: float
    v_w: float
    C_value: float
    Lambda: float


# ---------------------------------------------------------------------------
# I/O helpers: ler os JSONs do diagnostics_summary
# ---------------------------------------------------------------------------

def load_diagnostics_for_C(
    C: float,
    base_results_dir: str = ".",
) -> Dict[str, Any]:
    """
    Carrega o diagnostics_summary_C_<C>.json para um dado C.

    Espera encontrar o arquivo em:
        <base_results_dir>/results_C_<C>/diagnostics_summary_C_<C>.json
    """
    folder = os.path.join(base_results_dir, f"results_C_{C}")
    fname = os.path.join(folder, f"diagnostics_summary_C_{C}.json")
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            f"Could not find diagnostics file for C={C}: {fname}"
        )
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gw_params(di: Dict[str, Any], C_value: float) -> GWParams:
    """
    Extrai (alpha, beta/H, T*, g*, v_w) do dicionário de diagnostics.

    Usa diretamente os campos escritos pelo gather_diagnostics.
    """
    alpha = float(di.get("gw_alpha", np.nan))
    beta_over_H = float(di.get("gw_beta_over_H", np.nan))

    # T_* preferencial (do transitionFinder / GWCalculator)
    T_star = float(di.get("gw_T_star_GeV", np.nan))
    if not np.isfinite(T_star):
        # fallback: temperatura do SingleFieldInstanton, se existir
        T_star = float(di.get("temperature_GeV", np.nan))

    g_star = float(di.get("gw_g_star", np.nan))
    if not np.isfinite(g_star):
        g_star = 106.75  # padrão EW

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
        Lambda = Lambda
    )



# ---------------------------------------------------------------------------
# Curvas de sensibilidade "mock" para LISA, BBO e DECIGO
# (broken power law em Ω_GW)
# ---------------------------------------------------------------------------

def broken_power_law_sensitivity(
    f: np.ndarray,
    f_ref: float,
    omega_min: float,
    slope_low: float,
    slope_high: float,
) -> np.ndarray:
    """
    Sensibilidade do tipo power-law quebrada.

    Abaixo de f_ref: Ω ~ f^{-slope_low};
    acima de f_ref: Ω ~ f^{+slope_high},
    com Omega_min em f_ref.
    """
    x = f / f_ref
    sens = np.empty_like(f)
    mask_low = x < 1.0
    sens[mask_low] = omega_min * x[mask_low] ** (-slope_low)
    sens[~mask_low] = omega_min * x[~mask_low] ** (slope_high)
    return sens


def lisa_sensitivity(f: np.ndarray) -> np.ndarray:
    """Curva de sensibilidade mock para LISA."""
    return broken_power_law_sensitivity(
        f=f,
        f_ref=3e-3,      # ~ alguns mHz
        omega_min=1e-12, # mínimo em Ω_GW
        slope_low=4.0,
        slope_high=1.5,
    )


def bbo_sensitivity(f: np.ndarray) -> np.ndarray:
    """Curva de sensibilidade mock para BBO."""
    return broken_power_law_sensitivity(
        f=f,
        f_ref=0.3,       # ~ 0.1–1 Hz
        omega_min=1e-17,
        slope_low=3.0,
        slope_high=2.0,
    )


def decigo_sensitivity(f: np.ndarray) -> np.ndarray:
    """Curva de sensibilidade mock para DECIGO."""
    return broken_power_law_sensitivity(
        f=f,
        f_ref=0.2,
        omega_min=3e-18,
        slope_low=3.0,
        slope_high=2.0,
    )


# ---------------------------------------------------------------------------
# Rotina principal: multi-C + LISA/BBO/DECIGO em um único gráfico
# ---------------------------------------------------------------------------
def plot_multiC_spectra_from_diagnostics(
    C_list: Sequence[float] = C_LIST_DEFAULT,
    base_results_dir: str = ".",
    save_dir: str | None = "figs_multiC",
    filename: str = "fig_GW_multiC",
):
    """
    Figura única com:
      - bandas de h^2 Ω_tot(f) para cada C (ε_turb=0 → borda inferior,
        ε_turb=1 → borda superior, região hachurada/filled entre elas);
      - curvas de sensibilidade mock de LISA, BBO e DECIGO.

    Convenção visual:
      - curva cheia: ε_turb = 0;
      - curva tracejada: ε_turb = 1.

    Tudo construído APENAS a partir dos diagnostics_summary_C_<C>.json.
    """
    # Grade de frequências
    f = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Mapa de cores por valor de C
    # C = 3.83 -> roxo; 3.75 -> azul; 3.65 -> verde
    color_map = {
        3.83: "#9467bd",  # roxo
        3.75: "#1f77b4",  # azul
        3.65: "#2ca02c",  # verde
    }

    # Loop nos C's
    for idx, C in enumerate(C_list):
        di = load_diagnostics_for_C(C, base_results_dir=base_results_dir)

        params = extract_gw_params(di, C_value=C)
        alpha_val = params.alpha
        beta_over_H = params.beta_over_H
        T_star = params.T_star
        g_star = params.g_star
        v_w = params.v_w
        Lambda = params.Lambda

        # ε_turb = 0 e 1
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

        omega_tot_0 = spectra_eps0["total"]  # ε_turb = 0 (curva cheia)
        omega_tot_1 = spectra_eps1["total"]  # ε_turb = 1 (tracejada)

        # Escolhe cor pelo valor de C (com fallback pela ordem, se vier algum C inesperado)
        color = color_map.get(round(C, 2), f"C{idx}")

        # Curva cheia: ε_turb = 0
        ax.plot(
            f,
            omega_tot_0,
            color=color,
            lw=1.2,
            label=rf"$h^2\Omega_{{\rm tot}}(f),\, C={C:g}$",
        )
        # Curva tracejada: ε_turb = 1 (sem label na legenda)
        ax.plot(
            f,
            omega_tot_1,
            color=color,
            lw=1.2,
            ls="--",
        )

        # Região preenchida entre ε=0 e ε=1
        omega_low = np.minimum(omega_tot_0, omega_tot_1)
        omega_high = np.maximum(omega_tot_0, omega_tot_1)
        ax.fill_between(
            f,
            omega_low,
            omega_high,
            color=color,
            alpha=0.18,
        )

        # Print rápido dos parâmetros no terminal
        print(
            f"[C={C:g}] alpha={params.alpha:.3g}, "
            f"beta/H*={params.beta_over_H:.3g}, "
            f"T*={params.T_star:.3g} GeV"
        )

    # Curvas de sensibilidade dos detectores
    omega_LISA = lisa_sensitivity(f)
    omega_BBO = bbo_sensitivity(f)
    omega_DECIGO = decigo_sensitivity(f)

    # LISA: vermelho
    ax.plot(
        f,
        omega_LISA,
        color="red",
        lw=1.8,
        ls="-",
        label="LISA",
    )
    # DECIGO: laranja
    ax.plot(
        f,
        omega_DECIGO,
        color="orange",
        lw=1.8,
        ls="-",
        label="DECIGO",
    )
    # BBO: amarelo
    ax.plot(
        f,
        omega_BBO,
        color="yellow",
        lw=1.8,
        ls="-",
        label="BBO",
    )

    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_xlabel(r"$f$  [Hz]")
    ax.set_ylabel(r"$h^2 \Omega_{\rm GW}(f)$")
    ax.set_title(fr"Multi-$C$ GW spectra from diagnostics + LISA/BBO/DECIGO ($\Lambda=$ {Lambda})")
    ax.grid(True, which="both", alpha=0.3)

    # Remover entradas duplicadas da legenda (mantém só detectores + curvas cheias)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lbl in zip(handles, labels):
        if lbl not in unique:
            unique[lbl] = h
    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=8,
        loc="center right",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()

    # Salvar figura
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"{filename}.png")
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        print(f"[plot_multiC] Saved figure to: {outpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Basta rodar este script depois de já ter rodado o run_all
    # para os mesmos valores de C em C_LIST_DEFAULT.
    plot_multiC_spectra_from_diagnostics(
        C_list=C_LIST_DEFAULT,
        base_results_dir=".",      # ajuste se seus results_* estiverem em outro lugar
        save_dir="results",
        filename="fig_GW_multiC_LISA_BBO_DECIGO",
    )
