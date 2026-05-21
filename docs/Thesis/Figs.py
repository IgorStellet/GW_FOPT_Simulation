"""
chapter2_figures.py
===================

Author-style schematic figures for Chapter 2:
1. Phase transition intuition: water, ferromagnet, scalar field.
2. Landau second-order free energy.
3. Order parameter for a continuous transition.
4. Specific heat jump in Landau theory.
5. First-order Landau free energy with metastability and order-parameter jump.

Run:
    python chapter2_figures.py

Outputs:
    figures_chapter2/fig_2_1_phase_transition_intuition.pdf
    figures_chapter2/fig_2_2_landau_second_order_free_energy.pdf
    figures_chapter2/fig_2_3_landau_order_parameter.pdf
    figures_chapter2/fig_2_4_landau_specific_heat_jump.pdf
    figures_chapter2/fig_2_5_landau_first_order_free_energy.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUTDIR = Path("figures_chapter2")
OUTDIR.mkdir(exist_ok=True)


def setup_matplotlib() -> None:
    """Set a clean thesis-like style without requiring external LaTeX."""
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.2),
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def savefig(name: str) -> None:
    """Save current figure as PDF and PNG."""
    pdf_path = OUTDIR / f"{name}.pdf"
    png_path = OUTDIR / f"{name}.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ----------------------------------------------------------------------
# Figure for Section 2.1
# ----------------------------------------------------------------------
def fig_phase_transition_intuition() -> None:
    """
    Three-panel schematic:
    (a) Water: density changes around boiling temperature.
    (b) Ferromagnet: spontaneous magnetization below Tc.
    (c) Scalar field: finite-temperature effective potential shifts its minimum.
    """
    T = np.linspace(0.4, 1.6, 500)
    Tc = 1.0

    # Panel 1: schematic density drop, liquid -> vapor.
    # This is not a quantitative water model; it is a pedagogical transition profile.
    rho_liquid = 1.0
    rho_gas = 0.12
    sharpness = 55.0
    rho = rho_gas + (rho_liquid - rho_gas) / (1.0 + np.exp(sharpness * (T - Tc)))

    # Panel 2: ferromagnetic order parameter.
    M = np.zeros_like(T)
    below = T < Tc
    M[below] = np.sqrt(1.0 - T[below] / Tc)

    # Panel 3: scalar potentials at three temperatures.
    phi = np.linspace(-1.8, 1.8, 600)

    def V_scalar(phi: np.ndarray, a: float, b: float = 0.55) -> np.ndarray:
        return a * phi**2 + b * phi**4

    potentials = [
        (r"$T>T_c$", V_scalar(phi, a=+0.55)),
        (r"$T=T_c$", V_scalar(phi, a=0.00)),
        (r"$T<T_c$", V_scalar(phi, a=-0.55)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    # Water
    ax = axes[0]
    ax.plot(T, rho)
    ax.axvline(Tc, linestyle="--", linewidth=1.4)
    ax.text(0.58, 0.88, "liquid", transform=ax.transAxes)
    ax.text(0.70, 0.22, "vapor", transform=ax.transAxes)
    ax.set_title("(a) Water")
    ax.set_xlabel(r"$T/T_b$")
    ax.set_ylabel(r"schematic density")
    ax.set_ylim(0.0, 1.1)

    # Ferromagnet
    ax = axes[1]
    ax.plot(T, M)
    ax.axvline(Tc, linestyle="--", linewidth=1.4)
    ax.text(0.13, 0.70, "ordered", transform=ax.transAxes)
    ax.text(0.66, 0.14, "disordered", transform=ax.transAxes)
    ax.set_title("(b) Ferromagnet")
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"$M(T)$")
    ax.set_ylim(-0.03, 1.05)

    # Scalar field
    ax = axes[2]
    for label, V in potentials:
        V_shifted = V - np.min(V)
        ax.plot(phi, V_shifted, label=label)
    ax.set_title("(c) Scalar field")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$V_{\rm eff}(\phi,T)$")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 2.0)

    fig.suptitle("Different languages for the same idea: a phase changes when the preferred state changes", y=1.05)
    plt.tight_layout()
    savefig("fig_2_1_phase_transition_intuition")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figures for Section 2.2
# ----------------------------------------------------------------------
def fig_landau_second_order_free_energy() -> None:
    """
    Landau free energy:
        F(M,T) = a0 (T - Tc) M^2 + b M^4
    showing one minimum above Tc and two degenerate minima below Tc.
    """
    M = np.linspace(-1.6, 1.6, 800)
    Tc = 1.0
    a0 = 1.0
    b = 0.35

    temperatures = [1.25, 1.00, 0.75]
    labels = [r"$T>T_c$", r"$T=T_c$", r"$T<T_c$"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for T, label in zip(temperatures, labels):
        F = a0 * (T - Tc) * M**2 + b * M**4
        F = F - np.min(F)
        ax.plot(M, F, label=label)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"order parameter $M$")
    ax.set_ylabel(r"$F(M,T)-F_{\min}$")
    ax.set_title("Landau free energy for a continuous transition")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 1.8)

    savefig("fig_2_2_landau_second_order_free_energy")
    plt.close(fig)


def fig_landau_order_parameter() -> None:
    """
    Mean-field order parameter for the second-order Landau model:
        M = 0 for T >= Tc
        M = sqrt[a0 (Tc - T)/(2 b)] for T < Tc
    """
    T = np.linspace(0.0, 1.8, 600)
    Tc = 1.0
    a0 = 1.0
    b = 0.5

    M = np.zeros_like(T)
    below = T < Tc
    M[below] = np.sqrt(a0 * (Tc - T[below]) / (2.0 * b))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T / Tc, M, label=r"$|M_{\rm eq}(T)|$")
    ax.axvline(1.0, linestyle="--", linewidth=1.4, label=r"$T_c$")
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"$|M_{\rm eq}|$")
    ax.set_title("Order parameter in a continuous transition")
    ax.legend(frameon=False)
    ax.set_ylim(-0.03, 1.05)

    savefig("fig_2_3_landau_order_parameter")
    plt.close(fig)


def fig_landau_specific_heat_jump() -> None:
    """
    Schematic specific heat jump in the Landau mean-field model.
    This is not a microscopic model; it shows the finite discontinuity at Tc.
    """
    T = np.linspace(0.2, 1.8, 800)
    Tc = 1.0

    # Background plus Landau jump.
    C_background = 1.0 + 0.08 * T
    delta_C = 0.65
    C = C_background.copy()
    C[T < Tc] += delta_C

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T / Tc, C)
    ax.axvline(1.0, linestyle="--", linewidth=1.4, label=r"$T_c$")
    ax.text(0.56, 1.68, r"$C_V(T_c^-)$", transform=ax.transAxes)
    ax.text(0.68, 0.35, r"$C_V(T_c^+)$", transform=ax.transAxes)
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"schematic $C_V$")
    ax.set_title("Specific heat jump in Landau mean-field theory")
    ax.legend(frameon=False)

    savefig("fig_2_4_landau_specific_heat_jump")
    plt.close(fig)


def fig_landau_susceptibility_optional() -> None:
    """
    Optional alternative to the specific heat figure:
    mean-field susceptibility divergence near Tc.
    """
    T = np.linspace(0.2, 1.8, 1000)
    Tc = 1.0
    eps = 0.025

    chi = 1.0 / (np.abs(T - Tc) + eps)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(T / Tc, chi)
    ax.axvline(1.0, linestyle="--", linewidth=1.4, label=r"$T_c$")
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"schematic $\chi$")
    ax.set_title("Schematic susceptibility enhancement near a critical point")
    ax.legend(frameon=False)
    ax.set_ylim(0, np.percentile(chi, 99))

    savefig("fig_2_4_optional_landau_susceptibility")
    plt.close(fig)


def fig_landau_first_order_free_energy() -> None:
    """
    First-order Landau free energy:
        F(M,T) = a(T) M^2 + b M^4 + c M^6
    with b < 0 and c > 0.

    At criticality:
        a_c = b^2/(4c)
        M_c^2 = -b/(2c)

    The figure has two panels:
    (a) free energy curves around Tc
    (b) schematic order-parameter jump
    """
    M = np.linspace(-1.6, 1.6, 1200)
    b = -1.0
    c = 1.0
    a_c = b**2 / (4.0 * c)

    a_values = [0.34, a_c, 0.16]
    labels = [r"$T>T_c$", r"$T=T_c$", r"$T<T_c$"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Panel a: free energy.
    ax = axes[0]
    for a, label in zip(a_values, labels):
        F = a * M**2 + b * M**4 + c * M**6
        F = F - np.min(F)
        ax.plot(M, F, label=label)

    Mc = np.sqrt(-b / (2.0 * c))
    ax.axvline(+Mc, linestyle=":", linewidth=1.2)
    ax.axvline(-Mc, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"order parameter $M$")
    ax.set_ylabel(r"$F(M,T)-F_{\min}$")
    ax.set_title("(a) Coexisting minima and barrier")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 1.1)

    # Panel b: schematic jump.
    T = np.linspace(0.65, 1.35, 600)
    Tc = 1.0
    M_eq = np.zeros_like(T)
    M_eq[T < Tc] = Mc

    ax = axes[1]
    ax.plot(T / Tc, M_eq)
    ax.axvline(1.0, linestyle="--", linewidth=1.4, label=r"$T_c$")
    ax.scatter([1.0, 1.0], [0.0, Mc], zorder=5)
    ax.annotate(
        "jump",
        xy=(1.0, 0.5 * Mc),
        xytext=(1.08, 0.5 * Mc),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )
    ax.set_xlabel(r"$T/T_c$")
    ax.set_ylabel(r"$|M_{\rm eq}|$")
    ax.set_title("(b) Discontinuous order parameter")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 0.9)

    plt.tight_layout()
    savefig("fig_2_5_landau_first_order_free_energy")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()

    fig_phase_transition_intuition()
    fig_landau_second_order_free_energy()
    fig_landau_order_parameter()
    fig_landau_specific_heat_jump()
    fig_landau_susceptibility_optional()
    fig_landau_first_order_free_energy()


if __name__ == "__main__":
    main()