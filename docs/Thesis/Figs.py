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
from scipy.integrate import quad
import math

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



# ----------------------------------------------------------------------
# Section 2.5 figures
# ----------------------------------------------------------------------
def _JB_integrand(x: float, r: float) -> float:
    """
    Integrand for J_B(r^2), with r = m/T.

    J_B(r^2) = int_0^inf dx x^2 log(1 - exp[-sqrt(x^2 + r^2)]).
    """
    E = math.sqrt(x * x + r * r)

    # The logarithm is integrable at x=0 for r=0, but numerically
    # log(0) would generate NaN. This branch implements the limiting behavior.
    if E < 1e-12:
        return 0.0

    if E < 1e-8:
        log_term = math.log(E)
    else:
        log_term = math.log1p(-math.exp(-E))

    return x * x * log_term


def _JF_integrand(x: float, r: float) -> float:
    """
    Integrand for J_F(r^2), with r = m/T.

    J_F(r^2) = int_0^inf dx x^2 log(1 + exp[-sqrt(x^2 + r^2)]).
    """
    E = math.sqrt(x * x + r * r)
    return x * x * math.log1p(math.exp(-E))


def thermal_JB(r: float) -> float:
    """
    Bosonic thermal function as a function of r = m/T.
    """
    if abs(r) < 1e-12:
        return -math.pi**4 / 45.0

    value, _ = quad(
        _JB_integrand,
        0.0,
        80.0,
        args=(r,),
        epsabs=1e-9,
        epsrel=1e-9,
        limit=200,
    )
    return value


def thermal_JF(r: float) -> float:
    """
    Fermionic thermal function as a function of r = m/T.
    """
    if abs(r) < 1e-12:
        return 7.0 * math.pi**4 / 360.0

    value, _ = quad(
        _JF_integrand,
        0.0,
        80.0,
        args=(r,),
        epsabs=1e-9,
        epsrel=1e-9,
        limit=200,
    )
    return value


def fig_thermal_functions_JB_JF() -> None:
    """
    Plot the thermal functions J_B and J_F and the corresponding
    one-degree-of-freedom thermal free-energy contributions.

    Convention:
        Delta V_T^B / T^4 = + J_B / (2 pi^2)
        Delta V_T^F / T^4 = - J_F / (2 pi^2)
    with positive fermionic degeneracy.
    """
    r_values = np.logspace(-2, 1.4, 120)  # r = m/T

    JB_values = np.array([thermal_JB(float(r)) for r in r_values])
    JF_values = np.array([thermal_JF(float(r)) for r in r_values])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Panel 1: mathematical thermal functions.
    ax = axes[0]
    ax.semilogx(r_values, JB_values, label=r"$J_B(r^2)$")
    ax.semilogx(r_values, JF_values, label=r"$J_F(r^2)$")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$r=m/T$")
    ax.set_ylabel(r"thermal functions")
    ax.set_title(r"(a) $J_B$ and $J_F$")
    ax.legend(frameon=False)

    # Panel 2: actual thermal free-energy contribution per degree of freedom.
    ax = axes[1]
    ax.semilogx(
        r_values,
        JB_values / (2.0 * math.pi**2),
        label=r"boson: $J_B/(2\pi^2)$",
    )
    ax.semilogx(
        r_values,
        -JF_values / (2.0 * math.pi**2),
        label=r"fermion: $-J_F/(2\pi^2)$",
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$r=m/T$")
    ax.set_ylabel(r"$\Delta V_T/T^4$ per degree of freedom")
    ax.set_title("(b) Contribution to the free energy")
    ax.legend(frameon=False)

    fig.suptitle("Thermal functions for bosonic and fermionic fields", y=1.03)
    plt.tight_layout()
    savefig("fig_2_6_JB_JF_functions")
    plt.close(fig)


# ----------------------------------------------------------------------
# Section 2.6 figures
# ----------------------------------------------------------------------
def _finite_temperature_potential(
    phi: np.ndarray,
    T: float,
    *,
    D: float,
    E: float,
    lam: float,
    T0: float,
) -> np.ndarray:
    """
    Schematic high-temperature effective potential:
        V(phi,T) = D (T^2 - T0^2) phi^2 - E T phi^3 + lambda/4 phi^4.

    This is not meant to represent a full numerical model. It is a clean
    pedagogical potential showing how a cubic thermal term creates a barrier.
    """
    return D * (T**2 - T0**2) * phi**2 - E * T * phi**3 + 0.25 * lam * phi**4


def fig_finite_temperature_potential() -> None:
    """
    Plot a schematic first-order finite-temperature effective potential
    at T > Tc, T = Tc and T < Tc.
    """
    D = 0.18
    E = 0.04
    lam = 0.12
    T0 = 1.0

    Tc = T0 / math.sqrt(1.0 - E**2 / (D * lam))
    phic = 2.0 * E * Tc / lam

    phi = np.linspace(0.0, 1.6, 900)

    temperatures = [
        (1.12 * Tc, r"$T>T_c$"),
        (Tc, r"$T=T_c$"),
        (0.92 * Tc, r"$T<T_c$"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for T, label in temperatures:
        V = _finite_temperature_potential(phi, T, D=D, E=E, lam=lam, T0=T0)

        # Normalize by T0^4 only to keep the axis dimensionless.
        ax.plot(phi, V / T0**4, label=label)

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(phic, linestyle=":", linewidth=1.2, label=r"$\phi_c$ at $T_c$")

    ax.set_xlabel(r"scalar background $\phi$")
    ax.set_ylabel(r"schematic $V_{\rm eff}(\phi,T)$")
    ax.set_title("Schematic first-order finite-temperature effective potential")
    ax.legend(frameon=False)
    ax.set_ylim(-0.04, 0.09)

    savefig("fig_2_7_finite_temperature_potential")
    plt.close(fig)


# ----------------------------------------------------------------------
# Section 2.8 figures: Higgs crossover / Mexican hat
# ----------------------------------------------------------------------
def _mexican_hat_potential_2d(X, Y, T, *, D=1.1, lam=0.65, T0=1.0):
    """
    Schematic O(2)-symmetric Higgs-like finite-temperature potential:

        V(r,T) = D (T^2 - T0^2) r^2 + lambda/4 r^4,

    with r^2 = phi_1^2 + phi_2^2.

    This is a pedagogical potential. It is not meant to reproduce the
    full Standard Model electroweak crossover quantitatively. Its purpose
    is to visualize symmetry restoration at high temperature and the
    Mexican-hat structure at low temperature.
    """
    R2 = X**2 + Y**2
    return D * (T**2 - T0**2) * R2 + 0.25 * lam * R2**2


def fig_sm_higgs_mexican_hat_3d() -> None:
    """
    Three-panel 3D schematic Higgs potential.

    The panels show:
        1. high temperature: symmetric minimum at the origin;
        2. near the transition region: shallow potential;
        3. low temperature: Mexican-hat structure with degenerate minima.

    The low-temperature panel is intentionally displayed with a stronger
    vertical contrast so that the Mexican-hat geometry is visually clear.
    """
    x = np.linspace(-1.85, 1.85, 180)
    y = np.linspace(-1.85, 1.85, 180)
    X, Y = np.meshgrid(x, y)

    temperatures = [
        (1.28, r"$T>T_c$"),
        (1.00, r"$T\simeq T_c$"),
        (0.35, r"$T\ll T_c$"),
    ]

    fig = plt.figure(figsize=(14.0, 4.7))

    for i, (T, title) in enumerate(temperatures, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        V = _mexican_hat_potential_2d(X, Y, T)

        # Shift each panel so that its minimum starts at zero.
        # This makes the vacuum structure easier to compare visually.
        V = V - np.min(V)

        # A mild vertical rescaling helps avoid a visually flattened Mexican hat.
        # The factor is pedagogical and affects only the plot, not the potential.
        if T < 0.6:
            V_plot = 1.8 * V
        else:
            V_plot = V

        surf = ax.plot_surface(
            X,
            Y,
            V_plot,
            rstride=3,
            cstride=3,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
            cmap="plasma_r",
        )

        # Mark the origin. At high T it is the minimum; at low T it becomes
        # the top of the central bump.
        V_origin = _mexican_hat_potential_2d(
            np.array([[0.0]]),
            np.array([[0.0]]),
            T,
        )
        V_origin = float(V_origin - np.min(_mexican_hat_potential_2d(X, Y, T)))
        if T < 0.6:
            V_origin *= 1.8

        ax.scatter(
            [0.0],
            [0.0],
            [V_origin],
            s=32,
            color="black",
            depthshade=True,
        )

        ax.set_title(title, pad=8)
        ax.set_xlabel(r"$\phi_1$", labelpad=-2)
        ax.set_ylabel(r"$\phi_2$", labelpad=-2)
        ax.set_zlabel(r"$V_{\rm eff}$", labelpad=2)

        # Camera angle chosen to resemble the reference image and make the
        # central bump plus the circular valley visible.
        ax.view_init(elev=24, azim=-58)

        # Remove tick labels for a cleaner thesis-style schematic figure.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make the box less vertically compressed.
        ax.set_box_aspect((1.35, 1.05, 0.85))

    fig.suptitle(
        "Schematic thermal evolution of a Higgs-like potential",
        y=1.03,
    )
    plt.tight_layout()
    savefig("fig_2_8_sm_higgs_mexican_hat_3d")
    plt.close(fig)


# ----------------------------------------------------------------------
# Section 2.9 figures: first-order temperatures
# ----------------------------------------------------------------------
def _fopt_schematic_potential(phi, T, *, D=0.18, E=0.055, lam=0.12, T0=1.0):
    """
    Schematic high-temperature first-order potential:
        V(phi,T) = D (T^2 - T0^2) phi^2 - E T phi^3 + lambda/4 phi^4.

    T0 is also the spinodal temperature of the origin in this simple model.
    """
    return D * (T**2 - T0**2) * phi**2 - E * T * phi**3 + 0.25 * lam * phi**4


def fig_first_order_temperatures() -> None:
    """
    Qualitative figure showing the important temperatures of a first-order
    phase transition: appearance of the broken minimum, critical temperature,
    schematic nucleation temperature, spinodal temperature and T=0 limit.
    """
    D = 0.18
    E = 0.055
    lam = 0.12
    T0 = 1.0

    # Critical temperature for the quartic-cubic schematic potential.
    Tc = T0 / math.sqrt(1.0 - E**2 / (D * lam))

    # Temperature where the broken extrema first appear in this toy model.
    T1 = math.sqrt((8.0 * D * lam * T0**2) / (8.0 * D * lam - 9.0 * E**2))

    # Schematic nucleation temperature: model-dependent in reality.
    Tn = 0.94 * Tc

    # Spinodal of the origin in this toy model.
    Tsp = T0

    phi = np.linspace(0.0, 2.2, 1000)

    curves = [
        (1.06 * T1, r"$T>T_1$", "symmetric only"),
        (T1, r"$T=T_1$", "broken extrema appear"),
        (Tc, r"$T=T_c$", "degenerate minima"),
        (Tn, r"$T=T_n$", "nucleation, schematic"),
        (Tsp, r"$T=T_{\rm sp}$", "false vacuum loses stability"),
        (0.0, r"$T=0$", "zero-temperature limit"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for T, label, _ in curves:
        V = _fopt_schematic_potential(phi, T, D=D, E=E, lam=lam, T0=T0)
        V = V - V[0]
        ax.plot(phi, V, label=label)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$|\phi|$")
    ax.set_ylabel(r"schematic $V_{\rm eff}(\phi,T)-V_{\rm eff}(0,T)$")
    ax.set_title("Schematic first-order transition: important temperatures")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(-0.08, 0.16)
    ax.set_xlim(0.0, 2.2)

    ax.text(
        0.05,
        0.05,
        r"$T_c$: degenerate minima" "\n"
        r"$T_n$: transition actually starts" "\n"
        r"$T_{\rm sp}$: metastable minimum disappears",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )

    savefig("fig_2_9_first_order_temperatures")
    plt.close(fig)

def main() -> None:
    setup_matplotlib()

    #fig_phase_transition_intuition()
    #fig_landau_second_order_free_energy()
    #fig_landau_order_parameter()
    #fig_landau_specific_heat_jump()
    #fig_landau_susceptibility_optional()
    #fig_landau_first_order_free_energy()
    #fig_thermal_functions_JB_JF()
    #fig_finite_temperature_potential()
    fig_sm_higgs_mexican_hat_3d()
    fig_first_order_temperatures()

if __name__ == "__main__":
    main()