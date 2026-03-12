"""
opt_eta_temperature_example.py
==============================

Didactic example for the updated OPT module.

What this script does
---------------------
Part 1
^^^^^^
1. Choose model parameters and a temperature range.
2. Trace the physical phase over temperature.
3. Save a table with:
   - T
   - physical branch
   - physical eta, eta^2, phi, Veff
   - symmetric eta, eta^2, Veff
   - broken eta, eta^2, phi, Veff
   - d eta_phys / dT
4. Plot:
   - eta_phys(T)
   - eta_phys^2(T)
   - d eta_phys / dT
   - comparison plot with eta_sym(T), eta_broken(T), eta_phys(T)

Part 2
^^^^^^
1. Choose a specific temperature T_* and chemical potential mu_*.
2. Build the stationary phase candidates at that point.
3. Identify the physical phase.
4. Scan the on-shell effective potential V(phi) on a phi grid.
5. Print:
   - symmetric candidate
   - broken candidate (if it exists)
   - physical phase
   - grid minimum
   - local minima found on the grid
6. Plot V(phi) and mark the relevant points.

Updated behavior expected from OPT.py
-------------------------------------
- solve_physical_phase_at_T(...) should work seedlessly, i.e. without telling the
  code where the branch exchange happens.
- trace_physical_phases_over_T(...) should determine the broken -> symmetric
  exchange autonomously from the branch competition.
- scan_potential(...) should anchor itself on the autonomous physical phase at
  the chosen temperature.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CosmoTransitions import (
    OPTModelParams,
    SolverOptions,
    ThermalOptions,
    phi_min,
    scan_potential,
    solve_phase_candidates_at_T,
    solve_physical_phase_at_T,
    trace_physical_phases_over_T,
)


# ============================================================================
# User configuration
# ============================================================================

OUTPUT_DIR = Path("opt_example_outputs")

PARAMS = OPTModelParams(
    m2=-1.0,
    lam=1.0,
    M=1.0,
)

THERMAL = ThermalOptions(
    mode_mu0="auto",
    mode_muneq0="notebook_highT",
    ct_mu0_approx="exact",
    ct_mu0_n=20,
)

SOLVER = SolverOptions(
    root_tol=1e-12,
    max_iter=300,
    continuation=True,
)

PART1_CONFIG = {
    "mu": 0.5,
    "T_min": 2.0,
    "T_max": 7.0,
    "dT": 0.1,
    "table_filename": "physical_eta_vs_T_table.csv",
    "figure_eta_filename": "physical_eta_vs_T.png",
    "figure_eta2_filename": "physical_eta2_vs_T.png",
    "figure_deta_filename": "physical_deta_dT_vs_T.png",
    "figure_compare_filename": "eta_branches_comparison_vs_T.png",
}

PART2_CONFIG = {
    "T": 4.733359332047422 + 0.027,
    "mu": 0.5,
    "phi_min": -0.6,
    "phi_max": 0.6,
    "dphi": 0.01,
    "potential_figure_filename": "potential_scan.png",
}


# ============================================================================
# Small helpers
# ============================================================================

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_temperature_grid(T_min: float, T_max: float, dT: float) -> np.ndarray:
    T_min = float(T_min)
    T_max = float(T_max)
    dT = float(dT)

    if not np.isfinite(T_min) or not np.isfinite(T_max) or not np.isfinite(dT):
        raise ValueError("T_min, T_max, and dT must be finite.")
    if T_max <= T_min:
        raise ValueError(f"Require T_max > T_min, got T_min={T_min}, T_max={T_max}.")
    if dT <= 0.0:
        raise ValueError(f"dT must be > 0, got dT={dT}.")

    return np.arange(T_min, T_max + 0.5 * dT, dT, dtype=float)


def make_phi_grid(phi_min: float, phi_max: float, dphi: float) -> np.ndarray:
    phi_min = float(phi_min)
    phi_max = float(phi_max)
    dphi = float(dphi)

    if not np.isfinite(phi_min) or not np.isfinite(phi_max) or not np.isfinite(dphi):
        raise ValueError("phi_min, phi_max, and dphi must be finite.")
    if phi_max <= phi_min:
        raise ValueError(
            f"Require phi_max > phi_min, got phi_min={phi_min}, phi_max={phi_max}."
        )
    if dphi <= 0.0:
        raise ValueError(f"dphi must be > 0, got dphi={dphi}.")

    return np.arange(phi_min, phi_max + 0.5 * dphi, dphi, dtype=float)


def find_discrete_local_minima(phi: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    phi = np.asarray(phi, dtype=float)
    values = np.asarray(values, dtype=float)

    if phi.ndim != 1 or values.ndim != 1:
        raise ValueError("phi and values must be one-dimensional.")
    if phi.size != values.size:
        raise ValueError("phi and values must have the same length.")
    if phi.size < 3:
        return []

    minima: list[tuple[float, float]] = []
    for i in range(1, len(phi) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            minima.append((float(phi[i]), float(values[i])))

    return minima


def save_physical_eta_table(
    filepath: Path,
    trace_data: dict[str, np.ndarray],
    deta_dT_vals: np.ndarray,
) -> None:
    T = np.asarray(trace_data["T"], dtype=float)
    branch = np.asarray(trace_data["branch"], dtype=object)

    rows = zip(
        T,
        branch,
        np.asarray(trace_data["phi"], dtype=float),
        np.asarray(trace_data["phi2"], dtype=float),
        np.asarray(trace_data["eta"], dtype=float),
        np.asarray(trace_data["eta2"], dtype=float),
        np.asarray(trace_data["veff"], dtype=float),
        np.asarray(trace_data["eta_sym"], dtype=float),
        np.asarray(trace_data["eta2_sym"], dtype=float),
        np.asarray(trace_data["veff_sym"], dtype=float),
        np.asarray(trace_data["eta_broken"], dtype=float),
        np.asarray(trace_data["eta2_broken"], dtype=float),
        np.asarray(trace_data["phi_broken"], dtype=float),
        np.asarray(trace_data["phi2_broken"], dtype=float),
        np.asarray(trace_data["veff_broken"], dtype=float),
        np.asarray(deta_dT_vals, dtype=float),
    )

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "T",
                "branch_phys",
                "phi_phys",
                "phi2_phys",
                "eta_phys",
                "eta2_phys",
                "veff_phys",
                "eta_sym",
                "eta2_sym",
                "veff_sym",
                "eta_broken",
                "eta2_broken",
                "phi_broken",
                "phi2_broken",
                "veff_broken",
                "deta_phys_dT",
            ]
        )
        writer.writerows(rows)


def estimate_branch_switch_temperature(trace_data: dict[str, np.ndarray]) -> float | None:
    T = np.asarray(trace_data["T"], dtype=float)
    branch = np.asarray(trace_data["branch"], dtype=object)

    for i in range(1, len(T)):
        if branch[i] != branch[i - 1]:
            return float(T[i])
    return None


# ============================================================================
# Part 1: physical eta(T) study
# ============================================================================

def run_eta_temperature_study(
    *,
    output_dir: Path,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
    mu: float,
    T_min: float,
    T_max: float,
    dT: float,
    table_filename: str,
    figure_eta_filename: str,
    figure_eta2_filename: str,
    figure_deta_filename: str,
    figure_compare_filename: str,
) -> dict[str, np.ndarray]:
    ensure_output_dir(output_dir)

    mu = float(mu)
    T_grid = make_temperature_grid(T_min, T_max, dT)

    trace_data = trace_physical_phases_over_T(
        mu=mu,
        T_grid=T_grid,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    eta_phys = np.asarray(trace_data["eta"], dtype=float)
    eta2_phys = np.asarray(trace_data["eta2"], dtype=float)
    eta_sym = np.asarray(trace_data["eta_sym"], dtype=float)
    eta_broken = np.asarray(trace_data["eta_broken"], dtype=float)
    broken_exists = np.asarray(trace_data["broken_exists"], dtype=bool)
    branch_phys = np.asarray(trace_data["branch"], dtype=object)

    if len(T_grid) > 1:
        deta_dT_vals = np.gradient(eta_phys, T_grid)
    else:
        deta_dT_vals = np.zeros_like(T_grid)

    print("\n" + "=" * 79)
    print("PART 1 | Physical eta(T) study")
    print("=" * 79)
    print(f"Model parameters: m2={params.m2}, lambda={params.lam}, M={params.M}")
    print(f"Chemical potential: mu={mu}")
    print(f"Temperature range: [{T_grid[0]}, {T_grid[-1]}] with step dT={dT}")
    print("Autonomous mode: no branch-switch temperature is hard-coded.")

    switch_T = estimate_branch_switch_temperature(trace_data)
    if switch_T is None:
        print("Detected branch switch: none on the supplied grid.")
    else:
        print(f"Detected branch switch on this grid near T = {switch_T:.6f}")
    print()

    for i, T in enumerate(T_grid):
        broken_str = (
            f"{eta_broken[i]:14.10f}" if broken_exists[i] and np.isfinite(eta_broken[i]) else "      n/a     "
        )
        print(
            f"T = {T:10.6f} | "
            f"phys = {str(branch_phys[i]):9s} | "
            f"eta_phys = {eta_phys[i]:14.10f} | "
            f"eta_sym = {eta_sym[i]:14.10f} | "
            f"eta_broken = {broken_str}"
        )

    table_path = output_dir / table_filename
    save_physical_eta_table(
        filepath=table_path,
        trace_data=trace_data,
        deta_dT_vals=deta_dT_vals,
    )

    print("\nSaved table to:")
    print(f"  {table_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, eta_phys, marker="o", lw=1.8, color="black")
    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$\eta_{\mathrm{phys}}(T)$", fontsize=13)
    plt.title(r"Physical $\eta(T)$")
    plt.tight_layout()
    eta_fig_path = output_dir / figure_eta_filename
    plt.savefig(eta_fig_path, dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, eta2_phys, marker="o", lw=1.8, color="black")
    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$\eta^2_{\mathrm{phys}}(T)$", fontsize=13)
    plt.title(r"Physical $\eta^2(T)$")
    plt.tight_layout()
    eta2_fig_path = output_dir / figure_eta2_filename
    plt.savefig(eta2_fig_path, dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, deta_dT_vals, marker="o", lw=1.8, color="black")
    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$d\eta_{\mathrm{phys}}/dT$", fontsize=13)
    plt.title(r"Numerical derivative $d\eta_{\mathrm{phys}}/dT$")
    plt.tight_layout()
    deta_fig_path = output_dir / figure_deta_filename
    plt.savefig(deta_fig_path, dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(T_grid, eta_sym, lw=2.0, color="royalblue", label="symmetric")
    if np.any(broken_exists):
        plt.plot(
            T_grid[broken_exists],
            eta_broken[broken_exists],
            lw=2.0,
            color="crimson",
            label="broken",
        )
    plt.plot(T_grid, eta_phys, lw=2.4, color="black", label="physical")
    plt.xlabel(r"$T$", fontsize=13)
    plt.ylabel(r"$\eta(T)$", fontsize=13)
    plt.title(r"Comparison of OPT $\eta(T)$ branches")
    plt.legend(frameon=False)
    plt.tight_layout()
    compare_fig_path = output_dir / figure_compare_filename
    plt.savefig(compare_fig_path, dpi=180)
    plt.show()

    print("\nSaved figures to:")
    print(f"  {eta_fig_path}")
    print(f"  {eta2_fig_path}")
    print(f"  {deta_fig_path}")
    print(f"  {compare_fig_path}")

    return {
        "T": T_grid,
        "eta_phys": eta_phys,
        "eta2_phys": eta2_phys,
        "eta_sym": eta_sym,
        "eta_broken": eta_broken,
        "branch_phys": branch_phys,
        "deta_phys_dT": deta_dT_vals,
    }


# ============================================================================
# Part 2: single-temperature potential study
# ============================================================================

def run_single_temperature_potential_study(
    *,
    output_dir: Path,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
    T: float,
    mu: float,
    phi_min_value: float,
    phi_max_value: float,
    dphi: float,
    potential_figure_filename: str,
) -> dict[str, np.ndarray]:
    ensure_output_dir(output_dir)

    T = float(T)
    mu = float(mu)
    phi_grid = make_phi_grid(phi_min_value, phi_max_value, dphi)

    candidates = solve_phase_candidates_at_T(
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    physical_state = solve_physical_phase_at_T(
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    sym_state = candidates["symmetric"]
    broken_state = candidates["broken"]

    scan_data = scan_potential(
        phi_grid=phi_grid,
        T=T,
        mu=mu,
        params=params,
        thermal=thermal,
        solver=solver,
        subtract_origin=False,
    )

    phi_vals = np.asarray(scan_data["phi"], dtype=float)
    V_vals = np.asarray(scan_data["values"], dtype=float)

    phi_star_grid, V_star_grid = phi_min(
        T=T,
        mu=mu,
        phi_grid=phi_grid,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    local_minima = find_discrete_local_minima(phi_vals, V_vals)

    print("\n" + "=" * 79)
    print("PART 2 | Single-temperature potential study")
    print("=" * 79)
    print(f"Chosen temperature: T = {T}")
    print(f"Chosen chemical potential: mu = {mu}")
    print("Autonomous mode: the physical phase and the scan anchor are inferred by the module.")

    print("\nSymmetric stationary candidate:")
    if sym_state is None:
        print("  not found at this (T, mu).")
    else:
        print(f"  eta   = {np.sqrt(sym_state.eta2):.12f}")
        print(f"  eta^2 = {sym_state.eta2:.12f}")
        print(f"  phi   = {sym_state.phi:.12f}")
        print(f"  Veff  = {sym_state.veff:.12f}")
        print(f"  F_eta = {sym_state.F_eta:.3e}")

    if broken_state is None:
        print("\nBroken stationary candidate:")
        print("  not found at this (T, mu).")
    else:
        print("\nBroken stationary candidate:")
        print(f"  eta   = {np.sqrt(broken_state.eta2):.12f}")
        print(f"  eta^2 = {broken_state.eta2:.12f}")
        print(f"  phi   = {broken_state.phi:.12f}")
        print(f"  phi^2 = {broken_state.phi2:.12f}")
        print(f"  Veff  = {broken_state.veff:.12f}")
        print(f"  F_eta = {broken_state.F_eta:.3e}")
        print(f"  F_phi = {broken_state.F_phi:.3e}")

    print("\nPhysical phase:")
    print(f"  branch = {physical_state.branch}")
    print(f"  eta    = {np.sqrt(physical_state.eta2):.12f}")
    print(f"  eta^2  = {physical_state.eta2:.12f}")
    print(f"  phi    = {physical_state.phi:.12f}")
    print(f"  Veff   = {physical_state.veff:.12f}")

    print("\nGlobal minimum on the supplied phi grid:")
    print(f"  phi_min = {phi_star_grid:.12f}")
    print(f"  V_min   = {V_star_grid:.12f}")

    if local_minima:
        print("\nDiscrete local minima found on the grid:")
        for i, (phi_loc, V_loc) in enumerate(local_minima, start=1):
            print(f"  {i:2d}. phi = {phi_loc: .12f} | V = {V_loc: .12f}")
    else:
        print("\nNo discrete local minima were identified on the supplied phi grid.")

    plt.figure(figsize=(8, 5))
    plt.plot(phi_vals, V_vals, lw=2.0, color="black", label=r"$V_{\mathrm{eff}}(\phi)$")

    plt.scatter(
        [phi_star_grid],
        [V_star_grid],
        s=70,
        color="black",
        label="grid global minimum",
        zorder=5,
    )

    if sym_state is not None:
        plt.scatter(
            [0.0],
            [sym_state.veff],
            s=60,
            color="royalblue",
            label="symmetric stationary point",
            zorder=6,
        )

    if broken_state is not None:
        plt.scatter(
            [broken_state.phi, -broken_state.phi],
            [broken_state.veff, broken_state.veff],
            s=60,
            color="crimson",
            marker="x",
            label="broken stationary points",
            zorder=7,
        )

    plt.xlabel(r"$\phi$", fontsize=13)
    plt.ylabel(r"$V_{\mathrm{eff}}(\phi)$", fontsize=13)
    plt.title(rf"On-shell potential at $T={T}$, $\mu={mu}$")
    plt.legend(frameon=False)
    plt.tight_layout()

    potential_fig_path = output_dir / potential_figure_filename
    plt.savefig(potential_fig_path, dpi=180)
    plt.show()

    print("\nSaved figure to:")
    print(f"  {potential_fig_path}")

    return scan_data


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    run_eta_temperature_study(
        output_dir=OUTPUT_DIR,
        params=PARAMS,
        thermal=THERMAL,
        solver=SOLVER,
        **PART1_CONFIG,
    )

    run_single_temperature_potential_study(
        output_dir=OUTPUT_DIR,
        params=PARAMS,
        thermal=THERMAL,
        solver=SOLVER,
        T=PART2_CONFIG["T"],
        mu=PART2_CONFIG["mu"],
        phi_min_value=PART2_CONFIG["phi_min"],
        phi_max_value=PART2_CONFIG["phi_max"],
        dphi=PART2_CONFIG["dphi"],
        potential_figure_filename=PART2_CONFIG["potential_figure_filename"],
    )


if __name__ == "__main__":
    main()
