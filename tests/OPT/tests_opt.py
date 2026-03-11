import numpy as np
import matplotlib.pyplot as plt

from CosmoTransitions import (
    OPTModelParams,
    SolverOptions,
    ThermalOptions,
    effective_mass_sq,
    validate_effective_mass_sq,
    thermal_variables,
    H1_notebook_highT,
    H3_notebook_highT,
    H5_notebook_highT,
    H3,
    H5,
    opt_veff_off_shell,
    eta_gap_residual,
    phi_stationary_residual,
    solve_eta2_given_phi,
    solve_stationary_system,
    solve_symmetric_branch,
    solve_broken_branch,
    veff_on_shell,
    veff_difference_from_origin,
    scan_potential,
    trace_branches_over_T,
    phi_min,
    eta2_solution,
    Tc_opt,
    Tc_pt,
    reproduce_section3_scan,
    plot_potential_scan,
    plot_branches,
)

# Optional check: if the bosonic thermal backend is available through the package,
# some mu=0 comparison tests can be enabled.
try:
    from CosmoTransitions import Jb
    HAS_CT_THERMAL = True
except Exception:
    HAS_CT_THERMAL = False


# =============================================================================
# Global test/plot controls
# =============================================================================

SHOW_PLOTS = True          # Set False for non-interactive/CI runs
CLOSE_PLOTS_AFTER_SHOW = False


# =============================================================================
# Local helpers used only by the test file
# =============================================================================

def _eta2_physical_lower_bound_for_tests(
    mu: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    safety: float = 1e-10,
) -> float:
    """
    Local replica of the physical eta^2 lower-bound logic used by the module.

    We keep this helper here because the module's private helper is intentionally
    not part of the public API.
    """
    mu = float(mu)
    lower = max(0.0, -float(params.m2))

    if abs(mu) > thermal.mu_zero_tol and thermal.mode_muneq0 == "notebook_highT":
        lower = max(lower, mu**2 - float(params.m2))

    return float(lower + safety)


def _show_or_close_plots() -> None:
    """Utility to keep manual runs pleasant while allowing easy CI use."""
    if SHOW_PLOTS:
        plt.show()
    if CLOSE_PLOTS_AFTER_SHOW:
        plt.close("all")

def _dense_branch_phi_curve(
    mu: float,
    T_grid: np.ndarray,
    eta2_seed: float,
    phi2_seed: float,
    params: OPTModelParams,
    thermal: ThermalOptions,
    solver: SolverOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a dense |phi|(T) curve from branch tracing, not from brute-force
    phi-scans of the on-shell potential.
    """
    trace = trace_branches_over_T(
        mu=mu,
        T_grid=T_grid,
        eta2_seed=eta2_seed,
        phi2_seed=phi2_seed,
        params=params,
        thermal=thermal,
        solver=solver,
    )

    T_vals = np.asarray(trace["T"], dtype=float)
    phi_abs = np.abs(np.asarray(trace["phi"], dtype=float))
    return T_vals, phi_abs

def _plot_dense_minima_curves(
    T_mu0: np.ndarray,
    phi_mu0: np.ndarray,
    T_mu05: np.ndarray,
    phi_mu05: np.ndarray,
) -> None:
    """
    Plot |phi_min(T)| with many points for mu=0 and mu=0.5.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(T_mu0, phi_mu0, lw=2.0, marker="o", ms=3, label=r"$\mu/M = 0$")
    plt.plot(T_mu05, phi_mu05, lw=2.0, marker="o", ms=3, label=r"$\mu/M = 0.5$")
    plt.xlabel(r"$T/M$", fontsize=13)
    plt.ylabel(r"$|\phi_{\min}|/M$", fontsize=13)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()


def _plot_section2_phi_vs_T_mu_compare(
    trace_mu0: dict[str, np.ndarray],
    trace_mu05: dict[str, np.ndarray],
) -> None:
    """
    Plot a section-2-style comparison of phi(T) for mu/M=0 and mu/M=0.5.
    """
    T0 = np.asarray(trace_mu0["T"], dtype=float)
    phi0 = np.asarray(trace_mu0["phi"], dtype=float)

    T05 = np.asarray(trace_mu05["T"], dtype=float)
    phi05 = np.asarray(trace_mu05["phi"], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(T0, phi0, lw=2.0, marker="o", ms=4, label=r"$\mu/M = 0$")
    plt.plot(T05, phi05, lw=2.0, marker="s", ms=4, label=r"$\mu/M = 0.5$")
    plt.xlabel(r"$T/M$", fontsize=13)
    plt.ylabel(r"$\phi/M$", fontsize=13)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()


# =============================================================================
# Test blocks
# =============================================================================

def run_low_level_utility_tests(params: OPTModelParams) -> None:
    print("=== Low-level utilities tests ===")

    # --------------------------------------------------
    # Test 1: effective_mass_sq
    # --------------------------------------------------
    eta2_test = 2.1214029070500127
    m2_eff = effective_mass_sq(eta2_test, params)
    print(f"effective_mass_sq({eta2_test}) = {m2_eff:.12f}")

    assert np.isclose(m2_eff, 1.1214029070500127, atol=1e-14)

    # --------------------------------------------------
    # Test 2: validate_effective_mass_sq
    # --------------------------------------------------
    validate_effective_mass_sq(eta2_test, params)
    print("validate_effective_mass_sq: passed for a physical point.")

    try:
        validate_effective_mass_sq(0.5, params)
    except ValueError as exc:
        print("validate_effective_mass_sq correctly rejected unphysical Omega^2:")
        print(f"  {exc}")

    # --------------------------------------------------
    # Test 3: thermal_variables
    # --------------------------------------------------
    T_test = 4.733359332047422
    mu_test = 0.5
    y, r = thermal_variables(m2_eff, T_test, mu_test)

    print(f"thermal_variables(m2_eff={m2_eff:.12f}, T={T_test:.12f}, mu={mu_test:.12f})")
    print(f"  y = {y:.12f}")
    print(f"  r = {r:.12f}")

    y_expected = np.sqrt(m2_eff) / T_test
    r_expected = mu_test / np.sqrt(m2_eff)

    assert np.isclose(y, y_expected, atol=1e-14)
    assert np.isclose(r, r_expected, atol=1e-14)

    print("All low-level utility tests passed.")


def run_thermal_backend_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
) -> None:
    print("\n=== Thermal backend tests ===")

    # --------------------------------------------------
    # Test 4: notebook high-T backend at the section-3 point
    # --------------------------------------------------
    m2_eff_test = 1.1214029070500127
    T_test = 4.733359332047422
    mu_test = 0.5

    h1_val = H1_notebook_highT(m2_eff_test, T_test, mu_test)
    h3_val = H3_notebook_highT(m2_eff_test, T_test, mu_test)
    h5_val = H5_notebook_highT(m2_eff_test, T_test, mu_test)

    print(f"H1_notebook_highT = {h1_val:.12f}")
    print(f"H3_notebook_highT = {h3_val:.12f}")
    print(f"H5_notebook_highT = {h5_val:.12f}")

    assert np.isfinite(h1_val) and h1_val > 0.0
    assert np.isfinite(h3_val) and h3_val > 0.0
    assert np.isfinite(h5_val) and h5_val > 0.0

    # --------------------------------------------------
    # Test 5: derivative identities from notebook section 1
    # --------------------------------------------------
    m2_eff_id = 1.1214029070500127
    T_id = 4.733359332047422
    mu_id = 0.0
    eps = 1e-6

    dH5_fd = (
        H5_notebook_highT(m2_eff_id + eps, T_id, mu_id)
        - H5_notebook_highT(m2_eff_id - eps, T_id, mu_id)
    ) / (2.0 * eps)
    rhs_H5 = -H3_notebook_highT(m2_eff_id, T_id, mu_id) / (8.0 * T_id**2)

    dH3_fd = (
        H3_notebook_highT(m2_eff_id + eps, T_id, mu_id)
        - H3_notebook_highT(m2_eff_id - eps, T_id, mu_id)
    ) / (2.0 * eps)
    rhs_H3 = -H1_notebook_highT(m2_eff_id, T_id, mu_id) / (4.0 * T_id**2)

    print(f"dH5/dm2 (FD)       = {dH5_fd:.12e}")
    print(f"-H3/(8 T^2)        = {rhs_H5:.12e}")
    print(f"dH3/dm2 (FD)       = {dH3_fd:.12e}")
    print(f"-H1/(4 T^2)        = {rhs_H3:.12e}")

    assert np.isclose(dH5_fd, rhs_H5, rtol=1e-5, atol=1e-8)
    assert np.isclose(dH3_fd, rhs_H3, rtol=1e-5, atol=1e-8)

    # --------------------------------------------------
    # Test 6: unified dispatch with mu != 0
    # --------------------------------------------------
    h3_dispatch = H3(m2_eff_test, T_test, mu_test, thermal_nb)
    h5_dispatch = H5(m2_eff_test, T_test, mu_test, thermal_nb)

    assert np.isclose(h3_dispatch, h3_val, atol=1e-14)
    assert np.isclose(h5_dispatch, h5_val, atol=1e-14)
    print("Unified dispatch for mu != 0: passed.")

    # --------------------------------------------------
    # Test 7: compare notebook high-T vs CT backend at mu = 0
    # --------------------------------------------------
    if HAS_CT_THERMAL:
        thermal_ct = ThermalOptions(
            mode_mu0="ct_backend",
            mode_muneq0="notebook_highT",
            ct_mu0_approx="exact",
            ct_mu0_n=8,
        )

        T_cmp = 1.0
        x_cmp = 0.2
        m2_eff_cmp = x_cmp**2

        h3_nb_cmp = H3_notebook_highT(m2_eff_cmp, T_cmp, 0.0)
        h5_nb_cmp = H5_notebook_highT(m2_eff_cmp, T_cmp, 0.0)

        h3_ct_cmp = H3(m2_eff_cmp, T_cmp, 0.0, thermal_ct)
        h5_ct_cmp = H5(m2_eff_cmp, T_cmp, 0.0, thermal_ct)

        rel_h3 = abs(h3_ct_cmp - h3_nb_cmp) / abs(h3_ct_cmp)
        rel_h5 = abs(h5_ct_cmp - h5_nb_cmp) / abs(h5_ct_cmp)

        print(f"H3 notebook vs CT rel. diff. = {rel_h3:.3e}")
        print(f"H5 notebook vs CT rel. diff. = {rel_h5:.3e}")

        assert rel_h3 < 5e-2
        assert rel_h5 < 5e-2
        print("CT mu=0 backend comparison: passed.")
    else:
        print("CosmoTransitions thermal backend not available here; CT backend comparison skipped.")

    print("All thermal backend tests passed.")


def run_core_equation_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
) -> tuple[float, float, float, float]:
    print("\n=== Core physics equation tests ===")

    phi_test = 0.37
    eta2_test = 1.4
    T_test = 4.733359332047422 + 0.027
    mu_test = 0.5

    # --------------------------------------------------
    # Test 8: off-shell potential is finite and even in phi
    # --------------------------------------------------
    v_plus = opt_veff_off_shell(phi_test, eta2_test, T_test, mu_test, params, thermal_nb)
    v_minus = opt_veff_off_shell(-phi_test, eta2_test, T_test, mu_test, params, thermal_nb)

    print(f"V_eff(+phi) = {v_plus:.12f}")
    print(f"V_eff(-phi) = {v_minus:.12f}")

    assert np.isfinite(v_plus)
    assert np.isfinite(v_minus)
    assert np.isclose(v_plus, v_minus, atol=1e-12)

    # --------------------------------------------------
    # Test 9: eta-gap residual is even in phi
    # --------------------------------------------------
    eta_res_plus = eta_gap_residual(phi_test, eta2_test, T_test, mu_test, params, thermal_nb)
    eta_res_minus = eta_gap_residual(-phi_test, eta2_test, T_test, mu_test, params, thermal_nb)

    print(f"F_eta(+phi) = {eta_res_plus:.12f}")
    print(f"F_eta(-phi) = {eta_res_minus:.12f}")

    assert np.isfinite(eta_res_plus)
    assert np.isfinite(eta_res_minus)
    assert np.isclose(eta_res_plus, eta_res_minus, atol=1e-12)

    # --------------------------------------------------
    # Test 10: phi-stationary residual reproduces dV/dphi
    # --------------------------------------------------
    eps = 1e-6
    dV_dphi_fd = (
        opt_veff_off_shell(phi_test + eps, eta2_test, T_test, mu_test, params, thermal_nb)
        - opt_veff_off_shell(phi_test - eps, eta2_test, T_test, mu_test, params, thermal_nb)
    ) / (2.0 * eps)

    phi_res = phi_stationary_residual(phi_test, eta2_test, T_test, mu_test, params, thermal_nb)
    dV_dphi_from_residual = phi_test * phi_res / 24.0

    print(f"dV/dphi (finite diff.) = {dV_dphi_fd:.12e}")
    print(f"phi * F_phi / 24      = {dV_dphi_from_residual:.12e}")

    assert np.isclose(dV_dphi_fd, dV_dphi_from_residual, rtol=1e-6, atol=1e-8)

    # --------------------------------------------------
    # Test 11: find one real eta-gap root at phi = 0 manually
    # --------------------------------------------------
    T_root = 4.733359332047422 + 0.027
    mu_root = 0.5
    phi_root = 0.0

    def f_eta(e2: float) -> float:
        return eta_gap_residual(phi_root, e2, T_root, mu_root, params, thermal_nb)

    eta2_real_domain_min = -params.m2 + mu_root**2 + 1e-6
    x_grid = np.linspace(eta2_real_domain_min, 5.0, 400)
    bracket = None
    x_prev = x_grid[0]
    f_prev = f_eta(x_prev)

    for x in x_grid[1:]:
        f_now = f_eta(x)
        if np.isfinite(f_prev) and np.isfinite(f_now) and f_prev * f_now < 0.0:
            bracket = (x_prev, x)
            break
        x_prev = x
        f_prev = f_now

    assert bracket is not None, "Could not bracket a real eta-gap root in the test interval."

    a, b = bracket
    fa = f_eta(a)
    fb = f_eta(b)

    for _ in range(100):
        c = 0.5 * (a + b)
        fc = f_eta(c)
        if fa * fc <= 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    eta2_root = 0.5 * (a + b)
    f_root = f_eta(eta2_root)

    print(f"Bracketed eta^2 root at phi=0: eta2 = {eta2_root:.12f}")
    print(f"F_eta(root) = {f_root:.12e}")

    assert eta2_root > 1.0
    assert abs(f_root) < 1e-8

    # --------------------------------------------------
    # Test 12: off-shell potential is finite at the eta-gap root
    # --------------------------------------------------
    v_root = opt_veff_off_shell(phi_root, eta2_root, T_root, mu_root, params, thermal_nb)
    print(f"V_eff(phi=0, eta2=root) = {v_root:.12f}")
    assert np.isfinite(v_root)

    print("All core physics equation tests passed.")
    return T_root, mu_root, phi_root, eta2_root


def run_root_solver_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
    T_root: float,
    mu_root: float,
    phi_root: float,
    eta2_root: float,
) -> tuple[float, float, float]:
    print("\n=== Root-solving helper tests ===")

    # --------------------------------------------------
    # Test 13: solve_eta2_given_phi reproduces the phi=0 eta-gap root
    # --------------------------------------------------
    eta2_solver_root = solve_eta2_given_phi(
        phi=phi_root,
        T=T_root,
        mu=mu_root,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    f_eta_solver_root = eta_gap_residual(
        phi_root, eta2_solver_root, T_root, mu_root, params, thermal_nb
    )

    print(f"solve_eta2_given_phi(phi=0) = {eta2_solver_root:.12f}")
    print(f"F_eta(at solved root)       = {f_eta_solver_root:.12e}")

    assert eta2_solver_root > (-params.m2 + mu_root**2)
    assert np.isclose(eta2_solver_root, eta2_root, rtol=1e-8, atol=1e-10)
    assert abs(f_eta_solver_root) < 1e-8

    # --------------------------------------------------
    # Test 14: solve_eta2_given_phi at a nonzero phi inside the section-3 scan
    # --------------------------------------------------
    phi_scan = 0.30
    T_scan = 4.733359332047422 + 0.027
    mu_scan = 0.5

    eta2_nonzero = solve_eta2_given_phi(
        phi=phi_scan,
        T=T_scan,
        mu=mu_scan,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    f_eta_nonzero = eta_gap_residual(
        phi_scan, eta2_nonzero, T_scan, mu_scan, params, thermal_nb
    )

    eta2_min_nonzero = _eta2_physical_lower_bound_for_tests(mu_scan, params, thermal_nb)

    print(f"solve_eta2_given_phi(phi={phi_scan}) = {eta2_nonzero:.12f}")
    print(f"Physical eta2 lower bound            = {eta2_min_nonzero:.12f}")
    print(f"F_eta(at solved root)               = {f_eta_nonzero:.12e}")

    assert eta2_nonzero > eta2_min_nonzero
    assert abs(f_eta_nonzero) < 1e-8

    # --------------------------------------------------
    # Test 15: solve_stationary_system on a broken branch
    # --------------------------------------------------
    T_broken = 2.0
    mu_broken = 0.0

    eta2_broken, phi2_broken = solve_stationary_system(
        T=T_broken,
        mu=mu_broken,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    phi_broken = np.sqrt(phi2_broken)

    f_eta_broken = eta_gap_residual(
        phi_broken, eta2_broken, T_broken, mu_broken, params, thermal_auto
    )
    f_phi_broken = phi_stationary_residual(
        phi_broken, eta2_broken, T_broken, mu_broken, params, thermal_auto
    )

    print(f"Broken-branch solution at T={T_broken}, mu={mu_broken}:")
    print(f"  eta2 = {eta2_broken:.12f}")
    print(f"  phi2 = {phi2_broken:.12f}")
    print(f"  F_eta = {f_eta_broken:.12e}")
    print(f"  F_phi = {f_phi_broken:.12e}")

    assert eta2_broken > _eta2_physical_lower_bound_for_tests(mu_broken, params, thermal_auto)
    assert phi2_broken > 0.0
    assert abs(f_eta_broken) < 1e-7
    assert abs(f_phi_broken) < 1e-7

    print("All root-solving helper tests passed.")
    return T_broken, mu_broken, phi2_broken


def run_branch_solver_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
    T_root: float,
    mu_root: float,
    T_broken: float,
    mu_broken: float,
) -> None:
    print("\n=== Branch solver tests ===")

    # --------------------------------------------------
    # Test 16: solve_symmetric_branch must agree with solve_eta2_given_phi at phi=0
    # --------------------------------------------------
    eta2_sym_branch = solve_symmetric_branch(
        T=T_root,
        mu=mu_root,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    eta2_phi0_direct = solve_eta2_given_phi(
        phi=0.0,
        T=T_root,
        mu=mu_root,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    f_eta_sym = eta_gap_residual(
        phi=0.0,
        eta2=eta2_sym_branch,
        T=T_root,
        mu=mu_root,
        params=params,
        thermal=thermal_nb,
    )

    print(f"solve_symmetric_branch = {eta2_sym_branch:.12f}")
    print(f"solve_eta2_given_phi   = {eta2_phi0_direct:.12f}")
    print(f"F_eta(sym root)        = {f_eta_sym:.12e}")

    assert np.isclose(eta2_sym_branch, eta2_phi0_direct, rtol=1e-9, atol=1e-10)
    assert abs(f_eta_sym) < 1e-8

    # --------------------------------------------------
    # Test 17: solve_broken_branch must reproduce a valid broken-branch stationary solution
    # --------------------------------------------------
    eta2_broken_branch, phi2_broken_branch = solve_broken_branch(
        T=T_broken,
        mu=mu_broken,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    phi_broken_branch = np.sqrt(phi2_broken_branch)

    f_eta_broken_branch = eta_gap_residual(
        phi=phi_broken_branch,
        eta2=eta2_broken_branch,
        T=T_broken,
        mu=mu_broken,
        params=params,
        thermal=thermal_auto,
    )
    f_phi_broken_branch = phi_stationary_residual(
        phi=phi_broken_branch,
        eta2=eta2_broken_branch,
        T=T_broken,
        mu=mu_broken,
        params=params,
        thermal=thermal_auto,
    )

    print(f"solve_broken_branch eta2 = {eta2_broken_branch:.12f}")
    print(f"solve_broken_branch phi2 = {phi2_broken_branch:.12f}")
    print(f"F_eta(broken root)       = {f_eta_broken_branch:.12e}")
    print(f"F_phi(broken root)       = {f_phi_broken_branch:.12e}")

    assert phi2_broken_branch > 0.0
    assert abs(f_eta_broken_branch) < 1e-7
    assert abs(f_phi_broken_branch) < 1e-7

    # --------------------------------------------------
    # Test 18: broken-branch wrapper should be consistent with solve_stationary_system
    # --------------------------------------------------
    eta2_broken_direct, phi2_broken_direct = solve_stationary_system(
        T=T_broken,
        mu=mu_broken,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    print(f"solve_stationary_system eta2 = {eta2_broken_direct:.12f}")
    print(f"solve_stationary_system phi2 = {phi2_broken_direct:.12f}")

    assert np.isclose(eta2_broken_branch, eta2_broken_direct, rtol=1e-6, atol=1e-8)
    assert np.isclose(phi2_broken_branch, phi2_broken_direct, rtol=1e-6, atol=1e-8)

    print("All branch solver tests passed.")


def run_on_shell_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    solver_opts: SolverOptions,
    T_root: float,
    mu_root: float,
) -> tuple[float, float]:
    print("\n=== On-shell potential layer tests ===")

    # --------------------------------------------------
    # Test 19: veff_on_shell at phi=0 must agree with off-shell evaluation at the symmetric eta root
    # --------------------------------------------------
    veff_sym_on, eta2_sym_on = veff_on_shell(
        phi=0.0,
        T=T_root,
        mu=mu_root,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    eta2_sym_direct = solve_symmetric_branch(
        T=T_root,
        mu=mu_root,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    veff_sym_direct = opt_veff_off_shell(
        phi=0.0,
        eta2=eta2_sym_direct,
        T=T_root,
        mu=mu_root,
        params=params,
        thermal=thermal_nb,
    )

    print(f"veff_on_shell(phi=0)   = {veff_sym_on:.12f}")
    print(f"eta2_on_shell(phi=0)   = {eta2_sym_on:.12f}")
    print(f"veff_direct(phi=0)     = {veff_sym_direct:.12f}")
    print(f"eta2_direct(phi=0)     = {eta2_sym_direct:.12f}")

    assert np.isclose(eta2_sym_on, eta2_sym_direct, rtol=1e-9, atol=1e-10)
    assert np.isclose(veff_sym_on, veff_sym_direct, rtol=1e-9, atol=1e-10)

    # --------------------------------------------------
    # Test 20: veff_on_shell should remain even in phi
    # --------------------------------------------------
    phi_on = 0.30
    T_on = 4.733359332047422 + 0.027
    mu_on = 0.5

    veff_plus, eta2_plus = veff_on_shell(
        phi=phi_on,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )
    veff_minus, eta2_minus = veff_on_shell(
        phi=-phi_on,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"veff_on_shell(+phi) = {veff_plus:.12f}")
    print(f"veff_on_shell(-phi) = {veff_minus:.12f}")
    print(f"eta2(+phi)          = {eta2_plus:.12f}")
    print(f"eta2(-phi)          = {eta2_minus:.12f}")

    assert np.isclose(eta2_plus, eta2_minus, rtol=1e-9, atol=1e-10)
    assert np.isclose(veff_plus, veff_minus, rtol=1e-9, atol=1e-10)

    # --------------------------------------------------
    # Test 21: Delta V(0) must be zero
    # --------------------------------------------------
    dV_zero, eta2_phi_zero, eta2_zero_zero = veff_difference_from_origin(
        phi=0.0,
        T=T_on,
        mu=mu_on,
        eta2_seed_phi=2.1214029070500127,
        eta2_seed_zero=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"Delta V(phi=0) = {dV_zero:.12e}")
    print(f"eta2_phi(0)    = {eta2_phi_zero:.12f}")
    print(f"eta2_zero      = {eta2_zero_zero:.12f}")

    assert np.isclose(dV_zero, 0.0, atol=1e-10)
    assert np.isclose(eta2_phi_zero, eta2_zero_zero, rtol=1e-9, atol=1e-10)

    # --------------------------------------------------
    # Test 22: Delta V(phi) must match the difference of two independently computed on-shell potentials
    # --------------------------------------------------
    phi_diff = 0.30

    dV_func, eta2_phi_func, eta2_zero_func = veff_difference_from_origin(
        phi=phi_diff,
        T=T_on,
        mu=mu_on,
        eta2_seed_phi=2.1214029070500127,
        eta2_seed_zero=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    veff_phi_manual, eta2_phi_manual = veff_on_shell(
        phi=phi_diff,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    eta2_zero_manual = solve_symmetric_branch(
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    veff_zero_manual = opt_veff_off_shell(
        phi=0.0,
        eta2=eta2_zero_manual,
        T=T_on,
        mu=mu_on,
        params=params,
        thermal=thermal_nb,
    )

    dV_manual = veff_phi_manual - veff_zero_manual

    print(f"Delta V function = {dV_func:.12f}")
    print(f"Delta V manual   = {dV_manual:.12f}")
    print(f"eta2_phi func    = {eta2_phi_func:.12f}")
    print(f"eta2_phi manual  = {eta2_phi_manual:.12f}")
    print(f"eta2_zero func   = {eta2_zero_func:.12f}")
    print(f"eta2_zero manual = {eta2_zero_manual:.12f}")

    assert np.isclose(dV_func, dV_manual, rtol=1e-9, atol=1e-10)
    assert np.isclose(eta2_phi_func, eta2_phi_manual, rtol=1e-9, atol=1e-10)
    assert np.isclose(eta2_zero_func, eta2_zero_manual, rtol=1e-9, atol=1e-10)

    print("All on-shell potential layer tests passed.")
    return T_on, mu_on


def run_scan_and_continuation_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
    T_on: float,
    mu_on: float,
) -> dict[str, np.ndarray]:
    print("\n=== Scan and continuation tests ===")

    # --------------------------------------------------
    # Test 23: scan_potential with subtract_origin=True
    # --------------------------------------------------
    phi_grid_test = np.linspace(-0.5, 0.5, 11)

    scan_delta = scan_potential(
        phi_grid=phi_grid_test,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
        subtract_origin=True,
    )

    print(f"scan_potential keys = {list(scan_delta.keys())}")
    print(f"scan quantity       = {scan_delta['quantity'][0]}")
    print(f"number of points    = {len(scan_delta['phi'])}")

    assert len(scan_delta["phi"]) == len(phi_grid_test)
    assert len(scan_delta["values"]) == len(phi_grid_test)
    assert len(scan_delta["eta2"]) == len(phi_grid_test)
    assert len(scan_delta["eta2_zero"]) == len(phi_grid_test)
    assert np.all(np.isfinite(scan_delta["values"]))
    assert np.all(np.isfinite(scan_delta["eta2"]))
    assert np.all(np.isfinite(scan_delta["eta2_zero"]))

    idx_zero = np.argmin(np.abs(scan_delta["phi"]))
    print(f"Delta V at phi~0 = {scan_delta['values'][idx_zero]:.12e}")
    assert np.isclose(scan_delta["phi"][idx_zero], 0.0, atol=1e-14)
    assert np.isclose(scan_delta["values"][idx_zero], 0.0, atol=1e-10)

    # --------------------------------------------------
    # Test 24: scan_potential with subtract_origin=False
    # --------------------------------------------------
    phi_single = np.array([0.30], dtype=float)
    scan_abs = scan_potential(
        phi_grid=phi_single,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
        subtract_origin=False,
    )

    veff_direct_single, eta2_direct_single = veff_on_shell(
        phi=0.30,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"scan absolute V(phi=0.30) = {scan_abs['values'][0]:.12f}")
    print(f"direct V(phi=0.30)        = {veff_direct_single:.12f}")
    print(f"scan eta2(phi=0.30)       = {scan_abs['eta2'][0]:.12f}")
    print(f"direct eta2(phi=0.30)     = {eta2_direct_single:.12f}")

    assert np.isclose(scan_abs["values"][0], veff_direct_single, rtol=1e-9, atol=1e-10)
    assert np.isclose(scan_abs["eta2"][0], eta2_direct_single, rtol=1e-9, atol=1e-10)

    # --------------------------------------------------
    # Test 25: Delta V scan should be approximately even in phi
    # --------------------------------------------------
    left_vals = scan_delta["values"][:5]
    right_vals = scan_delta["values"][-5:][::-1]

    max_even_diff = np.max(np.abs(left_vals - right_vals))
    print(f"Max |DeltaV(-phi) - DeltaV(+phi)| on symmetric grid = {max_even_diff:.3e}")

    assert max_even_diff < 1e-6

    # --------------------------------------------------
    # Test 26: trace_branches_over_T should follow a broken branch at low T and eventually switch to symmetric
    # --------------------------------------------------
    T_grid_trace = np.linspace(2.0, 8.0, 13)

    trace_data = trace_branches_over_T(
        mu=0.0,
        T_grid=T_grid_trace,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    print(f"trace branches = {trace_data['branch']}")
    print(f"trace phi2     = {trace_data['phi2']}")

    assert len(trace_data["T"]) == len(T_grid_trace)
    assert len(trace_data["eta2"]) == len(T_grid_trace)
    assert len(trace_data["phi2"]) == len(T_grid_trace)
    assert len(trace_data["phi"]) == len(T_grid_trace)
    assert len(trace_data["is_broken"]) == len(T_grid_trace)
    assert len(trace_data["branch"]) == len(T_grid_trace)

    assert np.all(np.isfinite(trace_data["eta2"]))
    assert np.all(np.isfinite(trace_data["phi2"]))
    assert np.all(np.isfinite(trace_data["phi"]))
    assert np.all(trace_data["phi2"] >= 0.0)

    assert trace_data["branch"][0] == "broken"
    assert trace_data["branch"][-1] == "symmetric"

    sym_started = False
    for label in trace_data["branch"]:
        if label == "symmetric":
            sym_started = True
        if sym_started:
            assert label == "symmetric"

    print("All scan and continuation tests passed.")
    return trace_data


def run_observable_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
    T_on: float,
    mu_on: float,
) -> dict[str, np.ndarray]:
    print("\n=== Observable tests ===")

    # --------------------------------------------------
    # Test 27: eta2_solution must agree with veff_on_shell output
    # --------------------------------------------------
    eta2_obs = eta2_solution(
        phi=0.30,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    veff_obs, eta2_from_veff = veff_on_shell(
        phi=0.30,
        T=T_on,
        mu=mu_on,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"eta2_solution    = {eta2_obs:.12f}")
    print(f"eta2 via veff    = {eta2_from_veff:.12f}")
    print(f"veff_on_shell    = {veff_obs:.12f}")

    assert np.isclose(eta2_obs, eta2_from_veff, rtol=1e-9, atol=1e-10)

    # --------------------------------------------------
    # Test 28: phi_min should identify phi ~ 0 in a clearly symmetric high-temperature regime
    # --------------------------------------------------
    phi_grid_min = np.linspace(-2.0, 2.0, 81)

    phi_star, V_star = phi_min(
        T=8.0,
        mu=0.0,
        phi_grid=phi_grid_min,
        eta2_seed=2.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    print(f"phi_min at high T = {phi_star:.12f}")
    print(f"V_min at high T   = {V_star:.12f}")

    assert np.isclose(phi_star, 0.0, atol=1e-14)
    assert np.isfinite(V_star)

    # --------------------------------------------------
    # Test 29: Tc_pt should reproduce the section-2 benchmark
    # --------------------------------------------------
    mu_pt = 0.5
    Tc_pt_val = Tc_pt(mu_pt, params)
    Tc_pt_expected = np.sqrt(18.0 * (mu_pt**2 - params.m2))

    print(f"Tc_pt(mu=0.5)          = {Tc_pt_val:.12f}")
    print(f"Tc_pt expected formula = {Tc_pt_expected:.12f}")

    assert np.isclose(Tc_pt_val, Tc_pt_expected, rtol=1e-12, atol=1e-12)

    # --------------------------------------------------
    # Test 30: Tc_opt should match the first broken->symmetric switch temperature
    # --------------------------------------------------
    T_grid_tc = np.linspace(2.0, 8.0, 13)

    trace_tc = trace_branches_over_T(
        mu=0.0,
        T_grid=T_grid_tc,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    Tc_opt_val = Tc_opt(
        mu=0.0,
        T_grid=T_grid_tc,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    first_switch_T = None
    for i in range(1, len(trace_tc["T"])):
        if trace_tc["is_broken"][i - 1] and not trace_tc["is_broken"][i]:
            first_switch_T = float(trace_tc["T"][i])
            break

    print(f"Tc_opt = {Tc_opt_val:.12f}")
    print(f"First switch temperature from trace = {first_switch_T:.12f}")

    assert first_switch_T is not None
    assert np.isclose(Tc_opt_val, first_switch_T, atol=1e-14)

    print("All observable tests passed.")
    return trace_tc


def run_plotting_and_reproduction_helper_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    print("\n=== Plotting and notebook reproduction helper tests ===")

    # --------------------------------------------------
    # Test 31: reproduce_section3_scan default notebook-like setup
    # --------------------------------------------------
    sec3_data = reproduce_section3_scan()

    print(f"section 3 scan keys      = {list(sec3_data.keys())}")
    print(f"section 3 quantity       = {sec3_data['quantity'][0]}")
    print(f"section 3 number of pts  = {len(sec3_data['phi'])}")
    print(f"section 3 T              = {sec3_data['T'][0]:.12f}")
    print(f"section 3 mu             = {sec3_data['mu'][0]:.12f}")

    assert len(sec3_data["phi"]) == 101
    assert np.isclose(sec3_data["phi"][0], -0.5, atol=1e-14)
    assert np.isclose(sec3_data["phi"][-1], 0.5, atol=1e-14)
    assert np.isclose(sec3_data["dphi"][0], 0.01, atol=1e-14)
    assert sec3_data["quantity"][0] == "deltaV"
    assert np.all(np.isfinite(sec3_data["values"]))
    assert np.all(np.isfinite(sec3_data["eta2"]))
    assert np.all(np.isfinite(sec3_data["eta2_zero"]))

    idx_zero_sec3 = np.argmin(np.abs(sec3_data["phi"]))
    print(f"section 3 DeltaV(phi=0) = {sec3_data['values'][idx_zero_sec3]:.12e}")
    assert np.isclose(sec3_data["phi"][idx_zero_sec3], 0.0, atol=1e-14)
    assert np.isclose(sec3_data["values"][idx_zero_sec3], 0.0, atol=1e-10)

    # --------------------------------------------------
    # Test 32: generic reproduction helper for another T, mu and phi-range
    # --------------------------------------------------
    generic_scan = reproduce_section3_scan(
        phi_max=0.4,
        dphi=0.02,
        T=6.0,
        mu=0.3,
        eta2_seed=2.0,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
        subtract_origin=False,
    )

    print(f"generic scan number of pts = {len(generic_scan['phi'])}")
    print(f"generic scan quantity      = {generic_scan['quantity'][0]}")
    print(f"generic scan min(V)        = {np.min(generic_scan['values']):.12f}")
    print(f"generic scan max(V)        = {np.max(generic_scan['values']):.12f}")

    assert len(generic_scan["phi"]) == 41
    assert generic_scan["quantity"][0] == "veff"
    assert np.all(np.isfinite(generic_scan["values"]))
    assert np.all(np.isfinite(generic_scan["eta2"]))

    # --------------------------------------------------
    # Test 33: plot_potential_scan should run without error
    # --------------------------------------------------
    plot_potential_scan(sec3_data, show=SHOW_PLOTS)
    plot_potential_scan(generic_scan, show=SHOW_PLOTS)

    # --------------------------------------------------
    # Test 34: plot_branches should run without error
    # --------------------------------------------------
    trace_plot_data = trace_branches_over_T(
        mu=0.0,
        T_grid=np.linspace(2.0, 8.0, 13),
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    plot_branches(trace_plot_data, show=SHOW_PLOTS)

    print("All plotting and notebook reproduction helper tests passed.")
    return sec3_data, generic_scan


def run_new_extra_tests(
    params: OPTModelParams,
    thermal_nb: ThermalOptions,
    thermal_auto: ThermalOptions,
    solver_opts: SolverOptions,
    trace_data_mu0: dict[str, np.ndarray],
) -> None:
    print("\n=== Extra tests and requested figures ===")

    # --------------------------------------------------
    # Test 35: continuation=False should give a scan close to continuation=True
    # --------------------------------------------------
    phi_grid_cmp = np.linspace(-0.5, 0.5, 21)

    solver_no_cont = SolverOptions(
        root_tol=solver_opts.root_tol,
        max_iter=solver_opts.max_iter,
        continuation=False,
    )

    scan_cont = scan_potential(
        phi_grid=phi_grid_cmp,
        T=4.733359332047422 + 0.027,
        mu=0.5,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
        subtract_origin=True,
    )

    scan_no_cont = scan_potential(
        phi_grid=phi_grid_cmp,
        T=4.733359332047422 + 0.027,
        mu=0.5,
        eta2_seed=2.1214029070500127,
        params=params,
        thermal=thermal_nb,
        solver=solver_no_cont,
        subtract_origin=True,
    )

    max_diff_cont = np.max(np.abs(scan_cont["values"] - scan_no_cont["values"]))
    print(f"Max |DeltaV(cont=True) - DeltaV(cont=False)| = {max_diff_cont:.3e}")

    assert np.all(np.isfinite(scan_no_cont["values"]))
    assert np.all(np.isfinite(scan_no_cont["eta2"]))
    assert max_diff_cont < 1e-5

    # --------------------------------------------------
    # Test 36: dense minima figure with many points
    # --------------------------------------------------
    T_grid_dense = np.linspace(2.0, 8.0, 81)

    T_mu0, phiabs_mu0 = _dense_branch_phi_curve(
        mu=0.0,
        T_grid=T_grid_dense,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    T_mu05, phiabs_mu05 = _dense_branch_phi_curve(
        mu=0.5,
        T_grid=T_grid_dense,
        eta2_seed=2.2,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"dense minima curve mu=0   first/last = {phiabs_mu0[0]:.6f}, {phiabs_mu0[-1]:.6f}")
    print(f"dense minima curve mu=0.5 first/last = {phiabs_mu05[0]:.6f}, {phiabs_mu05[-1]:.6f}")

    assert np.all(np.isfinite(phiabs_mu0))
    assert np.all(np.isfinite(phiabs_mu05))
    assert np.all(phiabs_mu0 >= 0.0)
    assert np.all(phiabs_mu05 >= 0.0)

    _plot_dense_minima_curves(T_mu0, phiabs_mu0, T_mu05, phiabs_mu05)

    # --------------------------------------------------
    # Test 37: section-2-style phi(T) comparison for mu/M=0 and mu/M=0.5
    # --------------------------------------------------
    T_grid_section2 = np.linspace(2.0, 8.0, 41)

    trace_mu0 = trace_branches_over_T(
        mu=0.0,
        T_grid=T_grid_section2,
        eta2_seed=2.0,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_auto,
        solver=solver_opts,
    )

    trace_mu05 = trace_branches_over_T(
        mu=0.5,
        T_grid=T_grid_section2,
        eta2_seed=2.1214029070500127,
        phi2_seed=4.0,
        params=params,
        thermal=thermal_nb,
        solver=solver_opts,
    )

    print(f"mu=0   final branch label = {trace_mu0['branch'][-1]}")
    print(f"mu=0.5 final branch label = {trace_mu05['branch'][-1]}")

    assert np.all(np.isfinite(trace_mu0["phi"]))
    assert np.all(np.isfinite(trace_mu05["phi"]))
    assert np.all(trace_mu0["phi"] >= 0.0)
    assert np.all(trace_mu05["phi"] >= 0.0)

    _plot_section2_phi_vs_T_mu_compare(trace_mu0, trace_mu05)

    _show_or_close_plots()
    print("All extra tests and requested figure builds passed.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    params = OPTModelParams(m2=-1.0, lam=1.0, M=1.0)

    thermal_nb = ThermalOptions(
        mode_mu0="notebook_highT",
        mode_muneq0="notebook_highT",
    )

    thermal_auto = ThermalOptions(
        mode_mu0="auto",
        mode_muneq0="notebook_highT",
        ct_mu0_approx="exact",
        ct_mu0_n=20,
    )

    solver_opts = SolverOptions(
        root_tol=1e-12,
        max_iter=300,
        continuation=True,
    )

    run_low_level_utility_tests(params)
    run_thermal_backend_tests(params, thermal_nb)

    T_root, mu_root, phi_root, eta2_root = run_core_equation_tests(params, thermal_nb)

    T_broken, mu_broken, _ = run_root_solver_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
        T_root=T_root,
        mu_root=mu_root,
        phi_root=phi_root,
        eta2_root=eta2_root,
    )

    run_branch_solver_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
        T_root=T_root,
        mu_root=mu_root,
        T_broken=T_broken,
        mu_broken=mu_broken,
    )

    T_on, mu_on = run_on_shell_tests(
        params=params,
        thermal_nb=thermal_nb,
        solver_opts=solver_opts,
        T_root=T_root,
        mu_root=mu_root,
    )

    trace_data_mu0 = run_scan_and_continuation_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
        T_on=T_on,
        mu_on=mu_on,
    )

    run_observable_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
        T_on=T_on,
        mu_on=mu_on,
    )

    run_plotting_and_reproduction_helper_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
    )

    run_new_extra_tests(
        params=params,
        thermal_nb=thermal_nb,
        thermal_auto=thermal_auto,
        solver_opts=solver_opts,
        trace_data_mu0=trace_data_mu0,
    )

    print("\nAll OPT module tests completed successfully.")