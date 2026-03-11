"""CosmoTransitions (modernized): tools for phase transitions and GW forecasts."""

from .helper_functions import (set_default_args, monotonic_indices, clamp_val,
                               _rkck, rkqs,
                               deriv14, deriv14_const_dx, deriv23, deriv23_const_dx, deriv1n, gradientFunction,hessianFunction,
                               makeInterpFuncs, cubicInterpFunction, _safe_div, Nbspl, Nbspld1, Nbspld2)

from .finiteT import  (_asarray, _is_scalar, _apply_elementwise,
                      _Jf_exact_scalar, _Jb_exact_scalar, _Jf_exact2_scalar, _Jb_exact2_scalar,_dJf_exact_scalar, _dJb_exact_scalar,
                      Jf_exact, Jf_exact2, Jb_exact, Jb_exact2, dJf_exact, dJb_exact,
                       _build_Jb_dataset, _build_Jf_dataset, _save_Jb_cache, _save_Jf_cache, _ensure_Jb_spline, _ensure_Jf_spline,
                       Jb_spline, Jf_spline, _JF_CACHE_FILE, _JB_CACHE_FILE,
                       _series_tail_sum, Jb_low, Jf_low, _select_K, x2K2, dx2K2, d2x2K2, d3x2K2, Jb_high, Jf_high,
                       Jb, Jf)

from .tunneling1D import (PotentialError, SingleFieldInstanton)

from .transitionFinder import (traceMinimum, _traceMinimum_rval, Phase, traceMultiMin, findApproxLocalMin, removeRedundantPhases,
                               _removeRedundantPhase, getStartPhase, _solve_bounce, _tunnelFromPhaseAtT, _potentialDiffForPhase,
                               _maxTCritForPhase, tunnelFromPhase, secondOrderTrans, findAllTransitions, findCriticalTemperatures,
                               addCritTempsForFullTransitions, )

from .gravitational_Waves import (GravitationalWaveCalculator, gw_f_coll_peak, gw_f_sw_peak, gw_f_turb_peak, gw_omega_coll_h2,
                                  gw_h_star_Hz, gw_omega_turb_h2, gw_omega_sw_h2, gw_omega_total_h2, lisa_sensitivity_s_pis, bbo_sensitivity_s_pis, decigo_sensitivity_s_pis,)

from .generic_potential import (PotentialDerivatives, scalar_to_vector_potential_1d, build_finite_T_derivatives, ensure_dir,
                                savefig, tee_stdout, build_phi_grid, _extract_params_from_V, _build_phases_and_transitions,
                                _spinodal_data_for_phase, _closest_spinodal_to_T, _build_gw_calculator_from_summary,
                                _compute_gw_scales_from_calculator, compute_profile, gather_diagnostics, save_diagnostics_summary,)


from .OPT import (OPTModelParams, SolverOptions, ThermalOptions, effective_mass_sq, validate_effective_mass_sq, thermal_variables,
    h_e_odd, H1_notebook_highT, H3_notebook_highT, H5_notebook_highT, H3_ct_mu0, H5_ct_mu0, H3, H5, opt_veff_off_shell,
    eta_gap_residual, phi_stationary_residual, solve_eta2_given_phi, solve_stationary_system, solve_symmetric_branch,
    solve_broken_branch, veff_on_shell, veff_difference_from_origin, scan_potential, trace_branches_over_T, phi_min,
    eta2_solution, Tc_opt, Tc_pt, reproduce_section3_scan, plot_potential_scan, plot_branches, )

__version__ = "0.1.0"
