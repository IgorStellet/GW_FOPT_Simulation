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
                       _series_tail_sum, Jb_low, Jf_low, _select_K, x2K2, dx2K2, d2x2K2, d3x2K2, Jb_high, Jf_high)

__version__ = "0.1.0"
