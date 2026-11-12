"""Importa o mesmo núcleo instalado ou a partir de src na raiz do checkout."""

from importlib import import_module

try:
    CosmoTransitions = import_module("CosmoTransitions")
except ModuleNotFoundError as error:
    if error.name != "CosmoTransitions":
        raise
    CosmoTransitions = import_module("src.CosmoTransitions")

finiteT = import_module(f"{CosmoTransitions.__name__}.finiteT")
generic_potential = import_module(f"{CosmoTransitions.__name__}.generic_potential")
gravitational_Waves = import_module(f"{CosmoTransitions.__name__}.gravitational_Waves")
helper_functions = import_module(f"{CosmoTransitions.__name__}.helper_functions")

Jb = finiteT.Jb
Jf = finiteT.Jf
scalar_to_vector_potential_1d = generic_potential.scalar_to_vector_potential_1d
build_finite_T_derivatives = generic_potential.build_finite_T_derivatives
_build_phases_and_transitions = generic_potential._build_phases_and_transitions
gradientFunction = helper_functions.gradientFunction
hessianFunction = helper_functions.hessianFunction
GravitationalWaveCalculator = gravitational_Waves.GravitationalWaveCalculator
bbo_sensitivity_s_pis = gravitational_Waves.bbo_sensitivity_s_pis
decigo_sensitivity_s_pis = gravitational_Waves.decigo_sensitivity_s_pis
lisa_sensitivity_s_pis = gravitational_Waves.lisa_sensitivity_s_pis
gw_f_sw_peak = gravitational_Waves.gw_f_sw_peak
gw_omega_sw_h2 = gravitational_Waves.gw_omega_sw_h2
gw_omega_total_h2 = gravitational_Waves.gw_omega_total_h2
