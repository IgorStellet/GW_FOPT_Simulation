"""Scientific regression tests for the numerical APIs used by Articles.

Run with ``python -m unittest discover -s tests -p test_article_core.py``.
Analytic examples isolate differentiation, thermodynamic conventions, and
frequency units; the article smoke run exercises the actual bounce separately.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from CosmoTransitions.gravitational_Waves import (
    GravitationalWaveCalculator,
    gw_f_sw_peak,
    gw_f_turb_peak,
    gw_h_star_Hz,
    gw_omega_turb_h2,
)


class TemperatureDerivativeTests(unittest.TestCase):
    def calculator(self, action_over_T):
        calculator = object.__new__(GravitationalWaveCalculator)
        calculator._T_min, calculator._T_max = 1.0, 20.0
        self.evaluations = []

        def action(T):
            self.evaluations.append(T)
            return T * action_over_T(T)

        calculator._S3_at_T = action
        return calculator

    def test_second_order_uses_only_two_bounces_and_differentiates_S3_over_T(self):
        calculator = self.calculator(lambda T: T**2 + 2 * T + 140)
        self.assertAlmostEqual(calculator.beta_over_H(10, 0.5, order=2), 220)
        self.assertEqual(self.evaluations, [9.5, 10.5])

    def test_fourth_order_default_remains_exact_for_cubic(self):
        calculator = self.calculator(lambda T: T**3)
        self.assertAlmostEqual(calculator.dS_dT(10, 0.5), 300)
        self.assertEqual(self.evaluations, [9, 9.5, 10.5, 11])

    def test_invalid_stencil_is_rejected_before_evaluating_bounces(self):
        calculator = self.calculator(lambda T: T)
        for kwargs in (
            {"T": 1.1, "dT": 0.2, "order": 2},
            {"T": 10, "dT": 0.1, "order": 3},
            {"T": 10, "dT": np.nan, "order": 2},
        ):
            with self.assertRaises(ValueError):
                calculator.dS_dT(**kwargs)
        self.assertEqual(self.evaluations, [])


class ThermodynamicTests(unittest.TestCase):
    @staticmethod
    def calculator():
        # Both minima are analytically at 0 and 1, but the phase interpolation
        # guesses are deliberately inaccurate to check local refinement.
        def potential(x, T):
            phi = float(np.asarray(x).ravel()[0])
            return 100 * phi**2 * (1 - phi) ** 2 - (10 + 2 * T) * (
                3 * phi**2 - 2 * phi**3
            )

        def derivative_T(x, T):
            phi = float(np.asarray(x).ravel()[0])
            return -2 * (3 * phi**2 - 2 * phi**3)

        phases = {
            "high": SimpleNamespace(
                T=np.array([1, 10]), valAt=lambda T: np.array([0.1])
            ),
            "low": SimpleNamespace(
                T=np.array([1, 10]), valAt=lambda T: np.array([0.9])
            ),
        }
        return GravitationalWaveCalculator(
            potential,
            lambda x, T: np.zeros(1),
            derivative_T,
            phases,
            "high",
            "low",
            minima_phitol=1e-8,
        )

    def test_energy_and_trace_anomaly_are_distinct_and_signed(self):
        calculator = self.calculator()
        quantities = calculator.thermodynamics(5, 100)
        self.assertAlmostEqual(quantities["phi_high"][0], 0, places=6)
        self.assertAlmostEqual(quantities["phi_low"][0], 1, places=6)
        self.assertAlmostEqual(quantities["delta_V_GeV4"], 20)
        self.assertAlmostEqual(quantities["delta_dVdT_GeV3"], 2)
        self.assertAlmostEqual(quantities["delta_rho_GeV4"], 10)
        self.assertAlmostEqual(quantities["delta_theta_GeV4"], 17.5)
        self.assertAlmostEqual(
            quantities["alpha_energy"] * quantities["rho_rad_GeV4"], 10
        )
        self.assertEqual(calculator.alpha(5, 100), abs(quantities["alpha_trace"]))
        self.assertAlmostEqual(calculator.alpha(5, 100, return_delta_rho=True)[1], 17.5)


class FrequencyUnitTests(unittest.TestCase):
    def test_peak_frequency_convention_is_millihertz(self):
        self.assertAlmostEqual(gw_f_sw_peak(100, 100, 100, 1), 1.9)
        self.assertAlmostEqual(gw_f_turb_peak(100, 100, 100, 1), 2.7)
        self.assertAlmostEqual(gw_h_star_Hz(100, 100), 1.65e-5)

    def test_turbulence_shape_converts_hubble_hz_to_millihertz(self):
        parameters = {
            "alpha": 0.1,
            "beta_over_H": 100,
            "T_star": 100,
            "g_star": 100,
            "v_w": 1,
            "kappa_turb": 0.1,
        }
        frequency_mHz = 2.7
        # At f=f_turb, x=1. Evaluate the analytic shape with both frequencies
        # expressed in Hz, independently of the implementation's mHz API.
        shape = 1 / (2 ** (11 / 3) * (1 + 8 * np.pi * 0.0027 / 1.65e-5))
        expected = 3.35e-4 / 100 * (0.1 * 0.1 / 1.1) ** 1.5 * shape
        actual = float(gw_omega_turb_h2(frequency_mHz, **parameters))
        self.assertAlmostEqual(actual / expected, 1)
        calculator = object.__new__(GravitationalWaveCalculator)
        from_class = float(calculator.omega_turb_h2(frequency_mHz, **parameters))
        self.assertAlmostEqual(from_class / expected, 1)


if __name__ == "__main__":
    unittest.main()
