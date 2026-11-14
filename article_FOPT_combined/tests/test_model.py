"""Identidades físicas e contagem de modos, independentes do resolvedor de nucleação."""

import math

import numpy as np
import pytest
from article_FOPT_combined import core
from article_FOPT_combined.model import Parameters, Potential


@pytest.mark.parametrize(
    "m6,m8", [(math.inf, math.inf), (1000, math.inf), (1000, 668.740304976422), (1500, 400)]
)
def test_tree_higgs_and_goldstone_masses_at_v(m6, m8):
    model = Potential(Parameters(m6, m8, 1))
    masses = model.masses_squared(model.v)
    assert masses["h"] == pytest.approx(125**2, abs=1e-9)
    assert masses["G"] == pytest.approx(0, abs=1e-9)


def test_thermal_masses_from_independent_trace_hessian():
    # Laplaciano O(4) dos monômios: Delta r^(2n)=4n(n+1)r^(2n-2).
    # Sua curvatura radial/transversal dá as autoenergias escalares T²/24.
    model = Potential(Parameters(900, 400, 0.5))
    phi, T = 180, 90
    scalar_h = scalar_G = 0
    for n, coefficient in ((2, model.lam / 4), (3, 1 / (8 * 900**2)), (4, 1 / (16 * 400**4))):
        power = 2 * n - 2
        common = coefficient * 4 * n * (n + 1) * T**2 / 24
        scalar_h += common * power * (power - 1) * phi ** (power - 2)
        scalar_G += common * power * phi ** (power - 2)
    gauge_top = ((3 * model.g**2 + model.gp**2) / 16 + model.yt**2 / 4) * T**2
    h, G = model.scalar_self_energies(phi, T)
    assert h == pytest.approx(gauge_top + scalar_h)
    assert G == pytest.approx(gauge_top + scalar_G)


def test_symmetric_phase_and_pure_sm_limit():
    model = Potential(Parameters())
    h, G = model.scalar_self_energies(np.array([0, 200]), 100)
    expected = (
        (3 * model.g**2 + model.gp**2) / 16 + model.yt**2 / 4 + 125**2 / (4 * 246**2)
    ) * 100**2
    np.testing.assert_allclose(h, expected)
    np.testing.assert_allclose(G, expected)


def test_measure_does_not_enter_hard_one_loop_scalar_vertices():
    a, b = Potential(Parameters(1000, 700, 0)), Potential(Parameters(1000, 700, 3))
    np.testing.assert_allclose(a.scalar_self_energies(200, 100), b.scalar_self_energies(200, 100))


def test_no_T4_flowers_in_chosen_approximation():
    model = Potential(Parameters(1000, 400, 0.5))
    np.testing.assert_allclose(
        model.scalar_self_energies(200, 200), 4 * np.array(model.scalar_self_energies(200, 100))
    )


def test_scalar_ring_multiplicity_and_subtraction():
    model = Potential(Parameters(1000, 800, 2))
    phi, T = 220, 100
    masses = model.masses_squared(phi)
    ph, pg = model.scalar_self_energies(phi, T)
    expected = (
        -T
        / (12 * np.pi)
        * np.real(
            complex(masses["h"] + ph) ** 1.5
            - complex(masses["h"]) ** 1.5
            + 3 * (complex(masses["G"] + pg) ** 1.5 - complex(masses["G"]) ** 1.5)
        )
    )
    assert model.scalar_rings(phi, T) == pytest.approx(expected)
    assert abs(model.scalar_rings(phi, T)) > 1


def test_real_cubic_keeps_analytic_continuation():
    np.testing.assert_allclose(Potential.real_cubic([-4, 0, 4]), [0, 0, 8])


def test_longitudinal_matrix_eigenvalues_and_zero_temperature():
    model = Potential(Parameters())
    phi, T = 150, 100
    w, z, photon = model.longitudinal_masses_squared(phi, T)
    matrix = np.array(
        [
            [w, -model.g * model.gp * phi**2 / 4],
            [
                -model.g * model.gp * phi**2 / 4,
                model.gp**2 * phi**2 / 4 + 11 * model.gp**2 * T**2 / 6,
            ],
        ]
    )
    np.testing.assert_allclose([photon, z], np.linalg.eigvalsh(matrix))
    w0, z0, p0 = model.longitudinal_masses_squared(phi, 0)
    assert p0 == pytest.approx(0, abs=1e-10)
    assert z0 == pytest.approx(model.masses_squared(phi)["Z"])
    assert w0 == pytest.approx(model.masses_squared(phi)["W"])
    assert model.gauge_rings(phi, 0) == pytest.approx(0)
    assert model.scalar_rings(phi, 0) == pytest.approx(0)
    assert model.thermal(phi, 0) == pytest.approx(0)


def test_spline_inputs_are_squared_mass_ratios(monkeypatch):
    model = Potential(Parameters(1000, 700, 2))
    calls = []

    def spline(x, *, approx):
        assert approx == "spline"
        calls.append(np.asarray(x))
        return np.ones_like(x)

    monkeypatch.setattr(core.finiteT, "Jb", spline)
    monkeypatch.setattr(core.finiteT, "Jf", spline)
    value = model.thermal(180, 90)
    assert len(calls) == 5
    assert calls[0][0] == pytest.approx(model.masses_squared(180)["h"] / 90**2)
    assert value == pytest.approx(25 * 90**4 / (2 * np.pi**2))


def test_domain_and_broadcasting():
    model = Potential(Parameters(1000, 800, 3))
    assert np.isnan(model.measure(model.domain_limit * 1.001))
    values = model(np.array([[0], [100], [246]]), np.array([0, 100]))
    assert values.shape == (3, 2)
    assert np.all(np.isfinite(values))
    with pytest.raises(ValueError):
        model(0, -1)


@pytest.mark.parametrize(
    "kwargs", [{"m6_GeV": -1}, {"m8_GeV": float("nan")}, {"C": -1}, {"C": 100}]
)
def test_invalid_physical_parameters(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_parameter_identifiers_do_not_depend_on_int_vs_float():
    from article_FOPT_combined.storage import point_id

    assert point_id(Parameters(1000, math.inf, 2)) == point_id(Parameters(1000.0, math.inf, 2.0))
