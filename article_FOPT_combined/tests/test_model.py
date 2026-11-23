"""Identidades físicas e contagem de modos, independentes do resolvedor de nucleação."""

import math

import numpy as np
import pytest
from article_FOPT_combined import core
from article_FOPT_combined.model import Parameters, Potential


@pytest.mark.parametrize(
    "m6,m8", [(math.inf, math.inf), (1000, math.inf), (1000, 668.740304976422), (1500, 400)]
)
def test_full_zero_temperature_potential_preserves_v_and_curvature(m6, m8):
    model = Potential(Parameters(m6, m8, 1))
    step = 0.125
    far_left, left, center, right, far_right = model(model.v + step * np.arange(-2, 3), 0)
    slope = (far_left - 8 * left + 8 * right - far_right) / (12 * step)
    curvature = (-far_left + 16 * left - 30 * center + 16 * right - far_right) / (12 * step**2)
    assert slope == pytest.approx(0, abs=0.01)
    assert curvature == pytest.approx(model.mh**2, abs=0.01)


def test_coleman_weinberg_matches_sum_of_renormalized_w_z_top_terms():
    model = Potential(Parameters(900, 400, 2))
    phi = np.array([40.0, 180.0, 246.0, 360.0])
    masses = model.masses_squared(phi)
    physical = model.masses_squared(model.v)
    expected = sum(
        n
        * (
            masses[name] ** 2 * (np.log(masses[name] / physical[name]) - 1.5)
            + 2 * masses[name] * physical[name]
        )
        for name, n in (("W", 6), ("Z", 3), ("top", -12))
    ) / (64 * np.pi**2)
    np.testing.assert_allclose(model.coleman_weinberg(phi), expected)
    assert model.coleman_weinberg(0) == 0


def test_direct_loop_terms_depend_only_on_standard_model_inputs():
    reference = Potential(Parameters())
    combined = Potential(Parameters(900, 400, 2))
    assert set(combined.masses_squared(180)) == {"W", "Z", "top"}
    for field in (0, 180, 240):
        assert combined.coleman_weinberg(field) == reference.coleman_weinberg(field)
        assert combined.thermal(field, 95) == reference.thermal(field, 95)
        assert combined.gauge_rings(field, 95) == reference.gauge_rings(field, 95)


def test_measure_only_zero_temperature_degeneracy_brackets_published_value():
    # No limite puro e em T=0, V(v)=V(0) ocorre perto de C=4.14
    # na Eq. (20) de Berganholi et al., arXiv:2502.09593.
    below = Potential(Parameters(C=4.13))
    above = Potential(Parameters(C=4.15))
    assert below(below.v, 0) < below(0, 0)
    assert above(above.v, 0) > above(0, 0)


def test_longitudinal_matrix_eigenvalues_and_zero_temperature():
    model = Potential(Parameters())
    phi, T = 150, 100
    w, z, photon = model.longitudinal_masses_squared(phi, T)
    w_bare = model.mw**2 * (phi / model.v) ** 2
    z_bare = model.mz**2 * (phi / model.v) ** 2
    w_expected = w_bare + 11 * model.g**2 * T**2 / 6
    matrix = np.array(
        [
            [w_expected, -model.g * model.gp * phi**2 / 4],
            [
                -model.g * model.gp * phi**2 / 4,
                model.gp**2 * phi**2 / 4 + 11 * model.gp**2 * T**2 / 6,
            ],
        ]
    )
    neutral = np.linalg.eigvalsh(matrix)
    np.testing.assert_allclose([photon, z], neutral)
    assert w == pytest.approx(w_expected)
    expected_ring = (
        -T / (12 * np.pi) * (2 * (w_expected**1.5 - w_bare**1.5) + sum(neutral**1.5) - z_bare**1.5)
    )
    assert model.gauge_rings(phi, T) == pytest.approx(expected_ring)
    w0, z0, p0 = model.longitudinal_masses_squared(phi, 0)
    assert p0 == pytest.approx(0, abs=1e-10)
    assert z0 == pytest.approx(model.masses_squared(phi)["Z"])
    assert w0 == pytest.approx(model.masses_squared(phi)["W"])
    assert model.gauge_rings(phi, 0) == pytest.approx(0)
    assert model.thermal(phi, 0) == pytest.approx(0)


def test_spline_inputs_are_squared_mass_ratios(monkeypatch):
    model = Potential(Parameters(1000, 700, 2))
    calls = []

    def spline(species, value):
        def evaluate(x, *, approx):
            assert approx == "spline"
            calls.append((species, float(np.asarray(x).item())))
            return value * np.ones_like(x)

        return evaluate

    monkeypatch.setattr(core.finiteT, "Jb", spline("boson", -2))
    monkeypatch.setattr(core.finiteT, "Jf", spline("fermion", -1))
    value = model.thermal(180, 90)
    assert [species for species, _ in calls] == ["boson", "boson", "fermion"]
    np.testing.assert_allclose(
        [argument for _, argument in calls],
        [(mass * 180 / model.v / 90) ** 2 for mass in (model.mw, model.mz, model.mt)],
    )
    assert value == pytest.approx((-2 * (6 + 3) - 12) * 90**4 / (2 * np.pi**2))


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
