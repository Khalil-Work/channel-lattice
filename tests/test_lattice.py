"""Unit tests for the D2Q9 lattice definition, equilibrium, and Guo source."""

import numpy as np
import pytest

from channel_lattice.lattice import D2Q9


@pytest.fixture(scope="module")
def lat():
    return D2Q9.standard()


def test_velocity_set_layout(lat):
    # The exact D2Q9 stencil: rest, 4 axis links, 4 diagonal links.
    expected = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
         [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    np.testing.assert_array_equal(lat.velocities, expected)


def test_weights_are_exact(lat):
    np.testing.assert_allclose(
        lat.weights,
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        rtol=0, atol=1e-15,
    )
    assert lat.weights.sum() == pytest.approx(1.0, abs=1e-15)


def test_isotropic_moment_conditions(lat):
    # Zeroth/first/second moments of the quadrature: isotropy conditions
    # that make the scheme consistent with the continuum equations.
    w, c = lat.weights, lat.velocities.astype(float)
    np.testing.assert_allclose(w @ c, 0.0, atol=1e-15)
    second = np.einsum("q,qa,qb->ab", w, c, c)
    np.testing.assert_allclose(second, lat.cs2 * np.eye(2), atol=1e-15)
    assert lat.cs2 == pytest.approx(1.0 / 3.0)


def test_opposite_direction_map(lat):
    opp, c = lat.opposite, lat.velocities
    for q in range(9):
        np.testing.assert_array_equal(c[opp[q]], -c[q])
        assert opp[opp[q]] == q


def test_equilibrium_conserves_mass_and_momentum(lat):
    rng = np.random.default_rng(42)
    rho = 1.0 + 0.02 * rng.standard_normal((6, 5))
    u = 0.05 * rng.standard_normal((2, 6, 5))
    feq = lat.equilibrium(rho, u[0], u[1])
    np.testing.assert_allclose(feq.sum(axis=0), rho, rtol=0, atol=1e-13)
    c = lat.velocities.astype(float)
    mom_x = np.tensordot(c[:, 0], feq, axes=(0, 0))
    mom_y = np.tensordot(c[:, 1], feq, axes=(0, 0))
    np.testing.assert_allclose(mom_x, rho * u[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(mom_y, rho * u[1], rtol=0, atol=1e-13)


def test_guo_source_zeroth_and_first_moments(lat):
    # Guo et al. (2002) consistency: sum_q S_q = 0 (no mass input) and
    # sum_q S_q c_q = F = rho*g*e_x (exact discrete force).
    rng = np.random.default_rng(7)
    rho = 1.0 + 0.01 * rng.standard_normal((4, 4))
    u = 0.06 * rng.standard_normal((2, 4, 4))
    g = 1.3e-4
    src = lat.guo_source(rho, u[0], u[1], g)
    c = lat.velocities.astype(float)
    np.testing.assert_allclose(src.sum(axis=0), 0.0, atol=1e-18)
    fx = np.tensordot(c[:, 0], src, axes=(0, 0))
    fy = np.tensordot(c[:, 1], src, axes=(0, 0))
    np.testing.assert_allclose(fx, rho * g, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(fy, 0.0, atol=1e-18)


def test_invalid_lattice_rejected():
    from channel_lattice.exceptions import ChannelLatticeError
    c = np.zeros((9, 2))
    w = np.full(9, 1.0 / 9.0)
    with pytest.raises(ChannelLatticeError):
        D2Q9(velocities=c, weights=w)  # weights do not sum to 1
