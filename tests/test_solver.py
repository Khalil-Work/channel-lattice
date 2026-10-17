"""Solver construction, input validation, and discrete conservation tests."""

import numpy as np
import pytest

from channel_lattice import ChannelFlowLBM, ChannelLatticeError


@pytest.fixture(scope="module")
def small_flow():
    return ChannelFlowLBM(nx=8, ny=13, tau=0.8,
                          force_g=8 * 0.1 * 0.05 / 121)  # u_max ~ 0.05


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("tau", [0.5, 0.49, 0.0, -1.0, 20.5, float("inf"),
                                 float("nan")])
def test_tau_out_of_range_rejected(tau):
    with pytest.raises(ChannelLatticeError, match="tau"):
        ChannelFlowLBM(nx=16, ny=17, tau=tau, force_g=1e-4)


def test_tau_error_message_explains_viscosity():
    with pytest.raises(ChannelLatticeError, match="viscosity"):
        ChannelFlowLBM(nx=16, ny=17, tau=0.5, force_g=1e-4)


@pytest.mark.parametrize("ny", [24, 26, 100])
def test_even_ny_rejected_by_parity_rule(ny):
    with pytest.raises(ChannelLatticeError, match="odd"):
        ChannelFlowLBM(nx=16, ny=ny, tau=0.9, force_g=1e-4)


def test_too_small_grids_rejected():
    with pytest.raises(ChannelLatticeError, match="nx"):
        ChannelFlowLBM(nx=2, ny=17, tau=0.9, force_g=1e-4)
    with pytest.raises(ChannelLatticeError, match="ny"):
        ChannelFlowLBM(nx=16, ny=4, tau=0.9, force_g=1e-4)


@pytest.mark.parametrize("g", [0.0, -1e-4, -1.0, float("nan")])
def test_non_positive_forcing_rejected(g):
    with pytest.raises(ChannelLatticeError, match="force_g"):
        ChannelFlowLBM(nx=16, ny=17, tau=0.9, force_g=g)


def test_high_speed_forcing_rejected_with_low_mach_guidance():
    # A force that would drive u_max far above the low-Mach limit must be
    # refused with an actionable message.
    with pytest.raises(ChannelLatticeError, match="low-Mach"):
        ChannelFlowLBM(nx=16, ny=17, tau=0.9, force_g=0.5)


def test_non_integer_grid_raises_type_error():
    with pytest.raises(TypeError):
        ChannelFlowLBM(nx=16.5, ny=17, tau=0.9, force_g=1e-4)
    with pytest.raises(TypeError):
        ChannelFlowLBM(nx=16, ny=17.0, tau=0.9, force_g=1e-4)


def test_invalid_density_rejected():
    with pytest.raises(ChannelLatticeError):
        ChannelFlowLBM(nx=16, ny=17, tau=0.9, force_g=1e-4, rho0=-1.0)


# --------------------------------------------------------------------- #
# Construction conventions
# --------------------------------------------------------------------- #
def test_geometry_conventions(small_flow):
    flow = small_flow
    assert flow.h_lattice == 11                 # ny - 2 fluid links
    assert flow.wall_positions == (0.5, 11.5)   # half-way bounce-back walls
    assert flow.centerline_j == 6.0             # on-node centreline
    assert flow.nu == pytest.approx((0.8 - 0.5) / 3.0)
    assert flow.omega == pytest.approx(1.0 / 0.8)


def test_initial_state_is_rest_at_uniform_density(small_flow):
    flow = small_flow
    rho, ux, uy = flow.macroscopic()
    np.testing.assert_allclose(rho, flow.rho0, rtol=0, atol=1e-14)
    # Guo half-force correction: u = (sum f c)/rho + g/2 = g/2 from rest.
    np.testing.assert_allclose(ux, 0.5 * flow.force_g, rtol=0, atol=1e-18)
    np.testing.assert_allclose(uy, 0.0, atol=1e-16)


# --------------------------------------------------------------------- #
# Discrete conservation laws
# --------------------------------------------------------------------- #
def test_mass_is_conserved_to_roundoff(small_flow):
    flow = small_flow
    m0 = flow.mass()
    for _ in range(400):
        flow.step()
    assert abs(flow.mass() - m0) / m0 < 1e-10


def test_exact_discrete_momentum_balance(small_flow):
    # At EVERY step the raw fluid momentum change must equal the force
    # impulse plus the tracked bounce-back wall exchange, to round-off.
    flow = small_flow
    for _ in range(50):
        flow.step()
        f_tot = flow.last_force_momentum_x
        assert abs(flow.last_momentum_residual_x) < 1e-10 * abs(f_tot)
        assert abs(flow.last_momentum_residual_y) < 1e-10 * abs(f_tot)


def test_flow_moves_in_forcing_direction(small_flow):
    flow = small_flow
    for _ in range(200):
        flow.step()
    _, ux, uy = flow.macroscopic()
    assert float(ux.max()) > 0.01 * flow.u_max_analytic
    np.testing.assert_allclose(uy, 0.0, atol=1e-12)
