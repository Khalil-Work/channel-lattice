"""Unit tests for the relaxation-time/viscosity relation and unit mapping."""

import pytest

from channel_lattice import units


@pytest.mark.parametrize("tau", [0.5001, 0.55, 0.9, 1.0, 1.5, 5.0, 20.0])
def test_lattice_viscosity_relation(tau):
    # Core BGK mapping: nu = cs^2 (tau - 1/2) dt with cs^2 = 1/3, dt = 1.
    assert units.lattice_viscosity(tau) == pytest.approx((tau - 0.5) / 3.0,
                                                         rel=1e-15)


@pytest.mark.parametrize("tau", [0.5, 0.4, 0.0, -1.0])
def test_lattice_viscosity_nonpositive_for_tau_le_half(tau):
    assert units.lattice_viscosity(tau) <= 0.0


def test_physical_viscosity_round_trip():
    dx, dt = 1.0e-5, 1.0e-7  # metres, seconds
    nu_lat = units.lattice_viscosity(0.9)
    nu_phys = units.physical_viscosity(nu_lat, dx, dt)
    # nu_lat = nu_phys * dt / dx^2 must invert the conversion exactly.
    assert nu_lat == pytest.approx(nu_phys * dt / dx ** 2, rel=1e-15)


def test_timestep_realises_target_viscosity():
    dx, nu_phys = 1.0e-5, 1.0e-6  # water-like, in SI
    for tau in (0.55, 0.9, 1.5):
        dt = units.viscosity_timestep(nu_phys, dx, tau)
        got = units.physical_viscosity(units.lattice_viscosity(tau), dx, dt)
        assert got == pytest.approx(nu_phys, rel=1e-12)


def test_velocity_unit_conversions_round_trip():
    dx, dt = 2.0e-5, 3.0e-7
    u_phys = 0.15
    u_lat = units.velocity_lattice(u_phys, dx, dt)
    assert units.velocity_physical(u_lat, dx, dt) == pytest.approx(u_phys,
                                                                   rel=1e-15)


def test_lattice_spacing_uses_fluid_band():
    # Half-way bounce-back: a physical height H over ny rows spans ny - 2
    # lattice spacings, so dx = H / (ny - 2).
    assert units.lattice_spacing_for_height(0.01, 33) == pytest.approx(
        0.01 / 31)
    with pytest.raises(ValueError):
        units.lattice_spacing_for_height(0.01, 2)


def test_invalid_unit_inputs_raise():
    with pytest.raises(ValueError):
        units.viscosity_timestep(-1.0, 1e-5, 0.9)
    with pytest.raises(ValueError):
        units.viscosity_timestep(1e-6, 1e-5, 0.5)
