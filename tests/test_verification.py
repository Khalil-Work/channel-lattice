"""End-to-end verification of the steady channel flow against the analytic
parabolic solution, the wall-placement convention, and the steady global
momentum balance.

One shared steady simulation (module fixture) backs all assertions.  The
tolerances below are calibrated against the measured errors of this exact
setup (tau = 0.9, ny = 25, u_max = 0.06 -> measured relative L2 ~ 3.8e-4)
with a safety margin of roughly 5x, so they hold across platforms while
still catching genuine regressions.  See docs/verification.md.
"""

import numpy as np
import pytest

from channel_lattice import ChannelFlowLBM, analytic, units

TOL_L2 = 2.0e-3
TOL_LINF = 1.5e-3
TOL_UMAX = 5.0e-3
TOL_ROOT = 0.05


@pytest.fixture(scope="module")
def steady_flow():
    tau = 0.9
    nu = units.lattice_viscosity(tau)
    flow = ChannelFlowLBM(
        nx=16, ny=25, tau=tau,
        force_g=analytic.force_for_target_u_max(0.06, nu, 25 - 2),
    )
    flow.run_until_steady(tol=1e-10, max_steps=60000)
    assert flow.converged, "reference simulation failed to reach steady state"
    return flow


def test_steady_state_was_actually_reached(steady_flow):
    # Convergence evidence, not just an iteration count: the profile moves
    # by far less than the discretisation error between checks.
    flow = steady_flow
    assert flow.steady_residual < 1e-10 * flow.u_max_analytic


def test_profile_matches_analytic_parabola(steady_flow):
    flow = steady_flow
    l2, linf = flow.error_norms()
    assert l2 < TOL_L2
    assert linf < TOL_LINF


def test_amplitude_matches_analytic_u_max(steady_flow):
    # Shape alone is not a flow-rate check: the peak must also match.
    flow = steady_flow
    _, u = flow.profile()
    ratio = float(u.max()) / flow.u_max_analytic
    assert abs(ratio - 1.0) < TOL_UMAX


def test_profile_is_machine_symmetric(steady_flow):
    _, u = steady_flow.profile()
    assert analytic.symmetry_error(u) < 1e-12


def test_density_drifts_only_at_roundoff(steady_flow):
    flow = steady_flow
    drift = abs(flow.mass() - flow.rho0 * flow.nx * (flow.ny - 2))
    assert drift / flow.mass() < 1e-10


def test_bounce_back_walls_sit_halfway_between_nodes(steady_flow):
    # Fit u(j) = A (j - a)(b - j) to the computed profile.  The roots must
    # land on the half-way wall positions 0.5 and ny - 1.5, NOT on the
    # solid nodes (0, ny - 1) or the first/last fluid-node centres
    # (1, ny - 2).  This pins down the wall-location convention empirically.
    flow = steady_flow
    js, u = flow.profile()
    a, b = analytic.parabola_root_fit(js, u)
    assert abs(a - 0.5) < TOL_ROOT
    assert abs(b - (flow.ny - 1.5)) < TOL_ROOT
    assert abs(a - 0.0) > 0.3 and abs(a - 1.0) > 0.3
    assert abs(b - (flow.ny - 1)) > 0.3 and abs(b - (flow.ny - 2)) > 0.3


def test_steady_wall_drag_balances_body_force(steady_flow):
    # In steady state the momentum handed to the walls by bounce-back each
    # step must exactly cancel the force impulse added by collisions.
    flow = steady_flow
    imbalance = abs(flow.last_bb_momentum_x + flow.last_force_momentum_x)
    assert imbalance / abs(flow.last_force_momentum_x) < 1e-7


def test_flow_is_laminar_low_mach_regime(steady_flow):
    # Documented validity window of the verification: low Re, weak
    # compressibility.
    flow = steady_flow
    assert flow.reynolds < 50
    assert flow.mach < 0.15
