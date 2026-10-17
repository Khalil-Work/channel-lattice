"""Runnable version of the README "Architecture and API" snippet.

Run from the repository root::

    python examples/readme_api_demo.py

(The two lines below the imports only make the repository package
importable regardless of the caller's working directory.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_lattice import ChannelFlowLBM
from channel_lattice.analytic import force_for_target_u_max
from channel_lattice.units import lattice_viscosity

tau = 0.9
nu = lattice_viscosity(tau)                    # (tau - 1/2) / 3
g = force_for_target_u_max(0.06, nu, h_lattice=31)  # drive for u_max = 0.06

flow = ChannelFlowLBM(nx=64, ny=33, tau=tau, force_g=g)
flow.run_until_steady()                        # adaptive residual criterion

l2, linf = flow.error_norms()
print(f"steps={flow.steps}  rel_L2={l2:.3e}  rel_Linf={linf:.3e}")
print(f"walls at {flow.wall_positions}, h={flow.h_lattice}, "
      f"Re={flow.reynolds:.1f}, Ma={flow.mach:.3f}")
