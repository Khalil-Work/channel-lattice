"""channel_lattice — D2Q9 lattice-Boltzmann channel flow, lattice to continuum.

A minimal, dependency-light (NumPy only) lattice-Boltzmann BGK solver for
plane Poiseuille flow: D2Q9 streaming, half-way bounce-back walls, Guo body
forcing, and direct verification against the analytic parabolic profile.

Typical use::

    from channel_lattice import ChannelFlowLBM

    flow = ChannelFlowLBM(nx=64, ny=33, tau=0.9, force_g=6.66e-5)
    flow.run_until_steady()
    l2, linf = flow.error_norms()
"""

from .exceptions import ChannelLatticeError
from .lattice import D2Q9
from .solver import ChannelFlowLBM

__version__ = "1.0.0"

__all__ = [
    "ChannelLatticeError",
    "ChannelFlowLBM",
    "D2Q9",
    "analytic",
    "units",
]
