"""Lattice <-> physical unit conversions for the BGK scheme.

Everything inside the solver runs in *lattice units*: grid spacing
``dx = 1``, time step ``dt = 1``, link speed ``c = 1``, and sound speed
``cs^2 = 1/3``.  Physical quantities are recovered by choosing a physical
length scale ``dx_phys`` for one lattice spacing and a physical time step
``dt_phys``:

    nu_phys  = nu_lattice * dx_phys^2 / dt_phys
    u_phys   = u_lattice  * dx_phys    / dt_phys
    g_phys   = g_lattice  * dx_phys    / dt_phys^2

Because the walls sit half a link outside the fluid node band (half-way
bounce-back), a physical channel of height ``H`` resolved with ``ny`` rows
uses ``dx_phys = H / (ny - 2)``.

For a target physical viscosity ``nu_phys`` and a chosen relaxation time
``tau``, the required physical time step follows from the BGK relation
``nu_lattice = cs^2 (tau - 1/2) dt``:

    dt_phys = cs^2 (tau - 1/2) dx_phys^2 / nu_phys

Note what this mapping implies as ``tau -> 1/2``: the lattice viscosity
collapses to zero, so keeping a *fixed physical* viscosity forces
``dt_phys -> 0`` — an ever larger number of ever cheaper-but-numerically
fragile time steps.  Separately, the accuracy of bounce-back degrades as
``tau`` approaches 1/2 because the near-wall slip error of the scheme grows
relative to the vanishing physical viscosity.  ``docs/model.md`` spells out
both caveats.
"""

from __future__ import annotations


def lattice_viscosity(tau: float, dt: float = 1.0, cs2: float = 1.0 / 3.0) -> float:
    """Lattice viscosity of the BGK operator: ``nu = cs^2 (tau - 1/2) dt``."""
    return cs2 * (tau - 0.5) * dt


def physical_viscosity(nu_lattice: float, dx: float, dt: float) -> float:
    """Convert a lattice viscosity to physical units: ``nu * dx^2 / dt``."""
    return nu_lattice * dx * dx / dt


def viscosity_timestep(nu_phys: float, dx: float, tau: float,
                       cs2: float = 1.0 / 3.0) -> float:
    """Physical time step realising ``nu_phys`` at relaxation time ``tau``.

    ``dt = cs^2 (tau - 1/2) dx^2 / nu_phys``
    """
    if nu_phys <= 0.0:
        raise ValueError(f"nu_phys must be positive, got {nu_phys!r}")
    if tau <= 0.5:
        raise ValueError(f"tau must exceed 0.5 for a positive viscosity, got {tau!r}")
    return cs2 * (tau - 0.5) * dx * dx / nu_phys


def velocity_physical(u_lattice: float, dx: float, dt: float) -> float:
    """Convert a lattice velocity to physical units: ``u * dx / dt``."""
    return u_lattice * dx / dt


def velocity_lattice(u_phys: float, dx: float, dt: float) -> float:
    """Convert a physical velocity to lattice units: ``u * dt / dx``."""
    return u_phys * dt / dx


def lattice_spacing_for_height(height_phys: float, ny: int) -> float:
    """Grid spacing for a channel of physical height ``H`` on ``ny`` rows.

    With half-way bounce-back the fluid band measures ``ny - 2`` links, so
    ``dx = H / (ny - 2)``.
    """
    if ny < 3:
        raise ValueError(f"ny must be at least 3, got {ny!r}")
    return height_phys / (ny - 2)
