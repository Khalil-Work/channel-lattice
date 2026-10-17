"""D2Q9 lattice: velocities, weights, equilibrium, and the Guo forcing term.

This module defines the discrete velocity set used by the solver.  All
quantities are in *lattice units* where the grid spacing ``dx = 1`` and the
time step ``dt = 1``, so the link speed ``c = dx/dt = 1`` and the lattice
sound speed satisfies ``cs^2 = 1/3``.

Layout of the discrete velocities (index -> (cx, cy))::

        6   2   5
          \\ | //
    3 ----- 0 ----- 1     cy points "up" (+y, wall-normal)
          / | \\          cx points "right" (+x, streamwise)
        7   4   8
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import ChannelLatticeError


@dataclass(frozen=True, eq=False)
class D2Q9:
    """The two-dimensional nine-velocity lattice.

    Attributes
    ----------
    velocities : np.ndarray, shape (9, 2), dtype int
        Discrete velocity vectors ``(cx, cy)`` in lattice units (0 or +/-1).
    weights : np.ndarray, shape (9,)
        Quadrature weights ``w_q`` of the lattice.
    cs2 : float
        Squared lattice sound speed, ``cs^2 = 1/3`` in lattice units.
    """

    velocities: np.ndarray
    weights: np.ndarray
    cs2: float = 1.0 / 3.0

    @classmethod
    def standard(cls) -> "D2Q9":
        """Return the standard D2Q9 lattice with exact quadrature weights."""
        velocities = np.array(
            [
                [0, 0],    # 0: rest
                [1, 0],    # 1: east
                [0, 1],    # 2: north
                [-1, 0],   # 3: west
                [0, -1],   # 4: south
                [1, 1],    # 5: north-east
                [-1, 1],   # 6: north-west
                [-1, -1],  # 7: south-west
                [1, -1],   # 8: south-east
            ],
            dtype=np.int64,
        )
        weights = np.array(
            [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64
        )
        return cls(velocities=velocities, weights=weights)

    def __post_init__(self) -> None:
        c = np.asarray(self.velocities, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        if c.shape != (9, 2):
            raise ChannelLatticeError(
                f"D2Q9 velocities must have shape (9, 2), got {c.shape}"
            )
        if w.shape != (9,):
            raise ChannelLatticeError(
                f"D2Q9 weights must have shape (9,), got {w.shape}"
            )
        if not np.isclose(w.sum(), 1.0, atol=1e-12):
            raise ChannelLatticeError(
                f"D2Q9 weights must sum to 1, got {w.sum()!r}"
            )
        if not np.allclose(w, 0.0) and np.any(w < 0.0):
            raise ChannelLatticeError("D2Q9 weights must be non-negative")
        if not np.allclose(w @ c, 0.0, atol=1e-12):
            raise ChannelLatticeError("D2Q9 requires sum_q w_q c_q = 0")
        second = np.einsum("q,qa,qb->ab", w, c, c)
        if not np.allclose(second, self.cs2 * np.eye(2), atol=1e-12):
            raise ChannelLatticeError(
                "D2Q9 requires sum_q w_q c_qa c_qb = cs^2 delta_ab"
            )
        for q in range(9):
            if not np.array_equal(c[self.opposite[q]], -c[q]):
                raise ChannelLatticeError(
                    f"opposite[{q}] must map to the reversed velocity"
                )

    @property
    def opposite(self) -> np.ndarray:
        """Index array ``opp`` with ``c_opp[q] = -c_q``."""
        return np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

    def equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """Discrete Maxwell-Boltzmann equilibrium ``f_q^eq``.

        ``f_q^eq = w_q rho [ 1 + 3 (c_q . u) + 4.5 (c_q . u)^2 - 1.5 u^2 ]``

        Parameters
        ----------
        rho, ux, uy : np.ndarray, shape (nx, ny)
            Density and (Guo-corrected) velocity components in lattice units.

        Returns
        -------
        np.ndarray, shape (9, nx, ny)
        """
        c = self.velocities
        w = self.weights
        cu = (
            c[:, 0, None, None] * ux[None, :, :]
            + c[:, 1, None, None] * uy[None, :, :]
        ) * 3.0
        usq = 1.5 * (ux * ux + uy * uy)
        return w[:, None, None] * rho[None, :, :] * (
            1.0 + cu + 0.5 * cu * cu - usq[None, :, :]
        )

    def guo_source(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                   g: float) -> np.ndarray:
        """Guo et al. (2002) forcing term ``S_q`` for ``F = rho g e_x``.

        ``S_q = w_q rho g [ 3 (c_qx - ux) + 9 (c_q . u) c_qx ]``

        This is the exact expression ``w_q [(c_q - u)/cs^2 +
        (c_q . u) c_q / cs^4] . F`` written out for a streamwise body force.
        It satisfies ``sum_q S_q = 0`` (mass conservation) and
        ``sum_q S_q c_q = F`` (correct momentum input) to machine precision.
        """
        c = self.velocities
        w = self.weights
        cx = c[:, 0, None, None]
        cu = cx * ux[None, :, :] + c[:, 1, None, None] * uy[None, :, :]
        bracket = 3.0 * (cx - ux[None, :, :]) + 9.0 * cu * cx
        return w[:, None, None] * rho[None, :, :] * g * bracket
