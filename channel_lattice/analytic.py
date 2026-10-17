"""Analytic reference solutions and error metrics for plane channel flow.

The solver drives the flow with a uniform body force ``rho * g`` in the
streamwise (+x) direction and confines it with two no-slip walls.  With the
half-way bounce-back convention the walls are located at ``y = 0.5`` and
``y = ny - 1.5`` lattice units (see :mod:`channel_lattice.solver` and
``docs/model.md``), so the effective channel height is ``h = ny - 2``.

The steady laminar solution of the forced plane problem solves
``nu u'' + g = 0`` with ``u = 0`` on both walls, giving the parabola::

    u(y) = (g / (2 nu)) * (y - 0.5) * (ny - 1.5 - y)
    u_max = g h^2 / (8 nu)          at the centreline y = (ny - 1) / 2
"""

from __future__ import annotations

import numpy as np


def exact_channel_profile(js: np.ndarray, ny: int, force_g: float,
                          nu: float) -> np.ndarray:
    """Exact steady profile at wall-normal node positions ``js``.

    Parameters
    ----------
    js : array_like
        Wall-normal node coordinates ``j`` (fluid nodes are ``1 .. ny-2``).
    ny : int
        Total number of wall-normal lattice rows.
    force_g : float
        Body force per unit density (acceleration) in lattice units.
    nu : float
        Kinematic viscosity in lattice units, ``cs^2 (tau - 1/2) dt``.
    """
    js = np.asarray(js, dtype=np.float64)
    return 0.5 * force_g / nu * (js - 0.5) * (ny - 1.5 - js)


def force_for_target_u_max(u_max: float, nu: float, h_lattice: int) -> float:
    """Body force (per unit density) producing the requested centreline speed.

    Inverts ``u_max = g h^2 / (8 nu)`` for ``g``.
    """
    if u_max <= 0.0:
        raise ValueError(f"u_max must be positive, got {u_max!r}")
    if nu <= 0.0:
        raise ValueError(f"nu must be positive, got {nu!r}")
    if h_lattice <= 0:
        raise ValueError(f"h_lattice must be positive, got {h_lattice!r}")
    return 8.0 * nu * u_max / float(h_lattice) ** 2


def parabola_root_fit(js: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Fit ``u(j) = A (j - a)(b - j)`` and return the wall positions ``(a, b)``.

    A quadratic least-squares fit of the discrete profile is inverted to find
    where the fitted parabola reaches zero.  For the half-way bounce-back
    convention the roots must come out at ``a = 0.5`` and ``b = ny - 1.5``;
    an on-node wall convention would instead produce ``a = 0`` and
    ``b = ny - 1``.  This is the diagnostic used to verify the wall
    placement convention empirically.
    """
    js = np.asarray(js, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    coeffs = np.polyfit(js, u, 2)
    roots = np.roots(coeffs)
    roots = sorted(float(r.real) for r in roots)
    return roots[0], roots[1]


def error_norms(u_num: np.ndarray, u_ex: np.ndarray) -> tuple[float, float]:
    """Relative L2 and L-infinity error norms of a velocity profile.

    ``L2 = ||u_num - u_ex||_2 / ||u_ex||_2`` and
    ``Linf = max|u_num - u_ex| / max|u_ex|``.
    """
    u_num = np.asarray(u_num, dtype=np.float64)
    u_ex = np.asarray(u_ex, dtype=np.float64)
    diff = u_num - u_ex
    l2 = float(np.linalg.norm(diff) / np.linalg.norm(u_ex))
    linf = float(np.max(np.abs(diff)) / np.max(np.abs(u_ex)))
    return l2, linf


def symmetry_error(u: np.ndarray) -> float:
    """Maximum deviation of a profile from its own mirror image."""
    u = np.asarray(u, dtype=np.float64)
    return float(np.max(np.abs(u - u[::-1])))
