"""D2Q9 BGK lattice-Boltzmann solver for body-force-driven channel flow.

Geometry and conventions
------------------------
* ``f`` has shape ``(9, nx, ny)``: ``f[q, i, j]`` is the population moving
  with velocity ``q`` at streamwise node ``i`` (periodic) and wall-normal
  node ``j``.
* Rows ``j = 0`` and ``j = ny - 1`` are *solid*.  No-slip walls are imposed
  with **half-way bounce-back**: a population that streams towards a wall
  along a link is reflected back along the same link during the same time
  step.  The reflection happens at the link midpoint, so the physical walls
  sit at ``y = 0.5`` and ``y = ny - 1.5`` lattice units and the effective
  channel height is ``h = ny - 2`` (equal to the number of fluid rows).
* The flow is driven by a uniform body force ``F = rho * force_g * e_x``
  applied with the **Guo et al. (2002)** forcing term on fluid nodes only.
* ``ny`` must be odd so that the channel centreline ``(ny - 1) / 2``
  coincides with a fluid node; this keeps the discrete profile exactly
  symmetric and is validated at construction.

One time step
-------------
In lattice units (``dx = dt = 1``, ``cs^2 = 1/3``), with ``omega = 1/tau``::

    f_q(x + c_q, t + 1) = f_q(x, t)
                          - (1/tau) [ f_q(x, t) - f_q^eq(x, t) ]
                          + (1 - 1/(2 tau)) S_q(x, t)

    f_q^eq = w_q rho [ 1 + 3 (c_q.u) + 4.5 (c_q.u)^2 - 1.5 u^2 ]

    S_q    = w_q rho g [ 3 (c_qx - ux) + 9 (c_q.u) c_qx ]

    rho = sum_q f_q ,   rho u = sum_q c_q f_q + F / 2

Streaming is a periodic shift in ``x`` and ``y``; the populations that
arrive on solid rows are immediately reflected back to the fluid node they
came from (half-way bounce-back), with the momentum handed to the wall
tracked for the global momentum diagnostic.
"""

from __future__ import annotations

import numpy as np

from . import analytic, units
from .exceptions import ChannelLatticeError
from .lattice import D2Q9

#: Hard accuracy ceiling for the estimated steady centreline speed in lattice
#: units.  The low-Mach assumption of the method requires u_max << cs ~ 0.577.
_MAX_STEADY_U = 0.15

_TAU_MIN = 0.5
_TAU_MAX = 20.0


class ChannelFlowLBM:
    """Plane-channel lattice-Boltzmann flow driven by a body force.

    Parameters
    ----------
    nx : int
        Number of streamwise nodes (>= 4).  The direction is periodic.
    ny : int
        Number of wall-normal nodes (>= 5, **odd**).  Rows 0 and ``ny - 1``
        are solid; the walls sit at ``y = 0.5`` and ``y = ny - 1.5``.
    tau : float
        BGK relaxation time, strictly in ``(0.5, 20]``.  The lattice
        viscosity is ``nu = (tau - 0.5) / 3``.
    force_g : float
        Body force per unit density (acceleration) in lattice units; must be
        positive, i.e. the flow is driven in the ``+x`` direction.
    rho0 : float, optional
        Initial (and mean) density, lattice units.  Default 1.0.

    Raises
    ------
    ChannelLatticeError
        For any out-of-range or convention-violating input; the message
        states the constraint and the offending value.
    TypeError
        If ``nx``/``ny`` are not integers.
    """

    def __init__(self, nx: int, ny: int, tau: float, force_g: float,
                 rho0: float = 1.0) -> None:
        for name, val in (("nx", nx), ("ny", ny)):
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise TypeError(
                    f"{name} must be an integer number of lattice nodes, "
                    f"got {type(val).__name__} ({val!r})"
                )
        nx, ny = int(nx), int(ny)
        if nx < 4:
            raise ChannelLatticeError(
                f"nx must be >= 4 to be a meaningful periodic direction, got {nx}"
            )
        if ny < 5:
            raise ChannelLatticeError(
                f"ny must be >= 5 (2 solid rows + >= 3 fluid rows), got {ny}"
            )
        if ny % 2 == 0:
            raise ChannelLatticeError(
                f"ny must be odd so the centreline falls on a fluid node "
                f"(channel symmetry convention), got ny={ny}; use e.g. "
                f"{ny - 1} or {ny + 1}"
            )

        try:
            tau = float(tau)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"tau must be a real number, got {tau!r}") from exc
        if not np.isfinite(tau):
            raise ChannelLatticeError(f"tau must be finite, got {tau!r}")
        if not (_TAU_MIN < tau <= _TAU_MAX):
            raise ChannelLatticeError(
                f"tau must lie in ({_TAU_MIN}, {_TAU_MAX}]: "
                f"tau > 0.5 is required for a positive viscosity "
                f"cs^2*(tau-1/2) (at tau <= 0.5 the BGK viscosity is "
                f"non-positive and the scheme has no viscous term), and "
                f"very large tau only adds diffusion; got tau={tau}"
            )

        try:
            force_g = float(force_g)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"force_g must be a real number, got {force_g!r}") from exc
        if not np.isfinite(force_g):
            raise ChannelLatticeError(f"force_g must be finite, got {force_g!r}")
        if force_g <= 0.0:
            raise ChannelLatticeError(
                f"force_g must be positive so the flow is driven in +x "
                f"(forcing-sign convention of this solver), got {force_g}"
            )

        try:
            rho0 = float(rho0)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"rho0 must be a real number, got {rho0!r}") from exc
        if not np.isfinite(rho0) or rho0 <= 0.0:
            raise ChannelLatticeError(
                f"rho0 must be a positive finite density, got {rho0}"
            )

        nu = units.lattice_viscosity(tau)
        u_est = force_g * (ny - 2) ** 2 / (8.0 * nu)
        if u_est > _MAX_STEADY_U:
            raise ChannelLatticeError(
                f"estimated steady centreline speed u_max = {u_est:.4g} "
                f"exceeds {_MAX_STEADY_U} lattice units; the low-Mach "
                f"assumption (u << cs = {np.sqrt(1.0 / 3.0):.3f}) would be "
                f"violated. Reduce force_g, or increase tau/nu, or accept a "
                f"lower target speed."
            )

        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.force_g = force_g
        self.rho0 = rho0
        self._lat = D2Q9.standard()
        self._c = self._lat.velocities
        self._w = self._lat.weights

        # Initial state: equilibrium at uniform density and ZERO RAW
        # MOMENTUM (sum_q c_q f_q = 0).  Under the Guo velocity definition
        # the macroscopic velocity at t = 0 is therefore u = g/2 (the
        # half-step startup value), not exactly zero; it is far below the
        # steady u_max and relaxes into the forced solution.
        rho_field = np.full((nx, ny), rho0, dtype=np.float64)
        zero = np.zeros((nx, ny), dtype=np.float64)
        self.f = self._lat.equilibrium(rho_field, zero, zero)

        self.steps = 0
        self.converged = False
        self.steady_residual = float("inf")
        # Diagnostics of the most recent step() call.
        self.last_force_momentum_x = 0.0
        self.last_bb_momentum_x = 0.0
        self.last_bb_momentum_y = 0.0
        self.last_momentum_residual_x = 0.0
        self.last_momentum_residual_y = 0.0

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def lattice(self) -> D2Q9:
        """The D2Q9 lattice definition used by this solver."""
        return self._lat

    @property
    def nu(self) -> float:
        """Lattice viscosity ``cs^2 (tau - 1/2) dt`` with ``dt = 1``."""
        return units.lattice_viscosity(self.tau)

    @property
    def omega(self) -> float:
        """Collision frequency ``1 / tau``."""
        return 1.0 / self.tau

    @property
    def h_lattice(self) -> int:
        """Effective channel height in lattice units, ``ny - 2``."""
        return self.ny - 2

    @property
    def wall_positions(self) -> tuple[float, float]:
        """Wall-normal wall locations ``(0.5, ny - 1.5)`` (half-way BB)."""
        return (0.5, float(self.ny - 1.5))

    @property
    def centerline_j(self) -> float:
        """Wall-normal node index of the channel centreline, ``(ny-1)/2``."""
        return (self.ny - 1) / 2.0

    @property
    def u_max_analytic(self) -> float:
        """Analytic steady centreline speed ``g h^2 / (8 nu)``."""
        return self.force_g * self.h_lattice ** 2 / (8.0 * self.nu)

    @property
    def reynolds(self) -> float:
        """Reynolds number based on centreline speed and channel height."""
        return self.u_max_analytic * self.h_lattice / self.nu

    @property
    def mach(self) -> float:
        """Mach number of the steady flow, ``u_max / cs`` (weak-compressibility check)."""
        return self.u_max_analytic / float(np.sqrt(self._lat.cs2))

    # ------------------------------------------------------------------ #
    # Core algorithm
    # ------------------------------------------------------------------ #
    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Density and Guo velocity on the fluid nodes.

        Returns
        -------
        rho, ux, uy : np.ndarray, shape (nx, ny - 2)
            Arrays are aligned with wall-normal nodes ``j = 1 .. ny - 2``.
            The velocity includes the half-force correction
            ``u = (sum_q c_q f_q + F/2) / rho`` required by Guo forcing.
        """
        ff = self.f[:, :, 1:self.ny - 1]
        rho = ff.sum(axis=0)
        mx = np.tensordot(self._c[:, 0], ff, axes=(0, 0))
        my = np.tensordot(self._c[:, 1], ff, axes=(0, 0))
        return rho, mx / rho + 0.5 * self.force_g, my / rho

    def step(self) -> None:
        """Advance the flow by one lattice time step.

        Updates the per-step diagnostics ``last_force_momentum_x``,
        ``last_bb_momentum_x/y`` (momentum given to the fluid by wall
        reflections) and ``last_momentum_residual_x/y`` — the closure
        failure of the exact discrete balance
        ``dP = force + bounce-back exchange``, which is zero to round-off.
        """
        ny = self.ny
        lat = self._lat
        f = self.f
        fluid = (slice(None), slice(None), slice(1, ny - 1))

        # --- macroscopic fields and pre-step raw fluid momentum ----------
        rho, ux, uy = self.macroscopic()
        p_before_x = float(np.tensordot(self._c[:, 0], f[fluid], axes=(0, 0)).sum())
        p_before_y = float(np.tensordot(self._c[:, 1], f[fluid], axes=(0, 0)).sum())

        # --- BGK collision with Guo forcing, fluid nodes only ------------
        ff = f[:, :, 1:ny - 1]
        feq = lat.equilibrium(rho, ux, uy)
        src = lat.guo_source(rho, ux, uy, self.force_g)
        ff += (feq - ff) / self.tau + (1.0 - 0.5 / self.tau) * src
        force_x_total = float(rho.sum() * self.force_g)

        # --- streaming: periodic in x, shifted in y ----------------------
        # After the shift, f[q, i, j] holds the population that left node
        # (i - cx, j - cy); solid rows receive exactly the wall-bound
        # populations from the adjacent fluid rows.
        for q in range(9):
            f[q] = np.roll(f[q], (int(self._c[q, 0]), int(self._c[q, 1])),
                           axis=(0, 1))

        # --- half-way bounce-back ----------------------------------------
        # Populations that arrived on solid row j=0 left fluid row j=1 one
        # half link ago; reflect them back to the node they came from, with
        # reversed velocity.  c_opp - c gives the momentum returned to the
        # fluid per unit of reflected population.
        bb_x = 0.0
        bb_y = 0.0
        # Bottom wall (solid row 0 -> fluid row 1)
        v4 = f[4, :, 0]                      # q4 (0,-1)  -> q2 (0, 1): d(c) = ( 0, 2)
        f[2, :, 1] = v4
        v7 = np.roll(f[7, :, 0], 1, axis=0)   # q7 (-1,-1) -> q5 ( 1, 1): d(c) = ( 2, 2)
        f[5, :, 1] = v7
        v8 = np.roll(f[8, :, 0], -1, axis=0)  # q8 ( 1,-1) -> q6 (-1, 1): d(c) = (-2, 2)
        f[6, :, 1] = v8
        bb_x += 2.0 * float(v7.sum()) - 2.0 * float(v8.sum())
        bb_y += 2.0 * (float(v4.sum()) + float(v7.sum()) + float(v8.sum()))
        # Top wall (solid row ny-1 -> fluid row ny-2)
        v2 = f[2, :, ny - 1]                  # q2 (0, 1)  -> q4 (0,-1): d(c) = ( 0,-2)
        f[4, :, ny - 2] = v2
        v5 = np.roll(f[5, :, ny - 1], -1, axis=0)  # q5 (1,1) -> q7 (-1,-1): d(c) = (-2,-2)
        f[7, :, ny - 2] = v5
        v6 = np.roll(f[6, :, ny - 1], 1, axis=0)   # q6 (-1,1)-> q8 ( 1,-1): d(c) = ( 2,-2)
        f[8, :, ny - 2] = v6
        bb_x += -2.0 * float(v5.sum()) + 2.0 * float(v6.sum())
        bb_y -= 2.0 * (float(v2.sum()) + float(v5.sum()) + float(v6.sum()))

        # --- exact discrete momentum balance ------------------------------
        p_after_x = float(np.tensordot(self._c[:, 0], f[fluid], axes=(0, 0)).sum())
        p_after_y = float(np.tensordot(self._c[:, 1], f[fluid], axes=(0, 0)).sum())
        self.last_force_momentum_x = force_x_total
        self.last_bb_momentum_x = bb_x
        self.last_bb_momentum_y = bb_y
        self.last_momentum_residual_x = (p_after_x - p_before_x) - (force_x_total + bb_x)
        self.last_momentum_residual_y = (p_after_y - p_before_y) - bb_y

        self.steps += 1

    # ------------------------------------------------------------------ #
    # Run control and diagnostics
    # ------------------------------------------------------------------ #
    def mass(self) -> float:
        """Total fluid mass (sum of all populations on fluid nodes)."""
        return float(self.f[:, :, 1:self.ny - 1].sum())

    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Streamwise-averaged velocity profile on fluid nodes.

        Returns
        -------
        js, u_mean : np.ndarray, shape (ny - 2,)
            Wall-normal node coordinates ``j = 1 .. ny - 2`` and the mean of
            ``u_x`` over the periodic direction.
        """
        _, ux, _ = self.macroscopic()
        js = np.arange(1, self.ny - 1, dtype=np.float64)
        return js, ux.mean(axis=0)

    def analytic_profile(self) -> np.ndarray:
        """Exact steady parabolic profile at the fluid nodes."""
        js = np.arange(1, self.ny - 1, dtype=np.float64)
        return analytic.exact_channel_profile(js, self.ny, self.force_g, self.nu)

    def error_norms(self) -> tuple[float, float]:
        """Relative L2 and L-infinity errors against the analytic parabola."""
        _, u = self.profile()
        return analytic.error_norms(u, self.analytic_profile())

    def run_until_steady(self, tol: float = 1e-9, max_steps: int | None = None,
                         check_every: int = 250) -> int:
        """Iterate until the mean profile stops changing within ``tol``.

        Stopping criterion (measured, not a formal error bound): the run
        stops when the streamwise-averaged ``u_x`` profile changes by less
        than ``tol * u_max_analytic`` (infinity norm) between checks.  The
        last measured per-check profile change is stored in
        ``self.steady_residual`` (absolute, lattice units; ``inf`` if no
        check ran) as evidence of the steady state.  Because the diffusive
        relaxation time of the channel grows like ``h^2 / nu``, a step
        budget that suffices on a coarse grid may be exhausted on a fine
        one — callers who care should check ``self.converged``.

        Parameters
        ----------
        tol : float
            Finite, strictly positive; in units of ``u_max_analytic``.
        max_steps : int or None
            *Budget of additional steps for this call* (nonnegative
            integer).  Each check chunk is capped to the remaining budget,
            so the solver never overshoots.  ``None`` selects the automatic
            budget ``ceil(8 h^2 / nu) + 2000``.  Note that ``self.steps``
            is the cumulative step count over the solver's lifetime; the
            return value counts only this call.
        check_every : int
            Strictly positive integer; steps between convergence checks.

        Returns
        -------
        int
            Number of steps performed in this call.  ``self.converged`` is
            reset to ``False`` on entry and set only if the criterion was
            met during this call.

        Raises
        ------
        ChannelLatticeError
            If ``tol`` is not finite and positive, ``check_every`` is not
            a positive integer, or ``max_steps`` is not a nonnegative
            integer.
        """
        try:
            tol_val = float(tol)
        except (TypeError, ValueError):
            tol_val = float("nan")
        if isinstance(tol, bool) or not np.isfinite(tol_val) or tol_val <= 0.0:
            raise ChannelLatticeError(
                f"tol must be a finite positive tolerance in units of "
                f"u_max_analytic, got {tol!r}"
            )
        if isinstance(check_every, bool) or not isinstance(
                check_every, (int, np.integer)) or check_every < 1:
            raise ChannelLatticeError(
                f"check_every must be a positive integer number of steps "
                f"between convergence checks, got {check_every!r}"
            )
        if max_steps is not None and (isinstance(max_steps, bool)
                                      or not isinstance(max_steps,
                                                        (int, np.integer))
                                      or max_steps < 0):
            raise ChannelLatticeError(
                f"max_steps must be a nonnegative integer step budget for "
                f"this call (or None for the automatic budget), got "
                f"{max_steps!r}"
            )
        budget = (int(np.ceil(8.0 * self.h_lattice ** 2 / self.nu)) + 2000
                  if max_steps is None else int(max_steps))
        check_every = int(check_every)

        self.converged = False
        u_ref = max(self.u_max_analytic, 1e-300)
        start = self.steps
        prev = self.profile()[1].copy()
        self.steady_residual = float("inf")
        while self.steps - start < budget:
            chunk = min(check_every, budget - (self.steps - start))
            for _ in range(chunk):
                self.step()
            cur = self.profile()[1]
            self.steady_residual = float(np.max(np.abs(cur - prev)))
            if self.steady_residual < tol * u_ref:
                self.converged = True
                break
            prev = cur
        return self.steps - start

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ChannelFlowLBM(nx={self.nx}, ny={self.ny}, tau={self.tau}, "
            f"force_g={self.force_g:.6g}, steps={self.steps})"
        )
