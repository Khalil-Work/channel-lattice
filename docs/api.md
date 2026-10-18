# API reference

`channel_lattice` is deliberately small: one solver class, one lattice
record, two helper modules. NumPy is the only runtime dependency.

```python
from channel_lattice import ChannelFlowLBM, D2Q9, ChannelLatticeError
from channel_lattice import analytic, units
```

## `channel_lattice.exceptions`

### `ChannelLatticeError(ValueError)`

Raised for every rejected input. Messages name the constraint, the
offending value, and the accepted range/convention.

## `channel_lattice.lattice`

### `D2Q9`

```python
from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class D2Q9:
    velocities: np.ndarray   # (9, 2) int64 discrete velocities c_q
    weights: np.ndarray      # (9,) quadrature weights w_q
    cs2: float = 1 / 3       # squared lattice sound speed
```

Construction validates the isotropy conditions (`sum w = 1`,
`sum w c = 0`, `sum w c_a c_b = cs^2 delta_ab`) and raises
`ChannelLatticeError` if they fail.

| Member | Description |
|---|---|
| `D2Q9.standard()` | The standard D2Q9 lattice (class method). |
| `.opposite` | Index array with `c_opp[q] = -c_q`: `(0,3,4,1,2,7,8,5,6)`. |
| `.equilibrium(rho, ux, uy)` | `f^eq` from `(nx, ny)` fields, returns `(9, nx, ny)`. |
| `.guo_source(rho, ux, uy, g)` | Guo forcing term `S_q` for `F = rho g e_x`, same shapes. |

## `channel_lattice.solver`

### `ChannelFlowLBM(nx, ny, tau, force_g, rho0=1.0)`

Plane-channel flow on a D2Q9 lattice: rows `j=0` and `j=ny-1` solid,
half-way bounce-back walls at `y = 0.5` and `y = ny - 1.5`, periodic in
`x`, Guo body force `rho * force_g` in `+x`, initial state at rest.

**Parameters and validation** (all failures raise `ChannelLatticeError`
except type errors, which raise `TypeError`):

| Parameter | Constraint |
|---|---|
| `nx` | integer `>= 4` (periodic streamwise direction) |
| `ny` | integer `>= 5` and **odd** (centreline on a fluid node) |
| `tau` | `0.5 < tau <= 20` (positive viscosity `(tau-1/2)/3`) |
| `force_g` | finite and `> 0` (flow driven in `+x` by convention) |
| `rho0` | positive finite density |
| implied `u_max = g (ny-2)^2 / (8 nu)` | `<= 0.15` lattice units (low-Mach guard) |

**Properties**

| Property | Meaning |
|---|---|
| `lattice` | The `D2Q9` instance in use. |
| `nu` | Lattice viscosity `cs^2 (tau - 1/2) dt`. |
| `omega` | Collision frequency `1 / tau`. |
| `h_lattice` | Effective channel height `ny - 2`. |
| `wall_positions` | `(0.5, ny - 1.5)` — where bounce-back places the walls. |
| `centerline_j` | `(ny - 1) / 2`. |
| `u_max_analytic` | `g h^2 / (8 nu)`. |
| `reynolds` | `u_max h / nu`. |
| `mach` | `u_max / cs`. |
| `steps` | Number of time steps performed so far. |
| `converged`, `steady_residual` | Outcome of the last `run_until_steady`. |

**Methods**

| Method | Description |
|---|---|
| `macroscopic()` | `(rho, ux, uy)` on fluid nodes, shape `(nx, ny-2)`, velocity includes the Guo half-force correction. |
| `step()` | One lattice time step (collide, stream, bounce-back). Updates the `last_*` diagnostics. |
| `mass()` | Total fluid mass. |
| `profile()` | `(js, u_mean)`: fluid-node coordinates and streamwise-averaged `u_x`. |
| `analytic_profile()` | Exact steady parabola at the same nodes. |
| `error_norms()` | `(rel_L2, rel_Linf)` against the analytic profile. |
| `run_until_steady(tol=1e-9, max_steps=None, check_every=250)` | Iterate until the mean profile changes by `< tol * u_max` between checks (measured stopping criterion; last change stored in `steady_residual`). `max_steps` is a **budget of additional steps for this call** (nonnegative int or `None` = automatic); chunks are capped so the budget is never overshot; returns the steps performed in this call while `self.steps` stays cumulative; `converged` is reset on entry. Validates `tol` finite positive, `check_every` positive integer, `max_steps` nonnegative integer. |

**Per-step diagnostics** (attributes updated by `step()`):

| Attribute | Meaning |
|---|---|
| `last_force_momentum_x` | `sum(rho) * g` added by collisions this step. |
| `last_bb_momentum_x` / `_y` | Momentum returned to the fluid by wall reflections. |
| `last_momentum_residual_x` / `_y` | Closure failure of the exact discrete balance `dP = force + wall exchange` (round-off size). |

## `channel_lattice.analytic`

| Function | Description |
|---|---|
| `exact_channel_profile(js, ny, force_g, nu)` | `u(j) = (g/2nu)(j-0.5)(ny-1.5-j)`. |
| `force_for_target_u_max(u_max, nu, h_lattice)` | Inverts `u_max = g h^2/(8nu)` for `g`. |
| `parabola_root_fit(js, u)` | Fits `A(j-a)(b-j)`, returns wall positions `(a, b)`. |
| `error_norms(u_num, u_ex)` | Relative `L2` and `Linf` norms. |
| `symmetry_error(u)` | `max|u - u[::-1]|`. |

## `channel_lattice.units`

| Function | Description |
|---|---|
| `lattice_viscosity(tau, dt=1, cs2=1/3)` | `nu = cs^2 (tau - 1/2) dt`. |
| `physical_viscosity(nu_lattice, dx, dt)` | `nu * dx^2 / dt`. |
| `viscosity_timestep(nu_phys, dx, tau)` | `dt` realising `nu_phys` at this `tau`. |
| `velocity_physical(u_lattice, dx, dt)` / `velocity_lattice(u_phys, dx, dt)` | Velocity conversions. |
| `lattice_spacing_for_height(H, ny)` | `dx = H / (ny - 2)` (fluid band spans `ny - 2` links). |

## Command-line entry points

There is none by design. Use the scripts:

```
python -m pytest tests -q            # full test suite (< 60 s)
python examples/run_verification.py  # regenerates docs/results/
```
