# Model reference — D2Q9 channel flow from lattice to continuum

This document specifies every equation and convention implemented in
`channel_lattice`. All symbols are defined; lattice units are used
throughout unless stated otherwise.

## 1. Problem statement

Plane Poiseuille (channel) flow: a Newtonian fluid confined between two
infinite parallel no-slip walls, driven by a uniform body force. In the
continuum, the steady laminar solution of

```
nu * d^2u/dy^2 + g = 0 ,      u(y_wall) = 0
```

is the parabola `u(y) = (g / (2 nu)) * (y - y_bottom) * (y_top - y)` with
`u_max = g h^2 / (8 nu)` at the centreline, where `h = y_top - y_bottom` is
the true wall-to-wall channel height. The solver reproduces this solution
from the discrete kinetic equation, which is the whole point of the
verification: the walls enter the analytic comparison at their *physical*
locations (see Section 5), never at fluid-node centres.

## 2. Lattice: D2Q9

Two dimensions, nine velocities, unit grid spacing `dx = 1` and time step
`dt = 1`, so the link speed is `c = dx/dt = 1` and the lattice sound speed
is `cs^2 = 1/3`.

Velocity index map (the layout used in code, `y` is the wall-normal axis,
`x` the streamwise axis):

```
        6   2   5
          \ | /
    3 ---- 0 ---- 1      -> +x (streamwise, periodic)
          / | \           +y (wall-normal, walls top and bottom)
        7   4   8
```

| q | c_q = (cx, cy) | weight w_q |
|---|----------------|------------|
| 0 | ( 0,  0)       | 4/9        |
| 1 | ( 1,  0)       | 1/9        |
| 2 | ( 0,  1)       | 1/9        |
| 3 | (-1,  0)       | 1/9        |
| 4 | ( 0, -1)       | 1/9        |
| 5 | ( 1,  1)       | 1/36       |
| 6 | (-1,  1)       | 1/36       |
| 7 | (-1, -1)       | 1/36       |
| 8 | ( 1, -1)       | 1/36       |

The weights satisfy the isotropy conditions that make the quadrature
consistent with the continuum moments:

```
sum_q w_q = 1 ,   sum_q w_q c_q = 0 ,   sum_q w_q c_qa c_qb = cs^2 * delta_ab
```

The opposite-direction map used by bounce-back is
`opp = (0, 3, 4, 1, 2, 7, 8, 5, 6)`, i.e. `c_opp[q] = -c_q`.

## 3. Macroscopic fields and the equilibrium

Populations are stored as `f[q, i, j]` with streamwise node index `i`
(periodic) and wall-normal node index `j`. Density and velocity follow
from the discrete moments, with the half-force correction demanded by the
forcing scheme of Section 4 (`F = rho g e_x`):

```
rho   = sum_q f_q
rho u = sum_q c_q f_q  +  F dt / 2        (dt = 1)
```

**Startup convention.** The initial state is `f^eq(rho0, 0)` — zero *raw*
momentum, `sum_q c_q f_q = 0`. Under the definition above the
macroscopic velocity at `t = 0` is therefore `u = g/2`, the half-step
startup value, rather than exactly zero. This value is orders of magnitude
below the steady `u_max` (for the benchmark, `3.6e-5` vs `0.065`) and
relaxes into the forced solution; steady-state results are unaffected.
The animation and README state this convention instead of claiming an
exactly-at-rest start.

The equilibrium is the second-order (low-Mach) expansion:

```
f_q^eq = w_q rho [ 1 + 3 (c_q . u) + 4.5 (c_q . u)^2 - 1.5 u^2 ]
```

using `1/cs^2 = 3`, `1/(2 cs^4) = 4.5`, `1/(2 cs^2) = 1.5`. It conserves
mass and momentum exactly: `sum_q f_q^eq = rho`, `sum_q c_q f_q^eq = rho u`.

## 4. Collision, forcing, streaming

### 4.1 BGK collision with Guo forcing — the exact discrete equation

One time step of the implemented scheme is, verbatim:

```
f_q(x + c_q dt, t + dt) = f_q(x, t)
                          - (dt / tau) [ f_q(x, t) - f_q^eq(x, t) ]
                          + (1 - dt / (2 tau)) S_q(x, t)
```

with the Guo et al. (2002) source term for a body force `F = rho g e_x`:

```
S_q = w_q [ (c_q - u)/cs^2  +  (c_q . u) c_q / cs^4 ] . F
    = w_q rho g [ 3 (c_qx - u_x) + 9 (c_q . u) c_qx ]
```

The three consistency ingredients that must travel together are:

1. the force density is `F = rho * g` (per unit volume), stated explicitly;
2. the source carries the prefactor `(1 - 1/(2 tau))`, which places the
   force at the half time step where the trapezoidal collision acts;
3. the macroscopic velocity used in `f^eq` and in `S_q` includes the
   half-force correction `u = (sum c f)/rho + F/(2 rho)` from Section 3.

**Consistency caveat — where the forcing rules actually differ.** A simpler
source sometimes seen is the purely linear piece `w_q (c_q . F) / cs^2`.
On D2Q9 that simple source is *not* distinguished by its lowest moments:
with the isotropy identities `sum_q w_q c_q = 0` and
`sum_q w_q c_qa c_qb = cs^2 delta_ab` it also has exactly
`sum_q S_q = 0` and `sum_q S_q c_q = F` — mass conservation and correct
momentum input hold for both rules. The distinction appears in the
**second moment**, which feeds the momentum flux (stress). Expanding the
Guo source with the higher-order isotropy identities
`sum_q w_q c_qa c_qb c_qg = 0` and
`sum_q w_q c_qa c_qb c_qg c_qd = cs^4 (delta_ab delta_gd + delta_ag
delta_bd + delta_ad delta_bg)` gives:

```
Guo source:    sum_q S_q c_qa c_qb = u_a F_b + u_b F_a
linear source: sum_q S_q c_qa c_qb = 0
```

(Derivation sketch. First piece of the Guo rule:
`[sum w c_a c_b c_g F_g - (u.F) sum w c_a c_b]/cs^2 = -(u.F) delta_ab`,
using the third- and second-order identities. Second piece:
`[sum w c_a c_b c_g c_d] u_g F_d / cs^4 = (u.F) delta_ab + u_a F_b +
u_b F_a` via the fourth-order identity. The `(u.F) delta_ab` terms cancel,
leaving `u_a F_b + u_b F_a`. The linear source contracts
`F_g sum w c_a c_b c_g / cs^2 = 0` by the third-order identity. Both
results hold exactly, to machine precision.)

The forced Navier–Stokes momentum equation carries exactly the
`u (x) F + F (x) u` contribution in its convective flux; a source with a
vanishing second moment is inconsistent with the forced momentum equation
at the level of the stress, which is where the truncation error of naive
forcing schemes shows up. Separately, the prefactor `(1 - 1/(2 tau))` and
the half-force velocity correction of Section 3 must travel together:
they place the force at the half time step at which the trapezoidal
collision acts, and mixing a velocity-shift rule with only part of the
Guo rule breaks that timing. The zeroth and first source moments are
unit-tested in `tests/test_lattice.py`; the exact per-step momentum
balance of the assembled algorithm is tested in `tests/test_solver.py`.

### 4.2 Streaming and the periodic streamwise direction

Streaming shifts every population by its lattice velocity. In the
streamwise direction this is a pure periodic shift; there are no inlet or
outlet planes — the body force plays the role of the streamwise pressure
gradient, so the channel is statistically homogeneous in `x` and the
periodic direction is exact (not an approximation).

## 5. Walls: half-way bounce-back and where the wall really is

Node rows `j = 0` and `j = ny - 1` are **solid**; fluid lives on
`j = 1 .. ny - 2`. `ny` is required to be odd so the centreline
`(ny - 1)/2` is a fluid node and the discrete profile is exactly
symmetric.

After streaming, the populations sitting on a solid row are exactly those
that left the adjacent fluid row one half link earlier, heading for the
wall. Each is reflected back **to the fluid node it came from**, with its
velocity reversed:

```
bottom wall (row 0 -> row 1):   f_2(x,1) = f_4(x,0)
                                f_5(x,1) = f_7(x+1,0)
                                f_6(x,1) = f_8(x-1,0)

top wall (row ny-1 -> row ny-2): f_4(x,ny-2) = f_2(x,ny-1)
                                 f_7(x,ny-2) = f_5(x-1,ny-1)
                                 f_8(x,ny-2) = f_6(x+1,ny-1)
```

Geometrically: a population at the fluid row travels half a link toward
the wall, reverses at the link midpoint, and returns to its origin node
within the same time step. The momentum exchange with the wall therefore
happens **on the link midpoints**, and the no-slip planes are located at

```
y_bottom = 0.5 ,      y_top = ny - 1.5        (lattice units)
h = y_top - y_bottom = ny - 2                 (effective channel height)
```

Three candidate conventions exist for this node layout: walls on the solid
nodes (`y = 0, ny-1`), walls at the first/last fluid-node centres
(`y = 1, ny-2`), or the half-way convention above. They differ by a full
lattice spacing in effective `h`, which changes `u_max` by
`O(dx / h)` — far larger than the truncation error. **The verification
compares against the analytic parabola that vanishes at `y = 0.5` and
`y = ny - 1.5`**, i.e. at the true wall positions, and the wall-placement
test fits the computed profile to `u(j) = A (j - a)(b - j)` and checks
that the fitted roots `a, b` land on `0.5` and `ny - 1.5` (measured:
`a = 0.5016`, `b = 23.4984` for `ny = 25`), empirically excluding the
on-node and node-centre conventions.

Bounce-back is second-order accurate for straight walls aligned with the
lattice, but its slip error has a `tau`-dependent part; Section 7 and the
relaxation-time study quantify this.

## 6. Analytic steady solution used as reference

With walls at `y = 0.5` and `y = ny - 1.5`, the exact steady profile at
the fluid nodes `j = 1 .. ny-2` is

```
u_ex(j) = (g / (2 nu)) (j - 0.5)(ny - 1.5 - j)
u_max    = g h^2 / (8 nu) ,   h = ny - 2
Re       = u_max h / nu ,     Ma = u_max / cs
```

Error norms (documented definitions, used everywhere):

```
L2_rel  = || u_num - u_ex ||_2 / || u_ex ||_2          (over fluid nodes)
Linf_rel = max_j |u_num(j) - u_ex(j)| / u_max
```

The streamwise-averaged profile is used; since the initial condition,
force, and boundaries are x-uniform, the solution is exactly x-uniform at
all times, so the average is exact. The code still evolves the full 2D
population array.

## 7. Viscosity, lattice units, and the tau -> 1/2 limit

The BGK relaxation time sets the lattice viscosity:

```
nu_lattice = cs^2 (tau - 1/2) dt = (tau - 1/2) / 3        (dt = 1)
```

This relation is exact for the scheme (unit-tested), `tau > 1/2` is
required for a positive viscosity, and the code rejects `tau <= 1/2`.

### 7.1 Lattice-to-physical mapping

To simulate a physical channel of height `H [m]` with viscosity
`nu_phys [m^2/s]`, choose the grid and time step consistently:

```
dx_phys = H / (ny - 2)                  (fluid band spans ny - 2 links, Sec. 5)
dt_phys = cs^2 (tau - 1/2) dx_phys^2 / nu_phys
u_phys  = u_lattice * dx_phys / dt_phys
g_phys  = g_lattice * dx_phys / dt_phys^2
```

The conversions are implemented (and round-trip tested) in
`channel_lattice.units`.

### 7.2 Where the scheme deviates as tau -> 1/2

Two separate mechanisms, both visible in the relaxation-time study:

* **Viscosity collapse.** `nu_lattice -> 0` as `tau -> 1/2`. Holding a
  fixed *physical* viscosity then forces `dt_phys -> 0` (Section 7.1): an
  ever larger number of steps, while the diffusive relaxation time of the
  channel grows like `h^2 / nu_lattice`. The study case `tau = 0.55`
  needed 116 500 steps against 7 000 at `tau = 1.5`.
* **Relative accuracy loss.** The near-wall slip error of bounce-back
  carries `tau`-dependent factors; as the physical viscosity vanishes,
  that error is no longer small compared to `nu * du/dy`. Measured
  relative L2 error at `h = 31`: `1.4e-3` at `tau = 0.55` versus
  `4.8e-4` at `tau = 1.0` — the error grows toward the limit even though
  the grid is fixed. Round-off sensitivity also increases because the
  collision approaches an undamped, noise-amplifying operator.

For these reasons the verified working window is
`tau in [0.55, 1.5]`, `Ma < 0.15`, `Re < 50` for this code.

## 8. Conservation diagnostics

* **Mass.** Collision conserves mass per node; `sum_q S_q = 0` for the
  Guo source; streaming is a permutation; bounce-back returns exactly the
  departed populations. Total fluid mass is therefore conserved to
  round-off (measured drift after 16 250 steps: `9e-13` relative).
* **Momentum.** Per step, the raw fluid momentum change satisfies the
  *exact discrete* balance
  `dP = F_total + (bounce-back wall exchange)`,
  where the wall exchange is accumulated from the reflections of
  Section 5 (`v (c_opp - c_q)` per reflected population `v`). The solver
  computes and stores this residual every step; it is round-off size
  (`~1e-13` relative). In steady state the wall drag must cancel the
  force impulse exactly — measured closure failure `8e-11` relative —
  which is a non-trivial check that forcing and bounce-back are
  implemented consistently with each other.

## 9. References

* Krüger, Kusumaatmaja, Kuschnik, Silva, Viggen — *The Lattice Boltzmann
  Method: Principles and Practice*, Springer (2017). Standard reference
  for D2Q9, BGK, bounce-back, and forcing.
* Guo, Zheng, Shi — "Discrete lattice effects on the forcing term in the
  lattice Boltzmann method", *Phys. Rev. E* **65**, 046308 (2002),
  https://doi.org/10.1103/PhysRevE.65.046308 — the forcing rule of
  Section 4. The mathematics was re-derived and implemented
  independently for this project; no code was copied.
* Succi — *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*,
  Oxford University Press (2001).
