"""Reproduce the full Channel Lattice verification suite.

Run from the repository root::

    python examples/run_verification.py

Produces, in ``docs/results/``:

* ``channel_profile.png``   — steady profile vs the analytic parabola (cover)
* ``profile_development.gif``— animation of the parabola developing from rest
* ``convergence.csv/.png``  — grid-refinement study (second-order expected)
* ``tau_study.csv/.png``    — relaxation-time study (error growth near 0.5)
* ``run_summary.json``      — all metrics plus runtime versions

Every frame of the animation is an actual computed state of the solver; the
axes are fixed, the background white, and no grid lines are drawn.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_lattice import ChannelFlowLBM, analytic, units  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BENCHMARK = dict(nx=64, ny=33, tau=0.9, u_target=0.065)
GRID_STUDY_NY = (17, 25, 33, 49, 65)
TAU_STUDY_TAU = (0.55, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5)
STUDY_NX = 8
STUDY_U_TARGET = 0.06
STUDY_TOL = 1e-10

STUDY_FIELDS = [
    "study", "nx", "ny", "h_lattice", "tau", "nu", "force_g",
    "u_max_target", "u_max_measured", "u_max_ratio",
    "rel_l2", "rel_linf", "steady_residual",
    "mass_drift_rel", "steps", "converged", "wall_time_s",
]


def make_flow(nx: int, ny: int, tau: float, u_target: float) -> ChannelFlowLBM:
    nu = units.lattice_viscosity(tau)
    g = analytic.force_for_target_u_max(u_target, nu, ny - 2)
    return ChannelFlowLBM(nx=nx, ny=ny, tau=tau, force_g=g)


def measure(flow: ChannelFlowLBM, study: str, extra: dict) -> dict:
    js, u = flow.profile()
    ex = flow.analytic_profile()
    l2, linf = analytic.error_norms(u, ex)
    return {
        "study": study,
        "nx": flow.nx, "ny": flow.ny, "h_lattice": flow.h_lattice,
        "tau": flow.tau, "nu": flow.nu, "force_g": flow.force_g,
        "u_max_target": flow.u_max_analytic,
        "u_max_measured": float(u.max()),
        "u_max_ratio": float(u.max()) / flow.u_max_analytic,
        "rel_l2": l2, "rel_linf": linf,
        "steady_residual": flow.steady_residual,
        "mass_drift_rel": abs(flow.mass() - flow.rho0 * flow.nx
                              * (flow.ny - 2)) / flow.mass(),
        "steps": flow.steps, "converged": bool(flow.converged),
        **extra,
    }


def style_axes(ax, ny: int, u_max_ex: float, walls: bool = True) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlim(0.0, 1.08 * u_max_ex)
    ax.set_ylim(0.0, float(ny - 1))
    ax.set_xlabel(r"streamwise velocity $u_x$  [lattice units per step]")
    ax.set_ylabel(r"wall-normal coordinate $y$  [lattice units]")
    if walls:
        for yw in (0.5, ny - 1.5):
            ax.axhline(yw, color="0.55", lw=1.2)
        ax.annotate("wall (half-way bounce-back)", xy=(0.02, 0.5),
                    xycoords=("axes fraction", "data"), fontsize=8,
                    color="0.35", va="bottom")


def plot_cover(flow: ChannelFlowLBM, row: dict) -> Path:
    js, u = flow.profile()
    ex = flow.analytic_profile()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    fig.patch.set_facecolor("white")
    style_axes(ax, flow.ny, flow.u_max_analytic)
    ax.plot(ex, js, "k--", lw=1.4, label="analytic parabola (exact walls)")
    ax.plot(u, js, "o-", color="#1565c0", ms=4, lw=1.4,
            label="lattice-Boltzmann (steady)")
    text = (
        f"rel. $L_2$ = {row['rel_l2']:.2e}    rel. $L_\\infty$ = "
        f"{row['rel_linf']:.2e}\n"
        f"$u_{{max}}$ ratio = {row['u_max_ratio']:.5f}\n"
        f"$\\tau$ = {flow.tau}    $\\nu$ = {flow.nu:.4f}    "
        f"$g$ = {flow.force_g:.3e}\n"
        f"Re = {flow.reynolds:.1f}    Ma = {flow.mach:.3f}    "
        f"steps = {flow.steps}"
    )
    ax.text(0.97, 0.05, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))
    ax.set_title("Channel Lattice - steady D2Q9 channel flow vs analytic")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.30), frameon=False,
              fontsize=9)
    out = RESULTS / "channel_profile.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def capture_animation(params: dict, steady_steps: int) -> tuple[Path, list[int]]:
    """Rerun from the zero-momentum start, saving true computed states.

    The frame schedule includes ``t = 0`` (the Guo half-step startup
    profile, visually flat) and runs to the measured steady step count.
    Frames are indexed positionally into the schedule with an explicit
    ``init_func``, so the first encoded frame shows ``t = 0`` and the last
    shows the steady state, deterministically.  A concise static title is
    paired with a two-line time annotation placed *inside* the axes, so
    nothing can clip at the canvas edges.
    """
    flow = make_flow(params["nx"], params["ny"], params["tau"],
                     params["u_target"])
    u_max_ex = flow.u_max_analytic
    schedule = [0]
    schedule.extend(np.unique(
        np.round(np.geomspace(3, max(steady_steps, 100), 63)).astype(int)
    ).tolist())
    if not 30 <= len(schedule) <= 80:
        raise RuntimeError(f"animation schedule has {len(schedule)} frames")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    fig.patch.set_facecolor("white")
    style_axes(ax, flow.ny, u_max_ex)
    js, u_start = flow.profile()
    (line,) = ax.plot(u_start, js, color="#1565c0", lw=2.0,
                      label="computed profile $u_x(y,t)$")
    (exact,) = ax.plot(flow.analytic_profile(), js, "k--", lw=1.2,
                       label="analytic steady profile")
    ax.set_title("Development of the plane Poiseuille profile", fontsize=11)
    # Two-line, dynamic time annotation INSIDE the axes (cannot clip).
    annot = ax.text(0.98, 0.965, "", transform=ax.transAxes, ha="right",
                    va="top", fontsize=9, family="DejaVu Sans")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    recorded: list[int] = []

    def _annotate(n_steps: int, u: np.ndarray) -> None:
        frac = float(u.max()) / u_max_ex
        annot.set_text(f"t = {n_steps} lattice steps\n"
                       f"$u_{{\\max}}$ = {frac:.1%} of steady")

    def init():
        _annotate(0, u_start)
        return line, exact, annot

    def update(frame_idx: int):
        target = schedule[frame_idx]
        while flow.steps < target:
            flow.step()
        _, u = flow.profile()
        line.set_data(u, js)
        _annotate(flow.steps, u)
        recorded.append(flow.steps)
        return line, exact, annot

    out = RESULTS / "profile_development.gif"
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=range(len(schedule)), interval=125)
    anim.save(out, writer=PillowWriter(fps=8), savefig_kwargs={
        "facecolor": fig.get_facecolor()})
    plt.close(fig)
    # Reconcile the encoded frames with the schedule before reporting.
    if recorded != schedule:
        raise RuntimeError(
            f"encoded frame steps diverge from schedule: first={recorded[0] if recorded else None}, "
            f"last={recorded[-1] if recorded else None}, expected "
            f"({schedule[0]}, {schedule[-1]})")
    return out, schedule


def run_study(name: str, cases) -> list[dict]:
    rows = []
    for params in cases:
        flow = make_flow(STUDY_NX, params["ny"], params["tau"],
                         STUDY_U_TARGET)
        t0 = time.perf_counter()
        flow.run_until_steady(tol=STUDY_TOL)
        wall = time.perf_counter() - t0
        if not flow.converged:
            raise RuntimeError(
                f"{name} case {params} did not reach steady state")
        row = measure(flow, name, {"wall_time_s": round(wall, 3)})
        rows.append(row)
        print(f"  [{name}] ny={flow.ny:3d} tau={flow.tau:5.2f} "
              f"steps={flow.steps:7d} l2={row['rel_l2']:.3e} "
              f"linf={row['rel_linf']:.3e} ({wall:.1f}s)")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STUDY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_grid_study(rows: list[dict]) -> Path:
    h = np.array([r["h_lattice"] for r in rows], dtype=float)
    l2 = np.array([r["rel_l2"] for r in rows])
    linf = np.array([r["rel_linf"] for r in rows])
    # Error scales as (dx)^2 with dx = 1/h lattice links per channel height,
    # i.e. error ~ h^(-2): report the negated log-log slope as the order.
    order = -float(np.polyfit(np.log(h), np.log(l2), 1)[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.loglog(h, l2, "o-", color="#1565c0", label=r"relative $L_2$")
    ax.loglog(h, linf, "s-", color="#e07b39", label=r"relative $L_\infty$")
    ref = l2[-1] * (h[-1] / h) ** 2
    ax.loglog(h, ref, "k--", lw=1, label=r"$\propto h^{-2}$ (2nd order in $\Delta x$)")
    ax.set_xlabel("channel height $h$  [lattice units]")
    ax.set_ylabel("relative profile error")
    ax.set_title(f"Grid refinement at $\\tau$ = {rows[0]['tau']} "
                 f"(measured order p = {order:.2f})")
    ax.legend(frameon=False, fontsize=9)
    out = RESULTS / "convergence.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_tau_study(rows: list[dict]) -> Path:
    tau = np.array([r["tau"] for r in rows])
    l2 = np.array([r["rel_l2"] for r in rows])
    linf = np.array([r["rel_linf"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.semilogy(tau, l2, "o-", color="#1565c0", label=r"relative $L_2$")
    ax.semilogy(tau, linf, "s-", color="#e07b39", label=r"relative $L_\infty$")
    ax.axvline(0.5, color="0.6", lw=1, ls=":")
    ax.annotate(r"$\tau = 1/2$ ($\nu \to 0$, scheme limit)",
                xy=(0.5, max(l2)), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color="0.35")
    ax.set_xlabel(r"relaxation time $\tau$")
    ax.set_ylabel("relative profile error")
    ax.set_title(f"Relaxation-time study at $h$ = {rows[0]['h_lattice']} "
                 "lattice units")
    ax.legend(frameon=False, fontsize=9)
    out = RESULTS / "tau_study.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    t_start = time.perf_counter()
    print(f"channel_lattice verification run  ({datetime.now(timezone.utc).isoformat()})")
    print(f"python {platform.python_version()} | numpy {np.__version__} | "
          f"matplotlib {matplotlib.__version__}")

    # ------------------------------------------------------------------ #
    print("[1/4] benchmark channel + animation")
    flow = make_flow(**BENCHMARK)
    t0 = time.perf_counter()
    steady_steps = flow.run_until_steady(tol=1e-10)
    bench_wall = time.perf_counter() - t0
    if not flow.converged:
        raise RuntimeError("benchmark run failed to reach steady state")
    bench = measure(flow, "benchmark", {"wall_time_s": round(bench_wall, 3)})
    print(f"  steady after {flow.steps} steps ({bench_wall:.1f}s): "
          f"l2={bench['rel_l2']:.3e} linf={bench['rel_linf']:.3e} "
          f"u_max ratio={bench['u_max_ratio']:.5f}")
    cover = plot_cover(flow, bench)
    print(f"  cover still: {cover.name}")
    gif, schedule = capture_animation(BENCHMARK, steady_steps)
    with Image.open(gif) as im:
        frames = im.n_frames
        size = im.size
    if frames != len(schedule):
        raise RuntimeError(
            f"GIF encodes {frames} frames but schedule has {len(schedule)}")
    print(f"  animation: {gif.name} ({frames} frames, {size[0]}x{size[1]} px, "
          f"t = {schedule[0]} .. {schedule[-1]})")

    # ------------------------------------------------------------------ #
    print("[2/4] grid-refinement study")
    grid_rows = run_study("grid", [dict(ny=ny, tau=BENCHMARK["tau"])
                                   for ny in GRID_STUDY_NY])
    write_csv(RESULTS / "convergence.csv", grid_rows)
    conv_plot = plot_grid_study(grid_rows)

    # ------------------------------------------------------------------ #
    print("[3/4] relaxation-time study")
    tau_rows = run_study("tau", [dict(ny=BENCHMARK["ny"], tau=tau)
                                 for tau in TAU_STUDY_TAU])
    write_csv(RESULTS / "tau_study.csv", tau_rows)
    tau_plot = plot_tau_study(tau_rows)

    # ------------------------------------------------------------------ #
    print("[4/4] run summary")
    h = np.array([r["h_lattice"] for r in grid_rows], dtype=float)
    l2 = np.array([r["rel_l2"] for r in grid_rows])
    # Error ~ (dx)^2 with dx = 1/h, so the order is the negated log-log slope.
    order = -float(np.polyfit(np.log(h), np.log(l2), 1)[0])
    total = time.perf_counter() - t_start
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
            "platform": platform.platform(),
        },
        "benchmark": {
            **{k: v for k, v in BENCHMARK.items()},
            "nu_lattice": flow.nu,
            "force_g": flow.force_g,
            "mach": flow.mach,
            "reynolds": flow.reynolds,
            "steps_to_steady": flow.steps,
            "steady_residual": flow.steady_residual,
            "u_max_analytic": flow.u_max_analytic,
            **{k: bench[k] for k in
               ("u_max_measured", "u_max_ratio", "rel_l2", "rel_linf",
                "mass_drift_rel", "wall_time_s")},
            "wall_positions_lattice": list(flow.wall_positions),
            "h_lattice": flow.h_lattice,
        },
        "grid_study": {
            "tau": BENCHMARK["tau"],
            "u_max_target": STUDY_U_TARGET,
            "measured_order_l2": order,
            "cases": [{k: r[k] for k in STUDY_FIELDS} for r in grid_rows],
        },
        "tau_study": {
            "ny": BENCHMARK["ny"],
            "h_lattice": BENCHMARK["ny"] - 2,
            "u_max_target": STUDY_U_TARGET,
            "cases": [{k: r[k] for k in STUDY_FIELDS} for r in tau_rows],
        },
        "animation": {
            "file": gif.name,
            "frames": frames,
            "size_px": list(size),
            "fps": 8,
            "capture_steps_first_last": [schedule[0], schedule[-1]],
        },
        "artifacts": {
            "cover": cover.name,
            "convergence_plot": conv_plot.name,
            "tau_plot": tau_plot.name,
            "grid_csv": "convergence.csv",
            "tau_csv": "tau_study.csv",
        },
        "total_wall_time_s": round(total, 1),
    }
    out = RESULTS / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  {out.name} written; measured grid order p = {order:.2f}; "
          f"total {total:.0f}s")
    print("done.")


if __name__ == "__main__":
    main()
