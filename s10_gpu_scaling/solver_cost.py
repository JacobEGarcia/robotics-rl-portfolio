"""
The knob that actually sets the speed/fidelity trade: solver iterations.

Timestep gets all the attention, and it is the wrong first knob for a
contact-rich model on a GPU. Halving the timestep doubles the work exactly.
Halving the solver iteration count also roughly halves the constraint work,
but the accuracy it costs depends entirely on whether the solver was
converging in that budget to begin with, and on a hand holding a cube it
usually is, several iterations before the default cuts it off.

So the interesting measurement is a Pareto: for each iteration budget, what
throughput and what error, both against a reference whose budget is large
enough that spending more changes nothing.

Two things this is careful about.

**Accuracy is measured against the tight-solver trajectory, not against
penetration depth.** Penetration is tempting because it is a single number
with an obvious right answer of zero, but MuJoCo's contacts are soft by
design: the steady-state penetration is set by `solref` and is *supposed* to
be nonzero. A solver that drives it to zero is not more accurate, it is
running a different contact model. Trajectory divergence has no such
ambiguity.

**Throughput is measured on the device the answer is for.** Iteration count
costs differently on a GPU than on a CPU: the batched solver runs a fixed
number of iterations for the whole batch rather than exiting early per
environment, so on MJX the budget is paid in full by every environment on
every step, while on the CPU an environment whose constraints are already
satisfied leaves early. This is the single most common surprise when moving
an RL environment from MuJoCo to MJX, and it is why a setting tuned on CPU
is usually the wrong setting on GPU.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from s10_gpu_scaling import parity
from s10_gpu_scaling.throughput import mjx_throughput

# Iteration budgets to sweep. MuJoCo's own default is 100 for the Newton
# solver with 50 line-search iterations, and Menagerie models routinely
# override it downward; the low end here is deliberately below anything
# shipped, so the curve includes the regime where the solver visibly fails.
ITERATION_GRID = (1, 2, 4, 8, 16, 32, 64, 100)


def sweep_iterations(
    model: mujoco.MjModel,
    *,
    n_steps: int = 200,
    n_envs: int = 256,
    grid: tuple[int, ...] = ITERATION_GRID,
    seed: int = 0,
    measure_throughput: bool = True,
) -> dict:
    """
    Accuracy and speed against solver iteration budget, on one model.

    The reference is the same converged, tightly-solved, finely-integrated
    trajectory `parity.py` builds, so the numbers here sit on the same axis as
    the parity result and the two figures can be read together.
    """
    qpos0, qvel0 = parity.initial_state(model)
    ctrl = parity.make_controls(model, n_steps, seed=seed)

    conv = parity.converged_reference(model, qpos0, qvel0, ctrl)
    ref = conv.pop("reference")

    rows = []
    for iters in grid:
        m = copy.deepcopy(model)
        m.opt.iterations = iters
        # Line search is swept along with the main iteration count rather than
        # held fixed. Holding ls_iterations at its default while cutting
        # iterations to 1 does not produce a cheap solver, it produces an
        # expensive one that takes a single very carefully chosen step, and
        # the throughput curve comes out nearly flat for a reason that has
        # nothing to do with the variable being swept.
        m.opt.ls_iterations = max(1, iters // 2)

        traj = parity.rollout_mjc(m, qpos0, qvel0, ctrl, substeps=1,
                                  label=f"iters{iters}")
        curve = parity.divergence(ref, traj)

        row = {
            "iterations": iters,
            "ls_iterations": int(m.opt.ls_iterations),
            "final_m": float(curve[-1]),
            "max_m": float(curve.max()),
            "cpu_wall_s": traj.wall_s,
            "unstable": traj.extra.get("unstable", False),
            "warnings": traj.extra.get("warnings", {}),
        }
        if measure_throughput:
            p = mjx_throughput(m, n_envs, n_steps=100, seed=seed)
            row["mjx_steps_per_s"] = p.steps_per_s
            row["mjx_compile_s"] = p.compile_s
            row["mjx_error"] = p.error
        rows.append(row)
        print(f"    iters {iters:>4}  final {row['final_m']:.3e} m  "
              f"{row.get('mjx_steps_per_s') or 0:,.0f} steps/s", flush=True)

    return {
        "grid": list(grid),
        "n_envs": n_envs,
        "n_steps": n_steps,
        "reference_check": conv,
        "rows": rows,
        "shipped": {
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "solver": int(model.opt.solver),
        },
    }


def is_solver_limited(rows: list[dict], *, min_spread: float = 3.0) -> bool:
    """
    Did the iteration budget change the answer at all?

    If the worst and best errors across the whole grid are within `min_spread`
    of each other, the solver was never the binding constraint, and any
    "recommended" budget read off that grid is noise. This is not a corner
    case: it is what happens on every model whose contacts are absent or
    trivial, which is most arms in free space.

    Observed on `unitree_z1`: errors of 1.79e-2 through 1.97e-2 across
    iteration counts 1 to 100, a spread of 1.10x. Taken at face value the
    grid recommends 1 iteration against the shipped 100, and it recommends it
    because 1 happened to score marginally lowest, not because the solver
    converged. On a system the constraint solver barely touches, the ordering
    across the grid is chaotic scatter with no monotone trend to read.
    """
    ok = [r["final_m"] for r in rows if np.isfinite(r["final_m"])]
    if len(ok) < 2:
        return False
    lo, hi = min(ok), max(ok)
    return (hi / max(lo, 1e-12)) >= min_spread


def cheapest_within(rows: list[dict], *, factor: float = 2.0) -> dict | None:
    """
    The smallest iteration budget whose error is within `factor` of the best.

    Reported as the recommendation, because "use the shipped default" and "use
    the maximum" are both answers that were never measured. If this lands
    below the value the model ships, that is a concrete, defensible saving on
    every step of every environment for the whole run.

    Returns None when the grid is not solver-limited. Refusing to answer is
    the correct output there; a number would be read as a tuning
    recommendation and it would be scatter.
    """
    ok = [r for r in rows if np.isfinite(r["final_m"])]
    if not ok or not is_solver_limited(rows):
        return None
    best = min(r["final_m"] for r in ok)
    # Guard the degenerate case: if the tightest setting already matches the
    # reference to machine precision, `best` is ~0 and every ratio is huge.
    floor = max(best, 1e-12)
    for r in sorted(ok, key=lambda r: r["iterations"]):
        if r["final_m"] <= factor * floor:
            return r
    return None
