"""
How many environment steps per second, as a function of how many environments.

The number people quote for MJX is a single figure with no denominator: "two
million steps per second". On its own it says nothing, because it is a
throughput on one model at one batch size against an unstated baseline. Three
things have to be measured together before it means anything:

1.  **The CPU baseline on the same model.** MuJoCo's own `rollout` module
    threads across cores, and on a small model with few contacts it is
    genuinely competitive. Reporting a GPU number without it is how a 3x
    speedup gets sold as a 100x one.
2.  **Compile time, separately.** MJX pays a fixed cost in XLA compilation
    before the first step, and on a contact-rich model it is minutes. Folded
    into the throughput of a short run it disappears; a simulation engineer
    sizing a job needs it as its own line, because it decides whether an
    experiment that takes 90 seconds of compute is worth running at all.
3.  **Where it stops scaling, and why it stops.** Throughput rises with batch
    size until the GPU is saturated, then flattens, then the allocation
    fails. The batch size at the knee is the operationally useful number and
    it is not the largest one that fits.

Method notes that change the answer:

*   The timed region is a `jax.lax.scan` inside a single `jit`, so per-step
    Python dispatch and host round trips are outside the measurement. Timing
    a Python loop over `step()` measures the dispatch overhead, which on small
    models is most of the wall clock and has nothing to do with the simulator.
*   `block_until_ready` on the result, once, at the end. JAX dispatch is
    asynchronous: without it the loop returns futures and the measured time is
    how fast Python can enqueue work.
*   Compilation is triggered by a separate warm call on the same shapes and
    timed there. Shapes include the batch size, so every batch size in the
    sweep recompiles and each one gets its own compile number.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ThroughputPoint:
    n_envs: int
    steps_per_s: float | None       # None when the allocation failed
    compile_s: float | None
    wall_s: float | None
    per_env_us: float | None        # microseconds of wall clock per env-step
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "n_envs": self.n_envs,
            "steps_per_s": self.steps_per_s,
            "compile_s": self.compile_s,
            "wall_s": self.wall_s,
            "per_env_us": self.per_env_us,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# GPU / MJX
# --------------------------------------------------------------------------

def mjx_throughput(
    model: mujoco.MjModel,
    n_envs: int,
    *,
    n_steps: int = 100,
    seed: int = 0,
) -> ThroughputPoint:
    """
    Steps per second for `n_envs` copies stepped in lockstep.

    "Step" counts environment steps, not batched calls: a batch of 2048
    advanced once is 2048 steps. That is the unit an RL run consumes, and the
    unit S2's budget of 1e8 is denominated in.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    try:
        mx = mjx.put_model(model)
        dx = mjx.make_data(mx)

        key = jax.random.PRNGKey(seed)
        # Spread the batch out slightly. Identical copies are a best case for
        # the solver: every environment takes the same branch and resolves the
        # same contact set, so warp divergence and the constraint count are
        # both unrealistically uniform. A small qpos jitter costs nothing and
        # removes that flattery.
        qpos0 = jnp.asarray(model.key_qpos[0] if model.nkey else model.qpos0)
        jitter = 0.01 * jax.random.normal(key, (n_envs, model.nq))
        qpos_b = qpos0[None] + jitter

        batch = jax.vmap(lambda q: dx.replace(qpos=q))(qpos_b)
        batch = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))(mx, batch)

        step_v = jax.vmap(mjx.step, in_axes=(None, 0))

        @jax.jit
        def rollout(d):
            def body(carry, _):
                return step_v(mx, carry), None
            out, _ = jax.lax.scan(body, d, None, length=n_steps)
            return out

        t0 = time.perf_counter()
        warm = jax.block_until_ready(rollout(batch))
        compile_s = time.perf_counter() - t0
        del warm

        t0 = time.perf_counter()
        out = jax.block_until_ready(rollout(batch))
        wall = time.perf_counter() - t0
        del out, batch
        gc.collect()

        total = n_envs * n_steps
        return ThroughputPoint(
            n_envs=n_envs,
            steps_per_s=total / wall,
            compile_s=compile_s,
            wall_s=wall,
            per_env_us=1e6 * wall / total,
        )
    except Exception as exc:                       # noqa: BLE001
        # An out-of-memory failure is a data point, not a crash: it is the
        # right edge of the scaling curve. Anything else is recorded verbatim
        # so a real bug is not silently filed as "OOM".
        gc.collect()
        return ThroughputPoint(n_envs, None, None, None, None,
                               error=f"{type(exc).__name__}: {str(exc)[:300]}")


def mjx_sweep(
    model: mujoco.MjModel,
    batch_sizes: list[int],
    *,
    n_steps: int = 100,
    seed: int = 0,
    stop_after_failures: int = 2,
) -> list[ThroughputPoint]:
    """
    Sweep batch size upward, stopping once the device stops accepting work.

    Stops after two consecutive failures rather than one. A single failure can
    come from fragmentation left by the previous allocation rather than from a
    genuine capacity limit, and treating that as the ceiling understates the
    hardware.
    """
    pts: list[ThroughputPoint] = []
    consecutive = 0
    for n in batch_sizes:
        p = mjx_throughput(model, n, n_steps=n_steps, seed=seed)
        pts.append(p)
        status = (f"{p.steps_per_s:,.0f} steps/s  compile {p.compile_s:6.1f}s"
                  if p.steps_per_s else f"FAILED  {p.error}")
        print(f"    n_envs {n:>6,}  {status}", flush=True)
        consecutive = 0 if p.steps_per_s else consecutive + 1
        if consecutive >= stop_after_failures:
            print(f"    stopping sweep after {consecutive} consecutive failures", flush=True)
            break
    return pts


# --------------------------------------------------------------------------
# CPU baseline
# --------------------------------------------------------------------------

def mjc_throughput(
    model: mujoco.MjModel,
    *,
    n_steps: int = 2000,
    n_threads: int | None = None,
    seed: int = 0,
) -> ThroughputPoint:
    """
    The CPU denominator, using MuJoCo's own threaded rollout when available.

    `mujoco.rollout` releases the GIL and threads over independent initial
    states, which is what an honest CPU baseline looks like. Falling back to a
    single-threaded Python loop when it is missing is noted in the result
    rather than passed off as the same measurement, because the two differ by
    the core count and quoting the wrong one flatters the GPU by that factor.
    """
    n_threads = n_threads or _cpu_count()

    try:
        from mujoco import rollout as mjc_rollout
    except ImportError:
        return _mjc_single_thread(model, n_steps=n_steps, seed=seed)

    rng = np.random.default_rng(seed)
    qpos0 = model.key_qpos[0] if model.nkey else model.qpos0
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    initial = np.zeros((n_threads, nstate))
    d = mujoco.MjData(model)
    for i in range(n_threads):
        d.qpos[:] = qpos0 + 0.01 * rng.normal(size=model.nq)
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        mujoco.mj_getState(model, d, initial[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)

    ctrl = np.zeros((n_threads, n_steps, model.nu))
    datas = [mujoco.MjData(model) for _ in range(n_threads)]

    with mjc_rollout.Rollout(nthread=n_threads) as roll:
        roll.rollout(model, datas, initial, control=ctrl)     # warm the threads
        t0 = time.perf_counter()
        roll.rollout(model, datas, initial, control=ctrl)
        wall = time.perf_counter() - t0

    total = n_threads * n_steps
    return ThroughputPoint(n_threads, total / wall, 0.0, wall, 1e6 * wall / total)


def _mjc_single_thread(model: mujoco.MjModel, *, n_steps: int, seed: int) -> ThroughputPoint:
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    mujoco.mj_step(model, d)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        mujoco.mj_step(model, d)
    wall = time.perf_counter() - t0
    p = ThroughputPoint(1, n_steps / wall, 0.0, wall, 1e6 * wall / n_steps)
    p.error = "mujoco.rollout unavailable; single-threaded baseline"
    return p


def _cpu_count() -> int:
    import os
    try:
        return len(os.sched_getaffinity(0))       # respects cgroup limits
    except AttributeError:
        return os.cpu_count() or 1


# --------------------------------------------------------------------------
# Reading the curve
# --------------------------------------------------------------------------

def knee(points: list[ThroughputPoint], *, tol: float = 0.10) -> ThroughputPoint | None:
    """
    The smallest batch size within `tol` of the best throughput observed.

    This, not the largest batch that fits, is the number to run at. Beyond the
    knee each extra environment buys under 10% more throughput while costing
    proportional memory and lengthening every compile, and on a preemptible
    free-tier session a longer compile is a direct loss.
    """
    ok = [p for p in points if p.steps_per_s]
    if not ok:
        return None
    best = max(p.steps_per_s for p in ok)
    for p in ok:
        if p.steps_per_s >= (1 - tol) * best:
            return p
    return None
