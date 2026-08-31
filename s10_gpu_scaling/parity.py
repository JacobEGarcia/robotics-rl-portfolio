"""
Does the GPU simulator agree with the CPU one, and what does "agree" mean
for a contact-rich system?

The naive version of this measurement is worthless. Step MuJoCo C and MJX
from the same state with the same controls, plot the difference, watch it
grow, conclude that MJX is inaccurate. Every pair of implementations produces
that plot, because a robot in contact is chaotic: nearby trajectories separate
exponentially, so *any* perturbation, including one at the last bit of a
float, ends up macroscopic. The plot measures chaos, not correctness.

What makes the question answerable is a control that has no implementation
difference in it at all: the same code, at the same settings, started one
float32 ulp apart.

**The two questions are asked against different baselines, and mixing them up
invalidates both.**

*Question 1, the one that is actually about MJX.* Baseline is MuJoCo C at the
production timestep with the model's shipped solver settings. The candidates
are MJX at that same timestep, in float32 and in float64. Discretisation error
is then bit-for-bit identical on both sides and cancels out of the comparison,
so whatever is left is the port.

    mjx_f32 vs mjc_dt   what S2 actually trains in
    mjx_f64 vs mjc_dt   the same, with precision removed as an explanation
    envelope            floor A: what ONE float32-sized error becomes
    roundoff            floor B: what float32 state truncation costs per step

**There are two floors and using the wrong one inverts the answer.**
`chaos_envelope` perturbs the initial state once and integrates in float64.
`roundoff_envelope` truncates the carried state to float32 after every step,
in the same binary, in float64 arithmetic. They are not close: measured here,
floor B sits 24,000-42,000x above floor A on three different models. Floor A
answers "how chaotic is this system", which is interesting and is *not* the
question, because float32 does not make one rounding error, it makes one on
every operation of every step. Against floor A the Panda's MJX gap looks like
a 100,000x scandal; against floor B it is 4x, which is the same order of
magnitude and an entirely ordinary result. The first version of this file had
only floor A and would have reported the scandal.

Readings: `mjx_f32` at or below `roundoff` means precision explains the gap
and MJX is as right as float32 permits. `mjx_f32` above `roundoff` while
`mjx_f64` sits near zero still means precision, float32 arithmetic is worse
than float32 storage, and `roundoff` is a lower bound on it, not a simulation
of it. Only `mjx_f64` staying large is evidence of an algorithmic difference,
and then the place to look is the constraint solver and the contact set.

*Question 2, the denominator.* How much did the production timestep already
cost, before any of this? Reference is MuJoCo C at `dt/REFINE` with a tight
solver; the candidate is MuJoCo C at `dt`. If that gap dwarfs the MJX gap,
then moving to the GPU perturbed the trajectory less than the timestep choice
had already perturbed it, and no fidelity objection to MJX survives it.

This split is the whole design, and it exists because the obvious version does
not work. Comparing every candidate to one very fine reference *seems*
tidier, and it silently caps the resolution of question 1 at the reference's
own residual discretisation error. Measured here at REFINE=32, that residual
is ~1e-5 m while the chaos envelope is ~1e-9 m: four orders of magnitude of
headroom that the fine reference simply cannot see into. Every MJX number
would have come back sitting on the reference's error floor, looking like
agreement. `reference_check` reports that floor with every result so the claim
stays falsifiable.

float64 in JAX is a process-global setting (`jax_enable_x64`), and mjx models
are built out of arrays whose dtype is fixed at construction, so the two MJX
cases cannot share a process. `run.py` re-executes this module in a
subprocess with `JAX_ENABLE_X64=1` for the second one, and the loader below
asserts the dtype it actually got instead of trusting the environment
variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

# How much finer the reference integrates than the case under test. 32 is not
# arbitrary: below ~16 the reference's own discretisation error is within an
# order of magnitude of the effect being measured, and above ~64 it costs
# minutes per model for a curve that has stopped moving. `converged_reference`
# checks this rather than leaving it as a claim.
REFINE = 32

# One float32 ulp relative to a joint angle of order 1 rad. This is the
# smallest difference float32 cannot represent, so it is the smallest
# perturbation a float32 simulator is guaranteed to make.
F32_ULP = float(np.spacing(np.float32(1.0)))


@dataclass
class Trajectory:
    """qpos and body positions sampled at the coarse control rate."""
    qpos: np.ndarray        # (T+1, nq)
    xpos: np.ndarray        # (T+1, nbody, 3)
    label: str = ""
    wall_s: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Control sequence
# --------------------------------------------------------------------------

def make_controls(model: mujoco.MjModel, n_steps: int, seed: int = 0) -> np.ndarray:
    """
    A smooth, bounded, deterministic control sequence.

    Smooth on purpose. A white-noise control sequence excites the actuators at
    the Nyquist frequency of the coarse timestep, which the reference resolves
    and the coarse run cannot, so the comparison would be dominated by an
    input the two integrators do not even agree is the same signal. This is a
    sum of three low-frequency sinusoids per actuator, scaled into the control
    range, which every timestep in the sweep resolves identically.
    """
    rng = np.random.default_rng(seed)
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    unlimited = model.actuator_ctrllimited == 0
    lo[unlimited], hi[unlimited] = -1.0, 1.0

    t = np.arange(n_steps)[:, None] * model.opt.timestep
    freqs = rng.uniform(0.3, 1.5, size=(3, model.nu))
    phase = rng.uniform(0, 2 * np.pi, size=(3, model.nu))
    wave = sum(np.sin(2 * np.pi * freqs[i] * t + phase[i]) for i in range(3)) / 3.0

    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    # 60% of range, so the sequence exercises the actuators without spending
    # the whole trajectory saturated against a limit, where every integrator
    # agrees trivially.
    return mid + 0.6 * half * wave


def initial_state(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Keyframe 0 if the model ships one, otherwise the compiled qpos0."""
    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    else:
        mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    return d.qpos.copy(), d.qvel.copy()


# --------------------------------------------------------------------------
# MuJoCo C rollouts
# --------------------------------------------------------------------------

def rollout_mjc(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    *,
    substeps: int = 1,
    tighten: bool = False,
    round_f32: bool = False,
    label: str = "",
) -> Trajectory:
    """
    Integrate on the CPU, sampling at the coarse control rate.

    `substeps` divides the model's timestep. Each coarse interval holds one
    control value across all of its substeps, so the reference and the coarse
    run are driven by the same piecewise-constant signal and the comparison is
    not contaminated by a difference in the input.

    `round_f32` casts qpos and qvel through float32 and back after every
    substep. The arithmetic stays float64; only the state carried from one
    step to the next is truncated. See `roundoff_envelope` for why that is the
    floor the MJX comparison actually needs.
    """
    import time

    m = _clone(model)
    m.opt.timestep = model.opt.timestep / substeps
    if tighten:
        # The reference must not be solver-limited. These are far past the
        # point where the trajectory stops moving; `converged_reference`
        # verifies that rather than asserting it.
        m.opt.iterations = 500
        m.opt.ls_iterations = 100
        m.opt.tolerance = 1e-14
        m.opt.ls_tolerance = 1e-14

    d = mujoco.MjData(m)
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    mujoco.mj_forward(m, d)

    T = ctrl.shape[0]
    qpos = np.empty((T + 1, m.nq))
    xpos = np.empty((T + 1, m.nbody, 3))
    qpos[0], xpos[0] = d.qpos, d.xpos

    t0 = time.perf_counter()
    for k in range(T):
        d.ctrl[:] = ctrl[k]
        for _ in range(substeps):
            mujoco.mj_step(m, d)
            if round_f32:
                d.qpos[:] = d.qpos.astype(np.float32)
                d.qvel[:] = d.qvel.astype(np.float32)
                # Re-derive everything downstream of the truncated state.
                # Without this, xpos still holds the float64 kinematics from
                # before the rounding, and the recorded trajectory is a
                # mixture of the two precisions.
                mujoco.mj_forward(m, d)
        qpos[k + 1], xpos[k + 1] = d.qpos, d.xpos
    wall = time.perf_counter() - t0

    # MuJoCo reports instability through `d.warning`, on stderr, and then
    # carries on with a clamped state. So a run that went numerically unstable
    # still returns finite numbers and still produces a divergence curve, and
    # nothing downstream can tell it apart from a merely inaccurate run. The
    # counters make the difference explicit: a large error with warnings is a
    # simulation that failed, not a simulation that was imprecise, and those
    # are different claims.
    # `len(d.warning)` rather than a module constant: mujoco 3.11 does not
    # export mjNWARNING to Python, and hardcoding 8 would break silently the
    # next time a warning type is added.
    warn = {mujoco.mjtWarning(i).name: int(d.warning[i].number)
            for i in range(len(d.warning)) if d.warning[i].number}

    return Trajectory(qpos=qpos, xpos=xpos, label=label,
                      wall_s=wall,
                      extra={"substeps": substeps, "tighten": tighten,
                             "round_f32": round_f32,
                             "timestep": float(m.opt.timestep),
                             "iterations": int(m.opt.iterations),
                             "ls_iterations": int(m.opt.ls_iterations),
                             "warnings": warn,
                             "unstable": bool(warn)})


def _clone(model: mujoco.MjModel) -> mujoco.MjModel:
    """
    A private copy, so changing solver settings does not leak between cases.

    Mutating `model.opt` in place and restoring it afterwards works right up
    until an exception skips the restore, at which point every later case
    silently runs with the reference's solver settings and reports
    suspiciously good agreement.
    """
    import copy
    return copy.deepcopy(model)


# --------------------------------------------------------------------------
# MJX rollout
# --------------------------------------------------------------------------

def rollout_mjx(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    *,
    label: str = "",
) -> Trajectory:
    """
    Integrate the same problem through MJX, one environment, no batching.

    Single-environment MJX is the slowest way to run this model on any device
    and that is fine: this function measures agreement, never throughput. The
    throughput question lives in `throughput.py` and is asked with thousands
    of environments, which is the only regime where MJX is the right tool.
    """
    import time

    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    x64 = bool(jax.config.read("jax_enable_x64"))

    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=jnp.asarray(qpos0), qvel=jnp.asarray(qvel0))
    dx = mjx.forward(mx, dx)

    # The dtype MJX actually gave us, not the one the environment asked for.
    # A missing JAX_ENABLE_X64 turns the float64 case into a second float32
    # case that agrees with the first to the last bit, which reads as a
    # startlingly clean result rather than as the misconfiguration it is.
    got = np.dtype(dx.qpos.dtype)
    want = np.float64 if x64 else np.float32
    if got != want:
        raise RuntimeError(
            f"expected MJX arrays in {np.dtype(want).name}, got {got.name}. "
            f"jax_enable_x64={x64}. Set JAX_ENABLE_X64=1 before importing jax."
        )

    step = jax.jit(mjx.step)

    T = ctrl.shape[0]
    ctrl_j = jnp.asarray(ctrl)
    qpos = np.empty((T + 1, model.nq))
    xpos = np.empty((T + 1, model.nbody, 3))
    qpos[0] = np.asarray(dx.qpos)
    xpos[0] = np.asarray(dx.xpos)

    # Compile once, outside the timed section, by stepping a throwaway copy.
    t_c0 = time.perf_counter()
    _ = jax.block_until_ready(step(mx, dx.replace(ctrl=ctrl_j[0])))
    compile_s = time.perf_counter() - t_c0

    t0 = time.perf_counter()
    for k in range(T):
        dx = step(mx, dx.replace(ctrl=ctrl_j[k]))
        qpos[k + 1] = np.asarray(dx.qpos)
        xpos[k + 1] = np.asarray(dx.xpos)
    wall = time.perf_counter() - t0

    return Trajectory(
        qpos=qpos, xpos=xpos, label=label, wall_s=wall,
        extra={"x64": x64, "dtype": got.name, "compile_s": round(compile_s, 3),
               "device": str(jax.devices()[0])},
    )


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

def divergence(a: Trajectory, b: Trajectory, *, use: str = "xpos") -> np.ndarray:
    """
    Per-sample distance between two trajectories.

    Defaults to Cartesian body positions rather than qpos, for two reasons.
    A free joint's qpos mixes metres with quaternion components, so an
    L2 norm over qpos adds quantities with different units and the result
    depends on the modeller's choice of scale. And a hinge angle far up a
    kinematic chain moves an end effector much further than one near the tip,
    so a qpos norm underweights exactly the errors that matter. `xpos` is in
    metres everywhere and is what a downstream task would feel.
    """
    if use == "xpos":
        d = a.xpos - b.xpos
        return np.sqrt((d ** 2).sum(axis=-1)).max(axis=-1)   # worst body, per sample
    if use == "qpos":
        return np.abs(a.qpos - b.qpos).max(axis=-1)
    raise ValueError(use)


def summarize_divergence(curve: np.ndarray, dt: float) -> dict:
    """
    Reduce a divergence curve to numbers that can be compared across models.

    `t_1mm` is the headline: how long two implementations stay within a
    millimetre. It is reported as a step index and a wall time, and as None
    when the threshold is never crossed, never as the trajectory length, which
    would read as "diverged at the end" for a run that never diverged at all.
    """
    over = np.nonzero(curve > 1e-3)[0]
    return {
        "final_m": float(curve[-1]),
        "max_m": float(curve.max()),
        "steps_to_1mm": int(over[0]) if over.size else None,
        "seconds_to_1mm": float(over[0] * dt) if over.size else None,
    }


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

def converged_reference(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
) -> dict:
    """
    Check that the reference has stopped moving as it is refined.

    Compares REFINE against REFINE*2. If the gap between them is not small
    against the effects being measured, the "reference" is just another
    candidate and every number downstream is meaningless. This is the control
    that makes the rest of the file honest, so it runs by default rather than
    behind a flag.
    """
    a = rollout_mjc(model, qpos0, qvel0, ctrl, substeps=REFINE, tighten=True, label="ref")
    b = rollout_mjc(model, qpos0, qvel0, ctrl, substeps=REFINE * 2, tighten=True, label="ref2")
    c = divergence(a, b)
    return {
        "refine": REFINE,
        "self_gap_final_m": float(c[-1]),
        "self_gap_max_m": float(c.max()),
        "reference": a,
    }


def chaos_envelope(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    baseline: Trajectory,
    *,
    n_perturbations: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, list[Trajectory]]:
    """
    The noise floor: what the system does to a float32-sized rounding error.

    Perturbs the initial joint angles by one float32 ulp, in a random
    direction, and integrates **at the baseline's own settings**, same
    binary, same timestep, same solver. That is what makes this a floor and
    not just another candidate: the only difference between the two runs is a
    rounding error of the size float32 cannot avoid, so the separation between
    them is entirely the system's own amplification.

    Several directions, because a single one can land on a locally stable mode
    and understate the amplification; the envelope reported is the elementwise
    maximum, which is the conservative reading.

    Only qpos is perturbed, and only the non-quaternion entries. Adding a ulp
    to one component of a unit quaternion denormalises it, and MuJoCo will
    renormalise at the next `mj_forward`, so part of the perturbation would be
    silently discarded and the floor would come out too low.
    """
    rng = np.random.default_rng(seed)
    substeps = int(baseline.extra.get("substeps", 1))
    tighten = bool(baseline.extra.get("tighten", False))

    free_quat = _quaternion_indices(model)
    mask = np.ones(model.nq, dtype=bool)
    mask[free_quat] = False

    curves, trajs = [], []
    for i in range(n_perturbations):
        dq = np.zeros(model.nq)
        v = rng.normal(size=int(mask.sum()))
        dq[mask] = F32_ULP * v / np.linalg.norm(v) * np.sqrt(mask.sum())
        t = rollout_mjc(model, qpos0 + dq, qvel0, ctrl,
                        substeps=substeps, tighten=tighten, label=f"envelope{i}")
        curves.append(divergence(baseline, t))
        trajs.append(t)
    return np.maximum.reduce(curves), trajs


def roundoff_envelope(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    ctrl: np.ndarray,
    baseline: Trajectory,
) -> np.ndarray:
    """
    The floor that actually applies to a float32 simulator.

    `chaos_envelope` perturbs the initial state by one float32 epsilon and
    then integrates in full float64. That answers "what does the system do to
    a single rounding error", and it is the wrong question, because float32
    does not make one rounding error. It makes one on every operation, of
    every step, forever. On a system whose amplification is mild, a
    contact-free arm, say, a single initial perturbation stays small while
    continuously re-injected round-off random-walks upward, and the one-ulp
    envelope can understate the float32 floor by orders of magnitude. Reading
    an MJX result against it then indicts the port for something arithmetic
    was always going to do.

    This runs the same MuJoCo C, in float64, and truncates only the *state
    carried across the step boundary* to float32. Arithmetic inside the step
    stays exact, so this is a lower bound on float32's cost rather than a
    simulation of it, but it is a bound derived from the dynamics, using the
    same binary, with no implementation difference anywhere in it.

    An MJX float32 curve above this line is a real discrepancy. Below or on
    it, precision explains the whole thing.
    """
    substeps = int(baseline.extra.get("substeps", 1))
    tighten = bool(baseline.extra.get("tighten", False))
    t = rollout_mjc(model, qpos0, qvel0, ctrl, substeps=substeps,
                    tighten=tighten, round_f32=True, label="roundoff")
    return divergence(baseline, t)


def integrator_sweep(
    model: mujoco.MjModel,
    *,
    n_steps: int = 40,
    seed: int = 0,
    candidates: tuple[str, ...] = ("mjINT_EULER", "mjINT_IMPLICITFAST",
                                   "mjINT_RK4", "mjINT_IMPLICIT"),
) -> dict:
    """
    Localise an MJX/MuJoCo disagreement to the integrator.

    This is the experiment that turns "MJX diverges and it is not precision"
    into a cause. Run the identical comparison under each integrator the model
    could have been written with. If one of them collapses the divergence to
    the float64 noise level while another does not, the disagreement lives in
    that integrator's implementation and nowhere else, and the collapsing case
    doubles as the positive control the whole module needs: proof that the
    harness *can* show agreement, so a large number elsewhere is a result
    rather than a bug in the comparison.

    An integrator MJX refuses outright is recorded as `unsupported`, not
    skipped. "MJX does not implement this" is a harder and more useful fact
    than any divergence number, and a sweep that quietly dropped it would lose
    the strongest finding it produced.
    """
    import copy

    qpos0, qvel0 = initial_state(model)
    ctrl = make_controls(model, n_steps, seed=seed)
    shipped = mujoco.mjtIntegrator(model.opt.integrator).name

    rows = []
    for name in candidates:
        m = copy.deepcopy(model)
        m.opt.integrator = getattr(mujoco.mjtIntegrator, name)
        coarse = rollout_mjc(m, qpos0, qvel0, ctrl, substeps=1)
        floor = roundoff_envelope(m, qpos0, qvel0, ctrl, coarse)
        row = {"integrator": name, "shipped": name == shipped,
               "float32_floor_m": float(floor[-1]),
               "unstable": coarse.extra.get("unstable", False)}
        try:
            mx = rollout_mjx(m, qpos0, qvel0, ctrl)
            d = divergence(coarse, mx)
            row["mjx_vs_mjc_m"] = float(d[-1])
            row["ratio_to_floor"] = float(d[-1] / max(floor[-1], 1e-30))
        except NotImplementedError as exc:
            row["unsupported"] = str(exc)
        except Exception as exc:                      # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
    return {"shipped": shipped, "n_steps": n_steps, "rows": rows}


def _quaternion_indices(model: mujoco.MjModel) -> list[int]:
    """qpos slots belonging to a free or ball joint's quaternion."""
    idx: list[int] = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        t = model.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            idx += list(range(adr + 3, adr + 7))
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            idx += list(range(adr, adr + 4))
    return idx


def run_parity(
    model: mujoco.MjModel,
    *,
    n_steps: int = 300,
    seed: int = 0,
    include_mjx: bool = True,
    n_perturbations: int = 3,
) -> dict:
    """
    One model, both comparisons. Returns plain data, no plotting.

    Everything except the `mjx_*` curves runs on the CPU and is reproducible
    on any machine, which matters: the baseline and the noise floor must not
    depend on the accelerator whose fidelity is in question.

    Two groups come back, and they are kept apart in the output because they
    are measured against different baselines and are not comparable to each
    other:

        curves_vs_mjc   envelope, mjx_f32, mjx_f64   baseline = MuJoCo C at dt
        curves_vs_ref   mjc_dt                       baseline = MuJoCo C at dt/REFINE
    """
    dt = float(model.opt.timestep)
    qpos0, qvel0 = initial_state(model)
    ctrl = make_controls(model, n_steps, seed=seed)

    # --- question 1: is the port faithful, measured at the production dt ---
    coarse = rollout_mjc(model, qpos0, qvel0, ctrl, substeps=1, label="mjc_dt")
    env_curve, _ = chaos_envelope(model, qpos0, qvel0, ctrl, coarse,
                                  n_perturbations=n_perturbations, seed=seed)

    vs_mjc = {
        "envelope": env_curve,
        "roundoff": roundoff_envelope(model, qpos0, qvel0, ctrl, coarse),
    }
    walls = {"mjc_dt": coarse.wall_s}
    extra: dict = {}

    if include_mjx:
        import jax
        tag = "mjx_f64" if jax.config.read("jax_enable_x64") else "mjx_f32"
        mx = rollout_mjx(model, qpos0, qvel0, ctrl, label=tag)
        vs_mjc[tag] = divergence(coarse, mx)
        walls[tag] = mx.wall_s
        extra[tag] = mx.extra

    # --- question 2: what the production dt already cost ---
    conv = converged_reference(model, qpos0, qvel0, ctrl)
    ref: Trajectory = conv.pop("reference")
    vs_ref = {"mjc_dt": divergence(ref, coarse)}

    summary = {k: summarize_divergence(v, dt) for k, v in {**vs_mjc, **vs_ref}.items()}

    # A curve is only meaningful above the floor of the thing it was measured
    # against. Flag it in the data rather than leaving it to whoever reads the
    # figure: a number sitting on its baseline's own error is not agreement,
    # it is a measurement that ran out of resolution.
    floor_ref = conv["self_gap_max_m"]
    for k in vs_ref:
        summary[k]["below_baseline_floor"] = bool(summary[k]["max_m"] < floor_ref)
    for k in vs_mjc:
        # The MuJoCo-C-at-dt baseline has no discretisation floor relative to
        # itself; its floor is float64 round-off, which the envelope measures.
        summary[k]["baseline"] = "mjc_dt"
    summary["mjc_dt"]["baseline"] = f"mjc_dt_over_{REFINE}"

    return {
        "dt": dt,
        "n_steps": n_steps,
        "seed": seed,
        "f32_ulp": F32_ULP,
        "reference_check": conv,
        "curves_vs_mjc": {k: v.tolist() for k, v in vs_mjc.items()},
        "curves_vs_ref": {k: v.tolist() for k, v in vs_ref.items()},
        "summary": summary,
        "wall_s": walls,
        "mjx": extra,
        "solver_settings": {
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "solver": int(model.opt.solver),
            "integrator": int(model.opt.integrator),
        },
    }
