"""
S9: what muscle actuation changes, measured on MS-Human-700.

Three properties separate a Hill-type muscle from the position-controlled
motor behind every other humanoid in Menagerie. Each is measured here against
something external, not just plotted.

  1. Redundancy
     700 actuators over 85 DoF. The moment-arm matrix R = d(length)/dq is
     rank-deficient, so activation is not recoverable from joint torque: many
     activation patterns produce identical motion and differ only in stiffness.
     External check: the three base translation DoF must receive exactly zero
     generalised force from any activation pattern, because muscles are
     internal forces and internal forces cannot accelerate a system's centre of
     mass. If that came out non-zero the model would violate Newton's third
     law and nothing else here would be worth measuring.

  2. Force depends on state, not just command
     A motor's available torque is a constant. A muscle's is a function of its
     own length and shortening velocity. Measured against MuJoCo's documented
     closed form (`mju_muscleGain` / `mju_muscleBias`) evaluated independently
     of the simulator, which is the check that the sweep is reading the same
     model the integrator is.

  3. Activation lags command, asymmetrically
     Commanded activation is filtered by first-order dynamics whose time
     constant depends on direction and on current activation. Rise and fall are
     not the same speed. Measured against an independent RK4 integration of
     `mju_muscleDynamics`, so simulator and closed form have to agree.

Why any of this belongs in a sim-to-real portfolio: (3) is latency the
controller cannot command away, (2) means gain scheduling on pose is not
optional, and (1) is the reason a joint-space policy trained on a torque model
does not transfer to a tendon-driven one. Clone Robotics' Myofiber is not a
Hill muscle, but it is pull-only, state-dependent, and redundant, so it
inherits all three problems.

Run
---
    python -m s9_muscle.characterize

Writes media/s9/muscle_characterisation.png and a JSONL run log under runs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import PALETTE  # noqa: E402
from common.runlog import RunLogger, hash_mj_model  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from s9_muscle.control import equality_residual, project_equality  # noqa: E402
from s9_muscle.scene import (  # noqa: E402
    MEDIA_DIR,
    REPO_ROOT,
    actuator_names,
    describe,
    joint_name_for_dof,
    load,
    moment_arms,
)

SEED = 0
# Poses to evaluate rank at. Rank is a property of a configuration, not of the
# model, so reporting it at one pose would be reporting an anecdote.
N_POSES = 24
# The muscle swept in experiment 2. Vastus intermedius: mono-articular, large
# (1697 N peak), and crosses only the knee among the major joints, so sweeping
# knee_angle_r sweeps its length without dragging other joints along.
PROBE_MUSCLE = "vasint_r"
PROBE_JOINT = "knee_angle_r"


# ------------------------------------------------------- 1. redundancy


def measure_redundancy(model, data, *, rng) -> dict:
    """
    Rank of the moment-arm matrix across poses, and what lives in its null space.

    The null space of R^T is the set of joint-space directions no combination
    of muscles can drive. Its dimension is the honest measure of how much of
    this body is actually actuated.
    """
    qpos0 = data.qpos.copy()
    lo, hi = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
    limited = model.jnt_limited.astype(bool)

    ranks, null_weight, residual = [], np.zeros(model.nv), 0.0
    for k in range(N_POSES):
        data.qpos[:] = qpos0
        if k > 0:
            # Sample inside the joint limits. Unlimited joints (the base) are
            # left at the keyframe: randomising the root would change the pose
            # of the whole body without changing any relative configuration.
            for j in range(model.njnt):
                if not limited[j]:
                    continue
                adr = model.jnt_qposadr[j]
                data.qpos[adr] = rng.uniform(lo[j], hi[j])
        # Dependent joints are functions of their drivers, so a sampled pose is
        # not a pose until they are recomputed. See project_equality.
        residual = max(residual, project_equality(model, data))
        R = moment_arms(model, data)
        rank = int(np.linalg.matrix_rank(R, tol=1e-8))
        ranks.append(rank)
        # Right singular vectors past the rank span the unactuated directions.
        _, _, Vt = np.linalg.svd(R)
        null_weight += (Vt[rank:] ** 2).sum(axis=0)

    null_weight /= N_POSES
    order = np.argsort(-null_weight)
    top = [
        {"dof": joint_name_for_dof(model, int(i)), "weight": round(float(null_weight[i]), 4)}
        for i in order[:24]
        if null_weight[i] > 1e-3
    ]

    # External check: internal forces cannot accelerate the centre of mass, so
    # whatever activation pattern we pick, the generalised force on the three
    # base translation DoF must be zero. This is a physics identity, not a fit.
    base_dofs = [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")
    ]
    worst = 0.0
    for _ in range(8):
        data.qpos[:] = qpos0
        data.act[:] = rng.uniform(0.0, 1.0, size=model.na)
        mujoco.mj_forward(model, data)
        worst = max(worst, float(np.abs(data.qfrc_actuator[base_dofs]).max()))

    data.qpos[:] = qpos0
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)

    return {
        "n_poses": N_POSES,
        "nu": int(model.nu),
        "nv": int(model.nv),
        "rank_min": int(min(ranks)),
        "rank_max": int(max(ranks)),
        "rank_median": int(np.median(ranks)),
        "unactuated_dofs_median": int(model.nv - np.median(ranks)),
        "activation_null_space_dim_median": int(model.nu - np.median(ranks)),
        "top_null_space_dofs": top,
        "max_base_force_N": worst,
        "max_equality_residual": residual,
        "ranks": ranks,
    }


# ------------------------------------------------- 2. force-length curve


def measure_force_length(model, data, *, n_points=161) -> dict:
    """
    Sweep the probe joint, hold activation at 1, record force against length.

    Two curves come out and both matter. The simulator's own
    `data.actuator_force` is what the integrator uses. The independently
    evaluated `mju_muscleGain * act + mju_muscleBias` is MuJoCo's documented
    model of the same thing. They should agree to float precision; if they do
    not, the sweep is reading a different configuration than it thinks.
    """
    names = actuator_names(model)
    mus = names.index(PROBE_MUSCLE)
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROBE_JOINT)
    adr, dof = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
    lo, hi = model.jnt_range[jnt]

    # mju_muscleGain wants exactly 9 parameters; MuJoCo stores 10 per actuator
    # (mjNGAIN), the last unused for muscles. Passing all 10 is a TypeError.
    prm = model.actuator_gainprm[mus][:9].copy()
    lengthrange = model.actuator_lengthrange[mus].copy()
    acc0 = float(model.actuator_acc0[mus])
    # gainprm layout for a muscle: range[0], range[1], force, scale, lmin,
    # lmax, vmax, fpmax, fvmax. `force` is peak isometric force in newtons and
    # `range` is the operating span in units of optimal fibre length.
    peak_force, op_lo, op_hi = float(prm[2]), float(prm[0]), float(prm[1])

    angles = np.linspace(lo, hi, n_points)
    sim_force, ref_force, lengths, moment = [], [], [], []
    active, passive = [], []
    eq_residual = 0.0
    for angle in angles:
        data.qpos[adr] = angle
        eq_residual = max(eq_residual, project_equality(model, data))
        data.qvel[:] = 0.0
        data.act[:] = 0.0
        data.act[int(model.actuator_actadr[mus])] = 1.0
        data.ctrl[:] = 0.0
        data.ctrl[mus] = 1.0
        mujoco.mj_forward(model, data)

        length = float(data.actuator_length[mus])
        gain = mujoco.mju_muscleGain(length, 0.0, lengthrange, acc0, prm)
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, model.actuator_biasprm[mus][:9])
        sim_force.append(float(data.actuator_force[mus]))
        ref_force.append(gain * 1.0 + bias)
        # Total force is the sum of two very different things. `gain` is the
        # contractile element, scaled by activation, and it is what a
        # controller commands. `bias` is passive elasticity, present at zero
        # activation, and it is what the controller has to fight. Keeping them
        # separate is the difference between a readable plot and a misleading
        # one: past the optimal length the passive term grows fast enough to
        # hide the active element's descending limb entirely.
        active.append(gain)
        passive.append(bias)
        lengths.append(length)
        moment.append(float(moment_arms(model, data)[mus, dof]))

    sim_force = np.array(sim_force)
    ref_force = np.array(ref_force)
    active = np.abs(np.array(active))
    passive = np.abs(np.array(passive))
    lengths = np.array(lengths)
    # Normalised length, the axis the Hill model is actually defined on: the
    # operating range [op_lo, op_hi] maps onto the tendon's lengthrange.
    span = (lengthrange[1] - lengthrange[0]) / (op_hi - op_lo)
    norm_length = op_lo + (lengths - lengthrange[0]) / span

    # Muscles pull, so force is negative throughout. Magnitude is the readable
    # quantity.
    magnitude = np.abs(sim_force)
    peak_idx = int(np.argmax(magnitude))

    # The joint's own limits do not expose the whole force-length curve, so
    # evaluate the closed form over the full modelled span [lmin, lmax] too.
    # This part is not reachable in simulation and is plotted dashed: it is
    # what the muscle would do if the knee could bend that far, not a
    # measurement of what it does.
    lmin, lmax = float(prm[4]), float(prm[5])
    ext_norm = np.linspace(lmin, lmax, 400)
    ext_len = lengthrange[0] + (ext_norm - op_lo) * span
    ext_active = np.abs(
        np.array([mujoco.mju_muscleGain(float(L), 0.0, lengthrange, acc0, prm) for L in ext_len])
    )
    ext_passive = np.abs(
        np.array(
            [
                mujoco.mju_muscleBias(float(L), lengthrange, acc0, model.actuator_biasprm[mus][:9])
                for L in ext_len
            ]
        )
    )
    ext_force = ext_active + ext_passive
    ext_peak = int(np.argmax(ext_active))

    return {
        "muscle": PROBE_MUSCLE,
        "joint": PROBE_JOINT,
        "peak_isometric_force_N": round(peak_force, 1),
        "max_measured_force_N": round(float(magnitude[peak_idx]), 1),
        "force_ratio_measured_over_peak": round(float(magnitude[peak_idx] / peak_force), 4),
        "normalised_length_at_peak": round(float(norm_length[peak_idx]), 4),
        "joint_angle_at_peak_rad": round(float(angles[peak_idx]), 4),
        "force_at_shortest_N": round(float(magnitude[0]), 1),
        "force_at_longest_N": round(float(magnitude[-1]), 1),
        # Force varies this much across the joint's own range at a fixed,
        # maximal command. On a geared motor this ratio would be 1.
        "force_range_ratio_over_joint": round(
            float(magnitude.max() / magnitude.min()), 3
        ),
        "reachable_norm_length": [
            round(float(norm_length.min()), 4),
            round(float(norm_length.max()), 4),
        ],
        "modelled_norm_length": [lmin, lmax],
        "descending_limb_reachable": bool(
            norm_length.max() > 1.0 + 1e-6 and peak_idx < len(magnitude) - 1
        ),
        # Where the *active* element peaks, evaluated over the full modelled
        # span. The Hill model defines this to be the optimal fibre length, so
        # anything other than 1.0 here means the length normalisation is wrong.
        "active_peak_norm_length": round(float(ext_norm[ext_peak]), 4),
        "active_peak_force_N": round(float(ext_active[ext_peak]), 1),
        "passive_force_at_optimum_N": round(
            float(np.interp(1.0, ext_norm, ext_passive)), 1
        ),
        # The external check.
        "max_abs_error_vs_closed_form_N": float(np.abs(sim_force - ref_force).max()),
        "pull_only": bool(np.all(sim_force <= 1e-9)),
        "max_equality_residual": eq_residual,
        "moment_arm_min_m": round(float(np.min(moment)), 5),
        "moment_arm_max_m": round(float(np.max(moment)), 5),
        # What a controller actually sees. Force varies with length and the
        # moment arm varies with angle, and the two multiply. This is the
        # number that would be a constant on a geared motor.
        "joint_torque_min_Nm": round(float(np.min(magnitude * np.abs(moment))), 3),
        "joint_torque_max_Nm": round(float(np.max(magnitude * np.abs(moment))), 3),
        "joint_torque_range_ratio": round(
            float(
                np.max(magnitude * np.abs(moment)) / max(np.min(magnitude * np.abs(moment)), 1e-9)
            ),
            2,
        ),
        "_curve": {
            "angle": angles.tolist(),
            "norm_length": norm_length.tolist(),
            "force": magnitude.tolist(),
            "active": active.tolist(),
            "passive": passive.tolist(),
            "moment": moment,
            "ext_norm_length": ext_norm.tolist(),
            "ext_force": ext_force.tolist(),
            "ext_active": ext_active.tolist(),
            "ext_passive": ext_passive.tolist(),
        },
    }


# ------------------------------------------- 3. activation dynamics


def _reference_activation(cmd_at, *, prm, t_end, dt_ref=2e-5) -> callable:
    """
    RK4 integration of `mju_muscleDynamics` at a step small enough to treat as
    exact, returned as a sampler over time.

    Nothing from the simulator enters here except the three parameters, which
    is what makes the comparison a check rather than a tautology.
    """
    n = int(round(t_end / dt_ref)) + 1
    ts = np.arange(n) * dt_ref
    out = np.empty(n)
    a = 0.0
    out[0] = a
    for i in range(1, n):
        c = cmd_at(ts[i - 1])
        k1 = mujoco.mju_muscleDynamics(c, a, prm)
        k2 = mujoco.mju_muscleDynamics(c, a + 0.5 * dt_ref * k1, prm)
        k3 = mujoco.mju_muscleDynamics(c, a + 0.5 * dt_ref * k2, prm)
        k4 = mujoco.mju_muscleDynamics(c, a + dt_ref * k3, prm)
        a += dt_ref / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = a
    return lambda t: np.interp(t, ts, out)


def _crossing(values, t, level, rising) -> float:
    """
    Time at which `values` first crosses `level`, linearly interpolated.

    Returns NaN when the level is not bracketed by the samples, including the
    case where the very first sample is already past it. That case is the one
    that matters here: if the integrator jumps over the 10% level in a single
    step, the crossing time is not measured but assumed, and returning t[0]
    would quietly report the step size as a rise time.
    """
    mask = values >= level if rising else values <= level
    if not mask.any() or mask[0]:
        return float("nan")
    idx = int(np.argmax(mask))
    v0, v1, t0, t1 = values[idx - 1], values[idx], t[idx - 1], t[idx]
    if v1 == v0:
        return float(t1)
    return float(t0 + (level - v0) * (t1 - t0) / (v1 - v0))


def _nan_to_none(x):
    """JSON has no NaN. An unmeasurable quantity is null, not a number."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def measure_activation_dynamics(*, hold_s=0.25, timesteps=(2e-3, 1e-3, 5e-4, 2e-4, 1e-4)) -> dict:
    """
    Step the command up then down, and measure how well the simulator tracks it
    as a function of integration timestep.

    Two things come out of this and the second was not the plan.

    The intended measurement is the asymmetry: MuJoCo's muscle dynamics are not
    a plain first-order lag, because the time constant depends on activation
    and on the sign of (ctrl - act). Deactivation is slower than activation, so
    rise and fall are measured separately rather than assumed symmetric.

    The measurement that fell out is an integration error. Activation is
    advanced by explicit Euler at the model timestep, and MS-Human-700 ships
    with `timestep = 0.002`. The fastest activation time constant in the model
    is `tau_act * 0.5 = 5 ms`, so the default step is 40% of the time constant
    of the fastest state in the system. Sweeping the timestep and comparing
    against RK4 at 0.02 ms quantifies what that costs.

    The first pass at this measurement reported NaN for both rise and fall
    times, which is how the problem surfaced: at dt = 2 ms the very first step
    takes activation from 0 to about 0.4, straight past the 10% level, so a
    10-90% rise time is not resolvable at all at the shipped timestep.
    """
    results = []
    trace = None

    for dt in timesteps:
        model, data = load("full", with_floor=False)
        model.opt.timestep = dt
        names = actuator_names(model)
        mus = names.index(PROBE_MUSCLE)
        act_adr = int(model.actuator_actadr[mus])
        # mju_muscleDynamics wants 3 parameters (tau_act, tau_deact, smoothing
        # width); actuator_dynprm carries mjNDYN = 10.
        prm = model.actuator_dynprm[mus][:3].copy()
        tau_act, tau_deact = float(prm[0]), float(prm[1])
        n = int(round(hold_s / dt))

        data.act[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        times, sim_act, cmd = [], [], []
        for phase_ctrl in (1.0, 0.0):
            data.ctrl[mus] = phase_ctrl
            for _ in range(n):
                mujoco.mj_step(model, data)
                times.append(float(data.time))
                sim_act.append(float(data.act[act_adr]))
                cmd.append(phase_ctrl)
        times, sim_act, cmd = np.array(times), np.array(sim_act), np.array(cmd)

        reference = _reference_activation(
            lambda t: 1.0 if t < hold_s else 0.0, prm=prm, t_end=2 * hold_s
        )
        ref = reference(times)

        rise, fall = sim_act[:n], sim_act[n:]
        t_rise = times[:n] - times[0] + dt
        t_fall = times[n:] - times[n] + dt
        rise_10_90 = _crossing(rise, t_rise, 0.9, True) - _crossing(rise, t_rise, 0.1, True)
        fall_90_10 = _crossing(fall, t_fall, 0.1, False) - _crossing(fall, t_fall, 0.9, False)

        results.append(
            {
                "timestep_ms": round(dt * 1e3, 3),
                "max_abs_error_vs_rk4": round(float(np.abs(sim_act - ref).max()), 6),
                "rise_10_90_ms": _nan_to_none(round(rise_10_90 * 1e3, 3)),
                "fall_90_10_ms": _nan_to_none(round(fall_90_10 * 1e3, 3)),
                "measured_ratio": _nan_to_none(round(fall_90_10 / rise_10_90, 3)),
                # Null rise time means the first step already cleared the 10%
                # level, so the window was never bracketed and the rise is not
                # measurable at this step size.
                "rise_bracketed": bool(not np.isnan(rise_10_90)),
            }
        )
        if dt == timesteps[0]:
            trace = {
                "t": times.tolist(),
                "act": sim_act.tolist(),
                "ref": ref.tolist(),
                "cmd": cmd.tolist(),
            }
        finest = {"t": times, "act": sim_act, "ref": ref}

    trace["fine_t"] = finest["t"].tolist()
    trace["fine_act"] = finest["act"].tolist()

    shipped = results[0]
    converged = results[-1]
    return {
        "tau_act_s": tau_act,
        "tau_deact_s": tau_deact,
        "tau_ratio": round(tau_deact / tau_act, 3),
        "shipped_timestep_ms": shipped["timestep_ms"],
        "error_at_shipped_timestep": shipped["max_abs_error_vs_rk4"],
        "error_at_finest_timestep": converged["max_abs_error_vs_rk4"],
        "rise_10_90_ms": converged["rise_10_90_ms"],
        "fall_90_10_ms": converged["fall_90_10_ms"],
        "measured_ratio": converged["measured_ratio"],
        "rise_resolvable_at_shipped_timestep": shipped["rise_bracketed"],
        "finest_timestep_ms": converged["timestep_ms"],
        "sweep": results,
        "_trace": trace,
    }


# ------------------------------------------------------------- figure


def make_figure(redundancy: dict, fl: dict, dyn: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # Panel 1. A histogram would be the obvious choice and would be a bar of
    # height 24 at a single value: rank is the same at every pose sampled.
    # What varies, and what a reader needs, is *which* directions are missing.
    ax = axes[0]
    top = redundancy["top_null_space_dofs"][:12][::-1]
    ax.barh(
        [d["dof"] for d in top],
        [d["weight"] for d in top],
        color=PALETTE[0],
        alpha=0.9,
    )
    ax.set_xlabel("mean weight in unactuated subspace")
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(
        f"{redundancy['nu']} muscles, {redundancy['nv']} DoF, rank "
        f"{redundancy['rank_median']}"
    )

    ax = axes[1]
    curve = fl["_curve"]
    ax.plot(
        curve["ext_norm_length"],
        curve["ext_active"],
        color=PALETTE[0],
        lw=1.3,
        ls="--",
        label="active (contractile)",
    )
    ax.plot(
        curve["ext_norm_length"],
        curve["ext_passive"],
        color=PALETTE[4],
        lw=1.3,
        ls="--",
        label="passive (elastic)",
    )
    ax.plot(
        curve["norm_length"],
        curve["force"],
        color=PALETTE[1],
        lw=2.4,
        label="total, reachable",
    )
    ax.axvline(1.0, color=PALETTE[2], lw=1.1, ls=":", label="optimal fibre length")
    ax.set_xlabel("normalised muscle length  $L/L_0$")
    ax.set_ylabel("|force| (N)")
    ax.set_title(f"{fl['muscle']} at full activation")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    trace = dyn["_trace"]
    ax.plot(np.array(trace["t"]) * 1e3, trace["cmd"], color=PALETTE[3], lw=1.1, label="command")
    ax.plot(
        np.array(trace["fine_t"]) * 1e3,
        trace["fine_act"],
        color=PALETTE[0],
        lw=2.0,
        label=f"activation, dt = {dyn['sweep'][-1]['timestep_ms']:g} ms",
    )
    ax.plot(
        np.array(trace["t"]) * 1e3,
        trace["act"],
        color=PALETTE[1],
        lw=1.5,
        ls="--",
        label=f"dt = {dyn['shipped_timestep_ms']:g} ms (shipped)",
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("activation")
    ax.set_title(
        f"rise {dyn['rise_10_90_ms']:.1f} ms, fall {dyn['fall_90_10_ms']:.1f} ms"
    )
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------- main


def _relative(path: Path) -> Path:
    """
    Path relative to the repository root, for display.

    `RunLogger.root` defaults to a *relative* `Path("runs")` while REPO_ROOT is
    absolute, so a bare `relative_to` raises after every otherwise-successful
    run. Resolve both sides first, and fall back to the path as given rather
    than losing a completed run's output to a formatting error.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)

    model, data = load("full", with_floor=False)
    structure = describe(model, data)

    config = {
        "model": "ms_human_700/MS-Human-700.xml",
        "variant": "full",
        "n_poses": N_POSES,
        "probe_muscle": PROBE_MUSCLE,
        "probe_joint": PROBE_JOINT,
    }

    with RunLogger(
        "s9_muscle",
        config=config,
        seed=SEED,
        extra_manifest={"model_hash": hash_mj_model(model), "structure": structure},
    ) as log:
        redundancy = measure_redundancy(model, data, rng=rng)
        log.event("redundancy", **{k: v for k, v in redundancy.items() if k != "ranks"})

        fl = measure_force_length(model, data)
        log.event("force_length", **{k: v for k, v in fl.items() if not k.startswith("_")})

        dyn = measure_activation_dynamics()
        log.event("activation_dynamics", **{k: v for k, v in dyn.items() if not k.startswith("_")})

        figure = make_figure(redundancy, fl, dyn, MEDIA_DIR / "muscle_characterisation.png")
        log.event("figure", path=str(_relative(figure)))
        run_dir = log.path

    summary = {
        "structure": structure,
        "redundancy": {k: v for k, v in redundancy.items() if k != "ranks"},
        "force_length": {k: v for k, v in fl.items() if not k.startswith("_")},
        "activation_dynamics": {k: v for k, v in dyn.items() if not k.startswith("_")},
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nrun log:  {_relative(run_dir)}")
    print(f"figure:   {_relative(figure)}")


if __name__ == "__main__":
    main()
