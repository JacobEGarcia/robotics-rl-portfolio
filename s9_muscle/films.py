"""
S9 study films: one short video per control study.

`film.py` animates the plant characterisation. This animates the six studies
built on top of it. Each clip is driven by the same code that produces the
published numbers, not by a separate scripted animation, so the on-screen
readouts are measurements rather than captions.

    film_synergies.mp4    the 5 extracted synergies, one at a time, then a
                          stance cycle reconstructed from them
    film_redundancy.mp4   one stance problem solved at p = 1, 2, 3, 5, 10, with
                          recruitment visibly spreading as p rises
    film_impedance.mp4    the stiffness sign flipping across the knee's range,
                          then two perturbation tests: one joint returns, one
                          buckles
    film_transfer.mp4     a walking-cadence stance cycle commanded naively
                          against the same cycle with the activation inverse
    film_failure.mp4      muscles dying one at a time, with torque authority
                          and task feasibility falling as they go

Run
---
    python -m s9_muscle.films                 # all five, 1280x720
    python -m s9_muscle.films --only impedance
    python -m s9_muscle.films --draft         # 640x360, for composition checks
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from s9_muscle.control import (
    LEG_JOINTS,
    activation_rate,
    dof_indices,
    invert_activation,
    leg_plant,
    project_equality,
    solve_activation,
)
from s9_muscle.failure import mean_width, unit_directions
from s9_muscle.film import FPS, Overlay, body_camera
from s9_muscle.scene import (
    MEDIA_DIR,
    _save,
    actuator_names,
    colour_tendons,
    load,
    moment_arms,
)
from s9_muscle.synergies import build_dataset, nmf

SEED = 0
OPT = None          # set in main(); MjvOption needs mujoco imported first
DEAD_RGBA = (0.30, 0.31, 0.33, 1.0)     # limp muscle: visibly present, visibly off

# A stance posture used by several clips. Mid-range, feasible, and the same
# shape as the postures the studies sample.
POSTURE = {
    "hip_flexion_r": 0.45,
    "hip_adduction_r": 0.0,
    "hip_rotation_r": 0.0,
    "knee_angle_r": 0.85,
    "ankle_angle_r": -0.05,
}


def view_option():
    """
    Scene options shared by every clip.

    Geom group 1 is 14 translucent "skin" sleeves over the limbs, alpha 0.2.
    They look right in a whole-body shot and destroy a close one: the leg's
    tendons are behind them, and 0.2 alpha over a dense bundle washes the whole
    limb to a pale grey. Hidden here, kept in `film.py` where the shots are
    wide enough for them to help.
    """
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0
    return opt


# Idle and fully-active tendon colours. Deliberately further apart than the
# ramp in scene.py: these clips have to communicate a *level* at a glance, and
# the gamma lifts mid activations that would otherwise read as muddy violet.
IDLE_RGB = np.array([0.11, 0.17, 0.33])
HOT_RGB = np.array([1.00, 0.32, 0.06])
COLOUR_GAMMA = 0.50


# ------------------------------------------------------------- helpers


def set_posture(model, data, posture: dict) -> None:
    for name, value in posture.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = value
    data.qvel[:] = 0.0
    project_equality(model, data)


def stance_torque(model, data, *, grf_bw=0.9) -> np.ndarray:
    """Torque required to carry `grf_bw` body weights through the heel."""
    dofs = dof_indices(model, LEG_JOINTS)
    heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    point = np.array(data.xpos[heel], dtype=float)
    point[2] = 0.0
    mujoco.mj_jac(model, data, jacp, jacr, point, heel)
    force = np.array([0.0, 0.0, grf_bw * float(model.body_mass.sum()) * 9.81])
    return -jacp[:, dofs].T @ force + np.array(data.qfrc_bias)[dofs]


def paint(model, muscles: np.ndarray, values: np.ndarray, *, dead=None) -> None:
    """Colour the leg by activation, everything else idle, dead muscles grey."""
    level = np.full(model.nu, 0.0)
    level[muscles] = np.clip(values, 0.0, 1.0)
    t = np.power(level, COLOUR_GAMMA)[:, None]
    rgb = IDLE_RGB[None, :] * (1.0 - t) + HOT_RGB[None, :] * t
    tendon = model.actuator_trnid[:, 0]
    model.tendon_rgba[tendon, :3] = rgb
    model.tendon_rgba[tendon, 3] = 1.0
    if dead is not None and np.any(dead):
        model.tendon_rgba[model.actuator_trnid[muscles[dead], 0]] = DEAD_RGBA


def leg_camera(model, data, *, azimuth=140.0, distance=1.75, lift=-0.12):
    """
    Framed on the mid-thigh, not the femur's origin.

    `body_camera` looks at a body's frame origin, and the femur's origin is at
    the hip, so a positive lift walks the shot up into the torso. These clips
    are about what the leg is doing, so the lookat drops instead.
    """
    return body_camera(model, data, "femur_r", azimuth=azimuth,
                       distance=distance, elevation=-4.0, lift=lift)


def emphasise(model, muscles: np.ndarray, factor=3.0) -> np.ndarray:
    """
    Widen the leg's tendons so a 50-muscle subset reads at whole-body scale.

    Rendering only; no dynamics quantity depends on `tendon_width`. Returns the
    original array so a caller can restore it.
    """
    previous = model.tendon_width.copy()
    model.tendon_width[model.actuator_trnid[muscles, 0]] *= factor
    return previous


def write(frames, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.mp4"
    imageio.mimsave(path, frames, fps=FPS, quality=8, macro_block_size=1)
    _save(out_dir / f"{name}_poster.png", frames[len(frames) // 3])
    return path


# ------------------------------------------------------------- A. synergies


def render_synergies(w, h, overlay, *, rng) -> list[np.ndarray]:
    model, data = load("full")
    dataset = build_dataset(model, data, rng=rng, n=180)
    W = dataset["act"]
    H, S = nmf(W, 5, rng=np.random.default_rng(SEED + 5))
    labels = dataset["labels"]

    plant = leg_plant(model, data)
    muscles = plant.muscles
    previous_width = emphasise(model, muscles)
    frames: list[np.ndarray] = []

    # One synergy at a time. Each is a fixed pattern over the 50 muscles that
    # the controller would scale up and down; showing them individually is the
    # only way to see that they are anatomically coherent groups rather than
    # arbitrary vectors.
    order = np.argsort(-H.sum(axis=0))          # busiest synergy first
    set_posture(model, data, POSTURE)
    with mujoco.Renderer(model, height=h, width=w) as r:
        for rank, j in enumerate(order):
            weights = S[j] / max(S[j].max(), 1e-12)
            top = np.argsort(-S[j])[:3]
            paint(model, muscles, 0.05 + 0.90 * weights)
            for i in range(int(2.2 * FPS)):
                data.act[:] = 0.02
                data.act[muscles] = 0.05 + 0.90 * weights
                mujoco.mj_forward(model, data)
                cam = leg_camera(model, data, azimuth=128.0 + 24.0 * i / (2.2 * FPS))
                r.update_scene(data, camera=cam, scene_option=OPT)
                frames.append(overlay.draw(
                    r.render(),
                    title=f"Synergy {rank + 1} of 5",
                    subtitle="A fixed pattern over 50 muscles. Five of these cover 90% of "
                             "the activation variance in 387 stance problems",
                    hud=[f"share       {100 * H[:, j].sum() / H.sum():4.1f}% of total drive"]
                        + [f"{labels[t]:<12s}{S[j][t] / max(S[j].max(), 1e-12):5.2f}" for t in top],
                ))

        # Reconstruct the dataset from rank 5 and play through it, ordered by
        # how hard the leg is working, so the clip reads as one loading cycle.
        recon = (H @ S)
        seq = np.argsort(np.linalg.norm(dataset["tau"], axis=1))
        seq = np.concatenate([seq, seq[::-1]])[::max(1, len(seq) // (3 * FPS))]
        for k in seq:
            tau_hat = dataset["plant_A"][k] @ recon[k] + dataset["plant_c"][k]
            err = np.linalg.norm(tau_hat - dataset["tau"][k])
            demand = np.linalg.norm(dataset["tau"][k] - dataset["plant_c"][k])
            paint(model, muscles, recon[k])
            data.act[:] = 0.02
            data.act[muscles] = np.clip(recon[k], 0.0, 1.0)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=leg_camera(model, data), scene_option=OPT)
            frames.append(overlay.draw(
                r.render(),
                title="Rebuilt from those five patterns",
                subtitle="90% of the activation variance, and 70% of the torque. "
                         "Explaining activation is not the same as delivering force",
                hud=[
                    f"demand      {demand:6.1f} N·m",
                    f"delivered   {np.linalg.norm(tau_hat - dataset['plant_c'][k]):6.1f} N·m",
                    f"error       {err:6.1f} N·m  ({100 * err / demand:4.1f}%)",
                    f"rank        5 of 50",
                ],
            ))
    model.tendon_width[:] = previous_width
    return frames


# ----------------------------------------------------------- B. redundancy


def render_redundancy(w, h, overlay, *, rng) -> list[np.ndarray]:
    model, data = load("full")
    set_posture(model, data, POSTURE)
    plant = leg_plant(model, data)
    tau = stance_torque(model, data)
    muscles = plant.muscles
    labels = actuator_names(model)
    previous_width = emphasise(model, muscles)

    frames = []
    with mujoco.Renderer(model, height=h, width=w) as r:
        for p in (1.0, 2.0, 3.0, 5.0, 10.0):
            sol = solve_activation(plant, tau, p=p)
            act = sol["act"]
            note = {1.0: "a linear program: fewest muscles, each saturated",
                    2.0: "least norm: load spread over everything with a moment arm",
                    3.0: "Crowninshield and Brand, reported to match EMG best",
                    5.0: "approaching min-max",
                    10.0: "min-max: nobody works harder than they must"}[p]
            paint(model, muscles, act)
            data.act[:] = 0.02
            data.act[muscles] = act
            mujoco.mj_forward(model, data)
            for i in range(int(2.3 * FPS)):
                cam = leg_camera(model, data, azimuth=128.0 + 24.0 * i / (2.3 * FPS))
                r.update_scene(data, camera=cam, scene_option=OPT)
                frames.append(overlay.draw(
                    r.render(),
                    title=f"Same torque, minimising  sum(a^{p:g})",
                    subtitle=f"Identical stance problem every time. {note}",
                    hud=[
                        f"exponent    p = {p:g}",
                        f"recruited   {int(np.count_nonzero(act > 0.01)):2d} of 50 muscles",
                        f"peak        {act.max():.3f}",
                        f"total       {act.sum():.2f}",
                        f"hardest     {labels[muscles[int(np.argmax(act))]]}",
                    ],
                ))
    model.tendon_width[:] = previous_width
    return frames


# ------------------------------------------------------------ C. impedance


def _knee_stiffness(model, data, angle, level, dof, adr, delta=0.01):
    data.qpos[adr] = angle
    project_equality(model, data)
    spanning = np.abs(moment_arms(model, data)[:, dof]) > 1e-9
    data.act[:] = 0.0
    data.act[spanning] = level
    out = []
    for q in (angle + delta, angle - delta):
        data.qpos[adr] = q
        project_equality(model, data)
        out.append(float(data.qfrc_actuator[dof] + data.qfrc_passive[dof]))
    data.qpos[adr] = angle
    project_equality(model, data)
    return -(out[0] - out[1]) / (2.0 * delta), spanning


def _perturbation(model, data, angle, dof, adr, spanning, *, level=1.0,
                  seconds=1.4, dt=1e-3, kick=0.06):
    """
    One-degree-of-freedom release test at a held equilibrium.

    The rest of the body is held fixed and only the test joint moves, driven by
    the muscle torque it feels and its own effective inertia. An external load
    equal to the muscle torque at `angle` holds the joint there, so the joint
    starts in equilibrium and the only thing acting after the kick is the
    *change* in muscle torque, which is exactly what stiffness means.

    This is a reduced model, not a whole-body rollout, and it is labelled as
    one on screen. A free-standing body would collapse for reasons that have
    nothing to do with the joint under test.
    """
    data.qpos[adr] = angle
    project_equality(model, data)
    data.act[:] = 0.0
    data.act[spanning] = level
    mujoco.mj_forward(model, data)
    hold = float(data.qfrc_actuator[dof] + data.qfrc_passive[dof])
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    inertia = float(full[dof, dof])

    q, qd = angle + kick, 0.0
    trace = []
    for _ in range(int(seconds / dt)):
        data.qpos[adr] = q
        project_equality(model, data)
        data.act[:] = 0.0
        data.act[spanning] = level
        mujoco.mj_forward(model, data)
        tau = float(data.qfrc_actuator[dof] + data.qfrc_passive[dof]) - hold
        qd += dt * tau / inertia
        qd *= 0.995                     # light damping, so a stable joint settles
        q += dt * qd
        q = float(np.clip(q, model.jnt_range[model.dof_jntid[dof]][0] + 0.02,
                          model.jnt_range[model.dof_jntid[dof]][1] - 0.02))
        trace.append((q, qd))
    return trace, inertia


def render_impedance(w, h, overlay, *, rng) -> list[np.ndarray]:
    model, data = load("full")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    set_posture(model, data, POSTURE)
    plant = leg_plant(model, data)
    previous_width = emphasise(model, plant.muscles)
    frames = []

    # Phase 1: sweep the knee, colour the whole leg by the sign of K.
    lo, hi = model.jnt_range[jid]
    angles = np.linspace(lo + 0.1, hi - 0.1, int(3.4 * FPS))
    angles = np.concatenate([angles, angles[::-1]])
    with mujoco.Renderer(model, height=h, width=w) as r:
        for angle in angles:
            K, spanning = _knee_stiffness(model, data, angle, 1.0, dof, adr)
            shade = float(np.clip(abs(K) / 90.0, 0.15, 1.0))
            colour_tendons(model, np.full(model.nu, 0.02))
            tint = np.full(model.nu, 0.02)
            tint[spanning] = shade if K < 0 else 0.02
            colour_tendons(model, tint)
            if K > 0:
                # Stabilising: paint it in the cool end explicitly rather than
                # leaving it at the idle value, so "positive" is a colour and
                # not an absence of colour.
                model.tendon_rgba[model.actuator_trnid[spanning, 0]] = (
                    0.20, 0.45 + 0.45 * shade, 0.85, 1.0)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=leg_camera(model, data), scene_option=OPT)
            frames.append(overlay.draw(
                r.render(),
                title="Co-contraction stiffness changes sign with pose",
                subtitle="Every muscle crossing the knee at full activation. "
                         "Blue restores the joint, red drives it further away",
                hud=[
                    f"knee        {angle:+.2f} rad",
                    f"K           {K:+7.1f} N·m/rad",
                    f"sign        {'RESTORING' if K > 0 else 'DESTABILISING'}",
                    f"muscles     {int(spanning.sum())} spanning",
                ],
            ))

    # Phase 2: release tests at one angle of each sign, side by side.
    stable = 1.20
    unstable = 0.40
    half = w // 2
    panels = []
    for angle, tag in ((stable, "K > 0"), (unstable, "K < 0")):
        K, spanning = _knee_stiffness(model, data, angle, 1.0, dof, adr)
        trace, inertia = _perturbation(model, data, angle, dof, adr, spanning)
        shots = []
        step = max(1, len(trace) // int(2.6 * FPS))
        with mujoco.Renderer(model, height=h, width=half) as r:
            for q, qd in trace[::step]:
                data.qpos[adr] = q
                project_equality(model, data)
                data.act[:] = 0.0
                data.act[spanning] = 1.0
                mujoco.mj_forward(model, data)
                model.tendon_rgba[model.actuator_trnid[spanning, 0]] = (
                    (0.20, 0.65, 0.85, 1.0) if K > 0 else (0.92, 0.24, 0.20, 1.0))
                r.update_scene(data, camera=leg_camera(model, data), scene_option=OPT)
                shots.append((r.render(), q))
        panels.append((shots, tag, angle, K))

    (left, ltag, lang, lK), (right, rtag, rang, rK) = panels
    for i in range(min(len(left), len(right))):
        canvas = np.concatenate([left[i][0], right[i][0]], axis=1)
        frames.append(overlay.draw(
            canvas,
            title="Same command, opposite outcome",
            subtitle="One degree of freedom released from a held equilibrium after a "
                     "0.06 rad nudge. Rest of the body fixed",
            hud=[
                f"left        knee {lang:.2f} rad   K {lK:+6.1f}   now {left[i][1]:+.3f}",
                f"right       knee {rang:.2f} rad   K {rK:+6.1f}   now {right[i][1]:+.3f}",
                f"drift       {abs(left[i][1] - lang):.3f}  vs  {abs(right[i][1] - rang):.3f} rad",
                f"co-contraction  1.00 on both",
            ],
        ))
    model.tendon_width[:] = previous_width
    return frames


# ------------------------------------------------------------- E. transfer


def render_transfer(w, h, overlay, *, rng) -> list[np.ndarray]:
    from s9_muscle.transfer import CONTROL_HZ, POSTURE_AMP, POSTURE_MID, build_reference

    model, data = load("full")
    prm = model.actuator_dynprm[0][:3].copy()
    freq = 1.0
    n_control = int(round(CONTROL_HZ / freq))
    ref = build_reference(model, data, freq, n_control=n_control)
    muscles = ref["muscles"]
    target = ref["act"]

    # Integrate both commands once, then render the stored activation traces.
    dt = 1.0 / CONTROL_HZ
    runs = {}
    for mode in ("naive", "compensated"):
        act = target[0].copy()
        history = np.empty_like(target)
        for k in range(n_control):
            if mode == "compensated":
                nxt = target[min(k + 1, n_control - 1)]
                rate = (nxt - target[k]) * CONTROL_HZ
                cmd = np.clip(
                    [invert_activation(a, rr, prm) for a, rr in zip(target[k], rate)], 0.0, 1.0)
            else:
                cmd = target[k]
            for _ in range(4):
                sub = dt / 4.0
                rates = np.array([activation_rate(u, a, prm) for u, a in zip(cmd, act)])
                act = act + sub * rates
            history[k] = act
        runs[mode] = history

    half = w // 2
    previous_width = emphasise(model, muscles)
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in POSTURE_MID}
    step = max(1, n_control // int(5.0 * FPS))
    idx = list(range(0, n_control, step))

    panels = {}
    for mode in ("naive", "compensated"):
        shots = []
        with mujoco.Renderer(model, height=h, width=half) as r:
            for k in idx:
                phase = 2.0 * np.pi * freq * k / CONTROL_HZ
                for name in POSTURE_MID:
                    data.qpos[qadr[name]] = POSTURE_MID[name] + POSTURE_AMP[name] * np.sin(phase)
                data.qvel[:] = 0.0
                project_equality(model, data)
                data.act[:] = 0.02
                data.act[muscles] = np.clip(runs[mode][k], 0.0, 1.0)
                mujoco.mj_forward(model, data)
                model.tendon_rgba[:] = model.tendon_rgba
                paint(model, muscles, np.clip(runs[mode][k], 0.0, 1.0))
                r.update_scene(data, camera=leg_camera(model, data), scene_option=OPT)
                delivered = ref["A"][k] @ runs[mode][k] + ref["c"][k]
                err = float(np.linalg.norm(delivered - ref["tau"][k]))
                demand = float(np.linalg.norm(ref["tau"][k] - ref["c"][k]))
                shots.append((r.render(), err, demand))
        panels[mode] = shots

    frames = []
    for i in range(len(idx)):
        canvas = np.concatenate([panels["naive"][i][0], panels["compensated"][i][0]], axis=1)
        n_err, demand = panels["naive"][i][1], panels["naive"][i][2]
        c_err = panels["compensated"][i][1]
        frames.append(overlay.draw(
            canvas,
            title="A torque-model command, and the same command pre-inverted",
            subtitle="One stance cycle at walking cadence. Left commands the activation the "
                     "torque model asks for; right inverts the actuator dynamics first",
            hud=[
                f"phase       {100 * i / len(idx):5.1f}%   of a 1 Hz cycle",
                f"demand      {demand:6.1f} N·m",
                f"naive       {n_err:6.1f} N·m error  ({100 * n_err / max(demand, 1e-9):4.1f}%)",
                f"compensated {c_err:6.1f} N·m error  ({100 * c_err / max(demand, 1e-9):4.1f}%)",
            ],
        ))
    model.tendon_width[:] = previous_width
    return frames


# -------------------------------------------------------------- F. failure


def render_failure(w, h, overlay, *, rng) -> list[np.ndarray]:
    model, data = load("full")
    dataset = build_dataset(model, data, rng=rng, n=40)
    set_posture(model, data, POSTURE)
    plant = leg_plant(model, data)
    muscles = plant.muscles
    labels = actuator_names(model)
    directions = unit_directions(plant.A.shape[0], 240, rng=rng)
    full_width = mean_width(plant, directions)
    previous_width = emphasise(model, muscles)

    tau = stance_torque(model, data)
    nominal = solve_activation(plant, tau, p=3.0)["act"]

    order = rng.permutation(plant.n)
    scratch = leg_plant(model, data)
    frames = []
    hold = max(1, int(0.30 * FPS))

    with mujoco.Renderer(model, height=h, width=w) as r:
        for k in range(0, 31):
            dead = np.zeros(plant.n, dtype=bool)
            dead[order[:k]] = True
            width = mean_width(plant, directions, dead) / full_width

            solved = 0
            for i in range(dataset["n_kept"]):
                scratch.A[:] = np.where(dead[None, :], 0.0, dataset["plant_A"][i])
                scratch.c[:] = dataset["plant_c"][i]
                res = solve_activation(scratch, dataset["tau"][i], p=1.0)
                solved += int(res["success"] and res["residual_Nm"] < 1e-6)
            feasible = solved / dataset["n_kept"]

            scratch.A[:] = np.where(dead[None, :], 0.0, plant.A)
            scratch.c[:] = plant.c
            still = solve_activation(scratch, tau, p=3.0)
            holds = still["success"] and still["residual_Nm"] < 1e-6
            shown = np.where(dead, 0.0, still["act"] if holds else nominal)

            paint(model, muscles, shown, dead=dead)
            data.act[:] = 0.02
            data.act[muscles] = shown
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=leg_camera(model, data), scene_option=OPT)
            frame = overlay.draw(
                r.render(),
                title="Losing muscles, one at a time",
                subtitle="Grey is limp: no active force, passive elasticity unchanged. "
                         "Feasibility is over 40 held-out stance problems",
                hud=[
                    f"dead        {k:2d} of 50" + (f"   ({labels[muscles[order[k - 1]]]})" if k else ""),
                    f"authority   {100 * width:5.1f}%  of torque set",
                    f"feasible    {100 * feasible:5.1f}%  of stance tasks",
                    f"this pose   {'still held' if holds else 'CANNOT BE HELD'}",
                ],
            )
            frames.extend([frame] * hold)
    model.tendon_width[:] = previous_width
    return frames


# ------------------------------------------------------------------ main

FILMS = {
    "synergies": render_synergies,
    "redundancy": render_redundancy,
    "impedance": render_impedance,
    "transfer": render_transfer,
    "failure": render_failure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the S9 study films.")
    parser.add_argument("--only", choices=sorted(FILMS), help="render one film")
    parser.add_argument("--draft", action="store_true", help="640x360")
    args = parser.parse_args()

    global OPT
    OPT = view_option()
    w, h = (640, 360) if args.draft else (1280, 720)
    overlay = Overlay(w, h)
    names = [args.only] if args.only else list(FILMS)

    for name in names:
        start = time.time()
        frames = FILMS[name](w, h, overlay, rng=np.random.default_rng(SEED))
        suffix = "_draft" if args.draft else ""
        path = write(frames, f"film_{name}{suffix}", MEDIA_DIR)
        print(f"{name:12s} {len(frames):4d} frames  {len(frames) / FPS:5.1f} s  "
              f"rendered in {time.time() - start:5.1f} s  -> {path.name}")


if __name__ == "__main__":
    main()
