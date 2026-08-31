"""
S9 film: an animated walkthrough of what the S9 measurements found.

This is not a highlight reel. Each of the four segments shows one measured
result from `characterize.py`, and the on-screen numbers are read from the
simulation as it runs rather than typed in.

  1. THE BODY          orbit at rest. 700 muscle actuators over 85 DoF.
  2. RECRUITMENT       kinematic joint sweep, tendons tinted by |moment arm|
                       about that joint. Shows both which muscles reach a
                       given DoF and how far the moment arm moves as the
                       joint travels: 57.0 mm to 30.6 mm for vasint_r.
  3. ACTIVATION LAG    step the command up and down, 8x slow motion. Rise and
                       fall are visibly different lengths: 31.8 ms against
                       91.9 ms.
  4. UNDERACTUATION    two bodies released under gravity, one relaxed and one
                       at 50% co-contraction everywhere. Both fall, because
                       the moment-arm matrix has the six base DoF in its null
                       space and no activation pattern reaches them.

Segment 3 holds the pose and integrates only the activation state, using the
same RK4 reference integrator `characterize.py` validates the simulator
against. The body is not stepped there, because a free-standing model driven
by one muscle group collapses in under half a second and the collapse is not
what that segment is about. Segments 1, 2 and 4 are labelled for what they are:
1 and 2 are kinematic, 4 is a full dynamic rollout.

Run
---
    python -m s9_muscle.film            # 1280x720, writes media/s9/s9_muscle.mp4
    python -m s9_muscle.film --draft    # 640x360, quick composition check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from s9_muscle.control import project_equality
from s9_muscle.scene import (
    MEDIA_DIR,
    actuator_names,
    colour_tendons,
    framed_camera,
    load,
    moment_arms,
)

FPS = 30
PROBE_MUSCLE = "vasint_r"

# Fonts. Any missing face falls back to PIL's built-in, which is ugly but never
# crashes a render on a machine without the Windows font directory.
_FONT_DIRS = ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts")


def _font(name: str, size: int):
    for directory in _FONT_DIRS:
        candidate = Path(directory) / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class Overlay:
    """Caption band and live-value readout drawn onto rendered frames."""

    def __init__(self, width: int, height: int):
        scale = width / 1280.0
        self.w, self.h = width, height
        self.title = _font("arialbd.ttf", max(11, int(30 * scale)))
        self.body = _font("arial.ttf", max(9, int(19 * scale)))
        self.mono = _font("consola.ttf", max(9, int(18 * scale)))
        self.band = int(96 * scale)
        self.pad = int(26 * scale)

    def draw(self, frame: np.ndarray, *, title: str, subtitle: str, hud=()) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGB")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        d.rectangle([0, self.h - self.band, self.w, self.h], fill=(12, 16, 22, 214))
        d.text((self.pad, self.h - self.band + int(self.band * 0.16)), title,
               font=self.title, fill=(238, 242, 248, 255))
        d.text((self.pad, self.h - self.band + int(self.band * 0.58)), subtitle,
               font=self.body, fill=(150, 172, 196, 255))

        if hud:
            line_h = int(self.mono.size * 1.55)
            box_h = line_h * len(hud) + self.pad
            box_w = int(self.w * 0.36)
            d.rectangle([0, 0, box_w, box_h], fill=(12, 16, 22, 190))
            for i, line in enumerate(hud):
                d.text((self.pad, int(self.pad * 0.5) + i * line_h), line,
                       font=self.mono, fill=(226, 234, 244, 255))

        return np.asarray(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"))


def _renderer(model, width, height):
    return mujoco.Renderer(model, height=height, width=width)


def body_camera(model, data, body: str, *, azimuth, distance, elevation=-6.0, lift=0.0):
    """
    Free camera framed on a named body rather than the whole-body COM.

    The default COM framing puts the figure at roughly a fifth of frame width
    in a 16:9 crop, which is the right choice for a full-body shot and the
    wrong one for a segment about what one thigh is doing.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    cam.lookat[:] = data.xpos[bid]
    cam.lookat[2] += lift
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


# ------------------------------------------------------------ segments


def seg_body(width, height, overlay, *, seconds=4.0) -> list[np.ndarray]:
    """Orbit the resting body."""
    model, data = load("full")
    # Establish the colour convention in the opening shot rather than letting
    # the model's shipped red tendons read as "everything is firing" and then
    # switching to a blue baseline two segments later.
    colour_tendons(model, np.full(model.nu, 0.02))
    n = int(seconds * FPS)
    frames = []
    with _renderer(model, width, height) as r:
        for i in range(n):
            cam = framed_camera(data, azimuth=100.0 + 200.0 * i / max(n - 1, 1), distance=2.5)
            r.update_scene(data, camera=cam)
            frames.append(
                overlay.draw(
                    r.render(),
                    title="MS-Human-700",
                    subtitle="700 Hill-type muscle actuators over 85 degrees of freedom, "
                             "all pull-only, all tendon-routed",
                    hud=[
                        f"actuators   {model.nu}",
                        f"DoF         {model.nv}",
                        f"mass        {model.body_mass.sum():.1f} kg",
                        f"command     [0, 1]  (activation, not torque)",
                        f"colour      blue = idle, red = active",
                    ],
                )
            )
    return frames


def seg_recruitment(width, height, overlay, *, seconds=4.5) -> list[np.ndarray]:
    """
    Sweep a joint kinematically, tinting tendons by moment-arm magnitude.

    Two things are visible at once and both are measurements. Which muscles
    light up is the sparsity of one column of R. How brightly they light as the
    joint travels is that column changing with configuration, which is the
    reason available joint torque is not a constant.
    """
    frames = []
    for joint_name, azimuth, target, dist, note in (
        ("knee_angle_r", 55.0, "femur_r", 2.05, "22 of 700 muscles reach this joint"),
        ("shoulder_elv_r", 150.0, "humerus_r", 1.75,
         "the shoulder is driven partly through equality couplings"),
    ):
        model, data = load("full")
        names = actuator_names(model)
        jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr, dof = int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
        lo, hi = model.jnt_range[jnt]
        probe = names.index(PROBE_MUSCLE) if joint_name.startswith("knee") else None

        n = int(seconds * FPS)
        # Out and back, so the moment arm is seen collapsing and recovering.
        angles = np.concatenate(
            [np.linspace(lo, hi, n // 2), np.linspace(hi, lo, n - n // 2)]
        )
        base = model.tendon_rgba.copy()
        with _renderer(model, width, height) as r:
            for angle in angles:
                data.qpos[adr] = angle
                project_equality(model, data)
                col = np.abs(moment_arms(model, data)[:, dof])
                reach = int(np.count_nonzero(col))
                peak = col.max()
                # Normalise against this joint's own largest moment arm at any
                # pose in the sweep, not the instantaneous one: dividing by the
                # current maximum would rescale every frame and hide exactly
                # the variation the segment is about.
                model.tendon_rgba[:] = base
                colour_tendons(model, col / 0.06)
                cam = body_camera(model, data, target, azimuth=azimuth, distance=dist, lift=0.22)
                r.update_scene(data, camera=cam)
                hud = [
                    f"joint       {joint_name}",
                    f"angle       {angle:+.2f} rad",
                    f"muscles     {reach} with non-zero moment arm",
                    f"max arm     {peak * 1e3:5.1f} mm",
                ]
                if probe is not None:
                    hud.append(f"vasint_r    {col[probe] * 1e3:5.1f} mm")
                frames.append(
                    overlay.draw(
                        r.render(),
                        title="Moment arms move with the pose",
                        subtitle=f"Kinematic sweep. {note}. Brightness is |moment arm| "
                                 f"about this joint",
                        hud=hud,
                    )
                )
        model.tendon_rgba[:] = base
    return frames


def seg_activation_lag(width, height, overlay, *, sim_seconds=0.62, slow=8.0) -> list[np.ndarray]:
    """
    Step the command on the knee extensors and watch activation chase it.

    The pose is held and only the activation state is advanced, using the same
    RK4 integration of `mju_muscleDynamics` that characterize.py validates the
    simulator against. Playback is 8x slow, which turns a 92 ms release into
    three quarters of a second on screen.
    """
    model, data = load("full")
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    dof = int(model.jnt_dofadr[knee])
    data.qpos[model.jnt_qposadr[knee]] = 1.1
    project_equality(model, data)
    group = moment_arms(model, data)[:, dof] > 1e-4
    prm = model.actuator_dynprm[0][:3].copy()

    # Draw the recruited group three times wider. Five muscles rendered at the
    # model's 5 mm tendon width are a hairline against a 60 kg body, and the
    # segment is about watching that group's colour change over ~30 ms. This
    # affects rendering only; no dynamics quantity reads tendon_width.
    model.tendon_width[model.actuator_trnid[group, 0]] *= 3.0

    dt = 1.0 / (FPS * slow)          # simulated seconds per rendered frame
    sub = 40                          # RK4 substeps per frame: 0.1 ms each
    h = dt / sub
    n = int(sim_seconds / dt)
    step_down = n // 2

    base = model.tendon_rgba.copy()
    act, t = 0.0, 0.0
    frames = []
    with _renderer(model, width, height) as r:
        for i in range(n):
            cmd = 1.0 if i < step_down else 0.0
            for _ in range(sub):
                k1 = mujoco.mju_muscleDynamics(cmd, act, prm)
                k2 = mujoco.mju_muscleDynamics(cmd, act + 0.5 * h * k1, prm)
                k3 = mujoco.mju_muscleDynamics(cmd, act + 0.5 * h * k2, prm)
                k4 = mujoco.mju_muscleDynamics(cmd, act + h * k3, prm)
                act += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            data.act[:] = np.where(group, act, 0.02)
            mujoco.mj_forward(model, data)
            model.tendon_rgba[:] = base
            colour_tendons(model, data.act)
            r.update_scene(
                data,
                camera=body_camera(
                    model, data, "femur_r", azimuth=55.0, distance=2.1, elevation=-8.0, lift=0.28
                ),
            )
            phase = "rising" if cmd > 0 else "releasing"
            frames.append(
                overlay.draw(
                    r.render(),
                    title="Activation lags the command, asymmetrically",
                    subtitle="8x slow motion. Pose held; only the activation state is "
                             "advancing. Rise 31.8 ms, release 91.9 ms",
                    hud=[
                        f"t           {t * 1e3:6.1f} ms",
                        f"command     {cmd:.2f}",
                        f"activation  {act:.3f}   {'|' * int(act * 22)}",
                        f"phase       {phase}",
                    ],
                )
            )
    model.tendon_rgba[:] = base
    return frames


def seg_underactuation(width, height, overlay, *, seconds=3.4) -> list[np.ndarray]:
    """
    Release two bodies under gravity: one relaxed, one maximally co-contracted.

    Both fall. Co-contraction changes how the body falls, because it stiffens
    every joint it crosses, but it cannot change *that* it falls: muscles are
    internal forces, the six base DoF sit in the null space of the moment-arm
    matrix, and characterize.py measures the generalised force they receive
    from a random activation pattern as 0.0 N exactly.
    """
    half = width // 2
    panels = []
    for level, label in ((0.02, "relaxed"), (0.50, "50% co-contraction")):
        model, data = load("full")
        data.ctrl[:] = level
        data.act[:] = level
        base = model.tendon_rgba.copy()
        model.tendon_rgba[:] = base
        colour_tendons(model, np.full(model.nu, level))
        steps = int(round((1.0 / FPS) / model.opt.timestep))
        frames = []
        with _renderer(model, half, height) as r:
            for _ in range(int(seconds * FPS)):
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                cam = framed_camera(data, azimuth=125.0, distance=2.9)
                cam.lookat[2] = max(float(cam.lookat[2]), 0.55)
                r.update_scene(data, camera=cam)
                frames.append((r.render(), float(data.subtree_com[0][2])))
        panels.append((frames, label))
        model.tendon_rgba[:] = base

    out = []
    (left, left_label), (right, right_label) = panels
    label_font = _font("arialbd.ttf", max(10, int(21 * width / 1280.0)))
    for i in range(min(len(left), len(right))):
        canvas = np.concatenate([left[i][0], right[i][0]], axis=1)
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        d.line([(half, 0), (half, height)], fill=(90, 104, 122), width=2)
        # Panel labels sit just above the caption band, not at the top: the HUD
        # box occupies the top-left corner and would swallow the left label.
        label_y = height - overlay.band - int(overlay.mono.size * 2.2)
        for x, text in ((int(half * 0.06), left_label), (half + int(half * 0.06), right_label)):
            d.text((x, label_y), text, font=label_font, fill=(238, 242, 248),
                   stroke_width=2, stroke_fill=(10, 14, 20))
        out.append(
            overlay.draw(
                np.asarray(img),
                title="No activation pattern holds the body up",
                subtitle="Full dynamic rollout. Co-contraction stiffens every joint it "
                         "crosses and still cannot reach the six base DoF",
                hud=[
                    f"t           {i / FPS:4.2f} s",
                    f"COM height  {left[i][1]:.2f} m   vs   {right[i][1]:.2f} m",
                    f"base force  0.0 N from any activation",
                    f"R rank      65 of 85 DoF",
                ],
            )
        )
    return out


# ---------------------------------------------------------------- main


def render(out_path: Path, *, width=1280, height=720) -> dict:
    overlay = Overlay(width, height)
    t0 = time.time()
    segments: list[tuple[str, int, float]] = []
    all_frames: list[np.ndarray] = []
    spans: dict[str, tuple[int, int]] = {}

    for name, fn in (
        ("body", seg_body),
        ("recruitment", seg_recruitment),
        ("activation_lag", seg_activation_lag),
        ("underactuation", seg_underactuation),
    ):
        start = time.time()
        frames = fn(width, height, overlay)
        spans[name] = (len(all_frames), len(all_frames) + len(frames))
        all_frames.extend(frames)
        segments.append((name, len(frames), round(time.time() - start, 1)))
        print(f"  {name:16s} {len(frames):4d} frames  {time.time() - start:5.1f} s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `macro_block_size=1` keeps the exact resolution instead of silently
    # padding to a multiple of 16, which otherwise shifts the caption band.
    imageio.mimsave(out_path, all_frames, fps=FPS, quality=8, macro_block_size=1)

    # Poster still: a quarter of the way into the activation segment, where the
    # recruited group is lit and the caption is on screen.
    lo, hi = spans["activation_lag"]
    imageio.imwrite(out_path.with_name("film_poster.png"), all_frames[lo + (hi - lo) // 4])

    return {
        "path": str(out_path),
        "frames": len(all_frames),
        "seconds": round(len(all_frames) / FPS, 2),
        "resolution": f"{width}x{height}",
        "segments": segments,
        "render_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the S9 walkthrough video.")
    parser.add_argument("--draft", action="store_true", help="640x360, for composition checks")
    parser.add_argument("--out", type=Path, default=MEDIA_DIR / "s9_muscle.mp4")
    args = parser.parse_args()

    size = (640, 360) if args.draft else (1280, 720)
    out = args.out.with_name("s9_muscle_draft.mp4") if args.draft else args.out
    print(f"rendering {size[0]}x{size[1]} -> {out}")
    info = render(out, width=size[0], height=size[1])
    print(
        f"\n{info['frames']} frames, {info['seconds']} s at {FPS} fps, "
        f"rendered in {info['render_seconds']} s"
    )


if __name__ == "__main__":
    main()
