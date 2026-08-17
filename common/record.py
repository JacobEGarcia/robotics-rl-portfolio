"""
Render a trained policy to video, and capture still frames.

Why videos are part of the deliverable, not decoration
------------------------------------------------------
A learning curve tells you the return went up. It does not tell you *how*.
Policies routinely reach a good return via behaviour a human would reject --
vibrating against a joint limit, exploiting a contact artifact, or spinning to
farm a shaping term. On real hardware those policies destroy actuators.

Watching the rollout is the cheapest available check that the number means
what you think it means, and for a sim-to-real project it is the check that
matters most.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def record_gym_episode(
    env,
    policy,
    *,
    out_path: str | Path,
    seed: int = 0,
    max_steps: int = 1000,
    fps: int = 30,
    still_frames: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """
    Roll out one episode, writing an MP4 plus still frames.

    `env` must have been created with `render_mode="rgb_array"`.

    `still_frames` are fractions through the episode at which to also save a
    PNG. Stills matter because a README or a job application renders images
    inline; a video needs a click, and reviewers do not click.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total, steps = 0.0, 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        frames.append(env.render())
        action = policy.act(np.asarray(obs), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(
            np.clip(action, env.action_space.low, env.action_space.high)
        )
        total += float(reward)
        steps += 1

    if not frames:
        raise RuntimeError("No frames captured; was the env created with render_mode='rgb_array'?")

    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)

    still_paths = []
    for frac in still_frames:
        idx = min(len(frames) - 1, int(frac * (len(frames) - 1)))
        p = out_path.with_name(f"{out_path.stem}_t{int(frac * 100):03d}.png")
        imageio.imwrite(p, frames[idx])
        still_paths.append(str(p))

    return {
        "video": str(out_path),
        "stills": still_paths,
        "return": total,
        "length": steps,
        "n_frames": len(frames),
    }


def render_mujoco_rollout(
    model,
    data,
    controller,
    *,
    out_path: str | Path,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: str | int = -1,
) -> dict:
    """
    Render a raw MuJoCo rollout (no Gymnasium wrapper).

    Used by the tendon work (S4), where the object of study is the mechanism
    itself rather than a trained agent. `controller(model, data, t)` is called
    each physics step and should write into `data.ctrl`.
    """
    import mujoco

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    n_steps = int(duration / model.opt.timestep)
    # Render at `fps` regardless of the physics rate: MuJoCo commonly runs at
    # 500-1000 Hz, and writing every physics step would produce an enormous
    # file playing back in slow motion.
    every = max(1, int(1.0 / (fps * model.opt.timestep)))

    mujoco.mj_resetData(model, data)
    for i in range(n_steps):
        controller(model, data, data.time)
        mujoco.mj_step(model, data)
        if i % every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    renderer.close()
    return {"video": str(out_path), "n_frames": len(frames), "sim_time": float(data.time)}
