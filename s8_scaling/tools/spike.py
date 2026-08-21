"""Feasibility spike: scene sanity, gripper convention, expert success, throughput.

Run:  python -m s8_scaling.tools.spike [--shots]
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

from ..env import SUBSTEPS, TabletopEnv
from ..expert import ScriptedExpert
from ..tasks import FAMILIES, sample_instance


def main() -> None:
    env = TabletopEnv()
    print(f"model: nq={env.model.nq} nv={env.model.nv} nu={env.model.nu}")

    # --- gripper convention check: ctrl 255 should open, 0 should close.
    env.reset(seed=0, family="reach")
    env.data.ctrl[7] = 255
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)
    w_open = float(env.data.qpos[7] + env.data.qpos[8])
    env.data.ctrl[7] = 0
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)
    w_closed = float(env.data.qpos[7] + env.data.qpos[8])
    print(f"finger width: ctrl=255 -> {w_open:.4f} m, ctrl=0 -> {w_closed:.4f} m")
    assert w_open > 0.07 and w_closed < 0.01, "gripper convention is wrong"

    # --- IK tracking check: command a square, measure tracking error.
    obs, _ = env.reset(seed=1, family="reach")
    targets = [(0.5, 0.15, 0.25), (0.5, -0.15, 0.25), (0.4, -0.15, 0.10), (0.4, 0.15, 0.10)]
    worst = 0.0
    for tx, ty, tz in targets:
        for _ in range(40):
            sp = env.privileged_state()["setpoint"]
            delta = (np.array([tx, ty, tz]) - sp) / 0.04
            env.step(np.array([*np.clip(delta, -1, 1), 0.0, 1.0]))
        tcp = env.privileged_state()["tcp_pos"]
        err = float(np.linalg.norm(tcp - np.array([tx, ty, tz])))
        worst = max(worst, err)
        print(f"  waypoint ({tx:.2f},{ty:.2f},{tz:.2f}) -> tracking error {err*1000:.1f} mm")
    assert worst < 0.02, f"IK tracking error {worst*1000:.0f} mm is too large"

    # --- expert success per family + throughput.
    n = 10
    t0 = time.time()
    total_ctrl_steps = 0
    print(f"\nexpert success over {n} seeds per family:")
    for family in FAMILIES:
        expert = ScriptedExpert(rng=np.random.default_rng(123))
        wins = 0
        lengths = []
        for seed in range(n):
            env.reset_to(sample_instance(family, 10_000 + seed))
            expert.reset()
            for _ in range(env.instance.horizon):
                _, _, term, trunc, info = env.step(expert.act(env))
                total_ctrl_steps += 1
                if term or trunc:
                    break
            wins += bool(info["success"])
            lengths.append(info["steps"])
        print(f"  {family:8s} {wins}/{n}  mean length {np.mean(lengths):.0f}")

    dt = time.time() - t0
    sim_seconds = total_ctrl_steps * SUBSTEPS * 0.002
    print(f"\nthroughput: {total_ctrl_steps/dt:.0f} ctrl steps/s "
          f"= {sim_seconds/dt:.1f}x realtime single-core")

    if "--shots" in sys.argv:
        import imageio.v2 as imageio
        for family in FAMILIES:
            env.reset_to(sample_instance(family, 42))
            imageio.imwrite(f"s8_scaling/tools/shot_{family}.png", env.render())
        print("wrote scene shots to s8_scaling/tools/")


if __name__ == "__main__":
    main()
