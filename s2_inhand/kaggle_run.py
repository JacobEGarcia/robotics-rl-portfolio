"""
Kaggle session driver for S2.

Paste this into a Kaggle notebook cell (GPU T4 x2 or P100, internet on). It is
written to be run repeatedly: each session picks up where the last one died.

Why this file exists at all
---------------------------
Kaggle's free GPU tier gives about 30 GPU-hours a week, capped at roughly 12
hours per session, and destroys the machine at the end of one. A 1e8-step run
plus the reward-shaping iterations that in-hand reorientation actually needs
does not fit in a single session. So the unit of work is "one session", and
the trainer must survive the boundary.

Session chaining on Kaggle
--------------------------
`/kaggle/working` is persisted as the notebook version's *output*. To continue,
add the previous run's output as an input dataset to the next run and point
CKPT_DIR at it, or copy it into /kaggle/working on startup. The `SEED_CKPT`
path below handles the copy.

What to watch
-------------
The success rate and the curriculum value. If the curriculum is not widening
after a few million steps, the reward is not shaping the behaviour and the fix
is the reward, not more steps. That is the expected failure mode for this
project and it is worth catching in hour one rather than hour eleven.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/JacobEGarcia/robotics-rl-portfolio.git"
WORK = Path("/kaggle/working")
SRC = WORK / "robotics-rl-portfolio"
CKPT_DIR = WORK / "checkpoints"

# If a previous session's output is attached as an input dataset, point this at
# its checkpoints directory and they will be copied in before training starts.
SEED_CKPT = Path("/kaggle/input/s2-checkpoints/checkpoints")


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> None:
    if not SRC.exists():
        sh(f"git clone --depth 1 {REPO} {SRC}")

    # jax[cuda12] must match the driver Kaggle provides. Pinning the local
    # variant instead would silently give a CPU build, and the whole point of
    # the session is the GPU, so verify rather than assume.
    sh(f"{sys.executable} -m pip install -q --upgrade "
       f'"jax[cuda12]" mujoco mujoco-mjx optax')

    import jax
    print("jax", jax.__version__, "devices:", jax.devices(), flush=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit(
            "No GPU visible to JAX. Enable the GPU accelerator on this "
            "notebook. Running S2 on CPU is not a slower option, it is not an "
            "option: a single mjx.step of this model costs seconds."
        )


def seed_checkpoints() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if SEED_CKPT.exists():
        n = 0
        for f in SEED_CKPT.glob("ckpt_*.pkl"):
            shutil.copy2(f, CKPT_DIR / f.name)
            n += 1
        print(f"seeded {n} checkpoint(s) from a previous session", flush=True)
    else:
        print("no previous session attached, starting fresh", flush=True)


def train(total_steps: int = 100_000_000, num_envs: int = 2048) -> None:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    cmd = [
        sys.executable, "-u", "-m", "s2_inhand.train",
        "--total-steps", str(total_steps),
        "--num-envs", str(num_envs),
        "--ckpt-dir", str(CKPT_DIR),
        "--ckpt-every", "2000000",
        "--resume",
    ]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(SRC), env=env, check=True)


if __name__ == "__main__":
    setup()
    seed_checkpoints()
    train()
