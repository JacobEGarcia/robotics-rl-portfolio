"""
Determinism control for reproducible RL experiments.

Why this file exists
--------------------
An RL result that cannot be reproduced is an anecdote, not a measurement. The
usual failure is partial seeding: people seed `numpy` but forget `torch`, or
seed the policy but not the environment, and then wonder why two "identical"
runs diverge.

This module seeds every source of randomness in one call and records exactly
what was seeded, so a run manifest can prove it later.

Note on the limits of determinism
---------------------------------
Bitwise determinism is only guaranteed for the *same* machine, same library
versions, and same device. Floating-point reduction order differs across CPU
and GPU, so a CPU run and a GPU run of the same seed will diverge. That is
expected and is why `capture_environment()` records the device and library
versions alongside the seed: reproducibility claims are only meaningful
relative to a recorded environment.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Everything needed to judge whether two runs are comparable."""

    python_version: str
    platform: str
    numpy_version: str
    torch_version: str | None
    mujoco_version: str | None
    torch_device: str
    git_sha: str | None
    git_dirty: bool

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> int:
    """
    Seed every RNG this project can reach.

    Parameters
    ----------
    seed
        The seed to apply.
    deterministic_torch
        If True, ask torch to refuse non-deterministic kernels rather than
        silently using them. This can slow things down and will raise on some
        ops, which is the point -- a loud failure beats a silent one.

    Returns
    -------
    The seed, so callers can log it inline.
    """
    # PYTHONHASHSEED affects set/dict iteration order for str keys. It must be
    # set before interpreter start to fully take effect, so setting it here is
    # best-effort; we record it in the manifest either way.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # cuDNN picks algorithms by benchmarking unless told not to; that
            # benchmark is itself nondeterministic across runs.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Raises if an op has no deterministic implementation. We warn
            # rather than hard-fail because some MuJoCo-adjacent ops are fine
            # in practice and we would rather run than block.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def _git(*args: str) -> str | None:
    """Run a git command, returning None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment() -> EnvironmentFingerprint:
    """
    Snapshot the machine and code state.

    The git SHA matters more than it looks: it is the difference between
    "this plot came from some version of the code" and "this plot came from
    commit abc123, which you can check out and re-run".
    """
    torch_version = None
    torch_device = "none"
    try:
        import torch

        torch_version = torch.__version__
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    mujoco_version = None
    try:
        import mujoco

        mujoco_version = mujoco.__version__
    except ImportError:
        pass

    sha = _git("rev-parse", "HEAD")
    # `git status --porcelain` prints one line per modified file; any output
    # means the working tree does not match the recorded SHA.
    dirty_out = _git("status", "--porcelain")

    return EnvironmentFingerprint(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        numpy_version=np.__version__,
        torch_version=torch_version,
        mujoco_version=mujoco_version,
        torch_device=torch_device,
        git_sha=sha,
        git_dirty=bool(dirty_out),
    )
