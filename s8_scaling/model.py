"""
The policy family: five MLP sizes spanning ~400x in parameter count.

MLPs rather than transformers, deliberately. The observation is a single
69-D state vector, not a token sequence; what a transformer would add here
is attention over a sequence that does not exist. The scaling phenomena
under study - power-law loss curves, capacity saturation, transfer - are
properties of model capacity vs data scale, and MLP capacity is the
cleanest possible capacity axis. (GEN-0's laws are measured on
transformers because their inputs are genuinely sequential; the law is
about scale, not about attention.)

Output is an action chunk: the next K=8 actions, flattened. Chunked
prediction is standard in manipulation BC (ACT, diffusion policy, GEN-0's
action horizons) because it forces the model to commit to a short plan,
which suppresses the compounding-error jitter of single-step regression.

Sizes are geometric in parameters (~4x per step). `xl` sits deliberately
past the point where capacity stops paying on this task family - a scaling
study needs at least one size that has clearly saturated the task, or
"bigger is better" is just the left half of every curve.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dataset import CHUNK
from .env import ACT_DIM, OBS_DIM

SIZES: dict[str, list[int]] = {
    "t": [64, 64],
    "s": [128, 128],
    "m": [256, 256, 256],
    "l": [512, 512, 512],
    "xl": [1024, 1024, 1024, 1024],
}


class Policy(nn.Module):
    """MLP: normalised 69-D observation -> K*5 action chunk."""

    def __init__(self, size: str):
        super().__init__()
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}, got {size!r}")
        self.size = size
        widths = SIZES[size]
        layers: list[nn.Module] = []
        d = OBS_DIM
        for w in widths:
            layers += [nn.Linear(d, w), nn.GELU()]
            d = w
        layers.append(nn.Linear(d, CHUNK * ACT_DIM))
        self.net = nn.Sequential(*layers)
        # Normalisation is part of the model state so a checkpoint is
        # self-contained: no way to load a policy and forget its stats.
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))

    def set_normalisation(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.obs_mean.copy_(torch.from_numpy(mean))
        self.obs_std.copy_(torch.from_numpy(std))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.obs_mean) / self.obs_std)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ChunkedPolicyAdapter:
    """
    Wraps a trained Policy for closed-loop rollout: predicts a K-chunk,
    executes it action by action, re-plans when exhausted. Satisfies the
    S7 `Policy` protocol (`act(obs) -> action`).
    """

    def __init__(self, policy: Policy, execute: int = CHUNK):
        self.policy = policy.eval()
        self.execute = execute  # how many of the K predicted steps to run
        self._buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self._buffer = []

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        if not self._buffer:
            with torch.no_grad():
                chunk = self.policy(torch.from_numpy(obs[None].astype(np.float32)))
            chunk = chunk.numpy().reshape(CHUNK, ACT_DIM)
            self._buffer = list(chunk[: self.execute])
        return np.clip(self._buffer.pop(0), -1.0, 1.0)


def save_checkpoint(path, policy: Policy, config: dict) -> None:
    torch.save({"state_dict": policy.state_dict(), "size": policy.size, "config": config}, path)


def load_checkpoint(path) -> tuple[Policy, dict]:
    ckpt = torch.load(path, weights_only=True)
    policy = Policy(ckpt["size"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy, ckpt.get("config", {})
