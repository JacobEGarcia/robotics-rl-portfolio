"""
Closed-loop evaluation through the S7 harness.

Every number this module produces goes through `common.backends.SimClosedLoop`
and `common.metrics.success_rate` - the same interface and the same statistic
definitions as every other project in this portfolio. Success rates carry
bootstrap confidence intervals; nothing here invents its own metric.

The paired-comparison protocol ("blind A/B" in the write-up): evaluations
use seed range [base_seed, base_seed + n). Seed i deterministically maps to
task instance i, so two policies evaluated with the same base seed face
*identical* instance sequences, and their difference is a paired statistic
rather than two noisy marginals. Evaluation seeds live below 1,000,000,
a range no training corpus draws from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.backends import SimClosedLoop  # noqa: E402
from common.metrics import Interval, bootstrap_ci  # noqa: E402

from .env import TabletopEnv  # noqa: E402
from .model import ChunkedPolicyAdapter, Policy  # noqa: E402


class _FamilyEnv:
    """Adapter pinning a TabletopEnv to one family for SimClosedLoop."""

    def __init__(self, env: TabletopEnv, family: str, adapter: ChunkedPolicyAdapter):
        self.env = env
        self.family = family
        self.adapter = adapter

    def reset(self, *, seed: int):
        # A fresh episode must not execute a stale action chunk.
        self.adapter.reset()
        return self.env.reset(seed=seed, family=self.family)

    def step(self, action):
        return self.env.step(action)


def eval_policy(
    policy: Policy,
    family: str,
    *,
    n_episodes: int = 50,
    base_seed: int = 0,
    env: TabletopEnv | None = None,
) -> tuple[Interval, list[dict]]:
    """Success rate of `policy` on `family` over deterministic seeds."""
    env = env or TabletopEnv()
    adapter = ChunkedPolicyAdapter(policy)
    backend = SimClosedLoop(
        _FamilyEnv(env, family, adapter),
        adapter,
        max_steps=300,
        success_fn=lambda info: bool(info.get("success", False)),
    )
    records = backend.evaluate(n_episodes, base_seed=base_seed)
    flags = np.array([float(r["success"]) for r in records])
    return bootstrap_ci(flags, statistic=lambda v: float(np.mean(v))), records
