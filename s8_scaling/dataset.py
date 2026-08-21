"""
Corpus loading with the two properties the scaling axes depend on.

1. Prefix subsets. `load_split(hours=H)` takes episodes in stored order
   until H hours accumulate. Because the data engine interleaved families
   round-robin, every prefix is family-balanced, and the 1-hour dataset is
   literally the first hour of the 32-hour dataset. Scaling curves compare
   nested datasets, not six independently sampled ones - otherwise dataset-
   to-dataset sampling luck is confounded with the effect of scale.

2. Leakage-free splits. Train/val is split by *episode*, not by timestep,
   using a hash of the episode seed. Adjacent timesteps of one episode are
   nearly identical; putting them on opposite sides of the split would let
   the validation loss reward memorisation. Hashing the seed (rather than
   position) keeps an episode's split assignment stable across corpus sizes.

Normalisation statistics are computed on each run's own training subset -
computing them on the full corpus would leak information from data a small-
scale run is not supposed to have seen.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_engine import SECONDS_PER_STEP
from .env import ACT_DIM, OBS_DIM

CHUNK = 8  # action-chunk length K: the policy predicts K future actions


def _val_episode(seed: int, val_fraction: float = 0.05) -> bool:
    h = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4], "little")
    return (h % 10_000) < val_fraction * 10_000


@dataclass
class Split:
    """Index into a memory-mapped corpus: sample (obs_t, act_t..t+K)."""

    obs: np.ndarray  # mmap (N, OBS_DIM)
    act: np.ndarray  # mmap (N, ACT_DIM)
    starts: np.ndarray  # per-sample flat index of obs_t
    ep_start: np.ndarray  # per-sample episode start (for chunk clamping)
    ep_end: np.ndarray  # per-sample episode end (exclusive)

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def hours(self) -> float:
        return float(len(self.starts) * SECONDS_PER_STEP / 3600.0)

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.starts), size=size)
        return self.gather(idx)

    def gather(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = self.starts[idx]
        x = self.obs[t]
        # Action chunk act[t : t+K], clamped at the episode boundary by
        # repeating the final action - the standard chunking convention, and
        # honest here because every expert ends episodes in a quiescent
        # retreat, so the repeated action is genuinely what comes next.
        offsets = t[:, None] + np.arange(CHUNK)[None, :]
        clamped = np.minimum(offsets, self.ep_end[idx][:, None] - 1)
        y = self.act[clamped.reshape(-1)].reshape(len(idx), CHUNK * ACT_DIM)
        return x, y


def load_split(
    corpus_dir: str | Path,
    *,
    hours: float | None = None,
    max_episodes: int | None = None,
    val: bool = False,
    val_fraction: float = 0.05,
) -> Split:
    """
    Load the train (or val) split of the first `hours` of a corpus.

    `max_episodes` selects by episode count instead - used for the transfer
    experiments, whose budgets are "N demonstrations" not "H hours".
    """
    corpus_dir = Path(corpus_dir)
    obs = np.load(corpus_dir / "obs.npy", mmap_mode="r")
    act = np.load(corpus_dir / "act.npy", mmap_mode="r")
    episodes = np.load(corpus_dir / "episodes.npy")

    budget_steps = int(hours * 3600 / SECONDS_PER_STEP) if hours is not None else None
    starts, ep_start_col, ep_end_col = [], [], []
    used_steps = 0
    used_eps = 0
    for start, length, _fam, seed in episodes:
        if budget_steps is not None and used_steps >= budget_steps:
            break
        if max_episodes is not None and used_eps >= max_episodes:
            break
        used_steps += int(length)
        used_eps += 1
        if _val_episode(int(seed), val_fraction) != val:
            continue
        t = np.arange(start, start + length)
        starts.append(t)
        ep_start_col.append(np.full(length, start))
        ep_end_col.append(np.full(length, start + length))

    if not starts:
        raise ValueError(
            f"empty {'val' if val else 'train'} split from {corpus_dir} "
            f"(hours={hours}, max_episodes={max_episodes})"
        )
    return Split(
        obs=obs,
        act=act,
        starts=np.concatenate(starts),
        ep_start=np.concatenate(ep_start_col),
        ep_end=np.concatenate(ep_end_col),
    )


def normalisation_stats(split: Split, cap: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """
    Observation mean/std over (a subsample of) the training split.

    The subsample cap keeps this O(1) as corpora grow; 200k timesteps
    estimates 69 per-dimension means to well under 1% of a std.
    """
    rng = np.random.default_rng(0)
    idx = (
        np.arange(len(split))
        if len(split) <= cap
        else rng.choice(len(split), size=cap, replace=False)
    )
    x = split.obs[split.starts[idx]]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-4] = 1.0  # constant dims (one-hot zeros): leave unscaled
    return mean.astype(np.float32), std.astype(np.float32)
