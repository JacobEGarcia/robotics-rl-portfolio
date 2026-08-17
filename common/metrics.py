"""
Metric definitions, computed from run logs.

Why metrics live here and not inside training loops
---------------------------------------------------
If a metric is computed inline during training, it exists only in that run's
memory and can never be recomputed differently. Worse, two projects end up
with two subtly different definitions of "success rate" and their numbers get
compared anyway.

Here, training writes raw events, and every metric is a pure function of the
log. That means a metric can be redefined and every historical run
re-evaluated under the new definition, without retraining anything.

Statistical note
----------------
The standard practice of reporting mean +/- std over 3 seeds is not defensible
for RL, where returns are heavy-tailed and 3 samples cannot support a normal
approximation. Agarwal et al., "Deep Reinforcement Learning at the Edge of the
Statistical Precipice" (NeurIPS 2021), documented how badly this misleads.

We therefore report:
  * the interquartile mean (IQM), robust to the outlier runs RL produces
  * bootstrap confidence intervals rather than standard error
  * the seed count, always, so a reader can discount accordingly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def interquartile_mean(values: np.ndarray) -> float:
    """
    Mean of the middle 50% of values.

    More robust than the mean (which one lucky seed can dominate) and more
    informative than the median (which discards all magnitude information).
    This is the primary statistic recommended by the Precipice paper.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        # IQM is meaningless on tiny samples; fall back to the mean rather
        # than silently returning something that looks robust but isn't.
        return float(np.mean(values))
    lo, hi = np.percentile(values, [25, 75])
    middle = values[(values >= lo) & (values <= hi)]
    return float(np.mean(middle)) if middle.size else float(np.mean(values))


def bootstrap_ci(
    values: np.ndarray,
    statistic=interquartile_mean,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """
    Percentile bootstrap confidence interval for an arbitrary statistic.

    We resample the observed values with replacement `resamples` times,
    recompute the statistic each time, and take percentiles of that
    distribution. This makes no normality assumption, which matters because
    RL returns are frequently bimodal (runs that learned vs runs that didn't).

    The `seed` argument makes the interval itself reproducible -- otherwise
    re-running analysis produces slightly different error bars and readers
    reasonably lose confidence.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        v = float(values[0])
        return Interval(v, v, v, 1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    stats = np.array([statistic(values[row]) for row in idx])

    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return Interval(float(statistic(values)), float(low), float(high), n)


def episode_returns(events: list[dict]) -> np.ndarray:
    """Extract per-episode returns from a run log's events."""
    return np.array(
        [e["ret"] for e in events if e.get("type") == "episode" and "ret" in e],
        dtype=float,
    )


def success_rate(events: list[dict]) -> Interval:
    """
    Fraction of episodes flagged successful, with a bootstrap CI.

    Requires episodes to carry an explicit boolean `success` field. We do not
    infer success from a return threshold: thresholds drift between projects
    and make numbers incomparable, which is exactly what this module exists
    to prevent.
    """
    flags = np.array(
        [float(bool(e["success"])) for e in events if e.get("type") == "episode" and "success" in e]
    )
    return bootstrap_ci(flags, statistic=lambda v: float(np.mean(v)))


def learning_curve(events: list[dict], window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (steps, smoothed_return) for plotting.

    A trailing moving average is used rather than exponential smoothing so
    that the plotted value at step X depends only on data at or before X.
    Exponential smoothing with a poorly chosen factor can visually imply a
    run converged earlier than it did.
    """
    pts = [
        (e.get("step", i), e["ret"])
        for i, e in enumerate(events)
        if e.get("type") == "episode" and "ret" in e
    ]
    if not pts:
        return np.array([]), np.array([])
    steps, rets = map(np.asarray, zip(*pts))
    if window > 1 and rets.size >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rets, kernel, mode="valid")
        return steps[window - 1 :], smoothed
    return steps, rets.astype(float)


def summarize(events: list[dict]) -> dict:
    """One-line summary of a run, for tables and console output."""
    rets = episode_returns(events)
    if rets.size == 0:
        return {"episodes": 0}
    # Final performance uses the last 10% of episodes rather than the single
    # best, because best-of is a selection artifact, not a performance claim.
    tail = rets[max(1, int(0.9 * rets.size)) :]
    return {
        "episodes": int(rets.size),
        "return_final": bootstrap_ci(tail),
        "return_best_episode": float(np.max(rets)),
        "return_mean_all": float(np.mean(rets)),
    }
