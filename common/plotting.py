"""
Shared plotting. One visual style across every project so figures from
different experiments can sit on the same page without reading as a collage.

All functions take run-log directories, never in-memory training state, so any
figure can be regenerated from disk long after the run finished.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Agg backend: these run headless in CI and over SSH, where no display exists.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import learning_curve  # noqa: E402
from .runlog import read_log  # noqa: E402

# A restrained palette that stays legible in greyscale print and to the most
# common forms of colour blindness (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def plot_learning_curves(
    run_dirs: list[str | Path],
    labels: list[str] | None = None,
    *,
    title: str = "",
    target: float | None = None,
    target_label: str = "published baseline",
    window: int = 20,
    out: str | Path = "curve.png",
) -> Path:
    """
    Plot one or more learning curves on shared axes.

    A horizontal `target` line is strongly encouraged: a curve alone only shows
    that a number increased, which is not evidence of correctness. A curve
    against a published baseline is.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, run_dir in enumerate(run_dirs):
        _, events = read_log(run_dir)
        steps, rets = learning_curve(events, window=window)
        if steps.size == 0:
            continue
        label = labels[i] if labels else Path(run_dir).name[:28]
        ax.plot(steps, rets, color=PALETTE[i % len(PALETTE)], lw=1.6, label=label)

    if target is not None:
        ax.axhline(
            target, color="0.35", ls="--", lw=1.2,
            label=f"{target_label} ({target:g})",
        )

    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return ({window}-episode moving average)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_diagnostics(run_dir: str | Path, *, out: str | Path = "diagnostics.png") -> Path:
    """
    Four-panel training diagnostic.

    These are the panels that actually tell you whether PPO is healthy:

      explained variance -- is the critic doing anything? <= 0 means it is no
          better than predicting the mean, which is nearly always a bug.
      approx KL          -- how far the policy moves per update. Collapsing to
          0 means learning has stalled; spikes mean instability.
      clip fraction      -- share of samples hitting the clip. Healthy is
          roughly 0.05-0.3; near 0 means the updates are too timid.
      value loss         -- should fall and stay bounded. A value loss orders
          of magnitude above the policy loss means reward scaling is missing.
    """
    _, events = read_log(run_dir)
    ups = [e for e in events if e.get("type") == "update"]
    if not ups:
        raise ValueError(f"no update events in {run_dir}")

    steps = [e.get("step", i) for i, e in enumerate(ups)]
    panels = [
        ("explained_variance", "explained variance", 0.0),
        ("approx_kl", "approx KL", None),
        ("clipfrac", "clip fraction", None),
        ("v_loss", "value loss", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (key, label, ref) in zip(axes.ravel(), panels):
        vals = [e.get(key, np.nan) for e in ups]
        ax.plot(steps, vals, color=PALETTE[0], lw=1.4)
        if ref is not None:
            ax.axhline(ref, color="0.5", ls="--", lw=1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("steps")
        if key == "v_loss":
            ax.set_yscale("log")

    fig.suptitle(f"PPO diagnostics — {Path(run_dir).name}", fontsize=11)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_before_after(
    x: np.ndarray,
    reference: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    xlabel: str = "time (s)",
    ylabel: str = "state",
    title: str = "",
    out: str | Path = "before_after.png",
) -> Path:
    """
    The system-identification figure: reference vs simulation, pre- and
    post-fit. This is the single plot S5 exists to produce.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, reference, color="black", lw=2.0, label="real (reference)")
    ax.plot(x, before, color=PALETTE[1], lw=1.4, ls="--", label="sim, before sysID")
    ax.plot(x, after, color=PALETTE[2], lw=1.6, label="sim, after sysID")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Dark theme for the published site
# --------------------------------------------------------------------------
# The portfolio site is dark. A figure with a white background pasted onto a
# black page reads as an afterthought, so figures are regenerated in a
# matching palette rather than being visually patched with CSS filters.

DARK_BG = "#0a0a0a"
DARK_PANEL = "#111111"
DARK_INK = "#e8e8e6"
DARK_MUTED = "#8a8a86"


def use_dark_style() -> None:
    """Switch matplotlib to the site's dark palette."""
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "savefig.facecolor": DARK_BG,
            "savefig.edgecolor": DARK_BG,
            "text.color": DARK_INK,
            "axes.labelcolor": DARK_MUTED,
            "axes.edgecolor": "#2a2a2a",
            "xtick.color": DARK_MUTED,
            "ytick.color": DARK_MUTED,
            "grid.color": "#242424",
            "grid.alpha": 1.0,
            "legend.labelcolor": DARK_INK,
            "axes.titlecolor": DARK_INK,
        }
    )
