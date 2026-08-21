"""
Every published S8 figure, regenerated from disk.

Follows the portfolio convention: figures are pure functions of run logs
and results files, never of in-memory training state, so any figure can be
regenerated long after the runs finished. Dark variants match the site.

Run:  python -m s8_scaling.figures [--dark] [--out s8_scaling/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.plotting import PALETTE, use_dark_style  # noqa: E402

from .fits import fit_offset, fit_pure  # noqa: E402
from .model import SIZES  # noqa: E402
from .tasks import PRETRAIN_FAMILIES  # noqa: E402

SIZE_ORDER = list(SIZES)
SIZE_LABELS = {}  # filled from results (params per size)


def load_sweep(runs_root: Path) -> list[dict]:
    rows = []
    for result in runs_root.glob("*_s8_bc_*/result.json"):
        rows.append(json.loads(result.read_text()))
    return rows


def _by_size(rows: list[dict], metric: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """size -> (hours, mean over seeds, spread over seeds)."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["size"]][float(r["hours"])].append(float(r[metric]))
        SIZE_LABELS[r["size"]] = f"{r['size']} ({r['n_params']/1e3:.0f}k)" if r[
            "n_params"
        ] < 1e6 else f"{r['size']} ({r['n_params']/1e6:.1f}M)"
    out = {}
    for size, per_h in acc.items():
        hours = np.array(sorted(per_h))
        mean = np.array([np.mean(per_h[h]) for h in hours])
        spread = np.array([np.ptp(per_h[h]) / 2 for h in hours])
        out[size] = (hours, mean, spread)
    return out


def fig_scaling(rows: list[dict], out: Path, metric: str, title: str, fit_sizes=("l", "xl")) -> None:
    """Log-log loss vs data, one curve per model size, power-law fits."""
    data = _by_size(rows, metric)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, size in enumerate(SIZE_ORDER):
        if size not in data:
            continue
        hours, mean, spread = data[size]
        ax.errorbar(
            hours, mean, yerr=spread,
            color=PALETTE[i % len(PALETTE)], lw=1.7, marker="o", ms=4,
            capsize=2.5, label=SIZE_LABELS.get(size, size),
        )
        if size in fit_sizes and len(hours) >= 4:
            fit = fit_pure(hours, mean)
            xs = np.geomspace(hours.min(), hours.max(), 64)
            ax.plot(
                xs, fit.predict(xs),
                color=PALETTE[i % len(PALETTE)], ls="--", lw=1.0, alpha=0.7,
            )
            ax.annotate(
                f"α = {fit.alpha:.2f}, R² = {fit.r2:.3f}",
                xy=(xs[-1], fit.predict(xs[-1:])[0]),
                xytext=(-8, -14 if size == "xl" else 10),
                textcoords="offset points", ha="right", fontsize=8.5,
                color=PALETTE[i % len(PALETTE)],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pretraining data (hours of demonstrated experience)")
    ax.set_ylabel("next-action prediction MSE")
    ax.set_title(title)
    ax.legend(fontsize=8.5, ncol=2)
    fig.savefig(out)
    plt.close(fig)


def fig_capacity(rows: list[dict], out: Path) -> None:
    """Two panels: loss vs data per size; loss vs params per data scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    data = _by_size(rows, "best_val")
    for i, size in enumerate(SIZE_ORDER):
        if size not in data:
            continue
        hours, mean, _ = data[size]
        ax1.plot(hours, mean, color=PALETTE[i % len(PALETTE)], lw=1.7, marker="o",
                 ms=4, label=SIZE_LABELS.get(size, size))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("data (hours)")
    ax1.set_ylabel("held-in validation MSE")
    ax1.set_title("capacity saturation: small models flatten")
    ax1.legend(fontsize=8)

    # loss vs params at each data scale
    acc: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[float(r["hours"])][int(r["n_params"])].append(float(r["best_val"]))
    hours_sorted = sorted(acc)
    for i, h in enumerate(hours_sorted):
        params = np.array(sorted(acc[h]))
        mean = np.array([np.mean(acc[h][p]) for p in params])
        ax2.plot(params, mean, color=PALETTE[i % len(PALETTE)], lw=1.5, marker="s",
                 ms=3.5, label=f"{h:g} h")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("model parameters")
    ax2.set_ylabel("held-in validation MSE")
    ax2.set_title("returns to capacity grow with data")
    ax2.legend(fontsize=8, title="data", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_transfer(transfer_json: Path, out: Path) -> None:
    """Closed-loop stack success vs pretraining hours, per demo budget."""
    rows = json.loads(transfer_json.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    budgets = sorted({r["budget"] for r in rows if r["budget"] > 0})
    for i, budget in enumerate(budgets):
        pts = defaultdict(list)
        for r in rows:
            if r["budget"] == budget and r["pretrain_hours"] > 0:
                pts[float(r["pretrain_hours"])].append(r)
        hours = np.array(sorted(pts))
        mean = np.array([np.mean([p["success"] for p in pts[h]]) for h in hours])
        lo = np.array([np.min([p["ci_low"] for p in pts[h]]) for h in hours])
        hi = np.array([np.max([p["ci_high"] for p in pts[h]]) for h in hours])
        ax.plot(hours, mean, color=PALETTE[i], lw=1.8, marker="o", ms=4.5,
                label=f"pretrained, {budget} demos")
        ax.fill_between(hours, lo, hi, color=PALETTE[i], alpha=0.15, lw=0)

        scratch = [r for r in rows if r["budget"] == budget and r["pretrain_hours"] == 0]
        if scratch:
            s_mean = np.mean([r["success"] for r in scratch])
            ax.axhline(s_mean, color=PALETTE[i], ls=":", lw=1.3, alpha=0.8)
            ax.annotate(f"scratch, {budget} demos", xy=(hours[0], s_mean),
                        xytext=(2, 3), textcoords="offset points", fontsize=8,
                        color=PALETTE[i])

    zero = defaultdict(list)
    for r in rows:
        if r["budget"] == 0 and r["pretrain_hours"] > 0:
            zero[float(r["pretrain_hours"])].append(r["success"])
    if zero:
        hours = np.array(sorted(zero))
        mean = np.array([np.mean(zero[h]) for h in hours])
        ax.plot(hours, mean, color="0.55", lw=1.4, marker="d", ms=4, ls="--",
                label="zero-shot (0 demos)")

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("pretraining data (hours), stack never included")
    ax.set_ylabel("stack success rate (100 paired episodes)")
    ax.set_title("transfer scaling: pretraining hours buy downstream success")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def fig_multitask(multitask_json: Path, out: Path, size: str = "xl") -> None:
    """Per-family closed-loop success vs data scale for one model size."""
    rows = [r for r in json.loads(multitask_json.read_text()) if r["size"] == size]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, family in enumerate(PRETRAIN_FAMILIES):
        pts = sorted(
            [(float(r["hours"]), r) for r in rows if r["family"] == family]
        )
        hours = np.array([h for h, _ in pts])
        mean = np.array([r["success"] for _, r in pts])
        lo = np.array([r["ci_low"] for _, r in pts])
        hi = np.array([r["ci_high"] for _, r in pts])
        ax.plot(hours, mean, color=PALETTE[i], lw=1.7, marker="o", ms=4, label=family)
        ax.fill_between(hours, lo, hi, color=PALETTE[i], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("pretraining data (hours)")
    ax.set_ylabel("success rate (50 episodes)")
    ax.set_title(f"one {size} policy, five tasks: closed-loop success vs data")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def fig_datacard(card_json: Path, out: Path) -> None:
    """The data engine at a glance: hours, episodes, rejection per family."""
    card = json.loads(card_json.read_text())
    fams = list(card["per_family"])
    hours = [card["per_family"][f]["hours"] for f in fams]
    succ = [card["per_family"][f]["success_rate"] for f in fams]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(fams, hours, color=PALETTE[: len(fams)])
    ax1.set_ylabel("hours of demonstrations")
    ax1.set_title(
        f"corpus: {card['hours']:.1f} h, {card['episodes']:,} episodes, "
        f"collected at {card['realtime_factor']:.0f}x realtime"
    )
    ax2.bar(fams, [100 * s for s in succ], color=PALETTE[: len(fams)])
    ax2.set_ylabel("expert success rate (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("data engine acceptance rate (failures discarded)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate all S8 figures.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="s8_scaling/figures")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    if args.dark:
        use_dark_style()
    out = Path(args.out) / ("dark" if args.dark else "light")
    out.mkdir(parents=True, exist_ok=True)

    rows = load_sweep(Path(args.runs_root))
    if rows:
        fig_scaling(
            rows, out / "fig1_withheld_scaling.png", "best_withheld",
            "zero-shot prediction error on the withheld family (stack)",
        )
        fig_scaling(
            rows, out / "fig2_val_scaling.png", "best_val",
            "held-in validation error, five pretraining families",
        )
        fig_capacity(rows, out / "fig3_capacity.png")

    transfer_json = Path("s8_scaling/results/transfer.json")
    if transfer_json.exists():
        fig_transfer(transfer_json, out / "fig4_transfer.png")

    multitask_json = Path("s8_scaling/results/multitask.json")
    if multitask_json.exists():
        fig_multitask(multitask_json, out / "fig5_multitask.png")

    card_json = Path("s8_scaling/corpus/pretrain/card.json")
    if card_json.exists():
        fig_datacard(card_json, out / "fig6_datacard.png")

    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
