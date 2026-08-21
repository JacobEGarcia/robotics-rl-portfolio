"""
Numbers for the write-up, computed from disk and nothing else.

Produces `s8_scaling/results/summary.json` and a Markdown rendering
`summary.md`. Every headline number in the README and on the site is
copied from this output, so a reader can regenerate the page from the logs.

Reported:
  * power-law fits (pure and irreducible-floor forms) of held-in and
    withheld-family prediction error vs data, per model size
  * the capacity table: error vs parameters at the largest data scale,
    and the data scale at which each size stops improving (ossification)
  * transfer: zero-shot, post-trained, and scratch success on stack, with
    PAIRED differences (pretrained minus scratch, episode for episode)
  * multi-task closed-loop success per family for each size at 32 h

Run:  python -m s8_scaling.analysis
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.metrics import paired_difference, wilson_interval  # noqa: E402

from .figures import load_sweep  # noqa: E402
from .fits import fit_offset, fit_pure  # noqa: E402
from .model import SIZES  # noqa: E402
from .tasks import PRETRAIN_FAMILIES  # noqa: E402

RESULTS = Path("s8_scaling/results")


def _cells(rows: list[dict], metric: str) -> dict[str, dict[float, list[float]]]:
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["size"]][float(r["hours"])].append(float(r[metric]))
    return acc


def scaling_fits(rows: list[dict]) -> dict:
    out = {}
    params = {r["size"]: r["n_params"] for r in rows}
    for metric in ("val_common", "withheld_at_8ep", "withheld_min", "best_val", "best_withheld"):
        if not all(metric in r for r in rows):
            continue
        cells = _cells(rows, metric)
        out[metric] = {}
        for size in SIZES:
            if size not in cells:
                continue
            hours = np.array(sorted(cells[size]))
            mean = np.array([np.mean(cells[size][h]) for h in hours])
            pure = fit_pure(hours, mean)
            off = fit_offset(hours, mean)
            # Ossification proxy: fractional improvement over the last doubling.
            last_gain = float(1 - mean[-1] / mean[-2]) if len(mean) >= 2 else float("nan")
            total_gain = float(1 - mean[-1] / mean[0])
            out[metric][size] = {
                "n_params": params[size],
                "hours": hours.tolist(),
                "mean": mean.tolist(),
                "pure": {"alpha": pure.alpha, "d_c": pure.d_c, "r2": pure.r2},
                "offset": None
                if off is None
                else {"alpha": off.alpha, "d_c": off.d_c, "l_inf": off.l_inf, "r2": off.r2},
                "gain_last_doubling": last_gain,
                "gain_total": total_gain,
            }
    return out


def transfer_summary(path: Path) -> dict | None:
    """
    Transfer cells and paired comparisons.

    Statistics, and why each is what it is:
      * Success per cell is the mean over training seeds; its Wilson interval
        is computed at n = 100, the number of distinct evaluation instances.
        Both seeds face the SAME 100 instances, so pooling them as n = 200
        would double-count instance sampling.
      * The paired difference averages (pretrained - scratch) over seeds
        per instance first, then bootstraps over the 100 instances. It
        captures evaluation noise conditional on these trained policies;
        training-seed variance (2 seeds) is only partially represented.
      * An exact McNemar test on the pooled discordant pairs is reported
        alongside, because a percentile bootstrap cannot reach zero when
        there are only a handful of discordant pairs, all one way.
      * With 21 paired comparisons per model size, the Bonferroni threshold
        is 0.05 / 21; cells passing it are counted and listed.
    """
    if not path.exists():
        return None
    from scipy.stats import binomtest

    rows = json.loads(path.read_text())
    by = defaultdict(list)
    for r in rows:
        by[(float(r["pretrain_hours"]), int(r["budget"]))].append(r)

    n_inst = len(rows[0]["episodes"]) if rows else 0

    def cell_stats(key):
        rs = by.get(key, [])
        if not rs:
            return None
        per_seed = np.array([np.mean(r["episodes"]) for r in rs])
        mean_rate = float(per_seed.mean())
        ci = wilson_interval(int(round(mean_rate * n_inst)), n_inst)
        return {
            "success": mean_rate, "low": ci.low, "high": ci.high,
            "n_instances": n_inst, "seeds": len(rs),
            "per_seed": per_seed.tolist(),
        }

    hours_axis = sorted({h for h, _ in by if h > 0})
    budgets = sorted({b for _, b in by if b > 0})
    summary = {"hours": hours_axis, "budgets": budgets, "cells": {}, "paired": {},
               "n_comparisons": len(hours_axis) * len(budgets)}
    for h in [0.0] + hours_axis:
        for b in [0] + budgets:
            cell = cell_stats((h, b))
            if cell:
                summary["cells"][f"{h}h/{b}"] = cell

    alpha_bonf = 0.05 / max(summary["n_comparisons"], 1)
    for b in budgets:
        for h in hours_axis:
            diffs_by_seed, disc_pos, disc_neg = [], 0, 0
            for r in by.get((h, b), []):
                scratch = [s_ for s_ in by.get((0.0, b), []) if s_["seed"] == r["seed"]]
                if not scratch:
                    continue
                a = np.asarray(r["episodes"], dtype=float)
                bb = np.asarray(scratch[0]["episodes"], dtype=float)
                diffs_by_seed.append(a - bb)
                disc_pos += int(np.sum((a == 1) & (bb == 0)))
                disc_neg += int(np.sum((a == 0) & (bb == 1)))
            if not diffs_by_seed:
                continue
            per_instance = np.mean(diffs_by_seed, axis=0)  # seed-averaged, length n_inst
            d = paired_difference(per_instance, np.zeros_like(per_instance))
            n_disc = disc_pos + disc_neg
            p_mcnemar = float(binomtest(disc_pos, n_disc, 0.5).pvalue) if n_disc else 1.0
            summary["paired"][f"{h}h/{b}"] = {
                "difference": d.point, "low": d.low, "high": d.high, "n_instances": d.n,
                "pretrained_only_wins": disc_pos, "scratch_only_wins": disc_neg,
                "p_mcnemar_exact": p_mcnemar,
                "significant_005": p_mcnemar < 0.05,
                "significant_bonferroni": p_mcnemar < alpha_bonf,
            }
    summary["bonferroni_alpha"] = alpha_bonf
    summary["n_significant_005"] = sum(v["significant_005"] for v in summary["paired"].values())
    summary["n_significant_bonferroni"] = sum(v["significant_bonferroni"] for v in summary["paired"].values())
    return summary


def multitask_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    out = defaultdict(dict)
    for r in rows:
        out[f"{r['size']}/{float(r['hours'])}h"][r["family"]] = {
            "success": r["success"], "low": r["ci_low"], "high": r["ci_high"]
        }
    return dict(out)


def render_markdown(summary: dict) -> str:
    md = ["# S8 results summary\n"]
    md.append("## Data scaling: power-law fits (mean over seeds)\n")
    for metric, title in (("withheld_at_8ep", "withheld family (stack), zero-shot, at 8 epochs (fixed rule)"),
                          ("withheld_min", "withheld family (stack), minimum over training (ORACLE selection on the test family; reported for transparency only)"),
                          ("val_common", "validation on the common held-out set"),
                          ("best_withheld", "withheld family at the best-val checkpoint (raw)"),
                          ("best_val", "own-prefix validation (raw, NOT comparable across scales)")):
        md.append(f"### {title}\n")
        md.append("| size | params | loss @ first | loss @ last | α (pure) | R² | α (offset) | L∞ | gain last doubling |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for size, s in summary["scaling"].get(metric, {}).items():
            off = s["offset"] or {}
            md.append(
                f"| {size} | {s['n_params']:,} | {s['mean'][0]:.4f} | {s['mean'][-1]:.4f} | "
                f"{s['pure']['alpha']:.3f} | {s['pure']['r2']:.3f} | "
                f"{off.get('alpha', float('nan')):.3f} | {off.get('l_inf', float('nan')):.4f} | "
                f"{100*s['gain_last_doubling']:.1f}% |"
            )
        md.append("")
        md.append("Per-doubling fractional improvement (each column is one doubling of data):\n")
        any_size = next(iter(summary["scaling"].get(metric, {}).values()), None)
        if any_size:
            hrs = any_size["hours"]
            md.append("| size | " + " | ".join(f"{hrs[i]:g}→{hrs[i+1]:g} h" for i in range(len(hrs) - 1)) + " |")
            md.append("|---|" + "---|" * (len(hrs) - 1))
            for size, s in summary["scaling"].get(metric, {}).items():
                m_ = s["mean"]
                md.append(f"| {size} | " + " | ".join(f"{100*(1 - m_[i+1]/m_[i]):.1f}%" for i in range(len(m_) - 1)) + " |")
            md.append("")

    for key, title in (("transfer", "580k (l) policy"), ("transfer_xl", "3.26M (xl) policy")):
        t = summary.get(key)
        if not t:
            continue
        md.append(f"## Transfer, {title}: stack success (mean of seeds; Wilson 95% at n = 100 instances)\n")
        header = "| pretraining | " + " | ".join(f"{b} demos" for b in [0] + t["budgets"]) + " |"
        md.append(header)
        md.append("|---|" + "---|" * (len(t["budgets"]) + 1))
        for h in [0.0] + t["hours"]:
            cells = []
            for b in [0] + t["budgets"]:
                c = t["cells"].get(f"{h}h/{b}")
                cells.append("—" if c is None else f"{100*c['success']:.0f}% [{100*c['low']:.0f}, {100*c['high']:.0f}]")
            label = "scratch" if h == 0.0 else f"{h:g} h"
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
        md.append("### Paired difference, pretrained − scratch (seed-averaged per instance; bootstrap over 100 instances; exact McNemar p)\n")
        md.append("| pretraining | " + " | ".join(f"{b} demos" for b in t["budgets"]) + " |")
        md.append("|---|" + "---|" * len(t["budgets"]))
        for h in t["hours"]:
            cells = []
            for b in t["budgets"]:
                p = t["paired"].get(f"{h}h/{b}")
                if p is None:
                    cells.append("—")
                else:
                    star = "**" if p["significant_bonferroni"] else ("*" if p["significant_005"] else "")
                    cells.append(f"{100*p['difference']:+.0f} pts [{100*p['low']:+.0f}, {100*p['high']:+.0f}] p={p['p_mcnemar_exact']:.3f}{star}")
            md.append(f"| {h:g} h | " + " | ".join(cells) + " |")
        md.append("")
        md.append(f"{t['n_significant_005']} of {t['n_comparisons']} comparisons significant at p < 0.05 (*); "
                  f"{t['n_significant_bonferroni']} survive Bonferroni at p < {t['bonferroni_alpha']:.4f} (**).\n")

    if summary.get("multitask"):
        md.append("## Multi-task closed-loop success (seed-0 checkpoints, 50 episodes per family, Wilson half-width ~12 pts)\n")
        md.append("| checkpoint | " + " | ".join(PRETRAIN_FAMILIES) + " | mean |")
        md.append("|---|" + "---|" * (len(PRETRAIN_FAMILIES) + 1))
        for key in sorted(summary["multitask"]):
            fams = summary["multitask"][key]
            vals = [fams.get(f, {}).get("success", float("nan")) for f in PRETRAIN_FAMILIES]
            md.append(f"| {key} | " + " | ".join(f"{100*v:.0f}%" for v in vals) + f" | {100*np.nanmean(vals):.0f}% |")
        md.append("")
    return "\n".join(md)


def main() -> None:
    rows = load_sweep(Path("runs"))
    summary = {
        "n_runs": len(rows),
        "scaling": scaling_fits(rows) if rows else {},
        "transfer": transfer_summary(RESULTS / "transfer.json"),
        "transfer_xl": transfer_summary(RESULTS / "transfer_xl.json"),
        "multitask": multitask_summary(RESULTS / "multitask.json"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    md = render_markdown(summary)
    (RESULTS / "summary.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
