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
    for metric in ("val_common", "withheld_min", "best_val", "best_withheld"):
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
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    by = defaultdict(list)
    for r in rows:
        by[(float(r["pretrain_hours"]), int(r["budget"]))].append(r)

    def pooled(key):
        rs = by.get(key, [])
        if not rs:
            return None
        eps = np.concatenate([r["episodes"] for r in rs])
        ci = wilson_interval(int(eps.sum()), len(eps))
        return {"success": ci.point, "low": ci.low, "high": ci.high, "n": ci.n, "seeds": len(rs)}

    hours_axis = sorted({h for h, _ in by if h > 0})
    budgets = sorted({b for _, b in by if b > 0})
    summary = {"hours": hours_axis, "budgets": budgets, "cells": {}, "paired": {}}
    for h in [0.0] + hours_axis:
        for b in [0] + budgets:
            cell = pooled((h, b))
            if cell:
                summary["cells"][f"{h}h/{b}"] = cell

    # Paired: pretrained (each hours) minus scratch, same budget, matched by
    # seed so that both arms faced the identical 100 instances.
    for b in budgets:
        for h in hours_axis:
            a_all, b_all = [], []
            for r in by.get((h, b), []):
                scratch = [s for s in by.get((0.0, b), []) if s["seed"] == r["seed"]]
                if not scratch:
                    continue
                a_all.append(r["episodes"])
                b_all.append(scratch[0]["episodes"])
            if a_all:
                a = np.concatenate(a_all)
                bb = np.concatenate(b_all)
                d = paired_difference(a, bb)
                summary["paired"][f"{h}h/{b}"] = {
                    "difference": d.point,
                    "low": d.low,
                    "high": d.high,
                    "n_pairs": d.n,
                    # Discordant pairs are what carries the paired evidence.
                    "pretrained_only_wins": int(np.sum((a == 1) & (bb == 0))),
                    "scratch_only_wins": int(np.sum((a == 0) & (bb == 1))),
                }
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
    for metric, title in (("withheld_min", "withheld family (stack), zero-shot, best over training"),
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

    if summary.get("transfer"):
        t = summary["transfer"]
        md.append("## Transfer: stack success (pooled over seeds, Wilson 95% CI)\n")
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
        md.append("### Paired difference, pretrained − scratch (same instances, same seed)\n")
        md.append("| pretraining | " + " | ".join(f"{b} demos" for b in t["budgets"]) + " |")
        md.append("|---|" + "---|" * len(t["budgets"]))
        for h in t["hours"]:
            cells = []
            for b in t["budgets"]:
                p = t["paired"].get(f"{h}h/{b}")
                cells.append("—" if p is None else f"{100*p['difference']:+.0f} pts [{100*p['low']:+.0f}, {100*p['high']:+.0f}]")
            md.append(f"| {h:g} h | " + " | ".join(cells) + " |")
        md.append("")

    if summary.get("multitask"):
        md.append("## Multi-task closed-loop success (50 episodes per family)\n")
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
