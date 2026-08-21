"""
Transfer scaling: does pretraining data buy downstream success?

The experiment GEN-0 stakes its thesis on, at tabletop scale. Policies
pretrained on {0.5 ... 32} hours of the five-family corpus are post-trained
on small budgets {10, 30, 100} of stack demonstrations - a family none of
them ever saw - and evaluated closed-loop. The comparison arm is identical
training from scratch on the same demonstrations.

Fairness rules, fixed before results were seen:

* One post-training recipe for every arm: AdamW, lr 3e-4, 2000 steps,
  cosine decay, batch 256. No per-arm tuning, no early stopping (there is
  no validation set at a 10-demo budget to stop on honestly).
* Identical evaluation: 100 episodes, seeds 500_000..500_099, so every arm
  faces the same 100 sampled stack instances - paired, not marginal.
* Scratch baselines compute observation statistics from their own demo
  budget. Handing them the pretraining statistics would leak information
  the scratch condition is defined not to have.

Run:  python -m s8_scaling.transfer [--size l]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BUDGETS = [10, 30, 100]
EVAL_EPISODES = 100
EVAL_BASE_SEED = 500_000
FT_STEPS = 2000
FT_LR = 3e-4
BATCH = 256


def find_pretrained(runs_root: Path, size: str) -> dict[tuple[float, int], Path]:
    """Map (hours, seed) -> run dir of the pretraining sweep for `size`."""
    out: dict[tuple[float, int], Path] = {}
    for result in runs_root.glob("*_s8_bc_*/result.json"):
        r = json.loads(result.read_text())
        if r["size"] == size:
            out[(float(r["hours"]), int(r["seed"]))] = result.parent
    return out


def _transfer_cell(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "2"
    import numpy as np
    import torch

    from .dataset import load_split, normalisation_stats
    from .evaluate import eval_policy
    from .model import Policy, load_checkpoint

    torch.set_num_threads(2)
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])

    demos = load_split(
        Path(job["corpus"]) / "stack_demos",
        max_episodes=job["budget"],
        val=False,
        val_fraction=0.0,
    )

    if job["pretrain_dir"] is not None:
        policy, _ = load_checkpoint(Path(job["pretrain_dir"]) / "policy.pt")
    else:
        policy = Policy(job["size"])
        mean, std = normalisation_stats(demos)
        policy.set_normalisation(mean, std)

    if job["budget"] > 0:
        opt = torch.optim.AdamW(policy.parameters(), lr=FT_LR, weight_decay=1e-4)
        for step in range(1, FT_STEPS + 1):
            lr = 1e-5 + 0.5 * (FT_LR - 1e-5) * (1 + math.cos(math.pi * step / FT_STEPS))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = demos.batch(rng, BATCH)
            pred = policy(torch.from_numpy(x.astype(np.float32)))
            loss = torch.mean((pred - torch.from_numpy(y.astype(np.float32))) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    ci, _ = eval_policy(
        policy, "stack", n_episodes=EVAL_EPISODES, base_seed=EVAL_BASE_SEED
    )
    return {
        "size": job["size"],
        "pretrain_hours": job["pretrain_hours"],
        "budget": job["budget"],
        "seed": job["seed"],
        "success": ci.point,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "n": ci.n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Transfer scaling on stack.")
    ap.add_argument("--size", default="l")
    ap.add_argument("--corpus", default="s8_scaling/corpus")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="s8_scaling/results/transfer.json")
    args = ap.parse_args()

    pretrained = find_pretrained(Path(args.runs_root), args.size)
    if not pretrained:
        raise SystemExit(f"no pretrained {args.size} runs found - run the sweep first")

    jobs = []
    for (hours, seed), run_dir in sorted(pretrained.items()):
        for budget in [0] + BUDGETS:  # 0 = zero-shot
            jobs.append(
                {
                    "size": args.size,
                    "pretrain_hours": hours,
                    "pretrain_dir": str(run_dir),
                    "budget": budget,
                    "seed": seed,
                    "corpus": args.corpus,
                }
            )
    for seed in range(args.seeds):  # scratch baselines
        for budget in BUDGETS:
            jobs.append(
                {
                    "size": args.size,
                    "pretrain_hours": 0.0,
                    "pretrain_dir": None,
                    "budget": budget,
                    "seed": seed,
                    "corpus": args.corpus,
                }
            )

    print(f"{len(jobs)} transfer cells")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_transfer_cell, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(
                    f"[{len(results)}/{len(jobs)} {time.time()-t0:5.0f}s] "
                    f"pretrain {r['pretrain_hours']:>4}h budget {r['budget']:>3} "
                    f"seed{r['seed']}: {r['success']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
                )
            except Exception as e:  # noqa: BLE001
                print(f"cell {j['pretrain_hours']}h/{j['budget']} FAILED: {e!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
