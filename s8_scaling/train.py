"""
One behaviour-cloning run: (model size, data hours, seed) -> checkpoint.

Training recipe, and why each choice is what it is:

* AdamW, 500-step warmup, cosine decay. The most-standard recipe available
  on purpose: a scaling study's conclusions should survive the reader
  asking "or did you just tune the small runs harder?"  Every cell of the
  grid gets the identical recipe; only capacity and data vary.

* Budget scales with data (30 epoch-equivalents, capped at 150k steps),
  always run to completion, with the best held-in-validation checkpoint
  restored at the end. An earlier draft early-stopped on val
  loss; review showed that made the effective recipe differ by cell - the
  8k floor meant small-data cells could never stop early and always saw
  the full cosine anneal, while overfitting mid-size cells broke out
  mid-anneal. Fixed budget + best-checkpoint selection treats every cell
  identically, which is the property a scaling comparison needs.

* Validation is measured two ways every 1000 steps:
    val        held-out episodes of the five pretraining families
    withheld   episodes of `stack`, a family the model never trains on
  The second is GEN-0's Figure-1 metric (next-action prediction error on
  withheld downstream tasks) at tabletop scale, and is the quantity whose
  power-law behaviour the study is about.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.runlog import RunLogger  # noqa: E402

from .dataset import Split, load_split, normalisation_stats  # noqa: E402
from .model import Policy, save_checkpoint  # noqa: E402

EVAL_EVERY = 1000
BATCH = 256
EVAL_CAP = 60_000  # max timesteps per evaluation subset


def evaluate(policy: Policy, split: Split, idx: np.ndarray) -> float:
    """Mean MSE on a fixed index subset, in inference mode."""
    losses = []
    with torch.no_grad():
        for i in range(0, len(idx), 8192):
            x, y = split.gather(idx[i : i + 8192])
            pred = policy(torch.from_numpy(x.astype(np.float32)))
            losses.append(
                torch.mean((pred - torch.from_numpy(y.astype(np.float32))) ** 2).item()
                * len(x)
            )
    return float(sum(losses) / len(idx))


def train_run(
    *,
    size: str,
    hours: float,
    seed: int,
    corpus: str = "s8_scaling/corpus",
    threads: int = 4,
    max_steps_cap: int = 150_000,
    runs_root: str = "runs",
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    corpus = Path(corpus)
    train = load_split(corpus / "pretrain", hours=hours, val=False)
    val = load_split(corpus / "pretrain", hours=hours, val=True)
    withheld = load_split(corpus / "stack_val", val=False, val_fraction=0.0)

    policy = Policy(size)
    mean, std = normalisation_stats(train)
    policy.set_normalisation(mean, std)

    # 30 epoch-equivalents, capped. No floor: an earlier draft clamped to a
    # minimum of 8k steps, which silently gave the 0.5 h cells sixty epochs -
    # twice the recipe - at exactly the data scale where the small models'
    # curves looked flattest and the largest model's pretraining looked
    # harmful. Found in review; every cell now gets the same epoch budget
    # unless the cap binds (it binds only at 32 h: 17.6 epochs).
    max_steps = int(min(30 * len(train) / BATCH, max_steps_cap))
    warmup = 500
    base_lr = 1e-3

    opt = torch.optim.AdamW(policy.parameters(), lr=base_lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        if step < warmup:
            return base_lr * step / warmup
        t = (step - warmup) / max(max_steps - warmup, 1)
        return 1e-5 + 0.5 * (base_lr - 1e-5) * (1 + math.cos(math.pi * min(t, 1.0)))

    # Fixed evaluation subsets: the same indices every eval, so the val
    # curve moves only when the model does. Each subset gets its OWN
    # generator: drawing both from one generator would make the withheld
    # subset depend on len(val) - i.e. on the data scale - so every point
    # on the hours axis would be scored on a different withheld subset.
    # (Found in review; the withheld pool is the study's headline metric.)
    val_idx = np.random.default_rng(0).choice(
        len(val), size=min(len(val), EVAL_CAP), replace=False
    )
    wh_idx = np.random.default_rng(1).choice(
        len(withheld), size=min(len(withheld), EVAL_CAP), replace=False
    )

    config = {
        "size": size,
        "hours": hours,
        "corpus": str(corpus),
        "batch": BATCH,
        "max_steps": max_steps,
        "base_lr": base_lr,
        "n_params": policy.n_params(),
        "train_samples": len(train),
        "val_samples": len(val),
        "withheld_samples": len(withheld),
    }

    with RunLogger("s8_bc", config=config, seed=seed, root=Path(runs_root)) as log:
        best_val = float("inf")
        best_withheld = float("inf")
        best_step = 0
        best_state = None
        t0 = time.time()
        running = 0.0

        for step in range(1, max_steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y = train.batch(rng, BATCH)
            pred = policy(torch.from_numpy(x.astype(np.float32)))
            loss = torch.mean((pred - torch.from_numpy(y.astype(np.float32))) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()

            if step % EVAL_EVERY == 0:
                policy.eval()
                v = evaluate(policy, val, val_idx)
                w = evaluate(policy, withheld, wh_idx)
                policy.train()
                log.event(
                    "eval",
                    step=step,
                    train_loss=running / EVAL_EVERY,
                    val_loss=v,
                    withheld_loss=w,
                    lr=lr_at(step),
                    sps=round(step * BATCH / (time.time() - t0), 1),
                )
                running = 0.0
                if v < best_val:
                    best_val, best_withheld, best_step = v, w, step
                    best_state = {
                        k: t.detach().clone() for k, t in policy.state_dict().items()
                    }

        if best_state is not None:
            policy.load_state_dict(best_state)
        result = {
            "size": size,
            "hours": hours,
            "seed": seed,
            "n_params": policy.n_params(),
            "train_samples": len(train),
            "best_val": best_val,
            "best_withheld": best_withheld,
            "best_step": best_step,
            "steps_trained": step,
            "wall_seconds": round(time.time() - t0, 1),
        }
        log.event("result", **result)
        save_checkpoint(log.path / "policy.pt", policy, config)
        (log.path / "result.json").write_text(json.dumps(result, indent=2))
        result["run_dir"] = str(log.path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="One BC run of the scaling grid.")
    ap.add_argument("--size", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="s8_scaling/corpus")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    result = train_run(
        size=args.size,
        hours=args.hours,
        seed=args.seed,
        corpus=args.corpus,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
