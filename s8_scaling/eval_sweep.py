"""
Closed-loop evaluation of the pretrained multi-task policies.

Prediction error is the scaling-law metric, but a robot is judged in
closed loop; this measures actual task success of the swept checkpoints
across the five pretraining families, at every (size, hours) cell of the
grid, 50 episodes per family with the identical seed block per cell.

Output rows: {size, hours, seed, family, success, ci_low, ci_high}.

Run:  python -m s8_scaling.eval_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .tasks import PRETRAIN_FAMILIES

EVAL_EPISODES = 50
EVAL_BASE_SEED = 600_000


def _eval_cell(job: dict) -> list[dict]:
    os.environ["OMP_NUM_THREADS"] = "2"
    from .env import TabletopEnv
    from .evaluate import eval_policy
    from .model import load_checkpoint

    policy, _ = load_checkpoint(Path(job["run_dir"]) / "policy.pt")
    env = TabletopEnv()
    rows = []
    for family in PRETRAIN_FAMILIES:
        ci, records = eval_policy(
            policy, family, n_episodes=EVAL_EPISODES, base_seed=EVAL_BASE_SEED, env=env
        )
        rows.append(
            {
                "size": job["size"],
                "hours": job["hours"],
                "seed": job["seed"],
                "family": family,
                "success": ci.point,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "n": ci.n,
                "episodes": [int(bool(r["success"])) for r in records],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop eval of the sweep grid.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="which sweep seed to evaluate")
    ap.add_argument("--out", default="s8_scaling/results/multitask.json")
    args = ap.parse_args()

    jobs = []
    for result in Path(args.runs_root).glob("*_s8_bc_*/result.json"):
        r = json.loads(result.read_text())
        if int(r["seed"]) != args.seed:
            continue
        jobs.append(
            {
                "size": r["size"],
                "hours": float(r["hours"]),
                "seed": int(r["seed"]),
                "run_dir": str(result.parent),
            }
        )
    print(f"{len(jobs)} checkpoints x {len(PRETRAIN_FAMILIES)} families")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_eval_cell, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                cell = fut.result()
                rows.extend(cell)
                mean = sum(c["success"] for c in cell) / len(cell)
                print(
                    f"[{time.time()-t0:5.0f}s] {j['size']:>2}/{j['hours']:<4}h "
                    f"mean success {mean:.2f}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"{j['size']}/{j['hours']}h FAILED: {e!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
