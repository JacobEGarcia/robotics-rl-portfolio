"""
The scaling grid: 5 model sizes x 7 data scales x 2 seeds = 70 runs.

Step budget is 30 epoch-equivalents, capped at 150k steps. The cap binds
only at 32 h (15.5 epochs); every other cell trains its full 30 epochs.
(A first pass used a 60k cap, which bound at 16 h and 32 h - 14 and 7
epochs - and those cells were rerun with the higher cap, so the recipe is
uniform everywhere the claims are made.)

Scheduling: a process pool of 6 workers, 5 torch threads each (30 threads
on a 32-core box, leaving headroom for the OS). Runs are dispatched
largest-first - the xl/32h cells are the long poles, and bin-packing them
first keeps the tail of the sweep short. Each run is an independent
`train.train_run` invocation writing its own run directory; the sweep holds
no shared state, so a crashed run costs one cell, not the grid.

Completed cells are detected by scanning existing run manifests and skipped,
which makes the sweep resumable: rerunning this script after an interruption
does only the missing work.

Run:  python -m s8_scaling.sweep [--seeds 2]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HOURS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
SIZES = ["t", "s", "m", "l", "xl"]

# Rough relative cost of one training step per size, for largest-first order.
COST = {"t": 1, "s": 2, "m": 5, "l": 14, "xl": 45}


def _run_cell(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    from s8_scaling.train import train_run

    return train_run(
        size=job["size"],
        hours=job["hours"],
        seed=job["seed"],
        corpus=job["corpus"],
        threads=job["threads"],
        runs_root=job["runs_root"],
        max_steps_cap=job["max_steps_cap"],
    )


def completed_cells(runs_root: Path) -> set[tuple[str, float, int]]:
    done = set()
    for result in runs_root.glob("*_s8_bc_*/result.json"):
        try:
            r = json.loads(result.read_text())
            done.add((r["size"], float(r["hours"]), int(r["seed"])))
        except (KeyError, json.JSONDecodeError):
            continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the BC scaling grid.")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--corpus", default="s8_scaling/corpus")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--sizes", default=",".join(SIZES), help="comma-separated subset")
    ap.add_argument("--hours", default=",".join(str(h) for h in HOURS), help="comma-separated subset")
    ap.add_argument("--max-steps-cap", type=int, default=150_000)
    ap.add_argument(
        "--force", action="store_true",
        help="rerun selected cells even if complete, deleting their old run dirs first",
    )
    args = ap.parse_args()

    sizes = [x for x in args.sizes.split(",") if x]
    hours_list = [float(x) for x in args.hours.split(",") if x]

    if args.force:
        # Old run dirs are removed rather than shadowed: runs/ is what the
        # published numbers are computed from, and should contain exactly
        # the runs that produced them.
        import shutil

        for result in Path(args.runs_root).glob("*_s8_bc_*/result.json"):
            r = json.loads(result.read_text())
            if r["size"] in sizes and float(r["hours"]) in hours_list and int(r["seed"]) < args.seeds:
                shutil.rmtree(result.parent)

    done = completed_cells(Path(args.runs_root))
    jobs = [
        {
            "size": size,
            "hours": hours,
            "seed": seed,
            "corpus": args.corpus,
            "threads": args.threads,
            "runs_root": args.runs_root,
            "max_steps_cap": args.max_steps_cap,
        }
        for size in sizes
        for hours in hours_list
        for seed in range(args.seeds)
        if (size, hours, seed) not in done
    ]
    jobs.sort(key=lambda j: -COST[j["size"]] * min(j["hours"], 8))
    print(f"{len(done)} cells already complete; {len(jobs)} to run")

    t0 = time.time()
    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_cell, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            finished += 1
            try:
                r = fut.result()
                print(
                    f"[{finished}/{len(jobs)} {time.time()-t0:6.0f}s] "
                    f"{j['size']:>2}/{j['hours']:<4}h seed{j['seed']} "
                    f"val {r['best_val']:.4f}  withheld {r['best_withheld']:.4f}  "
                    f"({r['steps_trained']} steps, {r['wall_seconds']:.0f}s)"
                )
            except Exception as e:  # noqa: BLE001 - a dead cell must not kill the grid
                print(f"[{finished}/{len(jobs)}] {j['size']}/{j['hours']}h seed{j['seed']} FAILED: {e!r}")

    print(f"sweep done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
