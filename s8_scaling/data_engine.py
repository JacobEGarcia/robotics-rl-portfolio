"""
The data engine: parallel demonstration collection with a printed data card.

Layout on disk (one corpus = one directory):

    corpus/<name>/
        obs.npy        float32 (N, 69)   all timesteps, all episodes
        act.npy        float32 (N, 5)
        episodes.npy   int64   (E, 4)    start, length, family_index, seed
        card.json      the data card: counts, hours, rejection rates, params

Two npy files rather than per-episode files or npz archives because
`np.load(..., mmap_mode="r")` then works: six concurrent training runs can
share one corpus through the OS page cache instead of each holding a copy.

Retention policy: only successful demonstrations are kept - behaviour
cloning on failed attempts teaches failure. Rejection is not hidden: the
card records attempts per family, and the write-up reports them, because a
data engine's rejection rate is a fact about the data distribution.

Seed ranges (documented so no range can collide with another):
    eval seeds                < 1_000_000
    pretrain corpus           1_000_000 + family_index * 10_000_000 + i
    transfer (stack) demos    100_000_000 + i
    withheld-val (stack)      110_000_000 + i
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .tasks import FAMILIES, PRETRAIN_FAMILIES

SECONDS_PER_STEP = 0.05  # 20 Hz control

_ENV = None
_EXPERT = None


def _worker_init() -> None:
    """One environment per worker process, built once."""
    global _ENV, _EXPERT
    # Each worker is a single-threaded simulator; keep BLAS at one thread so
    # 28 workers do not fight over 32 cores.
    os.environ["OMP_NUM_THREADS"] = "1"
    from .env import TabletopEnv
    from .expert import ScriptedExpert

    _ENV = TabletopEnv()
    _EXPERT = ScriptedExpert()


def _collect_chunk(job: tuple[str, list[int]]) -> dict:
    """Run the expert over a list of seeds; return successful episodes."""
    family, seeds = job
    from .tasks import sample_instance

    obs_parts, act_parts, meta = [], [], []
    attempts = 0
    for seed in seeds:
        attempts += 1
        inst = sample_instance(family, seed)
        obs, _ = _ENV.reset_to(inst)
        _EXPERT.rng = np.random.default_rng(seed ^ 0x5EED)
        _EXPERT.reset()
        ep_obs, ep_act = [], []
        info = {"success": False}
        for _ in range(inst.horizon):
            action = _EXPERT.act(_ENV)
            ep_obs.append(obs)
            ep_act.append(action.astype(np.float32))
            obs, _, term, trunc, info = _ENV.step(action)
            if term or trunc:
                break
        if info["success"]:
            obs_parts.append(np.asarray(ep_obs, dtype=np.float32))
            act_parts.append(np.asarray(ep_act, dtype=np.float32))
            meta.append((len(ep_obs), FAMILIES.index(family), seed))
    return {
        "family": family,
        "attempts": attempts,
        "obs": obs_parts,
        "act": act_parts,
        "meta": meta,
    }


def collect_corpus(
    out_dir: str | Path,
    *,
    families: tuple[str, ...] = PRETRAIN_FAMILIES,
    target_hours: float = 32.0,
    seed_base_fn=None,
    workers: int = 28,
    chunk: int = 64,
) -> dict:
    """
    Collect at least `target_hours` of successful demonstrations, spread
    evenly across `families`, and write the corpus directory.

    Episodes are collected in interleaved seed order and stored family-
    round-robin, so "the first H hours" of this corpus is a well-defined,
    family-balanced subset - the property that makes the data-scaling axis
    a chain of strict prefixes rather than six unrelated datasets.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if seed_base_fn is None:
        seed_base_fn = lambda fam: 1_000_000 + FAMILIES.index(fam) * 10_000_000

    # Adaptive chunk size: whatever is in flight when the target is reached
    # still gets collected, so in-flight work (workers x chunk episodes)
    # bounds the overshoot. Small targets get small chunks.
    est_episodes = target_hours * 3600 / 7.0  # ~7 s per successful episode
    chunk = int(np.clip(est_episodes / (workers * 4), 8, chunk))

    t0 = time.time()
    per_family_hours = target_hours / len(families)
    results: dict[str, dict] = {
        f: {"obs": [], "act": [], "meta": [], "attempts": 0, "hours": 0.0}
        for f in families
    }

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        next_seed = {f: seed_base_fn(f) for f in families}
        pending = {}

        def submit(fam):
            seeds = list(range(next_seed[fam], next_seed[fam] + chunk))
            next_seed[fam] += chunk
            fut = pool.submit(_collect_chunk, (fam, seeds))
            pending[fut] = fam

        for fam in families:
            for _ in range(max(1, workers // len(families))):
                submit(fam)

        from concurrent.futures import FIRST_COMPLETED, wait

        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                fam = pending.pop(fut)
                r = fut.result()
                acc = results[fam]
                acc["attempts"] += r["attempts"]
                acc["obs"] += r["obs"]
                acc["act"] += r["act"]
                acc["meta"] += r["meta"]
                acc["hours"] = (
                    sum(m[0] for m in acc["meta"]) * SECONDS_PER_STEP / 3600.0
                )
                if acc["hours"] < per_family_hours:
                    submit(fam)
            hours_now = sum(a["hours"] for a in results.values())
            print(
                f"\r  {hours_now:6.2f} / {target_hours} hours "
                f"({time.time()-t0:5.0f}s elapsed)",
                end="",
                flush=True,
            )
    print()

    # Interleave families round-robin so any prefix is balanced.
    order = []
    counters = {f: 0 for f in families}
    while True:
        advanced = False
        for f in families:
            if counters[f] < len(results[f]["meta"]):
                order.append((f, counters[f]))
                counters[f] += 1
                advanced = True
        if not advanced:
            break

    obs_all, act_all, episodes = [], [], []
    cursor = 0
    for f, i in order:
        o, a = results[f]["obs"][i], results[f]["act"][i]
        length, fam_idx, seed = results[f]["meta"][i]
        obs_all.append(o)
        act_all.append(a)
        episodes.append((cursor, length, fam_idx, seed))
        cursor += length

    obs_arr = np.concatenate(obs_all, axis=0)
    act_arr = np.concatenate(act_all, axis=0)
    ep_arr = np.asarray(episodes, dtype=np.int64)
    np.save(out_dir / "obs.npy", obs_arr)
    np.save(out_dir / "act.npy", act_arr)
    np.save(out_dir / "episodes.npy", ep_arr)

    card = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "families": list(families),
        "target_hours": target_hours,
        "hours": float(cursor * SECONDS_PER_STEP / 3600.0),
        "episodes": int(len(episodes)),
        "transitions": int(cursor),
        "wall_seconds": round(time.time() - t0, 1),
        "realtime_factor": round(cursor * SECONDS_PER_STEP / max(time.time() - t0, 1e-9), 1),
        "per_family": {
            f: {
                "episodes": len(results[f]["meta"]),
                "attempts": results[f]["attempts"],
                "success_rate": round(
                    len(results[f]["meta"]) / max(results[f]["attempts"], 1), 4
                ),
                "hours": round(results[f]["hours"], 3),
            }
            for f in families
        },
        "bytes": int(obs_arr.nbytes + act_arr.nbytes),
    }
    (out_dir / "card.json").write_text(json.dumps(card, indent=2))
    return card


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Collect demonstration corpora.")
    ap.add_argument("--out", default="s8_scaling/corpus")
    ap.add_argument("--hours", type=float, default=32.0)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()

    root = Path(args.out)
    print(f"pretrain corpus -> {root/'pretrain'}")
    card = collect_corpus(
        root / "pretrain", target_hours=args.hours, workers=args.workers
    )
    print(json.dumps(card["per_family"], indent=2))

    print(f"withheld-family validation corpus (stack) -> {root/'stack_val'}")
    collect_corpus(
        root / "stack_val",
        families=("stack",),
        target_hours=1.0,
        workers=args.workers,
        seed_base_fn=lambda fam: 110_000_000,
    )

    print(f"transfer demonstration pool (stack) -> {root/'stack_demos'}")
    collect_corpus(
        root / "stack_demos",
        families=("stack",),
        target_hours=1.5,
        workers=args.workers,
        seed_base_fn=lambda fam: 100_000_000,
    )


if __name__ == "__main__":
    main()
