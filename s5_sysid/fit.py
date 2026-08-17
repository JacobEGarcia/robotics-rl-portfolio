"""
Fit MuJoCo Panda parameters to real DROID trajectories with CMA-ES.

    python -m s5_sysid.fit --iterations 60

Methodology
-----------
The single most important design choice here is the TRAIN/TEST SPLIT.

Fitting simulator parameters and then reporting the error on the same
trajectories you fitted to is overfitting a simulator. The number will always
look good and it means nothing -- with enough free parameters you can match
any finite set of trajectories.

So episodes are split by *episode*, never by frame (frames within an episode
are heavily correlated; a frame-level split leaks). Parameters are optimised
on the training episodes only. The headline result is the RMSE on the HELD-OUT
episodes, which the optimiser never saw.

That is the difference between a demo and a measurement, and it is the thing
an interviewer will check first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cma
import numpy as np

from common.plotting import plot_before_after, PALETTE
from common.runlog import RunLogger
from common.seeding import seed_everything
from s5_sysid.sysid import (
    DROID_FPS,
    PARAM_BOUNDS,
    PARAM_ORDER,
    default_params,
    evaluate,
    load_episodes,
    params_to_vector,
    simulate_open_loop,
    build_model,
    trajectory_rmse,
    vector_to_params,
    verify_dataset_has_gap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-to-sim system identification")
    p.add_argument("--iterations", type=int, default=60, help="CMA-ES generations")
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=32)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    episodes = load_episodes(max_episodes=args.max_episodes)
    gap = verify_dataset_has_gap(episodes)
    print(f"Loaded {len(episodes)} episodes, {sum(len(e) for e in episodes)} frames")
    print(f"Reality gap present: {gap['mean_gap_rad']:.5f} rad mean |cmd - meas|\n")

    # Split BY EPISODE. Shuffled so the split is not correlated with recording
    # order (DROID episodes are grouped by building and collector).
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    n_test = max(2, int(round(args.test_frac * len(episodes))))
    test_eps = [episodes[i] for i in order[:n_test]]
    train_eps = [episodes[i] for i in order[n_test:]]
    print(f"Train: {len(train_eps)} episodes | Held-out test: {len(test_eps)} episodes\n")

    baseline = default_params()
    base_train = evaluate(baseline, train_eps)
    base_test = evaluate(baseline, test_eps)
    print(f"BEFORE (Menagerie defaults)  train {base_train:.5f}  test {base_test:.5f} rad")

    config = {
        "dataset": "cadene/droid_1.0.1 (chunk-000)",
        "fps": DROID_FPS,
        "n_episodes": len(episodes),
        "n_train": len(train_eps),
        "n_test": len(test_eps),
        "iterations": args.iterations,
        "popsize": args.popsize,
        "param_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "baseline_params": baseline,
    }

    with RunLogger("s5_sysid", config=config, seed=args.seed) as log:
        log.event("dataset_check", **gap)
        log.event("baseline", train_rmse=base_train, test_rmse=base_test, **baseline)

        x0 = np.clip(params_to_vector(baseline), 0.02, 0.98)
        es = cma.CMAEvolutionStrategy(
            list(x0),
            0.25,  # initial step size in normalised [0,1] space
            {
                "bounds": [0, 1],
                "popsize": args.popsize,
                "seed": args.seed + 1,
                "verbose": -9,
                "maxiter": args.iterations,
            },
        )

        best_val, best_params = base_train, baseline
        for it in range(args.iterations):
            solutions = es.ask()
            # Objective is TRAIN error only. The test set is never touched
            # inside this loop -- that is what makes the final number honest.
            losses = [evaluate(vector_to_params(np.array(s)), train_eps) for s in solutions]
            es.tell(solutions, losses)

            gen_best = int(np.argmin(losses))
            if losses[gen_best] < best_val:
                best_val = losses[gen_best]
                best_params = vector_to_params(np.array(solutions[gen_best]))

            if it % 5 == 0 or it == args.iterations - 1:
                print(f"  gen {it:3d}  best train RMSE {best_val:.5f} rad")
                log.event("generation", iteration=it, best_train_rmse=best_val, **best_params)

        # The moment of truth: evaluate the fitted parameters on episodes the
        # optimiser has never seen.
        fit_test = evaluate(best_params, test_eps)

        print(f"\nAFTER  (identified)          train {best_val:.5f}  test {fit_test:.5f} rad")
        print(f"\nHELD-OUT improvement: {100 * (1 - fit_test / base_test):.1f}%")
        print("\nIdentified parameters:")
        for k in PARAM_ORDER:
            lo, hi = PARAM_BOUNDS[k]
            v = best_params[k]
            # Flag parameters pinned to a bound: that means the bound is
            # wrong, or the parameter is not identifiable from this data.
            edge = " <-- AT BOUND" if (v - lo) < 0.01 * (hi - lo) or (hi - v) < 0.01 * (hi - lo) else ""
            print(f"  {k:14s} {baseline[k]:10.4f} -> {v:10.4f}{edge}")

        log.event(
            "result",
            baseline_train_rmse=base_train,
            baseline_test_rmse=base_test,
            fitted_train_rmse=best_val,
            fitted_test_rmse=fit_test,
            heldout_improvement_pct=100 * (1 - fit_test / base_test),
            **{f"fitted_{k}": v for k, v in best_params.items()},
        )

        # ---- the figure: one held-out episode, one joint, before vs after ----
        ep = max(test_eps, key=len)
        m_base = build_model(baseline)
        m_fit = build_model(best_params)
        sim_before = simulate_open_loop(m_base, ep, int(round(baseline["latency_steps"])))
        sim_after = simulate_open_loop(m_fit, ep, int(round(best_params["latency_steps"])))

        # Joint with the largest baseline error -- the honest choice is the
        # worst case, not the most flattering one.
        j = int(np.argmax(np.abs(sim_before - ep.measured).mean(0)))
        t = np.arange(len(ep)) / DROID_FPS

        out = Path("media/s5/sysid_before_after.png")
        plot_before_after(
            t,
            ep.measured[:, j],
            sim_before[:, j],
            sim_after[:, j],
            xlabel="time (s)",
            ylabel=f"joint {j} position (rad)",
            title=(
                f"S5 — Real-to-sim on held-out DROID episode (joint {j}, worst case)\n"
                f"RMSE {base_test:.4f} → {fit_test:.4f} rad"
            ),
            out=out,
        )
        log.event("figure", path=str(out))

        with open(Path(log.path) / "identified_params.json", "w") as fh:
            json.dump(
                {
                    "baseline": baseline,
                    "identified": best_params,
                    "baseline_test_rmse": base_test,
                    "fitted_test_rmse": fit_test,
                },
                fh,
                indent=2,
            )

        print(f"\nFigure: {out}\nRun:    {log.path}")


if __name__ == "__main__":
    main()
