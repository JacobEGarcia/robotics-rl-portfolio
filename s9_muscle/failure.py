"""
S9-E: what a redundant actuator set actually buys when actuators die.

A machine with 700 soft actuators will not have 700 working actuators for long.
The usual argument for redundancy is that it degrades gracefully, and that
argument is rarely quantified. This measures it two ways on the right leg's 50
muscles, and the two disagree in a way that reverses partway through.

The short version: up to about half the muscles, capability outlasts raw
authority, because real tasks sit well inside the achievable set and do not
need most of it. Past that, capability falls *faster* than authority, because
what is left of the set stops covering the particular directions the tasks
need. Averaged torque authority is therefore a comforting number early and a
misleading one late, which is the wrong way round for a metric you would use
to set a maintenance interval.

    geometric   the achievable joint-torque set. For a direction d the support
                is max over a in [0,1]^n of d^T (A a + c), which has a vertex
                solution and therefore a closed form: switch each a_i to 1 where
                the corresponding entry of A^T d is positive. Averaging the
                support width over many directions is a mean-width measure of
                how much torque authority survives.

    functional  the stance tasks from the synergy study. After k failures, can
                the leg still produce the torque that holding up a body weight
                at that posture requires? Answered by LP feasibility, which is
                the same question the geometric measure answers in aggregate
                but restricted to the torques anybody actually needs.

A failed muscle is modelled as going limp: it contributes no activation-
dependent force, so its column of A is zeroed, but its passive bias stays in c.
Real soft actuators fail that way rather than vanishing.

Checks
------
* The closed-form support must upper-bound any sampled activation, and must be
  attained exactly by the vertex activation. Verified against brute force
  before any of it is used.
* Removing a muscle can never enlarge the achievable set. Verified per muscle.
* Both curves must be monotone in the number of failures, checked *within*
  each trial's nested failure sequence rather than on the means: means over
  independently drawn subsets are not required to be ordered, so checking them
  would be testing the sampler, not the physics.

Run
---
    python -m s9_muscle.failure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import PALETTE  # noqa: E402
from common.runlog import RunLogger, hash_mj_model  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from s9_muscle.control import (  # noqa: E402
    achievable_range,
    check_affine,
    leg_plant,
    muscle_labels,
    solve_activation,
)
from s9_muscle.scene import MEDIA_DIR, load  # noqa: E402
from s9_muscle.synergies import build_dataset  # noqa: E402

SEED = 0
N_DIRECTIONS = 400          # unit directions in the 5-D torque space
FAILURE_COUNTS = (0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30)
TRIALS = 24                 # random failure sets per count
N_TASKS = 60                # stance tasks re-tested for feasibility


def unit_directions(dim: int, n: int, *, rng) -> np.ndarray:
    d = rng.normal(size=(n, dim))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def mean_width(plant, directions: np.ndarray, dead: np.ndarray | None = None) -> float:
    """Mean support width of the achievable torque set over `directions`."""
    A = plant.A if dead is None else np.where(dead[None, :], 0.0, plant.A)
    w = directions @ A                      # (n_dir, n_muscle)
    # Support width along d is the sum of |w| over muscles: the max picks the
    # positive part and the min picks the negative part, and `c` cancels.
    return float(np.mean(np.sum(np.abs(w), axis=1)))


def verify_support(plant, *, rng, n_dir=24, n_samples=4000) -> dict:
    """Brute-force check that the closed-form support is tight and correct."""
    dirs = unit_directions(plant.A.shape[0], n_dir, rng=rng)
    samples = rng.uniform(0.0, 1.0, size=(n_samples, plant.n))
    torques = samples @ plant.A.T + plant.c
    worst_violation, worst_vertex_gap = 0.0, 0.0
    for d in dirs:
        lo, hi = achievable_range(plant, d)
        proj = torques @ d
        worst_violation = max(worst_violation, float(proj.max() - hi), float(lo - proj.min()))
        vertex = (plant.A.T @ d > 0).astype(float)
        worst_vertex_gap = max(
            worst_vertex_gap, abs(float(d @ (plant.A @ vertex + plant.c)) - hi)
        )
    return {
        "n_directions": n_dir,
        "n_random_activations": n_samples,
        "max_sample_exceeding_bound": float(worst_violation),
        "max_vertex_gap": float(worst_vertex_gap),
        "bound_is_valid": bool(worst_violation <= 1e-9),
        "bound_is_tight": bool(worst_vertex_gap <= 1e-9),
    }


def verify_monotone_removal(plant, directions) -> dict:
    """Removing any single muscle must not enlarge the set."""
    full = mean_width(plant, directions)
    worst = 0.0
    for i in range(plant.n):
        dead = np.zeros(plant.n, dtype=bool)
        dead[i] = True
        worst = max(worst, mean_width(plant, directions, dead) - full)
    return {"max_width_increase_from_removal": float(worst),
            "removal_never_enlarges": bool(worst <= 1e-9)}


def criticality(plant, directions, labels) -> list[dict]:
    """Mean-width loss from losing each muscle alone, largest first."""
    full = mean_width(plant, directions)
    rows = []
    for i in range(plant.n):
        dead = np.zeros(plant.n, dtype=bool)
        dead[i] = True
        rows.append({
            "muscle": labels[i],
            "width_loss_pct": round(100.0 * (1.0 - mean_width(plant, directions, dead) / full), 3),
        })
    return sorted(rows, key=lambda r: -r["width_loss_pct"])


def run(model, data, *, rng) -> dict:
    dataset = build_dataset(model, data, rng=rng, n=N_TASKS)
    plant = leg_plant(model, data)
    labels = muscle_labels(model, plant.muscles)
    directions = unit_directions(plant.A.shape[0], N_DIRECTIONS, rng=rng)

    support = verify_support(plant, rng=rng)
    monotone = verify_monotone_removal(plant, directions)
    full_width = mean_width(plant, directions)

    # Functional feasibility uses each task's own plant, since each was posed
    # differently. A single failure mask is applied across all of them, which
    # is what losing a muscle actually means.
    task_A = dataset["plant_A"]
    task_c = dataset["plant_c"]
    task_tau = dataset["tau"]
    n_tasks = dataset["n_kept"]
    scratch = leg_plant(model, data)

    # Nested failure sets: each trial draws one permutation of the muscles and
    # every k takes its first k entries. Independent draws per k would make the
    # curves sample means over *different* subsets, and then a small rise with
    # k is expected noise rather than a violation, which destroys the
    # monotonicity check. Nested sets make monotonicity exact within a trial:
    # a muscle that is dead at k stays dead at k+1, so the feasible set can
    # only shrink.
    widths = np.zeros((TRIALS, len(FAILURE_COUNTS)))
    feasible = np.zeros((TRIALS, len(FAILURE_COUNTS)))
    for trial in range(TRIALS):
        order = rng.permutation(plant.n)
        for col, k in enumerate(FAILURE_COUNTS):
            dead = np.zeros(plant.n, dtype=bool)
            dead[order[:k]] = True
            widths[trial, col] = mean_width(plant, directions, dead) / full_width

            solved = 0
            for i in range(n_tasks):
                scratch.A[:] = np.where(dead[None, :], 0.0, task_A[i])
                scratch.c[:] = task_c[i]
                # p = 1 is the LP: the question is feasibility, not effort, and
                # the LP answers it exactly instead of asking SLSQP to converge.
                r = solve_activation(scratch, task_tau[i], p=1.0)
                solved += int(r["success"] and r["residual_Nm"] < 1e-6)
            feasible[trial, col] = solved / n_tasks

    geometric = [
        {"k": k, "mean_width_fraction": round(float(widths[:, c].mean()), 4),
         "std": round(float(widths[:, c].std()), 4)}
        for c, k in enumerate(FAILURE_COUNTS)
    ]
    functional = [
        {"k": k, "tasks_feasible_fraction": round(float(feasible[:, c].mean()), 4),
         "std": round(float(feasible[:, c].std()), 4)}
        for c, k in enumerate(FAILURE_COUNTS)
    ]
    per_trial_monotone = {
        "geometric": bool(np.all(np.diff(widths, axis=1) <= 1e-12)),
        "functional": bool(np.all(np.diff(feasible, axis=1) <= 1e-12)),
        "trials": TRIALS,
    }

    geo = [g["mean_width_fraction"] for g in geometric]
    fun = [f["tasks_feasible_fraction"] for f in functional]
    return {
        "n_muscles": plant.n,
        "n_dofs": int(plant.A.shape[0]),
        "n_tasks": n_tasks,
        "full_mean_width_Nm": round(full_width, 2),
        "support_check": support,
        "monotone_check": monotone,
        "geometric": geometric,
        "functional": functional,
        "criticality_top": criticality(plant, directions, labels)[:10],
        "per_trial_monotone": per_trial_monotone,
        "geometric_monotone_mean": bool(np.all(np.diff(geo) <= 1e-9)),
        "functional_monotone_mean": bool(np.all(np.diff(fun) <= 1e-9)),
        # The comparison the study exists for.
        "width_at_k5": geo[FAILURE_COUNTS.index(5)],
        "feasible_at_k5": fun[FAILURE_COUNTS.index(5)],
        "width_at_k12": geo[FAILURE_COUNTS.index(12)],
        "feasible_at_k12": fun[FAILURE_COUNTS.index(12)],
        # Where feasibility stops outrunning authority and starts trailing it.
        "crossover_k": next(
            (k for k, g, f in zip(FAILURE_COUNTS, geo, fun) if k > 0 and f < g), None
        ),
        "single_muscle_worst_loss_pct": max(
            r["width_loss_pct"] for r in criticality(plant, directions, labels)
        ),
    }


def make_figure(result, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    ks = [g["k"] for g in result["geometric"]]

    ax = axes[0]
    geo = np.array([g["mean_width_fraction"] for g in result["geometric"]])
    geo_sd = np.array([g["std"] for g in result["geometric"]])
    fun = np.array([f["tasks_feasible_fraction"] for f in result["functional"]])
    fun_sd = np.array([f["std"] for f in result["functional"]])
    ax.plot(ks, geo, "o-", color=PALETTE[0], lw=2, ms=4, label="torque authority (mean width)")
    ax.fill_between(ks, geo - geo_sd, geo + geo_sd, color=PALETTE[0], alpha=0.18)
    ax.plot(ks, fun, "s-", color=PALETTE[2], lw=2, ms=4, label="stance tasks still feasible")
    ax.fill_between(ks, fun - fun_sd, fun + fun_sd, color=PALETTE[2], alpha=0.18)
    ax.set_xlabel(f"muscles disabled (of {result['n_muscles']})")
    ax.set_ylabel("fraction remaining")
    ax.set_ylim(0, 1.05)
    ax.set_title("capability outlasts authority, then stops doing so")
    ax.legend(fontsize=8)

    ax = axes[1]
    top = result["criticality_top"][::-1]
    ax.barh([r["muscle"] for r in top], [r["width_loss_pct"] for r in top], color=PALETTE[1])
    ax.set_xlabel("mean-width loss from losing this one muscle (%)")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("redundancy is not uniform")

    ax = axes[2]
    ax.plot(geo, fun, "o-", color=PALETTE[3], lw=1.8, ms=4)
    for g, f, k in zip(geo, fun, ks):
        if k in (0, 5, 12, 20, 30):
            ax.annotate(f"k={k}", (g, f), fontsize=7, xytext=(4, -8),
                        textcoords="offset points")
    ax.plot([0, 1], [0, 1], color="0.7", lw=1.0, ls=":")
    ax.set_xlabel("torque authority remaining")
    ax.set_ylabel("stance tasks feasible")
    ax.set_title("above the diagonal, authority understates capability")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model, data = load("full", with_floor=False)

    config = {"n_directions": N_DIRECTIONS, "failure_counts": list(FAILURE_COUNTS),
              "trials": TRIALS, "n_tasks": N_TASKS}
    with RunLogger("s9_failure", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("affine_check", **check_affine(model, data, rng=rng))
        result = run(model, data, rng=rng)
        log.event("support_check", **result["support_check"])
        log.event("monotone_check", **result["monotone_check"])
        for g, f in zip(result["geometric"], result["functional"]):
            # Both dicts carry a `std`, so they cannot be splatted into one
            # event without collision. Name them for what they measure.
            log.event(
                "failures",
                k=g["k"],
                mean_width_fraction=g["mean_width_fraction"],
                mean_width_std=g["std"],
                tasks_feasible_fraction=f["tasks_feasible_fraction"],
                tasks_feasible_std=f["std"],
            )
        figure = make_figure(result, MEDIA_DIR / "failure.png")
        run_dir = log.path

    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("geometric", "functional", "criticality_top")}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
