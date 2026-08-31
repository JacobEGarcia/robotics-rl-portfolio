"""
S9-B: how the effort cost exponent decides which muscles do the work.

50 muscles, 5 DoF. Any torque the leg can produce has a continuum of activation
patterns that produce it, and picking one requires an assumption. Biomechanics
settled on minimising `sum(a^p)`, and argued for decades about p.

    p = 1   a linear program. The optimum sits at a vertex of the feasible
            polytope: the fewest muscles that can do the job, each pushed as
            hard as it will go. Nothing shares load.
    p = 2   least norm. Spreads effort over everything with a moment arm.
    p = 3   Crowninshield and Brand, 1981, reported to predict measured EMG
            better than either neighbour.
    p = 10  approaches min-max: equalise activation, minimise the hardest-
            working muscle.

Two consequences follow from the optimisation rather than from this model,
which makes them real checks: raising p penalises one hard-working muscle more
than several easy ones, so the *peak* activation must fall and the *total*
activation must rise. Both are checked at every exponent.

A third quantity, the count of muscles above a small threshold, is the one
usually quoted and is *not* well behaved. It is reported here anyway, because
watching it turn over is more useful than not measuring it.

This study is the one the synergy study depends on, since the activation
patterns factorised there are p = 3 solutions to the same stance problems.

Run
---
    python -m s9_muscle.redundancy
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
from s9_muscle.control import check_affine, leg_plant, solve_activation  # noqa: E402
from s9_muscle.scene import MEDIA_DIR, load  # noqa: E402
from s9_muscle.synergies import build_dataset  # noqa: E402

SEED = 0
N_SAMPLES = 160
EXPONENTS = (1.0, 2.0, 3.0, 5.0, 10.0)
ACTIVE_THRESHOLD = 0.01


def run(model, data, *, rng) -> dict:
    """Solve the same stance problems at every exponent."""
    dataset = build_dataset(model, data, rng=rng, n=N_SAMPLES)
    per_p: dict[float, dict] = {}
    solutions: dict[float, np.ndarray] = {}

    # One plant object, overwritten per sample from the stored linearisation.
    # Re-posing the model for each exponent would give each of them a freshly
    # computed A that should be identical but need not be bitwise so; reusing
    # the stored one guarantees every exponent sees the same problem.
    plant = leg_plant(model, data)

    for p in EXPONENTS:
        acts, n_active, peak, residual, failed = [], [], [], [], 0
        for i in range(dataset["n_kept"]):
            plant.A[:] = dataset["plant_A"][i]
            plant.c[:] = dataset["plant_c"][i]
            result = solve_activation(plant, dataset["tau"][i], p=p)
            if not result["success"] or result["residual_Nm"] > 1e-5:
                failed += 1
                continue
            acts.append(result["act"])
            n_active.append(result["n_active"])
            peak.append(result["peak_act"])
            residual.append(result["residual_Nm"])

        acts = np.array(acts)
        solutions[p] = acts
        per_p[p] = {
            "n_solved": len(acts),
            "n_failed": failed,
            "mean_muscles_recruited": round(float(np.mean(n_active)), 2),
            "mean_peak_activation": round(float(np.mean(peak)), 4),
            "mean_total_activation": round(float(acts.sum(axis=1).mean()), 3),
            "max_residual_Nm": float(np.max(residual)) if residual else 0.0,
            # Fraction of activations pinned at a bound. p = 1 should be almost
            # entirely at bounds; higher p should not be.
            "fraction_at_bounds": round(
                float(np.mean((acts < 1e-6) | (acts > 1.0 - 1e-6))), 4
            ),
        }

    recruited = [per_p[p]["mean_muscles_recruited"] for p in EXPONENTS]
    peaks = [per_p[p]["mean_peak_activation"] for p in EXPONENTS]
    totals = [per_p[p]["mean_total_activation"] for p in EXPONENTS]
    return {
        "dataset": {k: v for k, v in dataset.items() if not isinstance(v, (np.ndarray, list))},
        "labels": dataset["labels"],
        "per_exponent": {str(p): v for p, v in per_p.items()},
        "solutions": solutions,
        # The two checks that follow from the optimisation.
        "peak_activation_monotone_decreasing": bool(np.all(np.diff(peaks) <= 1e-9)),
        "total_activation_monotone_increasing": bool(np.all(np.diff(totals) >= -1e-9)),
        # The quantity everyone quotes, which is not monotone: it peaks at
        # p = 3 and falls again as the objective approaches min-max.
        "recruitment_monotone_increasing": bool(np.all(np.diff(recruited) >= -1e-9)),
        "recruitment_peaks_at_p": float(EXPONENTS[int(np.argmax(recruited))]),
    }


def make_figure(result, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ps = list(EXPONENTS)
    per = result["per_exponent"]

    ax = axes[0]
    ax.plot(ps, [per[str(p)]["mean_muscles_recruited"] for p in ps], "o-",
            color=PALETTE[0], lw=2, ms=5)
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_xlabel("cost exponent  $p$")
    ax.set_ylabel(f"muscles above {ACTIVE_THRESHOLD}")
    ax.set_title(f"peaks at $p$ = {result['recruitment_peaks_at_p']:g}, then turns over")

    ax = axes[1]
    ax.plot(ps, [per[str(p)]["mean_peak_activation"] for p in ps], "s-",
            color=PALETTE[1], lw=2, ms=5)
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_xlabel("cost exponent  $p$")
    ax.set_ylabel("mean peak activation")
    ax.set_title("peak activation falls monotonically")

    # Where the load actually goes, for the four largest hip and knee muscles.
    ax = axes[2]
    labels = result["labels"]
    show = [labels.index(n) for n in ("recfem_r", "vaslat_r", "glmax2_r", "bflh_r")
            if n in labels]
    width = 0.8 / len(show)
    for j, idx in enumerate(show):
        vals = [result["solutions"][p][:, idx].mean() for p in ps]
        ax.bar(np.arange(len(ps)) + j * width, vals, width,
               color=PALETTE[j % len(PALETTE)], label=labels[idx])
    ax.set_xticks(np.arange(len(ps)) + 0.4 - width / 2)
    ax.set_xticklabels([f"p={p:g}" for p in ps])
    ax.set_ylabel("mean activation")
    ax.set_title("load sharing between synergists")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model, data = load("full", with_floor=False)

    config = {"n_samples": N_SAMPLES, "exponents": list(EXPONENTS),
              "active_threshold": ACTIVE_THRESHOLD}
    with RunLogger("s9_redundancy", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("affine_check", **check_affine(model, data, rng=rng))
        result = run(model, data, rng=rng)
        for p, stats in result["per_exponent"].items():
            log.event("exponent", p=p, **stats)
        log.event("checks", **{k: v for k, v in result.items()
                               if k.startswith(("peak_", "total_", "recruitment_"))})
        figure = make_figure(result, MEDIA_DIR / "redundancy.png")
        run_dir = log.path

    summary = {k: v for k, v in result.items() if k not in ("solutions", "labels")}
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
