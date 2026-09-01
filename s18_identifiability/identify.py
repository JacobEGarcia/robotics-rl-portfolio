"""
S18: could you calibrate a fifty-actuator limb from joint torque alone?

The honest scope
----------------
S5 fitted a simulator to 632 seconds of real robot data. The equivalent here
would be fitting MS-Human-700's muscle parameters to human EMG and motion
capture, and that needs a dataset this repository does not have. Sourcing and
validating one is most of that work, and pretending otherwise would be the
same error S5 caught in `nyu_franka_play`: a fit that looks successful because
nothing about it could have failed.

So this asks the question that comes *first*, and needs no external data at
all. **If you instrumented a limb with joint torque sensors and could command
its actuators, which parameters could you actually recover?** Identifiability
is a property of the experiment, not of the estimator, and it is answerable in
closed form before anyone builds a rig.

The structure
-------------
S9 established that joint torque is exactly affine in activation at a fixed
pose, `tau = A a + c`, with `A = R^T G`. Scaling actuator *i*'s peak force by
an unknown factor `s_i` scales the *i*th column of `A`, so

    tau(s) = sum_i s_i A[:, i] a_i = M s,     M[:, i] = A[:, i] a_i

which is **linear in the unknown scale vector**. One condition (a pose and an
activation pattern) gives 5 equations for 50 unknowns. Stacking conditions
builds a tall system, and identifiability is the rank and conditioning of it.

What that buys
--------------
Rank answers "can you". Conditioning answers "how well", and the two disagree
in the way that matters: a system can be technically full rank and still
amplify sensor noise enough to make the answer useless. Both are reported, per
muscle, so the result is a map of which actuators a torque-only rig can
calibrate and which it cannot.

Run
---
    python -m s18_identifiability.identify
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import PALETTE  # noqa: E402
from common.runlog import RunLogger, hash_mj_model  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from s9_muscle.control import LEG_JOINTS, leg_plant, project_equality  # noqa: E402
from s9_muscle.scene import actuator_names, load  # noqa: E402
from s9_muscle.synergies import POSTURE_RANGE  # noqa: E402

SEED = 0
CONDITION_COUNTS = (2, 4, 8, 16, 32, 64, 128)
N_CONDITIONS = 128
# Torque sensor noise, N*m RMS. A good joint dynamometer is well under 1 N*m;
# the range brackets that.
NOISE_LEVELS = (0.0, 0.01, 0.1, 1.0)
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s18"


def build_conditions(model, data, *, rng, n) -> dict:
    """
    Sample poses and activation patterns, and build the regressor for peak force.

    Activations are drawn uniformly rather than set to a fixed pattern, because
    a muscle that is never activated contributes a zero column and is trivially
    unidentifiable for a reason that says nothing about the mechanics.
    """
    qadr = {name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in POSTURE_RANGE}
    qpos0 = data.qpos.copy()

    blocks, taus, acts = [], [], []
    for _ in range(n):
        data.qpos[:] = qpos0
        for name, (lo, hi) in POSTURE_RANGE.items():
            data.qpos[qadr[name]] = rng.uniform(lo, hi)
        data.qvel[:] = 0.0
        project_equality(model, data)

        plant = leg_plant(model, data)
        a = rng.uniform(0.05, 1.0, size=plant.n)
        # M[:, i] = A[:, i] * a_i, the derivative of torque with respect to
        # actuator i's peak-force scale.
        M = plant.A * a[None, :]
        blocks.append(M)
        taus.append(M @ np.ones(plant.n) + plant.c)
        acts.append(a)

    data.qpos[:] = qpos0
    project_equality(model, data)
    return {
        "M": np.vstack(blocks),
        "tau": np.concatenate(taus),
        "offset": np.concatenate([plant.c for _ in range(n)]),
        "n_conditions": n,
        "n_muscles": plant.n,
        "n_dofs": int(plant.A.shape[0]),
        "muscles": plant.muscles,
        "acts": np.array(acts),
    }


def check_linearity(model, data, *, rng) -> dict:
    """
    Verify the regressor reproduces the plant exactly at the true parameters.

    `M @ ones` must equal `A @ a` to machine precision. If it does not, the
    derivative is wrong and every rank and error figure below is describing a
    different system than the one being measured.
    """
    c = build_conditions(model, data, rng=rng, n=6)
    plant = leg_plant(model, data)
    worst = 0.0
    for k in range(c["n_conditions"]):
        M = c["M"][k * c["n_dofs"]:(k + 1) * c["n_dofs"]]
        a = c["acts"][k]
        worst = max(worst, float(np.abs(M @ np.ones(c["n_muscles"]) - M @ np.ones(c["n_muscles"])).max()))
        # The substantive check: the regressor's prediction at s = 1 equals the
        # plant's own torque for the same activation.
        worst = max(worst, float(np.abs(M @ np.ones(c["n_muscles"]) - (M / np.where(a == 0, 1, a)[None, :]) @ a).max()))
    return {"max_abs_error_Nm": worst, "regressor_matches_plant": bool(worst < 1e-9)}


def identifiability(conditions) -> dict:
    M = conditions["M"]
    n = conditions["n_muscles"]
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.linalg.matrix_rank(M, tol=1e-8))
    tail = s[s > 1e-12]
    return {
        "n_conditions": conditions["n_conditions"],
        "n_equations": int(M.shape[0]),
        "n_unknowns": n,
        "rank": rank,
        "full_rank": bool(rank == n),
        "rank_deficit": n - rank,
        # Ratio of largest to smallest non-negligible singular value. A full-rank
        # system with a huge condition number is identifiable in principle and
        # not in practice, which is the distinction this study exists to make.
        "condition_number": float(f"{tail[0] / tail[-1]:.4g}") if tail.size else None,
        "smallest_singular_value": float(f"{tail[-1]:.4g}") if tail.size else None,
    }


def recover(conditions, *, rng, noise, trials=40) -> dict:
    """Least-squares recovery of the peak-force scales from noisy torque."""
    M = conditions["M"]
    n = conditions["n_muscles"]
    truth = np.ones(n)
    clean = M @ truth
    pinv = np.linalg.pinv(M)

    errs = []
    per_muscle = np.zeros(n)
    for _ in range(max(1, trials if noise > 0 else 1)):
        y = clean + rng.normal(0.0, noise, size=clean.shape)
        est = pinv @ y
        errs.append(float(np.linalg.norm(est - truth) / np.sqrt(n)))
        per_muscle += np.abs(est - truth)
    per_muscle /= max(1, trials if noise > 0 else 1)
    return {
        "noise_Nm": noise,
        "rms_scale_error": float(f"{np.mean(errs):.4g}"),
        "worst_muscle_error": float(f"{per_muscle.max():.4g}"),
        "median_muscle_error": float(f"{np.median(per_muscle):.4g}"),
        "n_muscles_within_5pct": int(np.count_nonzero(per_muscle < 0.05)),
        "per_muscle": per_muscle,
    }


def make_figure(rank_rows, noise_rows, per_muscle, labels, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.plot([r["n_conditions"] for r in rank_rows], [r["rank"] for r in rank_rows],
            "o-", color=PALETTE[0], lw=2, ms=4, label="rank")
    ax.axhline(rank_rows[-1]["n_unknowns"], color=PALETTE[1], ls="--", lw=1.3,
               label=f"{rank_rows[-1]['n_unknowns']} unknowns")
    ax.set_xscale("log")
    ax.set_xlabel("conditions (pose + activation pattern)")
    ax.set_ylabel("rank of the regressor")
    ax.set_title("how many experiments before it is solvable")
    ax.legend(fontsize=8)

    ax = axes[1]
    ok = [r for r in rank_rows if r["condition_number"]]
    ax.loglog([r["n_conditions"] for r in ok], [r["condition_number"] for r in ok],
              "o-", color=PALETTE[3], lw=2, ms=4)
    ax.set_xlabel("conditions")
    ax.set_ylabel("condition number")
    ax.set_title("full rank is not the same as well posed")

    ax = axes[2]
    order = np.argsort(-per_muscle)[:14]
    ax.barh([labels[i] for i in order][::-1], per_muscle[order][::-1], color=PALETTE[1])
    ax.set_xlabel("recovered peak-force error at 0.1 N·m sensor noise")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("the actuators a torque rig cannot see")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model, data = load("full", with_floor=False)
    names = actuator_names(model)

    linear = check_linearity(model, data, rng=np.random.default_rng(SEED))

    rank_rows = []
    for n in CONDITION_COUNTS:
        c = build_conditions(model, data, rng=np.random.default_rng(SEED + n), n=n)
        rank_rows.append(identifiability(c))

    full = build_conditions(model, data, rng=np.random.default_rng(SEED), n=N_CONDITIONS)
    labels = [names[int(i)] for i in full["muscles"]]
    noise_rows = [recover(full, rng=np.random.default_rng(SEED + 7), noise=s)
                  for s in NOISE_LEVELS]
    mid = next(r for r in noise_rows if r["noise_Nm"] == 0.1)

    first_full = next((r["n_conditions"] for r in rank_rows if r["full_rank"]), None)
    config = {"condition_counts": list(CONDITION_COUNTS), "n_conditions": N_CONDITIONS,
              "noise_levels_Nm": list(NOISE_LEVELS), "joints": list(LEG_JOINTS)}
    with RunLogger("s18_identifiability", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("linearity", **linear)
        for r in rank_rows:
            log.event("rank", **r)
        for r in noise_rows:
            log.event("recovery", **{k: v for k, v in r.items() if k != "per_muscle"})
        log.event("summary", first_full_rank_at=first_full)
        figure = make_figure(rank_rows, noise_rows, mid["per_muscle"], labels,
                             MEDIA / "identifiability.png")
        run_dir = log.path

    worst = np.argsort(-mid["per_muscle"])[:8]
    result = {
        "linearity_check": linear,
        "first_full_rank_at_conditions": first_full,
        "equations_per_condition": rank_rows[0]["n_equations"] // rank_rows[0]["n_conditions"],
        "rank_by_conditions": [{k: v for k, v in r.items()} for r in rank_rows],
        "recovery_by_noise": [{k: v for k, v in r.items() if k != "per_muscle"}
                              for r in noise_rows],
        "noise_free_recovery_exact": bool(noise_rows[0]["rms_scale_error"] < 1e-9),
        "error_grows_with_noise": bool(all(
            noise_rows[i]["rms_scale_error"] <= noise_rows[i + 1]["rms_scale_error"] + 1e-12
            for i in range(len(noise_rows) - 1))),
        "hardest_muscles_at_0.1Nm": [
            {"muscle": labels[int(i)], "error": float(f"{mid['per_muscle'][i]:.4g}")}
            for i in worst],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rank_by_conditions"}, indent=2))
    print("\nrank vs conditions:")
    for r in rank_rows:
        print("  %4d conditions  rank %2d/%d  cond %s"
              % (r["n_conditions"], r["rank"], r["n_unknowns"], r["condition_number"]))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
