"""
S9-A: how many synergies does it take to control 50 leg muscles?

The question
------------
The moment-arm matrix has a 635-dimensional null space, so "which muscles do
you fire" has enormously many answers for any given motion. Biology's apparent
answer is that it does not search that space: human EMG during walking is
well approximated by a handful of fixed activation patterns scaled up and down,
usually reported as four to six. If effort-optimal control of this model lands
in the same range, then the low dimensionality is a property of the mechanics
rather than a property of the nervous system, and a controller for a
many-actuator machine can exploit it.

The task set
------------
Not an invented trajectory. Each sample is an inverse-statics problem: hold a
leg posture while the foot carries a ground reaction force.

    tau_required = -J_contact^T F_ground + qfrc_bias

Postures are sampled over stance-relevant ranges rather than the joints' full
travel (a 2.4 rad knee is not a stance posture). Ground reactions are sampled
over the magnitudes gait produces: 0.4 to 1.2 body weight vertical, up to
0.25 BW fore-aft, up to 0.10 BW medio-lateral, with the contact point sweeping
heel to toe the way the centre of pressure does through stance.

Each sample is then solved for minimum-effort activation, `sum(a^3)`, and the
solutions are stacked into a (samples x 50) matrix for non-negative matrix
factorisation.

What this does and does not measure
-----------------------------------
This is the dimensionality of the *effort-optimal solution manifold* over this
task distribution. It is not a measurement of neural control, and a match to
the EMG literature would be a convergence of two different things rather than a
reproduction. Stated plainly because the distinction is easy to lose.

Two controls guard the result:

  shuffled null   each muscle's activations are permuted independently across
                  samples, destroying between-muscle structure while preserving
                  every marginal distribution. NMF can fit almost anything given
                  enough components, so a rank that looks low is only meaningful
                  against the rank the same procedure needs on structureless
                  data.
  functional VAF  reconstructed activations are pushed back through the plant.
                  Explaining activation variance is not the same as producing
                  the required torque, and the second is what matters.

Run
---
    python -m s9_muscle.synergies
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
from s9_muscle.control import (  # noqa: E402
    project_equality,
    LEG_JOINTS,
    check_affine,
    dof_indices,
    leg_plant,
    muscle_labels,
    solve_activation,
)
from s9_muscle.scene import MEDIA_DIR, load

SEED = 0
N_SAMPLES = 420
MAX_RANK = 12
COST_EXPONENT = 3.0

# Stance-relevant posture ranges, radians. Deliberately narrower than the
# model's joint limits: sampling the full 0 to 2.4 rad knee would spend most of
# the dataset in postures a standing leg never visits, and the synergy count
# would then describe a task set nobody has.
POSTURE_RANGE = {
    "hip_flexion_r": (-0.20, 0.60),
    "hip_adduction_r": (-0.15, 0.15),
    "hip_rotation_r": (-0.20, 0.20),
    "knee_angle_r": (0.05, 0.90),
    "ankle_angle_r": (-0.30, 0.25),
}
# Ground reaction as a fraction of body weight, spanning what walking produces.
GRF_VERTICAL = (0.40, 1.20)
GRF_FORE_AFT = (-0.25, 0.25)
GRF_LATERAL = (-0.10, 0.10)


# ------------------------------------------------------------ task set


def build_dataset(model, data, *, rng, n=N_SAMPLES) -> dict:
    """Sample stance problems and solve each for minimum-effort activation."""
    body_weight = float(model.body_mass.sum()) * 9.81
    dofs = dof_indices(model, LEG_JOINTS)
    heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    toes = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
    qadr = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in POSTURE_RANGE
    }
    qpos0 = data.qpos.copy()

    acts, taus, postures, grfs = [], [], [], []
    # Every sample is solved at its own posture, so it has its own plant. The
    # functional check below has to use the matching one: scoring a sample's
    # reconstructed activation against a plant linearised somewhere else
    # measures the distance between two poses, not reconstruction error.
    plant_A, plant_c = [], []
    residuals, failures = [], 0
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    labels = None

    for _ in range(n):
        data.qpos[:] = qpos0
        posture = {k: rng.uniform(*v) for k, v in POSTURE_RANGE.items()}
        for name, value in posture.items():
            data.qpos[qadr[name]] = value
        data.qvel[:] = 0.0
        # The knee's rolling contact and the shoulder's rhythm are equality
        # couplings; writing a driver joint leaves its dependents stale.
        project_equality(model, data)

        # Centre of pressure sweeping heel to toe, projected onto the floor.
        blend = rng.uniform(0.0, 1.0)
        point = (1.0 - blend) * data.xpos[heel] + blend * data.xpos[toes]
        point = np.array(point, dtype=float)
        point[2] = 0.0
        body = heel if blend < 0.5 else toes
        mujoco.mj_jac(model, data, jacp, jacr, point, body)

        force = body_weight * np.array([
            rng.uniform(*GRF_FORE_AFT),
            rng.uniform(*GRF_LATERAL),
            rng.uniform(*GRF_VERTICAL),
        ])
        tau = -jacp[:, dofs].T @ force + np.array(data.qfrc_bias)[dofs]

        plant = leg_plant(model, data)
        labels = labels or muscle_labels(model, plant.muscles)
        result = solve_activation(plant, tau, p=COST_EXPONENT)
        # A target outside the achievable set is a fact about the leg, not a
        # solver failure, but including a sample whose activation does not
        # produce the requested torque would put noise into the factorisation.
        if not result["success"] or result["residual_Nm"] > 1e-6:
            failures += 1
            continue
        acts.append(result["act"])
        plant_A.append(plant.A.copy())
        plant_c.append(plant.c.copy())
        taus.append(tau)
        postures.append([posture[k] for k in POSTURE_RANGE])
        grfs.append(force)
        residuals.append(result["residual_Nm"])

    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    return {
        "act": np.array(acts),
        "plant_A": np.array(plant_A),
        "plant_c": np.array(plant_c),
        "tau": np.array(taus),
        "posture": np.array(postures),
        "grf": np.array(grfs),
        "labels": labels,
        "n_requested": n,
        "n_kept": len(acts),
        "n_infeasible": failures,
        "max_residual_Nm": float(max(residuals)) if residuals else 0.0,
        "body_weight_N": round(body_weight, 1),
    }


# ------------------------------------------------------------------ NMF


def nmf(W: np.ndarray, k: int, *, rng, iters=600, restarts=4) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee-Seung multiplicative-update NMF, best of `restarts` random starts.

    Written out rather than imported because scikit-learn is not a dependency
    of this repository and the algorithm is fifteen lines. Multiplicative
    updates keep both factors non-negative by construction, which is the whole
    reason NMF and not PCA: a synergy that says "activate this muscle by minus
    0.3" is not a thing a muscle can do.
    """
    n, m = W.shape
    best, best_err = None, np.inf
    for _ in range(restarts):
        H = rng.uniform(0.1, 1.0, (n, k))
        S = rng.uniform(0.1, 1.0, (k, m))
        for _ in range(iters):
            S *= (H.T @ W) / (H.T @ H @ S + 1e-12)
            H *= (W @ S.T) / (H @ S @ S.T + 1e-12)
        err = float(np.linalg.norm(W - H @ S))
        if err < best_err:
            best, best_err = (H.copy(), S.copy()), err
    return best


def vaf(W: np.ndarray, W_hat: np.ndarray) -> float:
    """
    Variance accounted for, uncentred.

    The synergy literature uses the uncentred form, `1 - SSE/SST` with SST
    taken about zero rather than about the mean, so that is what is used here.
    Centring would report a different and non-comparable number.
    """
    return float(1.0 - np.sum((W - W_hat) ** 2) / np.sum(W ** 2))


def rank_needed(vafs: dict[int, float], threshold: float) -> int | None:
    for k in sorted(vafs):
        if vafs[k] >= threshold:
            return k
    return None


# ------------------------------------------------------------- analysis


def analyse(dataset: dict, *, rng) -> dict:
    W = dataset["act"]
    A, c = dataset["plant_A"], dataset["plant_c"]
    shuffled = np.column_stack([rng.permutation(col) for col in W.T])

    curves = {"real": {}, "shuffled": {}, "functional": {}}
    synergies = {}
    # Normalise against the muscle-generated part of the demand, which is what
    # activation is responsible for. The passive bias `c` arrives whether or
    # not anything is activated.
    tau_norm = float(np.linalg.norm(dataset["tau"] - c, axis=1).mean())

    for k in range(1, MAX_RANK + 1):
        H, S = nmf(W, k, rng=np.random.default_rng(SEED + k))
        curves["real"][k] = vaf(W, H @ S)
        synergies[k] = (H, S)

        Hs, Ss = nmf(shuffled, k, rng=np.random.default_rng(SEED + 1000 + k))
        curves["shuffled"][k] = vaf(shuffled, Hs @ Ss)

        # Functional check: push the rank-k reconstruction back through the
        # plant and ask how much of the demanded torque survives.
        tau_hat = np.einsum("sij,sj->si", A, H @ S) + c
        err = float(np.linalg.norm(dataset["tau"] - tau_hat, axis=1).mean())
        curves["functional"][k] = float(1.0 - err / tau_norm)

    return {
        "vaf": curves,
        "synergies": synergies,
        "rank_90_vaf": rank_needed(curves["real"], 0.90),
        "rank_95_vaf": rank_needed(curves["real"], 0.95),
        "rank_90_shuffled": rank_needed(curves["shuffled"], 0.90),
        "rank_90_functional": rank_needed(curves["functional"], 0.90),
        "mean_torque_demand_Nm": float(tau_norm),
    }


def make_figure(dataset, result, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    ks = sorted(result["vaf"]["real"])

    ax = axes[0]
    ax.plot(ks, [result["vaf"]["real"][k] for k in ks], "o-", color=PALETTE[0],
            lw=2, ms=4, label="activation VAF")
    ax.plot(ks, [result["vaf"]["functional"][k] for k in ks], "s-", color=PALETTE[2],
            lw=1.8, ms=4, label="torque reproduced")
    ax.plot(ks, [result["vaf"]["shuffled"][k] for k in ks], "^--", color=PALETTE[1],
            lw=1.6, ms=4, label="shuffled null")
    ax.axhline(0.90, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel("number of synergies  $k$")
    ax.set_ylabel("variance accounted for")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{result['rank_90_vaf']} synergies reach 90%")
    ax.legend(fontsize=8, loc="lower right")

    # The synergies themselves, at the rank that first clears 90%.
    k = result["rank_90_vaf"] or 4
    _, S = result["synergies"][k]
    order = np.argsort(-S.sum(axis=0))[:16]
    ax = axes[1]
    bottom = np.zeros(len(order))
    for j in range(k):
        vals = S[j, order]
        ax.bar(range(len(order)), vals, bottom=bottom, color=PALETTE[j % len(PALETTE)],
               label=f"synergy {j + 1}")
        bottom += vals
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([dataset["labels"][i] for i in order], rotation=70, ha="right", fontsize=6.5)
    ax.set_ylabel("weight")
    ax.set_title(f"composition at $k$ = {k} (16 largest muscles)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.hist(dataset["act"].max(axis=1), bins=24, color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("peak activation across the leg")
    ax.set_ylabel("samples")
    ax.set_title(f"{dataset['n_kept']} stance problems solved")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model, data = load("full", with_floor=False)

    config = {
        "model": "ms_human_700/MS-Human-700.xml",
        "joints": list(LEG_JOINTS),
        "n_samples": N_SAMPLES,
        "cost_exponent": COST_EXPONENT,
        "max_rank": MAX_RANK,
    }
    with RunLogger(
        "s9_synergies", config=config, seed=SEED,
        extra_manifest={"model_hash": hash_mj_model(model)},
    ) as log:
        log.event("affine_check", **check_affine(model, data, rng=rng))

        dataset = build_dataset(model, data, rng=rng)
        plant = leg_plant(model, data)
        log.event("dataset", **{k: v for k, v in dataset.items()
                                if not isinstance(v, (np.ndarray, list))})
        # Sanity: the stored per-sample plants must reproduce the torque each
        # sample was solved for. If this drifts, the functional curve is
        # meaningless and nothing else would say so.
        recon = np.einsum("sij,sj->si", dataset["plant_A"], dataset["act"]) + dataset["plant_c"]
        log.event("plant_roundtrip",
                  max_abs_err_Nm=float(np.abs(recon - dataset["tau"]).max()))

        result = analyse(dataset, rng=rng)
        log.event("vaf_real", **{str(k): round(v, 4) for k, v in result["vaf"]["real"].items()})
        log.event("vaf_shuffled",
                  **{str(k): round(v, 4) for k, v in result["vaf"]["shuffled"].items()})
        log.event("vaf_functional",
                  **{str(k): round(v, 4) for k, v in result["vaf"]["functional"].items()})
        log.event("ranks", **{k: v for k, v in result.items() if k.startswith("rank_")})

        figure = make_figure(dataset, result, MEDIA_DIR / "synergies.png")
        run_dir = log.path

    summary = {
        "n_muscles": int(plant.n),
        "n_dofs": int(len(plant.dofs)),
        "dataset": {k: v for k, v in dataset.items() if not isinstance(v, (np.ndarray, list))},
        "rank_90_vaf": result["rank_90_vaf"],
        "rank_95_vaf": result["rank_95_vaf"],
        "rank_90_shuffled": result["rank_90_shuffled"],
        "rank_90_functional": result["rank_90_functional"],
        "vaf": {kind: {str(k): round(v, 4) for k, v in curve.items()}
                for kind, curve in result["vaf"].items()},
        "mean_torque_demand_Nm": round(result["mean_torque_demand_Nm"], 2),
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
