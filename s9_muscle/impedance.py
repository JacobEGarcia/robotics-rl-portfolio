"""
S9-C: co-contraction as variable impedance.

The redundancy result says the leg has a 45-dimensional space of activation
patterns that all produce the same joint torque. That space is not wasted: it
sets stiffness. Two muscles pulling against each other across a joint cancel in
torque and add in stiffness, which is how an animal stiffens an ankle before
landing without moving it.

This measures that channel. At a fixed pose, drive every muscle spanning a
joint to a common level `c`, perturb the joint, and read the restoring torque:

    K = -d tau / d q,   by central difference on the model

Three components are separated, because conflating them would inflate the
result:

    passive     everything at c = 0. Muscle passive elasticity, joint limits,
                and any joint-level springs and damping. Present whether or not
                anything is commanded.
    total       at co-contraction level c.
    active      total minus passive. This is the part a controller commands.

Checks
------
* Stiffness must rise with co-contraction, and the agonist and antagonist
  contributions must *add* rather than cancel. Measured separately, because
  "co-contraction stiffens a joint" is the claim under test, not an assumption.
* The restoring torque must oppose the perturbation, so K > 0. A negative
  measured stiffness would mean the finite difference straddled a joint limit
  or a snap-through, which is exactly the failure S4 hit on the tendon finger.
* Magnitude against the human literature: ankle quasi-stiffness during stance
  is usually reported in the low hundreds of N*m/rad.

The result: co-contraction is not a stiffness knob
--------------------------------------------------
Two of the three checks pass. Stiffness is *exactly* proportional to
co-contraction level (deviation below 1e-4, which follows from force being
affine in activation), and the two sides of each joint add rather than cancel.

The third does not, and not in the direction expected. Sweeping each joint
through its range at full co-contraction:

    ankle   -276 to +481 N*m/rad,  40% of the range negative
    knee     -58 to  +92 N*m/rad,  48% negative, two sign changes
    hip     -225 to +350 N*m/rad,  40% negative

**Stiffness changes sign with joint angle at every joint.** Near full plantar-
flexion the ankle is at +481, comfortably inside the human band; at its neutral
angle it is +13.6, sitting almost exactly on a zero crossing; in dorsiflexion
it is -276, meaning uniform co-contraction makes the joint *easier* to
displace. The knee at 0.35 rad reads -55.5: the muscles with positive moment
arm give +8.9 and those with negative moment arm give -64.4. All of these are
stable to within 0.6% across a 25x range of finite-difference step size, so
they are properties of the model rather than of the measurement.

The first version of this study reported "15x below the human literature" from
a single neutral angle. That number was real and the conclusion drawn from it
was wrong: it happened to land on the ankle's zero crossing. A single-pose
stiffness comparison against literature can produce any answer you like on this
model, which is worth knowing before running one.

Why the sign moves: `tendon_stiffness` is zero for all 700 tendons, so there is
no series elasticity. A MuJoCo muscle with an inextensible tendon has exactly
two stiffness sources, the slope of the force-length curve (always stabilising)
and the derivative of the moment arm (either sign). With the first term small,
the second decides, and it flips with pose. In a real limb short-range
stiffness from series elasticity dominates both and keeps the total positive.

Consequence: "co-contract to stiffen up" is a standard move in robot impedance
control and it does not survive contact with this model. Using the null space
as an impedance channel here requires choosing *which* muscles to co-contract
and knowing where the joint is, not just how hard to squeeze.

Run
---
    python -m s9_muscle.impedance
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
from s9_muscle.control import check_affine, equality_residual, project_equality  # noqa: E402
from s9_muscle.scene import MEDIA_DIR, load, moment_arms  # noqa: E402

SEED = 0
# Joints to characterise, with the neutral angle each is held at. Chosen inside
# the joint's range and away from its limits: a difference taken across a limit
# measures the constraint solver, not the muscles.
JOINTS = {
    "ankle_angle_r": 0.0,
    "knee_angle_r": 0.35,
    "hip_flexion_r": 0.20,
}
CO_CONTRACTION = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
# Perturbation for the central difference. Large enough that the torque change
# is far above numerical noise, small enough to stay in the linear regime;
# both ends are checked by the step-size sweep in `convergence()`.
DELTA_RAD = 0.01
# Reported human ankle quasi-stiffness during stance, N*m/rad. Used only as an
# order-of-magnitude sanity band, not as a target to hit.
ANKLE_LITERATURE_BAND = (200.0, 400.0)


def _torque(model, data, dof: int) -> float:
    # Project before reading. These joints drive equality-coupled dependents,
    # so perturbing one and calling mj_forward alone evaluates the muscles at a
    # configuration the model does not admit.
    project_equality(model, data)
    return float(data.qfrc_actuator[dof] + data.qfrc_passive[dof])


def stiffness(model, data, joint: str, level: float, *, delta=DELTA_RAD) -> float:
    """
    Central-difference joint stiffness at a given uniform co-contraction.

    Only the muscles that actually span the joint are driven. Driving all 700
    would add torque from muscles with no moment arm here, which changes the
    pose the body settles into without changing this joint's stiffness, and
    would make the number depend on the rest of the body's posture.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    neutral = JOINTS[joint]

    data.qvel[:] = 0.0
    data.qpos[adr] = neutral
    project_equality(model, data)
    spanning = np.abs(moment_arms(model, data)[:, dof]) > 1e-9

    data.act[:] = 0.0
    data.act[spanning] = level
    data.ctrl[:] = 0.0
    data.ctrl[spanning] = level

    data.qpos[adr] = neutral + delta
    plus = _torque(model, data, dof)
    data.qpos[adr] = neutral - delta
    minus = _torque(model, data, dof)
    data.qpos[adr] = neutral

    return -(plus - minus) / (2.0 * delta)


def antagonist_split(model, data, joint: str, *, level=1.0) -> dict:
    """
    Stiffness from each side of the joint separately, and from both together.

    If co-contraction works the way it is supposed to, the two one-sided
    numbers add up to the two-sided one. If they cancelled instead, the low
    total would have a different explanation and a different fix.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    neutral = JOINTS[joint]
    data.qvel[:] = 0.0
    data.qpos[adr] = neutral
    project_equality(model, data)
    arms = moment_arms(model, data)[:, dof]
    spanning = np.abs(arms) > 1e-9

    def measure(mask, delta=DELTA_RAD):
        data.act[:] = 0.0
        data.act[mask] = level
        data.qpos[adr] = neutral + delta
        plus = _torque(model, data, dof)
        data.qpos[adr] = neutral - delta
        minus = _torque(model, data, dof)
        data.qpos[adr] = neutral
        return -(plus - minus) / (2.0 * delta)

    positive = spanning & (arms > 0)
    negative = spanning & (arms < 0)
    k_pos, k_neg, k_both = measure(positive), measure(negative), measure(spanning)
    return {
        "n_positive_arm": int(positive.sum()),
        "n_negative_arm": int(negative.sum()),
        "K_positive_only": round(k_pos, 3),
        "K_negative_only": round(k_neg, 3),
        "K_both": round(k_both, 3),
        # 1e-4 relative, not machine epsilon: these are finite differences of a
        # mildly nonlinear function, and the hip's residual is 4e-6 relative
        # from curvature rather than from the contributions failing to add.
        "contributions_add": bool(abs(k_pos + k_neg - k_both) < 1e-4 * max(abs(k_both), 1.0)),
        "one_side_destabilising": bool(min(k_pos, k_neg) < 0.0),
    }


def convergence(model, data, joint: str, level: float) -> dict:
    """
    Stiffness against perturbation size.

    A central difference has two failure modes in opposite directions: too
    small and it measures solver noise, too large and it leaves the linear
    regime. Reporting the value at several step sizes is cheaper than arguing
    about which one is right.
    """
    return {
        f"{d:g}": round(stiffness(model, data, joint, level, delta=d), 2)
        for d in (0.002, 0.005, 0.01, 0.02, 0.05)
    }


def angle_sweep(model, data, joint: str, *, level=1.0, n=25) -> dict:
    """
    Stiffness across the joint's range, at full co-contraction.

    Reported because the single-pose number is not representative: at the knee
    the sign changes twice over the range, so a study that measured one angle
    would draw the opposite conclusion depending on which angle it picked.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    lo, hi = model.jnt_range[jid]
    # Stay off the limits: a difference straddling one measures the constraint
    # solver rather than the muscles.
    margin = 0.05 * (hi - lo)
    angles = np.linspace(lo + margin, hi - margin, n)

    values = []
    for angle in angles:
        data.qvel[:] = 0.0
        data.qpos[adr] = angle
        project_equality(model, data)
        spanning = np.abs(moment_arms(model, data)[:, dof]) > 1e-9
        data.act[:] = 0.0
        data.act[spanning] = level
        data.qpos[adr] = angle + DELTA_RAD
        plus = _torque(model, data, dof)
        data.qpos[adr] = angle - DELTA_RAD
        minus = _torque(model, data, dof)
        values.append(-(plus - minus) / (2.0 * DELTA_RAD))
    data.qpos[adr] = JOINTS.get(joint, float(angles[0]))
    project_equality(model, data)

    values = np.array(values)
    sign_changes = int(np.count_nonzero(np.diff(np.sign(values)) != 0))
    return {
        "angles_rad": [round(float(a), 4) for a in angles],
        "K_Nm_per_rad": [round(float(v), 2) for v in values],
        "fraction_of_range_negative": round(float(np.mean(values < 0.0)), 3),
        "sign_changes": sign_changes,
        "min_K": round(float(values.min()), 2),
        "max_K": round(float(values.max()), 2),
    }


def run(model, data) -> dict:
    out = {}
    for joint in JOINTS:
        passive = stiffness(model, data, joint, 0.0)
        totals = [stiffness(model, data, joint, c) for c in CO_CONTRACTION]
        active = [t - passive for t in totals]
        # K/c at every non-zero level. Constant iff stiffness is exactly
        # proportional to co-contraction.
        ratios = [t / c for t, c in zip(totals[1:], CO_CONTRACTION[1:])]
        out[joint] = {
            "neutral_rad": JOINTS[joint],
            "passive_Nm_per_rad": round(passive, 2),
            "total_Nm_per_rad": [round(v, 2) for v in totals],
            "active_Nm_per_rad": [round(v, 2) for v in active],
            "max_total_Nm_per_rad": round(max(totals), 2),
            "monotone_in_co_contraction": bool(np.all(np.diff(totals) >= -1e-6)),
            # Zero co-contraction is excluded: with no series elasticity the
            # passive term is identically zero here, and "is 0.0 > 0" is not a
            # question worth asking.
            "positive_above_zero_co_contraction": bool(np.all(np.array(totals[1:]) > 0.0)),
            "stiffness_per_unit_co_contraction": round(float(np.mean(ratios)), 3),
            "linearity_max_rel_deviation": round(
                float(np.max(np.abs(np.array(ratios) / np.mean(ratios) - 1.0))), 8
            ),
            "antagonist_split": antagonist_split(model, data, joint),
            "angle_sweep": angle_sweep(model, data, joint),
            "step_size_sweep_at_full": convergence(model, data, joint, 1.0),
        }
    out["max_equality_residual"] = equality_residual(data)
    ankle = out["ankle_angle_r"]
    lo, hi = ANKLE_LITERATURE_BAND
    sweep = ankle["angle_sweep"]["K_Nm_per_rad"]
    # Judged over the joint's whole range, not at one neutral angle. At the
    # neutral angle alone the ankle reads 13.6 because that pose sits on a zero
    # crossing, and concluding "15x below human" from it would be an artifact
    # of where the sample was taken.
    out["ankle_reaches_literature_band_somewhere"] = bool(any(lo <= v <= hi for v in sweep))
    out["ankle_at_neutral_Nm_per_rad"] = ankle["max_total_Nm_per_rad"]
    out["ankle_peak_over_range_Nm_per_rad"] = max(sweep)
    out["ankle_min_over_range_Nm_per_rad"] = min(sweep)
    out["all_joints_change_sign_over_range"] = bool(
        all(out[j]["angle_sweep"]["sign_changes"] > 0 for j in JOINTS)
    )
    # The structural cause of the shortfall, read straight off the model rather
    # than inferred from the number being small.
    out["tendons_with_stiffness"] = int(np.count_nonzero(model.tendon_stiffness))
    out["n_tendons"] = int(model.ntendon)
    out["co_contraction_levels"] = list(CO_CONTRACTION)
    return out


def make_figure(result, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    levels = result["co_contraction_levels"]

    ax = axes[0]
    for i, joint in enumerate(JOINTS):
        ax.plot(levels, result[joint]["total_Nm_per_rad"], "o-",
                color=PALETTE[i], lw=2, ms=4, label=joint)
    lo, hi = ANKLE_LITERATURE_BAND
    ax.axhspan(lo, hi, color=PALETTE[2], alpha=0.12)
    ax.text(0.02, hi, " human ankle, stance (literature)", fontsize=7.5,
            va="bottom", color="0.35")
    ax.set_xlabel("co-contraction level (uniform activation)")
    ax.set_ylabel("joint stiffness  (N·m/rad)")
    ax.set_title("exactly linear in co-contraction")
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.26
    for i, joint in enumerate(JOINTS):
        sp = result[joint]["antagonist_split"]
        ax.bar(i - width, sp["K_positive_only"], width, color=PALETTE[0],
               label="positive moment arm" if i == 0 else None)
        ax.bar(i, sp["K_negative_only"], width, color=PALETTE[1],
               label="negative moment arm" if i == 0 else None)
        ax.bar(i + width, sp["K_both"], width, color=PALETTE[2],
               label="both together" if i == 0 else None)
    ax.set_xticks(range(len(JOINTS)))
    ax.set_xticklabels(list(JOINTS), fontsize=8)
    ax.set_ylabel("N·m/rad at full activation")
    ax.set_title("agonist and antagonist add, they do not cancel")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, joint in enumerate(JOINTS):
        sw = result[joint]["angle_sweep"]
        ax.plot(sw["angles_rad"], sw["K_Nm_per_rad"], "-", color=PALETTE[i], lw=1.9, label=joint)
    ax.axhline(0.0, color="0.45", lw=1.1)
    ax.set_xlabel("joint angle (rad)")
    ax.set_ylabel("K at full co-contraction (N·m/rad)")
    ax.set_title("the sign is not a constant")
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

    config = {"joints": JOINTS, "levels": list(CO_CONTRACTION), "delta_rad": DELTA_RAD}
    with RunLogger("s9_impedance", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("affine_check", **check_affine(model, data, rng=rng))
        result = run(model, data)
        for joint in JOINTS:
            log.event("joint", name=joint, **result[joint])
        log.event("checks", **{k: v for k, v in result.items()
                               if k.startswith(("ankle_", "all_joints_", "tendons_",
                                                "n_tendons", "max_equality"))})
        figure = make_figure(result, MEDIA_DIR / "impedance.png")
        run_dir = log.path

    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
