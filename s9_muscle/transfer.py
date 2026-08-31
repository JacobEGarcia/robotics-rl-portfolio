"""
S9-F: does a command computed on a torque model transfer to a muscle plant?

This is the claim the rest of S9 keeps gesturing at, made into a number. A
policy or controller developed against a torque-actuated humanoid outputs joint
torque. A muscle-actuated body does not accept joint torque; it accepts
activation, delivered through actuator dynamics, into a pose-dependent map. The
question is how much is lost in that translation and what it depends on.

Three conditions along the same reference
-----------------------------------------
The reference is a stance cycle: the leg tracks a joint trajectory while the
foot carries a ground reaction that rises and falls the way it does in gait.
The torque this requires, `tau_ref(t)`, comes from inverse statics at each
instant, the same construction the synergy study uses.

    A  ideal torque actuator     delivers tau_ref exactly, by definition. This
                                 is what the torque model promises.
    B  muscle plant, naive       solve for the activation that would produce
                                 tau_ref at each instant, command it directly.
                                 This is the honest translation of a
                                 torque-model command, and it is what fails.
    C  muscle plant, compensated identical activation targets, but the command
                                 is pre-inverted through the activation
                                 dynamics using S9-D's closed form.

B fails for exactly one reason, and it is worth being precise about which:
the *solve* is correct at every instant, so the target activation is right.
What is wrong is that commanding `a*` does not produce `a*`; it produces a
low-pass filtered version of it, and the plant multiplies that by a matrix that
does not commute with the filter. The faster the reference, the worse it gets,
and in the quasi-static limit it disappears entirely.

Checks
------
* Delivered torque computed through the linearised plant must match MuJoCo's
  own `qfrc_actuator` when the same activation is written into the model. This
  validates the whole pipeline end to end, not just the arithmetic.
* Condition B's error must vanish as the reference slows. If it does not, the
  error is a bug rather than a bandwidth limit.
* Condition C must beat B below the saturation frequency S9-D measured, and
  stop beating it above.

Run
---
    python -m s9_muscle.transfer
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
    LEG_JOINTS,
    activation_rate,
    check_affine,
    dof_indices,
    equality_residual,
    invert_activation,
    leg_plant,
    project_equality,
    solve_activation,
)
from s9_muscle.scene import MEDIA_DIR, load  # noqa: E402

SEED = 0
DT = 2e-4                       # integration step for the activation dynamics
CONTROL_HZ = 500.0              # how often the activation target is recomputed
FREQUENCIES = (0.25, 0.5, 1.0, 2.0, 4.0)
COST_EXPONENT = 3.0

# Stance cycle. Midpoints and amplitudes in radians, chosen inside the ranges
# the synergy study samples so the required torque stays feasible throughout.
POSTURE_MID = {"hip_flexion_r": 0.20, "hip_adduction_r": 0.0, "hip_rotation_r": 0.0,
               "knee_angle_r": 0.35, "ankle_angle_r": -0.05}
POSTURE_AMP = {"hip_flexion_r": 0.25, "hip_adduction_r": 0.05, "hip_rotation_r": 0.05,
               "knee_angle_r": 0.25, "ankle_angle_r": 0.15}
GRF_MID_BW, GRF_AMP_BW = 0.80, 0.35


def build_reference(model, data, freq: float, *, n_control: int) -> dict:
    """
    One stance cycle: poses, the torque they require, and the activation for it.

    Everything here is quasi-static by construction. The reference is a
    sequence of equilibria, not a dynamic trajectory, which is deliberate: it
    means condition A is exact and every bit of error in B and C is
    attributable to the actuator, not to the reference being unreachable.
    """
    dofs = dof_indices(model, LEG_JOINTS)
    heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in POSTURE_MID}
    body_weight = float(model.body_mass.sum()) * 9.81
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    t = np.arange(n_control) / CONTROL_HZ
    phase = 2.0 * np.pi * freq * t
    qpos0 = data.qpos.copy()

    taus, acts, A_list, c_list, residual = [], [], [], [], 0.0
    for k in range(n_control):
        data.qpos[:] = qpos0
        for name in POSTURE_MID:
            data.qpos[qadr[name]] = POSTURE_MID[name] + POSTURE_AMP[name] * np.sin(phase[k])
        data.qvel[:] = 0.0
        residual = max(residual, project_equality(model, data))

        point = np.array(data.xpos[heel], dtype=float)
        point[2] = 0.0
        mujoco.mj_jac(model, data, jacp, jacr, point, heel)
        vertical = body_weight * (GRF_MID_BW + GRF_AMP_BW * np.sin(phase[k]))
        force = np.array([0.0, 0.0, vertical])
        tau = -jacp[:, dofs].T @ force + np.array(data.qfrc_bias)[dofs]

        plant = leg_plant(model, data)
        sol = solve_activation(plant, tau, p=COST_EXPONENT)
        if not sol["success"] or sol["residual_Nm"] > 1e-6:
            raise RuntimeError(
                f"reference infeasible at sample {k}; residual {sol['residual_Nm']:.3e}. "
                "Shrink POSTURE_AMP or GRF_AMP_BW rather than tolerating it: an "
                "infeasible reference makes every condition below meaningless."
            )
        taus.append(tau)
        acts.append(sol["act"])
        A_list.append(plant.A.copy())
        c_list.append(plant.c.copy())

    data.qpos[:] = qpos0
    project_equality(model, data)
    return {
        "t": t,
        "tau": np.array(taus),
        "act": np.array(acts),
        "A": np.array(A_list),
        "c": np.array(c_list),
        "muscles": plant.muscles,
        "max_equality_residual": residual,
    }


def verify_plant(model, data, ref, *, samples=6) -> dict:
    """
    Check the linearised plant against MuJoCo, end to end.

    The whole study is computed through `A a + c`. If that disagrees with what
    the simulator produces for the same activation at the same pose, every
    number below is fiction, and the disagreement would not announce itself.
    """
    dofs = dof_indices(model, LEG_JOINTS)
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in POSTURE_MID}
    qpos0 = data.qpos.copy()
    worst = 0.0
    idx = np.linspace(0, len(ref["t"]) - 1, samples).astype(int)
    phase = 2.0 * np.pi * ref["_freq"] * ref["t"]
    for k in idx:
        data.qpos[:] = qpos0
        for name in POSTURE_MID:
            data.qpos[qadr[name]] = POSTURE_MID[name] + POSTURE_AMP[name] * np.sin(phase[k])
        data.qvel[:] = 0.0
        project_equality(model, data)
        data.act[:] = 0.0
        data.act[ref["muscles"]] = ref["act"][k]
        mujoco.mj_forward(model, data)
        predicted = ref["A"][k] @ ref["act"][k] + ref["c"][k]
        worst = max(worst, float(np.abs(np.array(data.qfrc_actuator)[dofs] - predicted).max()))
    data.qpos[:] = qpos0
    project_equality(model, data)
    return {"max_abs_err_Nm": worst, "samples": samples,
            "plant_matches_simulator": bool(worst < 1e-6)}


def roll(ref, prm, *, compensate: bool) -> dict:
    """Integrate the activation dynamics under the chosen command and score."""
    target = ref["act"]
    n_control = len(ref["t"])
    sub = int(round(1.0 / CONTROL_HZ / DT))
    n_mus = target.shape[1]

    act = target[0].copy()
    delivered = np.empty_like(ref["tau"])
    actual_act = np.empty_like(target)
    saturated = 0

    for k in range(n_control):
        if compensate:
            nxt = target[min(k + 1, n_control - 1)]
            rate = (nxt - target[k]) * CONTROL_HZ
            raw = np.array([invert_activation(a, r, prm) for a, r in zip(target[k], rate)])
            saturated += int(np.any((raw < -1e-12) | (raw > 1.0 + 1e-12)))
            cmd = np.clip(raw, 0.0, 1.0)
        else:
            cmd = target[k]
        for _ in range(sub):
            k1 = np.array([activation_rate(u, a, prm) for u, a in zip(cmd, act)])
            mid = act + 0.5 * DT * k1
            k2 = np.array([activation_rate(u, a, prm) for u, a in zip(cmd, mid)])
            act = act + DT * k2
        actual_act[k] = act
        delivered[k] = ref["A"][k] @ act + ref["c"][k]

    demand = ref["tau"] - ref["c"]
    err = delivered - ref["tau"]
    return {
        "rms_torque_error_Nm": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "relative_torque_error": float(
            np.mean(np.linalg.norm(err, axis=1)) / np.mean(np.linalg.norm(demand, axis=1))
        ),
        "max_abs_torque_error_Nm": float(np.abs(err).max()),
        "rms_activation_error": float(np.sqrt(np.mean((actual_act - target) ** 2))),
        "command_saturated_fraction": round(saturated / n_control, 4),
        "_trace": {"delivered": delivered, "target_act": target, "actual_act": actual_act},
    }


def make_figure(rows, traces, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    freqs = [r["freq_hz"] for r in rows]

    ax = axes[0]
    ax.plot(freqs, [r["naive"]["relative_torque_error"] for r in rows], "o-",
            color=PALETTE[1], lw=2, ms=5, label="naive torque-model command")
    ax.plot(freqs, [r["compensated"]["relative_torque_error"] for r in rows], "s-",
            color=PALETTE[0], lw=2, ms=5, label="activation-inverse compensated")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stance cycle frequency (Hz)")
    ax.set_ylabel("relative torque error")
    ax.set_title("the gap is a bandwidth, not a constant")
    ax.legend(fontsize=8)

    ref, naive, comp, freq = traces
    ax = axes[1]
    for j in range(ref["tau"].shape[1]):
        ax.plot(ref["t"], ref["tau"][:, j], color="0.6", lw=1.2)
        ax.plot(ref["t"], naive["_trace"]["delivered"][:, j], color=PALETTE[1], lw=1.4)
        ax.plot(ref["t"], comp["_trace"]["delivered"][:, j], color=PALETTE[0], lw=1.0, ls="--")
    ax.plot([], [], color="0.6", lw=1.5, label="required")
    ax.plot([], [], color=PALETTE[1], lw=1.5, label="naive")
    ax.plot([], [], color=PALETTE[0], lw=1.5, ls="--", label="compensated")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint torque (N·m)")
    ax.set_title(f"delivered torque, all 5 DoF at {freq:g} Hz")
    ax.legend(fontsize=8)

    ax = axes[2]
    idx = int(np.argmax(naive["_trace"]["target_act"].std(axis=0)))
    ax.plot(ref["t"], naive["_trace"]["target_act"][:, idx], color="0.6", lw=2, label="target")
    ax.plot(ref["t"], naive["_trace"]["actual_act"][:, idx], color=PALETTE[1], lw=1.5,
            label="naive")
    ax.plot(ref["t"], comp["_trace"]["actual_act"][:, idx], color=PALETTE[0], lw=1.4, ls="--",
            label="compensated")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("activation")
    ax.set_title("the busiest muscle")
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
    prm = model.actuator_dynprm[0][:3].copy()

    config = {"dt": DT, "control_hz": CONTROL_HZ, "frequencies": list(FREQUENCIES),
              "cost_exponent": COST_EXPONENT, "joints": list(LEG_JOINTS)}
    with RunLogger("s9_transfer", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("affine_check", **check_affine(model, data, rng=rng))

        rows, traces = [], None
        plant_check = None
        for freq in FREQUENCIES:
            n_control = int(round(CONTROL_HZ / freq))
            ref = build_reference(model, data, freq, n_control=n_control)
            ref["_freq"] = freq
            if plant_check is None:
                plant_check = verify_plant(model, data, ref)
                log.event("plant_check", **plant_check)

            naive = roll(ref, prm, compensate=False)
            comp = roll(ref, prm, compensate=True)
            rows.append({
                "freq_hz": freq,
                "max_equality_residual": ref["max_equality_residual"],
                "naive": {k: v for k, v in naive.items() if not k.startswith("_")},
                "compensated": {k: v for k, v in comp.items() if not k.startswith("_")},
                "improvement": round(
                    naive["relative_torque_error"] / max(comp["relative_torque_error"], 1e-15), 1
                ),
            })
            log.event("frequency", freq_hz=freq,
                      naive_rel=naive["relative_torque_error"],
                      compensated_rel=comp["relative_torque_error"],
                      improvement=rows[-1]["improvement"])
            if freq == 1.0:
                traces = (ref, naive, comp, freq)

        figure = make_figure(rows, traces, MEDIA_DIR / "transfer.png")
        run_dir = log.path

    naive_rel = [r["naive"]["relative_torque_error"] for r in rows]
    comp_rel = [r["compensated"]["relative_torque_error"] for r in rows]
    summary = {
        "plant_check": plant_check,
        "naive_error_grows_with_speed": bool(np.all(np.diff(naive_rel) > 0)),
        "naive_error_at_slowest": round(naive_rel[0], 5),
        "naive_error_at_fastest": round(naive_rel[-1], 5),
        "compensated_error_at_fastest": round(comp_rel[-1], 6),
        "improvement_at_1hz": rows[FREQUENCIES.index(1.0)]["improvement"],
        "per_frequency": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frequency"}, indent=2))
    for r in rows:
        print(f"  {r['freq_hz']:>5g} Hz   naive {r['naive']['relative_torque_error']:.4f}"
              f"   compensated {r['compensated']['relative_torque_error']:.5f}"
              f"   {r['improvement']:>7.1f}x")
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
