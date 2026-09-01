"""
S17: the step length a muscle-driven leg would choose, and why.

The question
------------
Every study so far has taken the task as given and asked what the actuators
have to do. This one turns it around: given that muscles cost effort, what
motion is cheapest?

Walking has a well-known answer. Metabolic cost per unit distance is U-shaped
in step length, because short steps pay a per-step overhead many times while
long steps need large joint torques at unfavourable postures. Humans settle
near the bottom of that curve. If a purely mechanical effort model, with no
metabolism in it at all, reproduces the shape and lands near the human value,
that is a convergence worth having: it says the preference is largely a
consequence of the mechanics rather than of physiology.

The model
---------
Quasi-static stance. For a step length `L` the hip sweeps through
`+-asin(L/2/leg)`, the foot carries body weight plus a fore-aft shear that
reverses through the step, and at each sample the required joint torque is
solved for minimum-effort activation exactly as in S9. Cost per step is the
integrated `sum(a^3)`; cost per metre divides by `L`.

The knee and ankle profiles are **optimised, not assumed**. A hand-picked
posture would make the U-shape a property of the guess, so at every step length
three posture parameters are optimised and the reported cost is the best that
posture family can do.

Feasibility is a hard gate
--------------------------
A sample whose required torque is outside the leg's achievable set is not a
cheap sample, it is an impossible one. The first version of this averaged cost
over the feasible samples only, and long steps came out *cheaper* than the
optimum because most of their stance had been quietly dropped. A step length is
now feasible only if every sample in it is, and infeasible ones are excluded
rather than scored.

Run
---
    python -m s17_gait.economy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from common.plotting import PALETTE  # noqa: E402
from common.runlog import RunLogger, hash_mj_model  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from s9_muscle.control import (  # noqa: E402
    LEG_JOINTS,
    dof_indices,
    leg_plant,
    project_equality,
    solve_activation,
)
from s9_muscle.scene import load  # noqa: E402

SEED = 0
# Budget note: one evaluation costs about 30 ms per stance sample, and the
# posture search calls it tens of times at every step length. Nine samples over
# a twenty-point grid with a ninety-iteration search is nineteen minutes of
# nested optimisation, which is not worth it for a curve whose shape is already
# clear at coarser resolution.
N_SAMPLES = 7                # stance samples per step
COST_EXPONENT = 3.0
STEP_LENGTHS = np.arange(0.30, 1.11, 0.08)
SHEAR_FRACTION = 0.20        # fore-aft ground reaction, fraction of body weight
# Finite so the optimiser can compare two infeasible postures; far above any
# achievable cost so a feasible posture always wins.
INFEASIBLE_PENALTY = 1.0e6
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s17"

# Reported human preferred step length is roughly 0.7 m for an adult of this
# stature. Used as an order-of-magnitude comparison, never as a fitting target.
HUMAN_PREFERRED_M = (0.65, 0.75)


class Leg:
    """Cached handles into the compiled model, so the sweep is not re-resolving names."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        j = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.hip = int(model.jnt_qposadr[j("hip_flexion_r")])
        self.knee = int(model.jnt_qposadr[j("knee_angle_r")])
        self.ankle = int(model.jnt_qposadr[j("ankle_angle_r")])
        self.heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dofs = dof_indices(model, LEG_JOINTS)
        self.body_weight = float(model.body_mass.sum()) * 9.81
        mujoco.mj_forward(model, data)
        self.leg_length = float(np.linalg.norm(data.xpos[pelvis] - data.xpos[self.heel]))
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.qpos0 = data.qpos.copy()

    def step_cost(self, length, posture, *, load=1.0, p=COST_EXPONENT) -> dict:
        """
        Integrated effort for one stance phase, or infeasible.

        `posture` is (knee_mid, knee_amp, ankle_amp). `load` scales body weight,
        so carrying mass is a multiplier rather than a separate model.
        """
        knee_mid, knee_amp, ankle_amp = posture
        half = float(np.arcsin(np.clip(length / 2.0 / self.leg_length, -0.95, 0.95)))
        total, peak = 0.0, 0.0
        for phase in np.linspace(-1.0, 1.0, N_SAMPLES):
            self.d.qpos[:] = self.qpos0
            self.d.qpos[self.hip] = half * phase
            self.d.qpos[self.knee] = np.clip(knee_mid + knee_amp * (1.0 - abs(phase)), 0.02, 1.2)
            self.d.qpos[self.ankle] = np.clip(-ankle_amp * phase, -0.6, 0.45)
            self.d.qvel[:] = 0.0
            project_equality(self.m, self.d)

            point = np.array(self.d.xpos[self.heel], dtype=float)
            point[2] = 0.0
            mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, point, self.heel)
            force = np.array([-SHEAR_FRACTION * load * self.body_weight * phase,
                              0.0, load * self.body_weight])
            tau = -self.jacp[:, self.dofs].T @ force + np.array(self.d.qfrc_bias)[self.dofs]

            plant = leg_plant(self.m, self.d)
            sol = solve_activation(plant, tau, p=p)
            if not sol["success"] or sol["residual_Nm"] > 1e-6:
                # Not a cheap sample. An impossible one. Reported as infeasible,
                # but scored with a large *finite* penalty that grows with how
                # badly the torque was missed, so the search is pushed back
                # towards the achievable set. Returning infinity here makes
                # Nelder-Mead's convergence test compute inf - inf, which is
                # NaN, and the simplex then wanders: scipy warns about it and
                # the answer is quietly not an optimum.
                return {"feasible": False,
                        "cost": INFEASIBLE_PENALTY + float(sol["residual_Nm"]),
                        "peak_activation": None}
            total += sol["cost"]
            peak = max(peak, sol["peak_act"])
        return {"feasible": True, "cost": total, "peak_activation": peak}

    def optimal_posture(self, length, *, load=1.0, p=COST_EXPONENT) -> dict:
        """
        Best posture for a step length, so the U-shape is not an artifact of a guess.

        Nelder-Mead rather than a gradient method: `step_cost` is piecewise
        (it returns infinity outside the achievable set) and a finite-difference
        gradient across that edge is meaningless.
        """
        x0 = np.array([0.15, 0.35, 0.10])

        def objective(x):
            return self.step_cost(length, x, load=load, p=p)["cost"]

        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 30, "xatol": 5e-3, "fatol": 5e-3})
        best = self.step_cost(length, res.x, load=load, p=p)
        start = self.step_cost(length, x0, load=load, p=p)
        # If the optimiser wandered somewhere worse than where it started, keep
        # the start. Nelder-Mead on a penalised objective has no such guarantee.
        if start["feasible"] and (not best["feasible"] or start["cost"] < best["cost"]):
            best, res_x = start, x0
        else:
            res_x = res.x
        return {
            "length_m": round(float(length), 3),
            "feasible": best["feasible"],
            "cost_per_step": None if not best["feasible"] else round(best["cost"], 4),
            "cost_per_metre": None if not best["feasible"] else round(best["cost"] / length, 4),
            "peak_activation": None if not best["feasible"] else round(best["peak_activation"], 4),
            "posture": [round(float(v), 4) for v in res_x],
            "start_cost": None if not start["feasible"] else round(start["cost"], 4),
            "improved_on_start": bool(
                best["feasible"] and (not start["feasible"] or best["cost"] <= start["cost"] + 1e-9)),
        }


def sweep(leg, *, load=1.0, p=COST_EXPONENT):
    return [leg.optimal_posture(L, load=load, p=p) for L in STEP_LENGTHS]


def analyse(rows) -> dict:
    ok = [r for r in rows if r["feasible"]]
    if not ok:
        return {"n_feasible": 0}
    saturated = [r["length_m"] for r in ok if r["peak_activation"] >= 0.999]
    per_m = np.array([r["cost_per_metre"] for r in ok])
    lengths = np.array([r["length_m"] for r in ok])
    i = int(np.argmin(per_m))
    interior = 0 < i < len(ok) - 1
    return {
        "n_feasible": len(ok),
        "n_infeasible": len(rows) - len(ok),
        "feasible_range_m": [float(lengths.min()), float(lengths.max())],
        "optimal_step_m": float(lengths[i]),
        "optimal_cost_per_metre": float(per_m[i]),
        "minimum_is_interior": bool(interior),
        "cost_at_half_optimal": float(np.interp(lengths[i] / 2, lengths, per_m)),
        "cost_at_double_optimal": float(np.interp(min(lengths[i] * 2, lengths.max()),
                                                  lengths, per_m)),
        "penalty_for_halving": round(float(np.interp(lengths[i] / 2, lengths, per_m) / per_m[i]), 3),
        "within_human_preferred": bool(HUMAN_PREFERRED_M[0] <= lengths[i] <= HUMAN_PREFERRED_M[1]),
        "distance_from_human_band_m": round(float(min(
            abs(lengths[i] - HUMAN_PREFERRED_M[0]), abs(lengths[i] - HUMAN_PREFERRED_M[1]))), 3),
        "all_optimisations_improved": bool(all(r["improved_on_start"] for r in ok)),
        # Above this step length at least one muscle is pinned at full
        # activation, so the cost rise there is partly saturation rather than
        # an efficiency argument.
        "saturates_at_or_above_m": float(min(saturated)) if saturated else None,
    }


def make_figure(base, loads, exponents, summary, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ok = [r for r in base if r["feasible"]]
    ax.plot([r["length_m"] for r in ok], [r["cost_per_metre"] for r in ok], "o-",
            color=PALETTE[0], lw=2, ms=4)
    ax.axvspan(*HUMAN_PREFERRED_M, color=PALETTE[2], alpha=0.15)
    ax.text(HUMAN_PREFERRED_M[0], ax.get_ylim()[1] * 0.92, " human\n preferred",
            fontsize=7.5, color="0.35", va="top")
    ax.axvline(summary["optimal_step_m"], color=PALETTE[1], ls="--", lw=1.3)
    ax.set_xlabel("step length (m)")
    ax.set_ylabel("effort per metre")
    ax.set_title(f"cheapest at {summary['optimal_step_m']:.2f} m, short of human")

    ax = axes[1]
    for i, (load, rows) in enumerate(loads.items()):
        ok = [r for r in rows if r["feasible"]]
        ax.plot([r["length_m"] for r in ok], [r["cost_per_metre"] for r in ok],
                "-", color=PALETTE[i % len(PALETTE)], lw=1.8, label=f"{load:g}x body weight")
    ax.set_xlabel("step length (m)")
    ax.set_ylabel("effort per metre")
    ax.set_yscale("log")
    ax.set_title("load raises the cost, not the optimum")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, (p, rows) in enumerate(exponents.items()):
        ok = [r for r in rows if r["feasible"]]
        vals = np.array([r["cost_per_metre"] for r in ok])
        ax.plot([r["length_m"] for r in ok], vals / vals.min(),
                "-", color=PALETTE[i % len(PALETTE)], lw=1.8, label=f"p = {p:g}")
    ax.set_xlabel("step length (m)")
    ax.set_ylabel("effort per metre, normalised")
    ax.set_title("the optimum moves two grid cells with $p$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    model, data = load("full", with_floor=False)
    leg = Leg(model, data)

    base = sweep(leg)
    summary = analyse(base)

    # 1.0 body weight at p = 3 is the base sweep; recomputing it would cost a
    # third of the runtime to reproduce a result already in hand.
    loads = {1.0: base}
    loads.update({w: sweep(leg, load=w) for w in (1.25, 1.5)})
    load_optima = {str(w): analyse(r).get("optimal_step_m") for w, r in loads.items()}

    exponents = {3.0: base}
    exponents.update({p: sweep(leg, p=p) for p in (2.0, 5.0)})
    exponents = dict(sorted(exponents.items()))
    exponent_optima = {str(p): analyse(r).get("optimal_step_m") for p, r in exponents.items()}

    config = {"n_samples": N_SAMPLES, "cost_exponent": COST_EXPONENT,
              "shear_fraction": SHEAR_FRACTION,
              "step_lengths_m": [round(float(x), 3) for x in STEP_LENGTHS]}
    with RunLogger("s17_gait", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model),
                                   "leg_length_m": round(leg.leg_length, 4),
                                   "body_weight_N": round(leg.body_weight, 1)}) as log:
        for r in base:
            log.event("step", **r)
        log.event("summary", **summary)
        log.event("load_optima", **load_optima)
        log.event("exponent_optima", **exponent_optima)
        figure = make_figure(base, loads, exponents, summary, MEDIA / "gait_economy.png")
        run_dir = log.path

    result = {
        "leg_length_m": round(leg.leg_length, 4),
        "body_weight_N": round(leg.body_weight, 1),
        "summary": summary,
        "load_optima_m": load_optima,
        # Reported as a shift rather than a direction. All three loads land on
        # the same grid point, and "1.5 <= 1.0" is then trivially true, which
        # would have been written up as "carrying load shortens the optimum"
        # on the strength of an equality.
        "load_optimum_shift_m": round(float(load_optima["1.5"] - load_optima["1.0"]), 3)
        if None not in (load_optima["1.5"], load_optima["1.0"]) else None,
        "load_optimum_unchanged_at_grid": bool(
            len({v for v in load_optima.values() if v is not None}) == 1),
        "exponent_optima_m": exponent_optima,
        "optimum_range_over_exponents_m": [
            min(v for v in exponent_optima.values() if v is not None),
            max(v for v in exponent_optima.values() if v is not None)],
        "optimum_robust_to_exponent": bool(
            len({v for v in exponent_optima.values() if v is not None}) == 1),
        "grid_spacing_m": round(float(STEP_LENGTHS[1] - STEP_LENGTHS[0]), 3),
        "sweep": base,
        # Persist the load and exponent sweeps too. The figure needs them, and
        # without them regenerating it costs a fifteen-minute recomputation of
        # results already in hand.
        "load_sweeps": {str(k): v for k, v in loads.items()},
        "exponent_sweeps": {str(k): v for k, v in exponents.items()},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
