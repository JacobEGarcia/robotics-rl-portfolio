"""
S12: what a braided actuator would have to be, to replace a human leg's muscles.

Four measurements.

  1. The excursion ceiling   eps_max(theta0) is monotone with a supremum of
                             1 - 1/sqrt(3) = 42.26%. Closed form, checked
                             against a numeric sweep. Muscles past that cannot
                             be matched by any braid at any braid angle.
  2. Transmission cost       the ratio each muscle needs, and the fluid volume
                             that ratio buys with (volume scales linearly in
                             the ratio, because force does and radius goes as
                             its square root).
  3. Packing                 fibre radii, total water mass, and whether that
                             mass is plausible against the limb it sits in.
  4. Pump demand             the whole leg's stroke volume against the
                             published 40 L/min, continuing S11's flow budget
                             from one actuator to one limb.

Run
---
    python -m s12_myofiber_leg.analyse
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
from s11_myofiber.myofiber import PUBLISHED  # noqa: E402
from s12_myofiber_leg.leg import (  # noqa: E402
    size_actuator,
    BRAID_CEILING,
    eps_max,
    leg_totals,
    size_leg,
    verify_sizing,
)
from s9_muscle.control import leg_plant  # noqa: E402
from s9_muscle.synergies import build_dataset  # noqa: E402
from s9_muscle.scene import actuator_names, load  # noqa: E402

SEED = 0
THETA0 = 30.0
SUPPLY_PSI = 100.0
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s12"


def ceiling_check(*, n=4000) -> dict:
    """
    Confirm the closed-form ceiling against a numeric sweep.

    The supremum is approached as theta0 -> 0 and never attained, so the sweep
    should climb towards 0.42265 without passing it. A sweep that exceeded it
    would mean the closed form is wrong, not that a better braid exists.
    """
    angles = np.linspace(0.05, 89.0, n)
    values = np.array([eps_max(a) for a in angles])
    return {
        "ceiling_closed_form": round(float(BRAID_CEILING), 6),
        "sweep_max": round(float(values.max()), 6),
        "sweep_never_exceeds_ceiling": bool(values.max() <= BRAID_CEILING + 1e-12),
        "monotone_decreasing_in_theta0": bool(np.all(np.diff(values) < 0)),
        "eps_max_at_30deg": round(eps_max(30.0), 4),
        "eps_max_at_20deg": round(eps_max(20.0), 4),
        "eps_max_at_10deg": round(eps_max(10.0), 4),
    }


def analyse(sized) -> dict:
    excursion = np.array([s.required_excursion for s in sized])
    ratio = np.array([s.transmission_ratio for s in sized])
    e30 = eps_max(THETA0)

    impossible = [s.name for s in sized if not s.feasible_at_any_braid_angle]
    return {
        "n_muscles": len(sized),
        "excursion_min": round(float(excursion.min()), 4),
        "excursion_median": round(float(np.median(excursion)), 4),
        "excursion_max": round(float(excursion.max()), 4),
        "n_over_30deg_braid": int((excursion > e30).sum()),
        "n_over_20deg_braid": int((excursion > eps_max(20.0)).sum()),
        "n_over_ceiling": int((excursion > BRAID_CEILING).sum()),
        "impossible_muscles": impossible,
        "median_transmission_ratio": round(float(np.median(ratio)), 3),
        "max_transmission_ratio": round(float(ratio.max()), 3),
        # Volume scales linearly with the transmission ratio, because force
        # does and radius goes as its square root. This is what the fix costs.
        "volume_penalty_from_transmission": round(float(np.mean(ratio)), 3),
    }


def pump_demand(totals, *, cycle_hz=1.0) -> dict:
    """
    Flow the leg needs to cycle its actuators once per gait cycle.

    Fills only. Venting returns fluid to the reservoir and does not draw from
    the pump, so counting both would double the demand.
    """
    stroke_m3 = totals["total_stroke_volume_mL"] * 1e-6
    pump_m3s = PUBLISHED["pump_lpm"] / 60.0 / 1000.0
    # Half a cycle fills, half vents, so the fill has half a period to happen in.
    fill_seconds = 0.5 / cycle_hz
    demand = stroke_m3 / fill_seconds
    return {
        "cycle_hz": cycle_hz,
        "leg_stroke_volume_mL": totals["total_stroke_volume_mL"],
        "demand_lpm": round(demand * 60000.0, 2),
        "pump_lpm": PUBLISHED["pump_lpm"],
        "fraction_of_pump_for_one_leg": round(demand / pump_m3s, 3),
        "legs_the_pump_supports": round(pump_m3s / demand, 2),
        "pump_supports_one_leg": bool(demand <= pump_m3s),
    }


def force_budget(model, plant, names, *, rng) -> dict:
    """
    Size to the force a muscle *can* make, or to the force it actually makes?

    Peak isometric force is the obvious sizing target and the wrong one. S9
    measured that real stance tasks use a median peak activation of about 0.42
    per muscle and leave most of the achievable torque set untouched, because
    a human muscle is enormously over-provisioned relative to standing on it.

    Both targets are sized here. The gap between them is the difference between
    a machine built to match a human's peak capability and one built to do a
    human's actual work, and it is large enough to change what is buildable.
    """
    peak = model.actuator_gainprm[plant.muscles, 2]
    dataset = build_dataset(model, data_of(model), rng=rng, n=300)
    used = np.percentile(dataset["act"], 95, axis=0) * peak

    fibre_force = PUBLISHED["min_force_N"]
    lengthrange = model.actuator_lengthrange[plant.muscles]
    task_sized = [
        size_actuator(names[int(mus)], max(used[i], 1.0),
                      lengthrange[i, 0], lengthrange[i, 1],
                      theta0_deg=THETA0, supply_psi=SUPPLY_PSI)
        for i, mus in enumerate(plant.muscles)
    ]
    task_totals = leg_totals(task_sized)
    return {
        "leg_peak_isometric_sum_N": round(float(peak.sum()), 1),
        "leg_task_p95_sum_N": round(float(used.sum()), 1),
        "task_fraction_of_peak": round(float(used.sum() / peak.sum()), 4),
        "median_p95_activation": round(float(np.median(
            np.percentile(dataset["act"], 95, axis=0))), 4),
        "n_stance_tasks": dataset["n_kept"],
        "published_fibre_force_N": fibre_force,
        "published_body_muscle_count": PUBLISHED["protoclone_muscles"],
        "fibres_for_one_leg_at_peak": int(round(peak.sum() / fibre_force)),
        "fibres_for_one_leg_at_task": int(round(used.sum() / fibre_force)),
        "published_whole_body_force_N": round(fibre_force * PUBLISHED["protoclone_muscles"], 1),
        "task_sized_totals": task_totals,
        "task_sized_pump": pump_demand(task_totals),
    }


def data_of(model):
    """A fresh MjData for the compiled model, for the sampler's own use."""
    import mujoco

    d = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, d, key)
    mujoco.mj_forward(model, d)
    return d


def make_figure(sized, ceiling, summary, budget, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    excursion = np.array([s.required_excursion for s in sized])
    order = np.argsort(-excursion)

    ax = axes[0]
    angles = np.linspace(0.5, 60.0, 400)
    ax.plot(angles, [eps_max(a) for a in angles], color=PALETTE[0], lw=2)
    ax.axhline(BRAID_CEILING, color=PALETTE[1], ls="--", lw=1.4)
    ax.text(30, BRAID_CEILING + 0.008, f"ceiling {BRAID_CEILING:.4f}", fontsize=8,
            color=PALETTE[1])
    ax.axvline(THETA0, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("rest braid angle $\\theta_0$ (deg)")
    ax.set_ylabel("max contraction $\\epsilon_{max}$")
    ax.set_title("no braid contracts past 42.3%")

    ax = axes[1]
    colours = [PALETTE[1] if not sized[i].feasible_at_any_braid_angle
               else (PALETTE[4] if not sized[i].feasible_without_transmission else PALETTE[0])
               for i in order]
    ax.barh(range(len(order)), excursion[order], color=colours)
    ax.axvline(eps_max(THETA0), color="0.35", ls=":", lw=1.3)
    ax.axvline(BRAID_CEILING, color=PALETTE[1], ls="--", lw=1.3)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("required excursion (fraction of longest length)")
    ax.set_ylabel(f"{len(sized)} leg muscles")
    ax.set_title(f"{summary['n_over_ceiling']} are impossible at any braid angle")

    ax = axes[2]
    ratio = np.array([s.transmission_ratio for s in sized])
    vol = np.array([s.rest_volume_mL for s in sized])
    ax.scatter(ratio, vol, s=18, color=PALETTE[2], alpha=0.85)
    ax.set_xlabel("transmission ratio needed")
    ax.set_ylabel("actuator fluid volume (mL)")
    ax.set_yscale("log")
    ax.set_title("the fix is paid for in water")

    ax = axes[3]
    labels = ["sized to peak\nisometric force", "sized to force\nactually used"]
    vols = [sum(s.rest_volume_mL for s in sized) / 1000.0,
            budget["task_sized_totals"]["total_rest_volume_mL"] / 1000.0]
    ax.bar(labels, vols, color=[PALETTE[1], PALETTE[2]], width=0.55)
    for i, v in enumerate(vols):
        ax.text(i, v * 1.02, f"{v:.2f} L", ha="center", fontsize=9)
    ax.set_ylabel("fluid in one leg's actuators (L)")
    ax.set_title(f"sizing to the task costs "
                 f"{100 * budget['task_fraction_of_peak']:.0f}% as much")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    model, data = load("full", with_floor=False)
    plant = leg_plant(model, data)
    names = actuator_names(model)

    sized = size_leg(model, plant.muscles, names, theta0_deg=THETA0, supply_psi=SUPPLY_PSI)
    ceiling = ceiling_check()
    check = verify_sizing(sized, theta0_deg=THETA0, supply_psi=SUPPLY_PSI)
    summary = analyse(sized)
    totals = leg_totals(sized)
    demand = pump_demand(totals)
    budget = force_budget(model, plant, names, rng=np.random.default_rng(SEED))

    config = {"theta0_deg": THETA0, "supply_psi": SUPPLY_PSI,
              "joints": "right leg, 5 DoF", "n_muscles": len(sized)}
    with RunLogger("s12_myofiber_leg", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model),
                                   "published": PUBLISHED}) as log:
        log.event("ceiling", **ceiling)
        log.event("sizing_check", **check)
        log.event("excursion", **{k: v for k, v in summary.items()
                                  if k != "impossible_muscles"})
        log.event("impossible", muscles=summary["impossible_muscles"])
        log.event("totals", **totals)
        log.event("pump", **demand)
        log.event("force_budget", **{k: v for k, v in budget.items()
                                     if not isinstance(v, dict)})
        log.event("task_sized_totals", **budget["task_sized_totals"])
        log.event("task_sized_pump", **budget["task_sized_pump"])
        figure = make_figure(sized, ceiling, summary, budget, MEDIA / "myofiber_leg.png")
        run_dir = log.path

    result = {
        "ceiling": ceiling,
        "sizing_check": check,
        "excursion": summary,
        "totals": totals,
        "pump": demand,
        "force_budget": budget,
        "actuators": [vars(s) for s in sized],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "actuators"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
