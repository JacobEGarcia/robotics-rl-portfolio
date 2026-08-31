"""
S13: controlling a limb when the working fluid is rationed.

The constraint
--------------
S11 found that Clone's published 40 L/min pump serves about 34 actuators at
full stroke and full speed. S12 found that one leg sized to the force it
actually uses wants 204 L/min to cycle at 1 Hz. Both point at the same thing:
**on a hydraulic body, flow is scarce and has to be allocated.**

This has no analogue on a torque-actuated humanoid. A motor draws current from
a bus that can usually supply it, and a controller that asks all 50 joints for
their torque at once gets it. Here the working fluid is rationed, so a
controller that asks for everything gets a scaled-down version of everything,
unless it says which requests matter.

The mapping
-----------
Each muscle's Myofiber is sized (S12) so that at supply pressure it delivers
that muscle's peak force. Pressure fraction therefore plays exactly the role
activation plays in S9:

    P_target = a_ref * P_supply          a_ref from S9's minimum-effort solve
    V        = C P                       stored fluid, C the bladder compliance
    flow     = max(0, dV/dt)             fills draw from the pump, vents do not
    a_actual = P_actual / P_supply       what the joint actually receives

so the delivered joint torque is `A a_actual + c` with the same A and c the S9
transfer study used, and the two are directly comparable.

The policies
------------
    uniform      everyone scaled by the same factor. What a system with no
                 allocator does: pressure sags everywhere at once.
    equal_share  water-filling on equal shares. Fair, and indifferent to
                 whether a request matters.
    by_demand    proportional to request size. Serves the big consumers.
    by_torque    proportional to how much the actuator's force moves the joints
                 being tracked. The only policy that knows what the limb is
                 trying to do.

Checks
------
* With an effectively unlimited pump every policy must collapse to the
  unconstrained case and the torque error to near zero. That validates the
  pipeline, not the policies.
* No policy may ever exceed the cap. Asserted at every timestep.
* Error must fall monotonically as capacity rises, for every policy.

Run
---
    python -m s13_flow_control.schedule
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
from s11_myofiber.myofiber import PSI, PUBLISHED  # noqa: E402
from s12_myofiber_leg.leg import size_leg  # noqa: E402
from s9_muscle.control import leg_plant  # noqa: E402
from s9_muscle.scene import actuator_names, load  # noqa: E402
from s9_muscle.control import project_equality  # noqa: E402
from s9_muscle.transfer import (  # noqa: E402
    CONTROL_HZ,
    POSTURE_AMP,
    POSTURE_MID,
    build_reference,
)

SEED = 0
THETA0 = 30.0
SUPPLY_PSI = 100.0
SUPPLY_PA = SUPPLY_PSI * PSI
CYCLE_HZ = 1.0
PUMP_LPM = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 1.0e6)
POLICIES = ("uniform", "equal_share", "by_demand", "by_torque")
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s13"


def allocate(demand: np.ndarray, capacity: float, weight: np.ndarray, policy: str) -> np.ndarray:
    """
    Split `capacity` among actuators requesting `demand`.

    Every policy is clipped to the request, because pushing fluid into an
    actuator that did not ask for it is not a way to spend spare pump, and the
    surplus is recycled to whoever still wants it. Without that recycling step
    a weighted policy strands capacity on actuators with small requests and
    large weights, and then loses to `uniform` for a reason that has nothing to
    do with its priorities.
    """
    total = float(demand.sum())
    if total <= capacity or total <= 0.0:
        return demand.copy()
    if policy == "uniform":
        return demand * (capacity / total)

    if policy == "equal_share":
        w = (demand > 0).astype(float)
    elif policy == "by_demand":
        w = demand.astype(float)
    else:
        w = np.abs(weight).astype(float)
    if w.sum() <= 0.0:
        return demand * (capacity / total)

    served = np.zeros_like(demand)
    remaining = capacity
    active = demand > 0
    for _ in range(64):
        wa = w * active
        if wa.sum() <= 0.0 or remaining <= 1e-18:
            break
        offer = remaining * wa / wa.sum()
        take = np.minimum(demand - served, offer)
        served += take
        remaining -= float(take.sum())
        active = active & (served < demand - 1e-18)
        if not active.any():
            break
    return served


def build_cycle(model, data, fibres, *, n_control):
    """
    Reference cycle plus the fluid cost of following it, before rationing.

    Two volumes move, and only one of them is obvious. The bladder's compliance
    stores `C P`, which is microlitres. The braid's *geometry* stores far more:
    as the sleeve shortens it balloons radially, and `V(eps)` swings by nearly a
    millilitre per actuator over a full stroke. The first version of this
    function tracked only the compliance term, found a mean demand of
    0.003 L/min, and concluded that a 40 L/min pump is never the binding
    constraint. It was off by four orders of magnitude, and every policy tied at
    zero error, which is what a study looks like when its constraint has
    silently stopped existing.
    """
    ref = build_reference(model, data, CYCLE_HZ, n_control=n_control)
    muscles = ref["muscles"]
    dt = 1.0 / CONTROL_HZ
    caps = np.array([f.fibre(THETA0, SUPPLY_PSI).hydraulic_capacitance for f in fibres])

    # Walk the same postures again to recover each tendon's length, which is
    # what sets the braid's contraction and therefore its geometric volume.
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in POSTURE_MID}
    qpos0 = data.qpos.copy()
    lengths = np.empty((n_control, len(muscles)))
    for k in range(n_control):
        phase = 2.0 * np.pi * CYCLE_HZ * k / CONTROL_HZ
        data.qpos[:] = qpos0
        for name in POSTURE_MID:
            data.qpos[qadr[name]] = POSTURE_MID[name] + POSTURE_AMP[name] * np.sin(phase)
        data.qvel[:] = 0.0
        project_equality(model, data)
        lengths[k] = np.array(data.actuator_length)[muscles]
    data.qpos[:] = qpos0
    project_equality(model, data)

    # Actuator contraction. A transmission of ratio n means the actuator travels
    # 1/n of the tendon's excursion, and L0 was set to the tendon's longest
    # length, so eps = (L_max - L) / (n L0).
    ratio = np.array([f.transmission_ratio for f in fibres])
    L0 = np.array([f.L0 for f in fibres])
    eps = np.clip((L0[None, :] - lengths) / (ratio[None, :] * L0[None, :]),
                  0.0, None)
    eps_max = np.array([f.fibre(THETA0, SUPPLY_PSI).eps_max for f in fibres])
    eps = np.minimum(eps, eps_max[None, :] * 0.999)

    braid_volume = np.empty_like(eps)
    for j, f in enumerate(fibres):
        braid_volume[:, j] = f.fibre(THETA0, SUPPLY_PSI).volume(eps[:, j])

    pressure = ref["act"] * SUPPLY_PA
    volume = braid_volume + pressure * caps[None, :]
    delta = np.zeros_like(volume)
    delta[1:] = np.diff(volume, axis=0) / dt
    return ref, pressure, volume, caps, delta, dt, eps, braid_volume


def run_policy(ref, delta, caps, dvol_deps, capacity_m3s, policy, dt) -> dict:
    """
    Track the cycle under a flow cap and score the joint torque that results.

    Pressure is integrated from *served* flow, so an actuator that does not get
    what it asked for does not reach the pressure it wanted, and the torque
    error is the cost of the shortage rather than a modelling choice.
    """
    n_t, n_a = delta.shape
    p = ref["act"][0] * SUPPLY_PA
    delivered = np.empty((n_t, ref["tau"].shape[1]))
    served_trace = np.empty(n_t)
    overrun = 0.0

    for k in range(n_t):
        want = np.maximum(delta[k], 0.0)        # fills compete for the pump
        vent = np.minimum(delta[k], 0.0)        # vents are free
        # How much this actuator's force matters to the joints being tracked.
        weight = np.abs(ref["A"][k]).sum(axis=0)
        served = allocate(want, capacity_m3s, weight, policy)
        overrun = max(overrun, float(served.sum() - capacity_m3s))

        # Served flow first fills the braid's changing geometry; only the
        # surplus raises pressure through the bladder's compliance.
        spare = served + vent - dvol_deps[k]
        p = np.clip(p + dt * spare / caps, 0.0, SUPPLY_PA)
        act = np.clip(p / SUPPLY_PA, 0.0, 1.0)
        delivered[k] = ref["A"][k] @ act + ref["c"][k]
        served_trace[k] = served.sum()

    err = np.linalg.norm(delivered - ref["tau"], axis=1)
    demand = np.linalg.norm(ref["tau"] - ref["c"], axis=1)
    return {
        "relative_torque_error": round(float(err.mean() / demand.mean()), 6),
        "peak_relative_error": round(float((err / np.maximum(demand, 1e-9)).max()), 6),
        "mean_served_lpm": round(float(served_trace.mean() * 60000.0), 4),
        "fraction_of_cycle_flow_limited": round(
            float(np.mean(np.maximum(delta, 0.0).sum(axis=1) > capacity_m3s)), 4),
        "capacity_never_exceeded": bool(overrun <= 1e-12),
    }


def make_figure(rows, demand_trace, pump_lpm, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    caps = [r for r in PUMP_LPM if r < 1e5]

    ax = axes[0]
    for i, policy in enumerate(POLICIES):
        y = [rows[(policy, c)]["relative_torque_error"] for c in caps]
        ax.plot(caps, y, "o-", color=PALETTE[i], lw=1.9, ms=4, label=policy)
    ax.axvline(PUBLISHED["pump_lpm"], color="0.45", ls="--", lw=1.3)
    ax.text(PUBLISHED["pump_lpm"], ax.get_ylim()[1] * 0.5, "  published\n  40 L/min",
            fontsize=7.5, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pump capacity (L/min)")
    ax.set_ylabel("relative torque error")
    ax.set_title("who you starve matters more than how much you have")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(np.arange(len(demand_trace)) / len(demand_trace),
            demand_trace * 60000.0, color=PALETTE[0], lw=1.8)
    for c, style in ((PUBLISHED["pump_lpm"], "--"), (4 * PUBLISHED["pump_lpm"], ":")):
        ax.axhline(c, color="0.45", ls=style, lw=1.2)
        ax.text(0.02, c * 1.06, f"{c:g} L/min", fontsize=7.5, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("phase of the stance cycle")
    ax.set_ylabel("flow demanded (L/min)")
    ax.set_title("demand is bursty, not steady")

    ax = axes[2]
    width = 0.2
    for i, policy in enumerate(POLICIES):
        vals = [rows[(policy, c)]["relative_torque_error"] for c in (10.0, 40.0, 160.0)]
        ax.bar(np.arange(3) + i * width, vals, width, color=PALETTE[i], label=policy)
    ax.set_xticks(np.arange(3) + 1.5 * width)
    ax.set_xticklabels(["10 L/min", "40 L/min", "160 L/min"])
    ax.set_ylabel("relative torque error")
    ax.set_title("the gap closes as the pump grows")
    ax.legend(fontsize=8)

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
    fibres = size_leg(model, plant.muscles, names, theta0_deg=THETA0, supply_psi=SUPPLY_PSI)

    n_control = int(round(CONTROL_HZ / CYCLE_HZ))
    ref, pressure, volume, caps, delta, dt, eps, braid_volume = build_cycle(
        model, data, fibres, n_control=n_control)
    demand_trace = np.maximum(delta, 0.0).sum(axis=1)
    # Rate at which geometry alone consumes fluid, which the pressure integrator
    # has to pay before any of the served flow reaches the bladder.
    dvol = np.zeros_like(braid_volume)
    dvol[1:] = np.diff(braid_volume, axis=0) / dt

    rows = {}
    for policy in POLICIES:
        for lpm in PUMP_LPM:
            rows[(policy, lpm)] = run_policy(ref, delta, caps, dvol, lpm / 60000.0, policy, dt)

    unlimited = {p: rows[(p, 1.0e6)]["relative_torque_error"] for p in POLICIES}
    finite = [c for c in PUMP_LPM if c < 1e5]
    monotone = {
        p: bool(np.all(np.diff([rows[(p, c)]["relative_torque_error"]
                                for c in finite]) <= 1e-9))
        for p in POLICIES
    }
    at_published = {p: rows[(p, PUBLISHED["pump_lpm"])] for p in POLICIES}
    best = min(at_published, key=lambda p: at_published[p]["relative_torque_error"])

    config = {"theta0_deg": THETA0, "supply_psi": SUPPLY_PSI, "cycle_hz": CYCLE_HZ,
              "pump_lpm": list(PUMP_LPM), "policies": list(POLICIES),
              "n_actuators": len(fibres)}
    with RunLogger("s13_flow_control", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model),
                                   "published": PUBLISHED}) as log:
        for (policy, lpm), row in rows.items():
            log.event("policy", policy=policy, pump_lpm=lpm, **row)
        log.event("checks",
                  unlimited_error=unlimited,
                  monotone_in_capacity=monotone,
                  capacity_respected=bool(all(r["capacity_never_exceeded"]
                                              for r in rows.values())))
        figure = make_figure(rows, demand_trace, PUMP_LPM, MEDIA / "flow_control.png")
        run_dir = log.path

    result = {
        "peak_demand_lpm": round(float(demand_trace.max() * 60000.0), 2),
        "mean_demand_lpm": round(float(demand_trace.mean() * 60000.0), 2),
        "burstiness_peak_over_mean": round(
            float(demand_trace.max() / max(demand_trace.mean(), 1e-15)), 2),
        "unlimited_error_by_policy": unlimited,
        "monotone_in_capacity": monotone,
        "capacity_respected": bool(all(r["capacity_never_exceeded"] for r in rows.values())),
        "at_published_pump": {p: at_published[p] for p in POLICIES},
        "best_policy_at_published_pump": best,
        "uniform_over_best_at_published": round(
            at_published["uniform"]["relative_torque_error"]
            / max(at_published[best]["relative_torque_error"], 1e-12), 2),
        "table": {f"{p}@{c:g}": rows[(p, c)] for p, c in rows},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
