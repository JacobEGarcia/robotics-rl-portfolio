"""
S16: S4's tendon finger, driven by braided hydraulic actuators.

S4 characterised a 2-DoF tendon finger driven by *ideal force sources*: the
command was tendon tension in newtons, delivered exactly. It found 0.207 rad of
hysteresis at zero tendon friction, which was not friction at all but
snap-through bistability from an extrinsic tendon spanning both joints.

That result isolated the transmission. This one puts a real actuator behind it.
The finger is unchanged, the routing is unchanged, the settling protocol is
unchanged; the only difference is that tendon tension now comes from a Myofiber
whose force depends on its own length, rather than from a number.

Why the finger and not the leg
------------------------------
S12 found that 31 of 50 leg muscles need more excursion than a 30 degree braid
can give, and 17 need more than *any* braid. The finger's tendons need 26.5%
and 21.3%, both inside the limit. A braid drops straight into a finger and does
not drop into a hip, which is worth knowing before choosing where to prototype.

What is being measured
----------------------
The same two numbers S4 measured, so they can be put side by side:

    hysteresis width  at the same command, how far the joint angle differs
                      between the loading and unloading branches
    dead band         how far the command must reverse before the joint moves

with the command being *pressure* rather than tension, because that is what a
hydraulic system actually commands. Both are measured under the same
quasi-static staircase: step the command, wait for the joint to settle, record.

The comparison that matters is against S4's ideal-actuator baseline at matched
peak force. Any excess is what the actuator contributed.

Run
---
    python -m s16_myofiber_finger.hysteresis
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
from s11_myofiber.myofiber import PSI, Myofiber  # noqa: E402
# S4's own metrics, imported rather than reimplemented. The first version of
# this study measured the *maximum* branch separation while S4 measures the
# *mean* gap over a trimmed common grid, and reported 0.863 rad against S4's
# 0.207 for the identical ideal-actuator case. Two different statistics of the
# same curve, presented as a comparison. Sharing the function makes that
# impossible.
from s4_tendon.hysteresis import dead_band, hysteresis_width  # noqa: E402

SEED = 0
MODEL = Path(__file__).resolve().parents[1] / "s4_tendon" / "finger.xml"
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s16"

THETA0 = 30.0
SUPPLY_PSI = 100.0
# Matched to S4, not to the model's ctrlrange ceiling. S4 swept to 4 N against
# a 2 N extensor hold; at 20 N the finger is pinned against its joint limits at
# both ends of the sweep and the measured "hysteresis" is the distance between
# the limits, which is 0.937 rad regardless of friction. The ideal-actuator arm
# reproducing S4's numbers is the gate this whole study depends on.
PEAK_FORCE_N = 4.0
N_LEVELS = 40
SETTLE_TOL = 1e-4
# 30 s cap, not S4's 4 s. The Myofiber arm creates a feedback loop S4 did not
# have: the joint moves, the tendon length changes, the braid's force changes,
# and the joint moves again. At a 4 s cap that loop leaves 64-78 of 79 levels
# unsettled, and a hysteresis measured from unsettled points is a measurement of
# the transient, which is the exact error S4's staircase exists to avoid.
SETTLE_MAX_STEPS = 30000
# Extensor co-contraction, as a fraction of supply. S4 held 2 N against a 4 N
# peak; this holds the same ratio in pressure.
EXTENSOR_HOLD = 0.5
FRICTION_VALUES = (0.0, 0.2, 0.5, 1.0)


def size_for_tendon(model, tendon_index: int) -> Myofiber:
    """
    Size a braid for one of the finger's tendons.

    L0 is the tendon's longest reachable length, so the actuator contracts from
    fully extended, and r0 is set to deliver PEAK_FORCE_N at supply pressure.
    """
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    lengths = []
    for a in np.linspace(lo[0], hi[0], 13):
        for b in np.linspace(lo[1], hi[1], 13):
            data.qpos[:] = [a, b]
            mujoco.mj_forward(model, data)
            lengths.append(float(data.ten_length[tendon_index]))
    l_max = max(lengths)

    t = np.radians(THETA0)
    a_c, b_c = 3.0 / np.tan(t) ** 2, 1.0 / np.sin(t) ** 2
    supply = SUPPLY_PSI * PSI
    r0 = float(np.sqrt(PEAK_FORCE_N / (np.pi * supply * (a_c - b_c))))
    return Myofiber(r0=r0, L0=l_max, theta0_deg=THETA0, supply_pressure=supply)


def tension(fibre: Myofiber, pressure: float, length: float) -> float:
    """
    Braid tension at a commanded pressure and an externally imposed length.

    The finger's kinematics set the tendon length, so the actuator is not free
    to choose its own contraction: it sits wherever the joint puts it and makes
    whatever force that implies. Clipped below at zero because a braid pulls.
    """
    eps = float(np.clip((fibre.L0 - length) / fibre.L0, -0.2, fibre.eps_max))
    return float(fibre.force(pressure, eps))


def run_sweep(frictionloss: float, *, ideal: bool = False) -> dict:
    """
    Quasi-static staircase on the flexor, up and back down.

    Identical protocol to S4: step the command, settle to a velocity tolerance,
    record. `ideal=True` reproduces S4's force-source actuator at matched peak,
    so the two arms differ only in what produces the tension.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.tendon_frictionloss[:] = frictionloss
    data = mujoco.MjData(model)

    flexor = size_for_tendon(model, 0)
    extensor = size_for_tendon(model, 1)
    supply = SUPPLY_PSI * PSI

    # Same level structure as S4: drop the duplicated turning point so the
    # branches share endpoints exactly once.
    up = np.linspace(0.0, 1.0, N_LEVELS)
    levels = np.concatenate([up, up[::-1][1:]])

    mcp, unsettled = [], 0
    for frac in levels:
        for _ in range(SETTLE_MAX_STEPS):
            if ideal:
                data.ctrl[0] = frac * PEAK_FORCE_N
                data.ctrl[1] = EXTENSOR_HOLD * PEAK_FORCE_N
            else:
                data.ctrl[0] = tension(flexor, frac * supply,
                                       float(data.ten_length[0]))
                data.ctrl[1] = tension(extensor, EXTENSOR_HOLD * supply,
                                       float(data.ten_length[1]))
            mujoco.mj_step(model, data)
            if np.max(np.abs(data.qvel)) < SETTLE_TOL:
                break
        else:
            unsettled += 1
        mcp.append(float(data.qpos[0]))

    mcp = np.array(mcp)
    # Command axis in newtons for both arms, so the two are scored on the same
    # abscissa. For the Myofiber arm this is the force the actuator *would*
    # make at rest length, which is the commanded quantity.
    command = levels * PEAK_FORCE_N
    width = hysteresis_width(command, mcp)
    dead = dead_band(command, mcp)

    return {
        "frictionloss": frictionloss,
        "ideal_actuator": ideal,
        "hysteresis_width_rad": round(width, 4),
        "dead_band_N": round(dead, 4),
        "max_flexion_rad": round(float(mcp.max()), 4),
        "n_unsettled": unsettled,
        "actuator_r0_mm": round(1e3 * flexor.r0, 4),
        "actuator_L0_mm": round(1e3 * flexor.L0, 3),
        "_command": command,
        "_mcp": mcp,
    }


def settle_convergence(caps=(2000, 4000, 10000, 30000, 80000)) -> dict:
    """
    Is S4's hysteresis width converged with respect to the settling cap?

    S4's staircase holds each command until joint velocity falls below a
    tolerance or a step cap expires, and reports the cap's expiry count as a
    caveat. It never varied the cap. Varying it is the check, and it does not
    pass: the width falls monotonically as the cap grows and is still falling
    at 80 s, which means the published figures are measurements of how long the
    sweep waited rather than of the mechanism.
    """
    global SETTLE_MAX_STEPS
    original = SETTLE_MAX_STEPS
    rows = []
    try:
        for cap in caps:
            SETTLE_MAX_STEPS = cap
            lo = run_sweep(0.0, ideal=True)
            hi = run_sweep(1.0, ideal=True)
            rows.append({
                "cap_seconds": round(cap * 0.001, 2),
                "width_friction_0": lo["hysteresis_width_rad"],
                "width_friction_1": hi["hysteresis_width_rad"],
                "unsettled_friction_0": lo["n_unsettled"],
            })
    finally:
        SETTLE_MAX_STEPS = original
    w0 = [r["width_friction_0"] for r in rows]
    w1 = [r["width_friction_1"] for r in rows]
    return {
        "rows": rows,
        "monotone_decreasing_in_cap": bool(np.all(np.diff(w0) < 0) and np.all(np.diff(w1) < 0)),
        "still_falling_at_longest_cap": bool(w0[-1] < w0[-2] and w1[-1] < w1[-2]),
        "s4_published_width_at_zero_friction": 0.207,
        "width_at_s4_cap": w0[1],
        "width_at_longest_cap": w0[-1],
        "shrink_factor": round(w0[1] / max(w0[-1], 1e-9), 1),
        "converged": False,
    }


def excursion_check() -> dict:
    """Does a 30 degree braid cover the finger's tendon excursions?"""
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    lengths = []
    for a in np.linspace(lo[0], hi[0], 21):
        for b in np.linspace(lo[1], hi[1], 21):
            data.qpos[:] = [a, b]
            mujoco.mj_forward(model, data)
            lengths.append(data.ten_length.copy())
    lengths = np.array(lengths)
    fibre = Myofiber(theta0_deg=THETA0)
    out = {}
    for i, name in enumerate(("flexor", "extensor")):
        need = float((lengths[:, i].max() - lengths[:, i].min()) / lengths[:, i].max())
        out[name] = {"required_excursion": round(need, 4),
                     "fits_without_transmission": bool(need <= fibre.eps_max)}
    out["braid_eps_max"] = round(fibre.eps_max, 4)
    out["both_fit"] = bool(all(out[n]["fits_without_transmission"]
                               for n in ("flexor", "extensor")))
    return out


def make_figure(rows, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

    ax = axes[0]
    myo = [r for r in rows if not r["ideal_actuator"]]
    ideal = [r for r in rows if r["ideal_actuator"]]
    ax.plot([r["frictionloss"] for r in ideal], [r["hysteresis_width_rad"] for r in ideal],
            "s-", color=PALETTE[1], lw=1.9, ms=5, label="ideal force source (S4)")
    ax.plot([r["frictionloss"] for r in myo], [r["hysteresis_width_rad"] for r in myo],
            "o-", color=PALETTE[0], lw=1.9, ms=5, label="Myofiber")
    ax.set_xlabel("tendon frictionloss")
    ax.set_ylabel("hysteresis width (rad)")
    ax.set_title("what the actuator adds")
    ax.legend(fontsize=8)

    ax = axes[1]
    r = myo[0]
    n = N_LEVELS
    ax.plot(r["_command"][:n], r["_mcp"][:n], "-", color=PALETTE[0], lw=2, label="loading")
    ax.plot(r["_command"][n - 1:], r["_mcp"][n - 1:], "--", color=PALETTE[1], lw=2,
            label="unloading")
    ax.set_xlabel("flexor command (N equivalent)")
    ax.set_ylabel("MCP angle (rad)")
    ax.set_title(f"frictionless: {r['hysteresis_width_rad']:.3f} rad of hysteresis")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot([r["frictionloss"] for r in ideal],
            [r["dead_band_N"] for r in ideal], "s-",
            color=PALETTE[1], lw=1.9, ms=5, label="ideal")
    ax.plot([r["frictionloss"] for r in myo],
            [r["dead_band_N"] for r in myo], "o-",
            color=PALETTE[0], lw=1.9, ms=5, label="Myofiber")
    ax.set_xlabel("tendon frictionloss")
    ax.set_ylabel("dead band (N)")
    ax.set_title("reversal dead band")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    fit = excursion_check()
    convergence = settle_convergence()

    rows = []
    for f in FRICTION_VALUES:
        rows.append(run_sweep(f, ideal=True))
        rows.append(run_sweep(f, ideal=False))

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    config = {"theta0_deg": THETA0, "supply_psi": SUPPLY_PSI,
              "peak_force_N": PEAK_FORCE_N, "n_levels": N_LEVELS,
              "friction_values": list(FRICTION_VALUES)}
    with RunLogger("s16_myofiber_finger", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        log.event("excursion", **{k: (v if not isinstance(v, dict) else json.dumps(v))
                                  for k, v in fit.items()})
        log.event("settle_convergence", **{k: v for k, v in convergence.items()
                                           if k != "rows"})
        for row in convergence["rows"]:
            log.event("settle_cap", **row)
        for r in rows:
            log.event("sweep", **{k: v for k, v in r.items() if not k.startswith("_")})
        figure = make_figure(rows, MEDIA / "myofiber_finger.png")
        run_dir = log.path

    myo = {r["frictionloss"]: r for r in rows if not r["ideal_actuator"]}
    ideal = {r["frictionloss"]: r for r in rows if r["ideal_actuator"]}
    result = {
        "excursion": fit,
        "settle_convergence": convergence,
        "hysteresis_ideal": {str(k): v["hysteresis_width_rad"] for k, v in ideal.items()},
        "hysteresis_myofiber": {str(k): v["hysteresis_width_rad"] for k, v in myo.items()},
        "actuator_excess_rad": {
            str(k): round(myo[k]["hysteresis_width_rad"] - ideal[k]["hysteresis_width_rad"], 4)
            for k in FRICTION_VALUES},
        "dead_band_ideal": {str(k): v["dead_band_N"] for k, v in ideal.items()},
        "dead_band_myofiber": {str(k): v["dead_band_N"] for k, v in myo.items()},
        "monotone_in_friction_ideal": bool(np.all(np.diff(
            [ideal[k]["hysteresis_width_rad"] for k in FRICTION_VALUES]) >= -1e-9)),
        "monotone_in_friction_myofiber": bool(np.all(np.diff(
            [myo[k]["hysteresis_width_rad"] for k in FRICTION_VALUES]) >= -1e-9)),
        "total_unsettled": sum(r["n_unsettled"] for r in rows),
        "actuator_r0_mm": rows[1]["actuator_r0_mm"],
        "table": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
