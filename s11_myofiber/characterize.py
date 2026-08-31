"""
S11: characterising a Myofiber-class hydraulic actuator, and what it fixes.

Five measurements. The first three establish that the model is the actuator it
claims to be; the last two are the reason it was worth building.

  1. Specification check   the sized actuator against Clone's published
                           figures, every comparison an inequality in the
                           direction the specification states.
  2. Force and volume      F(P, eps) across the operating range, and the fluid
                           volume that goes with it. The volume is the half
                           people forget, and it is what the pump has to pay
                           for.
  3. Valve dynamics        orifice flow into a compliant bladder. Fill and vent
                           are asymmetric, and unlike a Hill muscle's fixed
                           time constants the asymmetry *depends on operating
                           pressure*, which is a different control problem.
  4. Flow budget           roughly 1000 muscles, one 40 L/min pump. How many
                           can actually move at once.
  5. Antagonist stiffness  the S9-C result, redone. MS-Human-700's
                           co-contraction stiffness changes sign over ~40% of
                           every joint's range because the model has no series
                           elasticity. With a real series element the sign
                           stops flipping.

Run
---
    python -m s11_myofiber.characterize
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import PALETTE  # noqa: E402
from common.runlog import RunLogger  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from s11_myofiber.myofiber import PSI, PUBLISHED, Myofiber  # noqa: E402

SEED = 0
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s11"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- 3. valve dynamics


def step_response(fibre: Myofiber, *, dt=2e-6, seconds=0.25) -> dict:
    """
    Isometric pressurisation, then venting, then a loaded contraction.

    The isometric part measures the hydraulics alone, the way a real actuator
    is bench-tested. Note what it does *not* measure well: orifice flow scales
    with sqrt(supply - P), so the driving head vanishes as the actuator
    approaches supply and the last few percent take forever. A 10-90% time
    against *supply pressure* is therefore the wrong statistic, and the first
    version of this function returned NaN for it, correctly. Time to 63% of
    supply, the natural time constant of the fill, is reported instead.

    The loaded contraction is what a specification means by "response time":
    a 1 kg load, the published rated force, lifted from rest.
    """
    n = int(seconds / dt)
    half = n // 2
    supply = fibre.supply_pressure
    pressure = 0.0
    trace = np.empty(n)
    for i in range(n):
        command = 1.0 if i < half else -1.0
        pressure = float(np.clip(
            pressure + dt * fibre.pressure_rate(pressure, 0.0, 0.0, command),
            0.0, supply))
        trace[i] = pressure
    t = np.arange(n) * dt

    def cross(values, times, level, rising):
        mask = values >= level if rising else values <= level
        if not mask.any() or mask[0]:
            return float("nan")
        j = int(np.argmax(mask))
        v0, v1 = values[j - 1], values[j]
        if v1 == v0:
            return float(times[j])
        return float(times[j - 1] + (level - v0) * (times[j] - times[j - 1]) / (v1 - v0))

    fill, vent = trace[:half], trace[half:]
    t_fill, t_vent = t[:half], t[half:] - t[half]
    fill_63 = cross(fill, t_fill, 0.632 * supply, True)
    vent_37 = cross(vent, t_vent, 0.368 * supply, False)

    # Loaded contraction against the published rated load.
    load = PUBLISHED["min_force_N"] / 9.81        # kg
    eps, rate, pressure = 0.0, 0.0, 0.0
    lo_n = int(0.12 / dt)
    contraction = np.empty(lo_n)
    for i in range(lo_n):
        force = float(fibre.force(pressure, eps))
        accel = (force - load * 9.81) / (load * fibre.L0)
        rate = max(0.0, rate + dt * accel)
        eps = float(np.clip(eps + dt * rate, 0.0, fibre.eps_max))
        if eps in (0.0, fibre.eps_max):
            rate = 0.0
        pressure = float(np.clip(
            pressure + dt * fibre.pressure_rate(pressure, eps, rate, 1.0), 0.0, supply))
        contraction[i] = eps
    t_load = np.arange(lo_n) * dt
    settled = float(contraction[-1])
    t90 = cross(contraction, t_load, 0.90 * settled, True) if settled > 0 else float("nan")

    # What the published response time actually demands of the valve. The
    # mean-head estimate that sized the default ignores the pressure build-up
    # transient and the load's inertia, so it undershoots; solving for the area
    # that hits the spec turns "under 50 ms" into a component requirement.
    required = required_valve_area(fibre)

    return {
        "required_valve_area_m2": required["area_m2"],
        "required_orifice_diameter_mm": required["diameter_mm"],
        "valve_undersize_factor": required["undersize_factor"],
        "fill_to_63pct_ms": round(1e3 * fill_63, 3),
        "vent_to_37pct_ms": round(1e3 * vent_37, 3),
        "fill_vent_asymmetry": round(fill_63 / vent_37, 3),
        "loaded_contraction_final": round(settled, 4),
        "loaded_t90_ms": round(1e3 * t90, 2),
        "load_kg": round(load, 3),
        "meets_published_response": bool(1e3 * t90 < PUBLISHED["response_ms"]),
        "_trace": {"t": t, "p": trace, "supply": supply, "half": half,
                   "t_load": t_load, "eps": contraction},
    }


def loaded_t90(fibre: Myofiber, *, dt=2e-6, seconds=0.12) -> float:
    """Time to 90% of final contraction under the published rated load."""
    load = PUBLISHED["min_force_N"] / 9.81
    supply = fibre.supply_pressure
    eps = rate = pressure = 0.0
    n = int(seconds / dt)
    trace = np.empty(n)
    for i in range(n):
        accel = (float(fibre.force(pressure, eps)) - load * 9.81) / (load * fibre.L0)
        rate = max(0.0, rate + dt * accel)
        eps = float(np.clip(eps + dt * rate, 0.0, fibre.eps_max))
        if eps in (0.0, fibre.eps_max):
            rate = 0.0
        pressure = float(np.clip(
            pressure + dt * fibre.pressure_rate(pressure, eps, rate, 1.0), 0.0, supply))
        trace[i] = eps
    settled = trace[-1]
    if settled <= 0.0:
        return float("inf")
    hit = np.argmax(trace >= 0.90 * settled)
    return float("inf") if trace[hit] < 0.90 * settled else float(hit * dt)


def required_valve_area(fibre: Myofiber, *, target_ms=None) -> dict:
    """
    Bisect on orifice area for the valve that just meets the published response.

    Reported as a derived component requirement rather than used to re-tune the
    default: the point is that the naive mean-head sizing is optimistic, and by
    how much.
    """
    target = (target_ms or PUBLISHED["response_ms"]) / 1000.0
    baseline = fibre.valve_area
    lo, hi = baseline, baseline * 16.0
    trial = Myofiber(**{k: v for k, v in vars(fibre).items()})
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial.valve_area = mid
        if loaded_t90(trial) <= target:
            hi = mid
        else:
            lo = mid
    area = hi
    return {
        "area_m2": float(f"{area:.4g}"),
        "diameter_mm": round(2e3 * np.sqrt(area / np.pi), 3),
        "undersize_factor": round(area / baseline, 2),
    }


# ---------------------------------------------------------- 4. flow budget


def flow_budget(fibre: Myofiber, *, n_muscles=None, stroke_ms=None) -> dict:
    """
    How many actuators the published pump can drive at once.

    A full stroke moves a fixed volume, so demanding it in a fixed time sets a
    flow rate per actuator. The pump rating then divides into a count. This is
    a hard resource constraint that no muscle model carries and no
    torque-actuated humanoid has an analogue of: a motor draws current from a
    bus that can usually supply it, whereas here the working fluid itself is
    rationed.
    """
    n_muscles = n_muscles or PUBLISHED["protoclone_muscles"]
    stroke_ms = stroke_ms or PUBLISHED["response_ms"]
    pump = PUBLISHED["pump_lpm"] / 60.0 / 1000.0            # m^3/s
    stroke_volume = abs(float(fibre.volume(fibre.eps_max) - fibre.volume(0.0)))
    per_actuator = stroke_volume / (stroke_ms / 1000.0)

    fractions = np.linspace(0.05, 1.0, 20)
    simultaneous = pump / per_actuator
    curve = [
        {
            "stroke_fraction": round(float(f), 3),
            "actuators_at_full_speed": int(np.floor(simultaneous / f)),
            "fraction_of_body": round(float(min(1.0, (simultaneous / f) / n_muscles)), 4),
        }
        for f in fractions
    ]
    return {
        "pump_lpm": PUBLISHED["pump_lpm"],
        "n_muscles": n_muscles,
        "stroke_volume_mL": round(1e6 * stroke_volume, 4),
        "flow_per_actuator_lpm": round(per_actuator * 60000.0, 3),
        "full_stroke_simultaneous": int(np.floor(simultaneous)),
        "fraction_of_body_at_full_stroke": round(float(simultaneous / n_muscles), 4),
        "whole_body_stroke_volume_L": round(1e3 * stroke_volume * n_muscles, 3),
        "seconds_to_fill_whole_body": round(stroke_volume * n_muscles / pump, 2),
        "curve": curve,
    }


# ------------------------------------------------- 5. antagonist stiffness


def antagonist_stiffness(fibre: Myofiber, *, moment_arm=0.03, pressures=None) -> dict:
    """
    Joint stiffness from an antagonist pair, swept over the joint's range.

    Two identical actuators on opposite sides of a revolute joint, both held at
    the same pressure, so the pair produces no net torque at the neutral angle
    and pure stiffness around it. Exactly the configuration S9-C measured on
    MS-Human-700, where the sign flipped over roughly 40% of every joint's
    range.

    Stiffness is a central difference of the net torque, the same protocol as
    S9-C, so the two are directly comparable.
    """
    pressures = pressures or [20.0, 40.0, 60.0, 80.0, 100.0]
    angles = np.linspace(-0.5, 0.5, 41)
    delta = 1e-4
    rest = fibre.L0 + fibre.slack

    def torque(pressure, angle):
        # Agonist shortens as the angle grows, antagonist lengthens.
        s1 = fibre.solve_length(pressure, rest - moment_arm * angle)
        s2 = fibre.solve_length(pressure, rest + moment_arm * angle)
        ok = s1["converged"] and s2["converged"]
        return moment_arm * (s1["tension_N"] - s2["tension_N"]), ok

    out = {}
    for psi in pressures:
        p = psi * PSI
        k, valid = [], []
        for angle in angles:
            plus, ok_p = torque(p, angle + delta)
            minus, ok_m = torque(p, angle - delta)
            k.append(-(plus - minus) / (2.0 * delta))
            valid.append(ok_p and ok_m)
        k = np.array(k)
        valid = np.array(valid)
        good = k[valid]
        out[f"{psi:g}"] = {
            "K_Nm_per_rad": [round(float(v), 3) if ok else None
                             for v, ok in zip(k, valid)],
            "n_converged": int(valid.sum()),
            "n_pinned": int((~valid).sum()),
            "min_K": round(float(good.min()), 3) if good.size else None,
            "max_K": round(float(good.max()), 3) if good.size else None,
            "fraction_negative": round(float(np.mean(good < 0.0)), 3) if good.size else None,
            "sign_changes": int(np.count_nonzero(np.diff(np.sign(good)) != 0)) if good.size else 0,
            "K_at_neutral": round(float(k[len(angles) // 2]), 3),
        }
    highest = out[f"{max(pressures):g}"]
    lowest = out[f"{min(pressures):g}"]
    neutral = [out[f"{p:g}"]["K_at_neutral"] for p in pressures]
    return {
        "moment_arm_m": moment_arm,
        "angles_rad": [round(float(a), 4) for a in angles],
        "per_pressure": out,
        "any_negative_anywhere": bool(any(v["fraction_negative"] > 0 for v in out.values())),
        "any_sign_change": bool(any(v["sign_changes"] > 0 for v in out.values())),
        # Compared at the neutral angle, where both actuators are on the same
        # side of their equilibrium. Comparing maxima across pressures compares
        # different joint angles and is not the same claim.
        "K_scales_with_pressure": bool(np.all(np.diff(neutral) > 0)),
        "K_at_neutral_by_pressure": {f"{p:g}": v for p, v in zip(pressures, neutral)},
        # Reported as a measured deviation rather than a pass/fail against an
        # arbitrary threshold. It comes out at 2%, which a 2% test would have
        # called a failure and a 3% test a success, and neither would have been
        # informative.
        "K_pressure_proportionality_max_deviation": round(float(np.max(np.abs(
            np.array(neutral) / np.array(pressures) / (neutral[0] / pressures[0]) - 1.0))), 4),
        "K_at_neutral_max_pressure": highest["K_at_neutral"],
        "max_K_overall": highest["max_K"],
        "total_pinned_points": int(sum(v["n_pinned"] for v in out.values())),
    }


# ------------------------------------------------------------------ figure


def make_figure(spec, surface, dyn, budget, stiff, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.1))

    ax = axes[0]
    for i, psi in enumerate(surface["pressures_psi"]):
        ax.plot(surface["eps"], surface["force_N"][i], color=PALETTE[i % len(PALETTE)],
                lw=1.9, label=f"{psi:g} psi")
    ax.axvline(spec["eps_max"], color="0.45", ls=":", lw=1.2)
    ax.text(spec["eps_max"], ax.get_ylim()[1] * 0.6, f"  blocking\n  {spec['eps_max']:.3f}",
            fontsize=7.5, color="0.35")
    ax.set_xlabel("contraction  $\\epsilon$")
    ax.set_ylabel("force (N)")
    ax.set_title("force falls to zero at the braid's blocking angle")
    ax.legend(fontsize=7.5)

    # Only the loaded contraction goes on this axis. The isometric fill has a
    # 0.35 ms time constant against a 70 ms contraction, and putting both on
    # one time axis renders the fast one as a vertical line and tells nobody
    # anything.
    ax = axes[1]
    tr = dyn["_trace"]
    ax.plot(tr["t_load"] * 1e3, tr["eps"], color=PALETTE[1], lw=2.0)
    ax.axhline(dyn["loaded_contraction_final"], color="0.5", ls="--", lw=1.1)
    ax.axvline(PUBLISHED["response_ms"], color=PALETTE[3], ls=":", lw=1.4)
    ax.text(PUBLISHED["response_ms"], 0.05, "  published 50 ms", fontsize=7.5,
            color=PALETTE[3])
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("contraction $\epsilon$ under 1 kg")
    ax.set_title(f"t90 {dyn['loaded_t90_ms']:.0f} ms: valve is "
                 f"{dyn['valve_undersize_factor']:g}x undersized")

    ax = axes[2]
    fr = [c["stroke_fraction"] for c in budget["curve"]]
    n = [c["actuators_at_full_speed"] for c in budget["curve"]]
    ax.plot(fr, n, "o-", color=PALETTE[1], lw=1.9, ms=3.5)
    ax.axhline(budget["n_muscles"], color="0.45", ls="--", lw=1.2)
    ax.text(0.5, budget["n_muscles"] * 1.05, f" {budget['n_muscles']} muscles on the body",
            fontsize=7.5, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("stroke fraction demanded")
    ax.set_ylabel(f"actuators at once, in {PUBLISHED['response_ms']:g} ms")
    ax.set_title(f"a {budget['pump_lpm']:g} L/min pump is the real budget")

    ax = axes[3]
    for i, (psi, row) in enumerate(stiff["per_pressure"].items()):
        vals = [np.nan if v is None else v for v in row["K_Nm_per_rad"]]
        ax.plot(stiff["angles_rad"], vals,
                color=PALETTE[i % len(PALETTE)], lw=1.8, label=f"{psi} psi")
    ax.axhline(0.0, color="0.45", lw=1.1)
    ax.set_xlabel("joint angle (rad)")
    ax.set_ylabel("stiffness (N·m/rad)")
    ax.set_title("positive everywhere, unlike the Hill model")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    fibre = Myofiber()

    spec = fibre.specification_check()
    eps = np.linspace(0.0, fibre.eps_max, 120)
    pressures_psi = [20.0, 40.0, 60.0, 80.0, 100.0]
    surface = {
        "eps": eps.tolist(),
        "pressures_psi": pressures_psi,
        "force_N": [fibre.force(p * PSI, eps).tolist() for p in pressures_psi],
        "volume_mL": (1e6 * fibre.volume(eps)).tolist(),
        "radius_mm": (1e3 * fibre.radius(eps)).tolist(),
    }
    dyn = step_response(fibre)
    # Same actuator, realistic return-line back-pressure. The asymmetry a Hill
    # muscle has by construction has to be plumbed in here.
    back = Myofiber(**{k: v for k, v in vars(fibre).items()})
    back.return_pressure = 15.0 * PSI
    dyn_back = step_response(back)
    dyn["with_15psi_return_fill_ms"] = dyn_back["fill_to_63pct_ms"]
    dyn["with_15psi_return_vent_ms"] = dyn_back["vent_to_37pct_ms"]
    dyn["with_15psi_return_asymmetry"] = dyn_back["fill_vent_asymmetry"]
    budget = flow_budget(fibre)
    stiff = antagonist_stiffness(fibre)

    config = {"r0": fibre.r0, "L0": fibre.L0, "theta0_deg": fibre.theta0_deg,
              "k_series": fibre.k_series, "valve_area": fibre.valve_area,
              "hydraulic_capacitance": fibre.hydraulic_capacitance}
    with RunLogger("s11_myofiber", config=config, seed=SEED,
                   extra_manifest={"published": PUBLISHED}) as log:
        log.event("specification", **spec)
        log.event("dynamics", **{k: v for k, v in dyn.items() if not k.startswith("_")})
        log.event("flow_budget", **{k: v for k, v in budget.items() if k != "curve"})
        log.event("antagonist_stiffness",
                  **{k: v for k, v in stiff.items()
                     if k not in ("per_pressure", "angles_rad")})
        figure = make_figure(spec, surface, dyn, budget, stiff, MEDIA / "myofiber.png")
        run_dir = log.path

    summary = {
        "specification": spec,
        "dynamics": {k: v for k, v in dyn.items() if not k.startswith("_")},
        "flow_budget": budget,
        "antagonist_stiffness": stiff,
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "specification": spec,
        "dynamics": {k: v for k, v in dyn.items() if not k.startswith("_")},
        "flow_budget": {k: v for k, v in budget.items() if k != "curve"},
        "antagonist_stiffness": {k: v for k, v in stiff.items()
                                 if k not in ("per_pressure", "angles_rad")},
    }, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
