"""
S14: how much can you know about a hydraulic limb from pressure alone?

A musculoskeletal robot is hard to instrument. There is no clean place to put a
joint encoder on a tendon-routed skeleton, and Clone's own material talks about
pressure rather than position. So: what does a pressure vector actually tell
you about where the limb is?

The mechanism
-------------
Close an actuator's valve and its fluid is trapped, so

    V_braid(eps) + C P = constant

and any change in contraction has to be absorbed by pressure. That makes P a
readout of eps, and eps a readout of joint angle through the tendon's moment
arm. A closed Myofiber is, in principle, its own encoder.

In practice the same equation says something more useful and less convenient.

Three results, and the first one is not what the study was for
--------------------------------------------------------------
1. **A closed valve is a lock, not a sensor.** Water is near-incompressible, so
   trapping it makes the actuator 18x to 200x stiffer than the same actuator at
   constant pressure, depending on contraction. That is a genuinely useful
   property (a joint can be held with zero flow and zero power) and it is the
   reason for result 2.

2. **Pressure is an exquisite sensor over a uselessly small range.** Because the
   actuator is nearly rigid when closed, the entire 0 to 100 psi span
   corresponds to between 0.24 and 0.94 mm of travel, under 3% of a 33 mm
   stroke. Sub-micron resolution, and saturation before the joint has moved a
   degree.

3. **Pose is observable, and the conditioning says how well.** With valves
   closed, dP/dq is the moment-arm matrix scaled row-wise, so its rank is the
   rank of R restricted to the joints of interest. Rank answers "can you", the
   condition number answers "how well", and they give different advice.

Run
---
    python -m s14_proprioception.pressure
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
from s11_myofiber.myofiber import PSI, Myofiber  # noqa: E402
from s12_myofiber_leg.leg import size_leg  # noqa: E402
from s9_muscle.control import LEG_JOINTS, leg_plant  # noqa: E402
from s9_muscle.scene import actuator_names, load, moment_arms  # noqa: E402

SEED = 0
THETA0 = 30.0
SUPPLY_PSI = 100.0
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s14"
# Pressure sensor resolutions to evaluate, in bits over the 0-100 psi span.
SENSOR_BITS = (8, 10, 12, 14, 16)


def axial_stiffness(fibre: Myofiber, eps0: float, p0: float, *, h=1e-7) -> dict:
    """
    Actuator stiffness with the valve open (constant pressure) and closed
    (constant fluid volume).

    Sign convention, because it is easy to get backwards: `eps` is contraction,
    so length is `L0 (1 - eps)` and `dL = -L0 deps`. Stretching a pressurised
    braid *increases* its pull, which resists the stretch, so a positive
    stiffness is `k = dF/dL = -(dF/deps) / L0`.
    """
    def force(p, e):
        return float(fibre.force(p, e))

    open_k = -(force(p0, eps0 + h) - force(p0, eps0 - h)) / (2 * h) / fibre.L0

    trapped = fibre.volume(eps0) + fibre.hydraulic_capacitance * p0

    def closed_p(e):
        return (trapped - fibre.volume(e)) / fibre.hydraulic_capacitance

    closed_k = -(force(closed_p(eps0 + h), eps0 + h)
                 - force(closed_p(eps0 - h), eps0 - h)) / (2 * h) / fibre.L0
    return {
        "eps": eps0,
        "open_k_N_per_m": round(open_k, 2),
        "closed_k_N_per_m": round(closed_k, 2),
        "lock_ratio": round(closed_k / max(abs(open_k), 1e-12), 1),
        "stiffer_than_series_element": bool(closed_k > fibre.k_series),
    }


def sensing_range(fibre: Myofiber, eps0: float, *, bits=12) -> dict:
    """
    Travel that spans the full pressure scale, and the resolution that implies.

    The two numbers trade against each other through the same derivative: a
    stiff closed actuator gives a large dP/deps, which is fine resolution and a
    short range. There is no design that gets both without changing C.
    """
    h = 1e-7
    dV = (fibre.volume(eps0 + h) - fibre.volume(eps0 - h)) / (2 * h)
    dp_deps = abs(dV / fibre.hydraulic_capacitance)
    span_eps = fibre.supply_pressure / dp_deps
    counts = 2 ** bits
    return {
        "eps": eps0,
        "dP_deps_Pa": float(f"{dp_deps:.4g}"),
        "full_scale_travel_mm": round(1e3 * span_eps * fibre.L0, 4),
        "fraction_of_stroke": round(span_eps / fibre.eps_max, 5),
        "resolution_um": round(1e6 * span_eps * fibre.L0 / counts, 4),
        "bits": bits,
    }


def observability(model, data, plant) -> dict:
    """
    Rank and conditioning of the pressure-to-pose map, valves closed.

    With trapped fluid, dP_j/dq_i = (dV_j/deps_j) / (C_j L0_j) * R_ji, which is
    the moment-arm matrix with each row scaled by a positive constant. Row
    scaling cannot change rank, so pose observability from pressure is exactly
    the rank question S9 already answered for actuation. Conditioning is a
    different matter and is where the useful information is.
    """
    R = moment_arms(model, data)[np.ix_(plant.muscles, plant.dofs)]
    names = actuator_names(model)
    fibres = size_leg(model, plant.muscles, names, theta0_deg=THETA0,
                      supply_psi=SUPPLY_PSI)

    h = 1e-7
    scale = np.empty(len(fibres))
    for j, f in enumerate(fibres):
        fib = f.fibre(THETA0, SUPPLY_PSI)
        eps0 = 0.10
        dV = (fib.volume(eps0 + h) - fib.volume(eps0 - h)) / (2 * h)
        scale[j] = abs(dV) / (fib.hydraulic_capacitance * fib.L0)

    J = scale[:, None] * R
    s_R = np.linalg.svd(R, compute_uv=False)
    s_J = np.linalg.svd(J, compute_uv=False)
    return {
        "n_pressures": int(J.shape[0]),
        "n_dofs": int(J.shape[1]),
        "rank_moment_arms": int(np.linalg.matrix_rank(R, tol=1e-9)),
        "rank_pressure_jacobian": int(np.linalg.matrix_rank(J, tol=1e-9)),
        "rank_preserved_by_row_scaling": bool(
            np.linalg.matrix_rank(R, tol=1e-9) == np.linalg.matrix_rank(J, tol=1e-9)),
        "pose_fully_observable": bool(np.linalg.matrix_rank(J, tol=1e-9) == J.shape[1]),
        "condition_number_moment_arms": round(float(s_R[0] / s_R[-1]), 2),
        "condition_number_pressure": round(float(s_J[0] / s_J[-1]), 2),
        "singular_values_pressure": [float(f"{v:.4g}") for v in s_J],
    }


def estimator_error(model, data, plant, *, rng, noise_levels=None, trials=200) -> dict:
    """
    Least-squares pose from noisy pressures, against the true perturbation.

    Local: a small pose perturbation dq produces dP = J dq, and the estimate is
    the least-squares inverse. This is the best any estimator can do with this
    measurement at this operating point, so the numbers are a bound rather than
    a proposal for a particular filter.
    """
    obs = observability(model, data, plant)
    R = moment_arms(model, data)[np.ix_(plant.muscles, plant.dofs)]
    names = actuator_names(model)
    fibres = size_leg(model, plant.muscles, names, theta0_deg=THETA0, supply_psi=SUPPLY_PSI)
    h = 1e-7
    scale = np.empty(len(fibres))
    for j, f in enumerate(fibres):
        fib = f.fibre(THETA0, SUPPLY_PSI)
        dV = (fib.volume(0.10 + h) - fib.volume(0.10 - h)) / (2 * h)
        scale[j] = abs(dV) / (fib.hydraulic_capacitance * fib.L0)
    J = scale[:, None] * R
    pinv = np.linalg.pinv(J)

    supply = SUPPLY_PSI * PSI
    noise_levels = noise_levels or [supply / 2 ** b for b in SENSOR_BITS]
    rows = []
    for bits, sigma in zip(SENSOR_BITS, noise_levels):
        errs = []
        for _ in range(trials):
            dq = rng.normal(0.0, 1e-4, size=J.shape[1])       # 0.1 mrad perturbation
            dP = J @ dq + rng.normal(0.0, sigma, size=J.shape[0])
            errs.append(np.linalg.norm(pinv @ dP - dq))
        errs = np.array(errs)
        rows.append({
            "bits": bits,
            "sensor_sigma_Pa": float(f"{sigma:.4g}"),
            "rms_angle_error_rad": float(f"{errs.mean():.4g}"),
            "rms_angle_error_mdeg": round(float(np.degrees(errs.mean()) * 1e3), 4),
        })
    # Noise-free limit: the estimator must be exact when the sensors are.
    dq = rng.normal(0.0, 1e-4, size=J.shape[1])
    exact = float(np.linalg.norm(pinv @ (J @ dq) - dq))
    return {
        "per_sensor": rows,
        "noise_free_error_rad": float(f"{exact:.3g}"),
        "estimator_exact_without_noise": bool(exact < 1e-12),
        "condition_number": obs["condition_number_pressure"],
    }


def drift(fibre: Myofiber, *, seconds=10.0, flow_error_pct=1.0, cycle_hz=1.0) -> dict:
    """
    The alternative when valves are open: integrate flow to get volume, and
    volume to get contraction.

    Open-valve operation destroys the trapped-volume relation, so the only
    route to position is integrating what went in. Integration has no
    restoring term, so a proportional flow-sensor bias walks off linearly and
    never comes back.
    """
    stroke = abs(float(fibre.volume(fibre.eps_max) - fibre.volume(0.0)))
    per_cycle = 2.0 * stroke                       # in and out
    total = per_cycle * cycle_hz * seconds
    bias_volume = total * flow_error_pct / 100.0
    h = 1e-7
    dV = abs((fibre.volume(0.10 + h) - fibre.volume(0.10 - h)) / (2 * h))
    return {
        "seconds": seconds,
        "flow_error_pct": flow_error_pct,
        "fluid_moved_mL": round(1e6 * total, 3),
        "accumulated_volume_error_mL": round(1e6 * bias_volume, 4),
        "eps_error": round(float(bias_volume / dV), 5),
        "length_error_mm": round(1e3 * float(bias_volume / dV) * fibre.L0, 3),
        "as_fraction_of_stroke": round(float(bias_volume / dV) / fibre.eps_max, 4),
    }


def make_figure(stiff, sensing, obs, est, drifts, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.1))

    ax = axes[0]
    eps = [s["eps"] for s in stiff]
    ax.plot(eps, [s["open_k_N_per_m"] for s in stiff], "o-", color=PALETTE[0],
            lw=1.9, ms=4, label="valve open")
    ax.plot(eps, [s["closed_k_N_per_m"] for s in stiff], "s-", color=PALETTE[1],
            lw=1.9, ms=4, label="valve closed")
    ax.set_yscale("log")
    ax.set_xlabel("contraction $\\epsilon$")
    ax.set_ylabel("axial stiffness (N/m)")
    ax.set_title("closing the valve is a lock")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot([s["eps"] for s in sensing], [s["full_scale_travel_mm"] for s in sensing],
            "o-", color=PALETTE[2], lw=1.9, ms=4)
    ax.set_xlabel("contraction $\\epsilon$")
    ax.set_ylabel("travel spanning full pressure scale (mm)")
    ax.set_title("the whole sensor range is under a millimetre")

    ax = axes[2]
    ax.semilogy([r["bits"] for r in est["per_sensor"]],
                [r["rms_angle_error_mdeg"] for r in est["per_sensor"]],
                "o-", color=PALETTE[3], lw=1.9, ms=4)
    ax.set_xlabel("pressure sensor resolution (bits)")
    ax.set_ylabel("pose RMS error (millidegrees)")
    ax.set_title(f"cond(J) = {obs['condition_number_pressure']:.0f}")

    ax = axes[3]
    ax.plot([d["seconds"] for d in drifts], [d["length_error_mm"] for d in drifts],
            "o-", color=PALETTE[1], lw=1.9, ms=4)
    ax.set_xlabel("seconds of open-valve operation")
    ax.set_ylabel("accumulated length error (mm)")
    ax.set_title("integrating flow drifts, and never returns")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model, data = load("full", with_floor=False)
    plant = leg_plant(model, data)
    fibre = Myofiber()

    p0 = 50.0 * PSI
    stiff = [axial_stiffness(fibre, e, p0) for e in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)]
    sensing = [sensing_range(fibre, e) for e in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)]
    obs = observability(model, data, plant)
    est = estimator_error(model, data, plant, rng=rng)
    drifts = [drift(fibre, seconds=s) for s in (1.0, 2.0, 5.0, 10.0, 20.0, 60.0)]

    config = {"theta0_deg": THETA0, "supply_psi": SUPPLY_PSI,
              "joints": list(LEG_JOINTS), "sensor_bits": list(SENSOR_BITS)}
    with RunLogger("s14_proprioception", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model)}) as log:
        for s in stiff:
            log.event("stiffness", **s)
        for s in sensing:
            log.event("sensing", **s)
        log.event("observability", **{k: v for k, v in obs.items()
                                      if k != "singular_values_pressure"})
        log.event("estimator", **{k: v for k, v in est.items() if k != "per_sensor"})
        for r in est["per_sensor"]:
            log.event("estimator_noise", **r)
        for dd in drifts:
            log.event("drift", **dd)
        figure = make_figure(stiff, sensing, obs, est, drifts, MEDIA / "proprioception.png")
        run_dir = log.path

    result = {
        "stiffness": stiff,
        "lock_ratio_range": [min(s["lock_ratio"] for s in stiff),
                             max(s["lock_ratio"] for s in stiff)],
        "sensing": sensing,
        "full_scale_travel_mm_range": [min(s["full_scale_travel_mm"] for s in sensing),
                                       max(s["full_scale_travel_mm"] for s in sensing)],
        "max_fraction_of_stroke": max(s["fraction_of_stroke"] for s in sensing),
        "observability": obs,
        "estimator": est,
        "drift": drifts,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("stiffness", "sensing", "drift")}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
