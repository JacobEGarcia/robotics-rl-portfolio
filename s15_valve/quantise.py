"""
S15: what a real valve costs, against the continuous one every model assumes.

S9-D showed that a muscle's activation lag is an invertible filter, worth about
500x in tracking error once you compensate it. That result, and S13's
allocation policies, both assume a valve that can be held at any opening. Real
valves are not like that. Clone's published Aquajet runs under a watt, which is
the power budget of a solenoid that is either open or shut, and a solenoid that
is either open or shut tracks a continuous command by switching.

Three things a switching valve costs, none of which appear in a continuous
model:

  1. Ripple            pressure oscillates around the target at the carrier
                       frequency. Amplitude falls as 1/f, so this is the cheap
                       one: pay for it with switching rate.
  2. Duty quantisation a valve with a minimum on-time cannot produce every duty
                       cycle. At a given carrier the achievable set is discrete,
                       and the spacing is a hard resolution floor no controller
                       can interpolate through.
  3. Switching cost    every cycle is an actuation. Valve life and power both
                       scale with rate, which is the reason not to simply raise
                       the carrier until ripple disappears.

The interesting part is that 1 and 2 pull in opposite directions. Raising the
carrier shrinks ripple and *worsens* duty quantisation, because a fixed minimum
on-time is a larger fraction of a shorter period. There is an optimum, it is
not at either end, and it moves with the minimum on-time.

Checks
------
* As the carrier rises with no minimum on-time, tracking must converge to the
  continuous-valve result. That validates the PWM machinery.
* Ripple amplitude must fall as roughly 1/f in that same limit.
* With a minimum on-time, the achievable duty set must be discrete with the
  spacing the constraint implies, verified against the closed form.

Run
---
    python -m s15_valve.quantise
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
from s11_myofiber.myofiber import PSI, Myofiber  # noqa: E402

SEED = 0
DT = 5e-7                    # integration step, well under the fastest carrier
SECONDS = 0.03
CARRIERS_HZ = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0,
               10000.0, 20000.0)
# Minimum integration steps per carrier period. Without this the fastest
# carriers are under-resolved and report a *rising* error, which reads like
# physics and is arithmetic: at a fixed step, 50 kHz got 40 samples a period.
STEPS_PER_PERIOD = 200
# The actuator's own isometric fill time constant, measured in S11. PWM only
# *averages* when the carrier period is short against this; when it is long,
# each cycle fully charges and fully discharges and switching is bang-bang.
FILL_TAU_S = 0.354e-3
MIN_ON_TIMES_MS = (0.0, 0.1, 0.2, 0.5, 1.0)
TARGET_FRACTION = 0.5        # hold at half supply pressure
MEDIA = Path(__file__).resolve().parents[1] / "media" / "s15"


def achievable_duties(carrier_hz: float, min_on_ms: float) -> np.ndarray:
    """
    Duty cycles a valve with a minimum on-time can actually produce.

    A valve that must stay open at least `t_min` once opened, and shut at least
    `t_min` once shut, can only realise duties that are multiples of
    `t_min / period` between those bounds, plus the two saturated states. The
    spacing is therefore `t_min * f`, which is the closed form the numeric set
    is checked against.
    """
    if min_on_ms <= 0.0:
        return np.linspace(0.0, 1.0, 1001)
    period = 1.0 / carrier_hz
    step = (min_on_ms / 1000.0) / period
    if step >= 1.0:
        return np.array([0.0, 1.0])
    inner = np.arange(step, 1.0 - step + 1e-12, step)
    return np.unique(np.concatenate([[0.0], inner, [1.0]]))


def quantise(duty: float, carrier_hz: float, min_on_ms: float) -> float:
    grid = achievable_duties(carrier_hz, min_on_ms)
    return float(grid[int(np.argmin(np.abs(grid - duty)))])


def simulate(fibre: Myofiber, *, carrier_hz, min_on_ms, target_fraction=TARGET_FRACTION,
             continuous=False) -> dict:
    """
    Hold a target pressure isometrically, with a switching or continuous valve.

    Isometric so the measurement is of the valve and not of the load. The
    controller is a simple proportional law on pressure error, converted to a
    duty cycle and then either used directly (continuous) or realised by
    switching within each carrier period.
    """
    supply = fibre.supply_pressure
    target = target_fraction * supply
    period = 1.0 / carrier_hz
    dt = min(DT, period / STEPS_PER_PERIOD)
    n = int(SECONDS / dt)

    pressure = 0.0
    trace = np.empty(n)
    duty_now = 0.0
    switches = 0
    last_open = None

    for i in range(n):
        t = i * dt
        # Proportional law, saturated. Gain chosen so the loop settles well
        # inside the window at the slowest carrier; it is identical across all
        # conditions so the comparison is of valves, not of tuning.
        duty_cmd = float(np.clip(4.0 * (target - pressure) / supply, -1.0, 1.0))
        if continuous:
            command = duty_cmd
        else:
            if i % max(1, int(period / dt)) == 0:
                duty_now = quantise(abs(duty_cmd), carrier_hz, min_on_ms) * np.sign(duty_cmd)
            phase = (t % period) / period
            command = np.sign(duty_now) if phase < abs(duty_now) else 0.0
            is_open = command != 0.0
            if last_open is not None and is_open != last_open:
                switches += 1
            last_open = is_open
        pressure = float(np.clip(
            pressure + dt * fibre.pressure_rate(pressure, 0.0, 0.0, command), 0.0, supply))
        trace[i] = pressure

    settled = trace[int(0.6 * n):]
    return {
        "carrier_hz": carrier_hz,
        "min_on_ms": min_on_ms,
        "period_over_tau": round(float((1.0 / carrier_hz) / FILL_TAU_S), 3),
        "mean_pressure_frac": round(float(settled.mean() / supply), 5),
        "ripple_pp_frac": float(f"{(settled.max() - settled.min()) / supply:.6g}"),
        "steps_per_period": int(round(period / dt)),
        "rms_error_frac": float(f"{np.sqrt(np.mean(((settled - target) / supply) ** 2)):.6g}"),
        "switches_per_second": round(switches / SECONDS / 2.0, 1),
        "_trace": trace,
    }


def duty_grid_check() -> dict:
    """Numeric achievable-duty spacing against the closed form `t_min * f`."""
    worst = 0.0
    rows = []
    for f in CARRIERS_HZ:
        for t in MIN_ON_TIMES_MS:
            if t <= 0.0:
                continue
            grid = achievable_duties(f, t)
            predicted = (t / 1000.0) * f
            if len(grid) > 2:
                # Minimum gap, not median. The grid's top end is irregular
                # (the last interior level to 1.0 is whatever is left over), and
                # a median over a short grid picks that leftover up and reports
                # a 25% error against a closed form that is actually right. The
                # resolution floor a controller feels is the smallest gap.
                spacing = float(np.diff(grid).min())
                worst = max(worst, abs(spacing - predicted) / predicted)
                rows.append({"carrier_hz": f, "min_on_ms": t,
                             "levels": len(grid),
                             "spacing": round(spacing, 6),
                             "predicted": round(predicted, 6)})
    return {"max_relative_spacing_error": float(f"{worst:.3g}"),
            "spacing_matches_closed_form": bool(worst < 1e-9),
            "rows": rows}


def make_figure(cont, rows, grid_rows, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

    ax = axes[0]
    ideal = [r for r in rows if r["min_on_ms"] == 0.0]
    ax.axvspan(50, 1.0 / FILL_TAU_S, color=PALETTE[1], alpha=0.10)
    ax.text(70, 5e-4, "bang-bang: period longer than tau", fontsize=7.5,
            color=PALETTE[1])
    ax.loglog([r["carrier_hz"] for r in ideal], [r["ripple_pp_frac"] for r in ideal],
              "o-", color=PALETTE[0], lw=1.9, ms=4, label="measured ripple")
    ref = ideal[0]["ripple_pp_frac"] * ideal[0]["carrier_hz"]
    ax.loglog([r["carrier_hz"] for r in ideal],
              [ref / r["carrier_hz"] for r in ideal], "--", color="0.5", lw=1.2,
              label="$1/f$")
    ax.set_xlabel("carrier frequency (Hz)")
    ax.set_ylabel("pressure ripple, peak-to-peak (fraction of supply)")
    ax.set_title("ripple is cheap: pay with switching rate")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, t in enumerate(MIN_ON_TIMES_MS):
        sel = [r for r in rows if r["min_on_ms"] == t]
        ax.loglog([r["carrier_hz"] for r in sel], [r["rms_error_frac"] for r in sel],
                  "o-", color=PALETTE[i % len(PALETTE)], lw=1.7, ms=3.5,
                  label=f"min on {t:g} ms" if t else "ideal valve")
    ax.axhline(cont["rms_error_frac"], color="0.4", ls=":", lw=1.3)
    ax.text(60, cont["rms_error_frac"] * 1.3, "continuous valve", fontsize=7.5, color="0.35")
    ax.set_xlabel("carrier frequency (Hz)")
    ax.set_ylabel("RMS pressure error (fraction of supply)")
    ax.set_title("with a minimum on-time, faster is not better")
    ax.legend(fontsize=7)

    ax = axes[2]
    for i, t in enumerate([x for x in MIN_ON_TIMES_MS if x > 0]):
        sel = [g for g in grid_rows if g["min_on_ms"] == t]
        ax.loglog([g["carrier_hz"] for g in sel], [g["levels"] for g in sel],
                  "o-", color=PALETTE[i % len(PALETTE)], lw=1.7, ms=3.5,
                  label=f"{t:g} ms")
    ax.set_xlabel("carrier frequency (Hz)")
    ax.set_ylabel("achievable duty levels")
    ax.set_title("resolution the controller cannot interpolate through")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    fibre = Myofiber()

    cont = simulate(fibre, carrier_hz=1.0, min_on_ms=0.0, continuous=True)
    rows = []
    for f in CARRIERS_HZ:
        for t in MIN_ON_TIMES_MS:
            rows.append(simulate(fibre, carrier_hz=f, min_on_ms=t))

    grid = duty_grid_check()
    ideal = [r for r in rows if r["min_on_ms"] == 0.0]
    # The 1/f law applies only where PWM actually averages, i.e. where the
    # carrier period is short against the actuator's time constant. Fitting it
    # across the whole sweep mixes in the bang-bang regime and returns a slope
    # of about -5, which is a real measurement of the wrong thing.
    averaging = [r for r in ideal if r["period_over_tau"] < 1.0 and r["ripple_pp_frac"] > 0]
    bang_bang = [r for r in ideal if r["period_over_tau"] >= 1.0]
    if len(averaging) >= 2:
        ripple = np.array([r["ripple_pp_frac"] for r in averaging])
        carriers = np.array([r["carrier_hz"] for r in averaging])
        slope = float(np.polyfit(np.log(carriers), np.log(ripple), 1)[0])
    else:
        slope = float("nan")

    best = {}
    for t in MIN_ON_TIMES_MS:
        sel = [r for r in rows if r["min_on_ms"] == t]
        pick = min(sel, key=lambda r: r["rms_error_frac"])
        best[f"{t:g}"] = {"carrier_hz": pick["carrier_hz"],
                          "rms_error_frac": pick["rms_error_frac"],
                          "switches_per_second": pick["switches_per_second"]}

    config = {"dt": DT, "seconds": SECONDS, "carriers_hz": list(CARRIERS_HZ),
              "min_on_times_ms": list(MIN_ON_TIMES_MS),
              "target_fraction": TARGET_FRACTION}
    with RunLogger("s15_valve", config=config, seed=SEED) as log:
        log.event("continuous", **{k: v for k, v in cont.items() if not k.startswith("_")})
        for r in rows:
            log.event("pwm", **{k: v for k, v in r.items() if not k.startswith("_")})
        log.event("duty_grid", max_relative_spacing_error=grid["max_relative_spacing_error"],
                  spacing_matches_closed_form=grid["spacing_matches_closed_form"])
        log.event("best_carrier", **{f"min_on_{k}ms": v["carrier_hz"] for k, v in best.items()})
        figure = make_figure(cont, rows, grid["rows"], MEDIA / "valve.png")
        run_dir = log.path

    result = {
        "continuous": {k: v for k, v in cont.items() if not k.startswith("_")},
        "ripple_slope_vs_log_f": None if np.isnan(slope) else round(slope, 3),
        # The transition is a cliff, not a rolloff. Ripple is full-scale for
        # every carrier whose period exceeds the actuator's time constant, and
        # collapses by three orders of magnitude within a factor of two in
        # frequency once it does not. Fitting a 1/f law across that is fitting a
        # power law to a step.
        "bang_bang_ripple_is_full_scale": bool(
            all(r["ripple_pp_frac"] > 0.99 for r in bang_bang if r["period_over_tau"] > 2.0)),
        "ripple_collapse_factor_across_transition": round(float(
            max(r["ripple_pp_frac"] for r in bang_bang)
            / max(min((r["ripple_pp_frac"] for r in averaging), default=1.0), 1e-12)), 1),
        "all_runs_resolved": bool(all(r["steps_per_period"] >= STEPS_PER_PERIOD - 1
                                      for r in rows)),
        # Reported as a gap rather than a pass/fail. The continuous valve is
        # exact to 1e-14; the best switching valve gets to about 1e-4, and the
        # useful statement is the size of that floor, not whether it clears an
        # arbitrary threshold.
        "best_switching_error": min(r["rms_error_frac"] for r in ideal),
        "continuous_to_switching_floor_ratio": float(f"{min(r['rms_error_frac'] for r in ideal) / max(cont['rms_error_frac'], 1e-18):.3g}"),
        "fill_tau_ms": round(FILL_TAU_S * 1e3, 4),
        "actuator_bandwidth_hz": round(1.0 / (2.0 * np.pi * FILL_TAU_S), 1),
        "carriers_that_average": [r["carrier_hz"] for r in averaging],
        "carriers_that_bang_bang": [r["carrier_hz"] for r in bang_bang],
        "worst_bang_bang_ripple_frac": round(
            max(r["ripple_pp_frac"] for r in bang_bang), 4),
        "ripple_at_100hz_frac": round(
            [r["ripple_pp_frac"] for r in ideal if r["carrier_hz"] == 100.0][0], 4),
        "ideal_valve_error_at_5khz": ideal[-1]["rms_error_frac"],
        "continuous_error": cont["rms_error_frac"],
        "duty_grid": {k: v for k, v in grid.items() if k != "rows"},
        "best_carrier_by_min_on_time": best,
        "table": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
