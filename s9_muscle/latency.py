"""
S9-D: cancelling activation lag, and finding the wall behind it.

S9 measured a 31.8 ms rise and a 91.9 ms release on a step command. That lag is
a first-order filter with a known, invertible form, so most of it is not a
physical limit at all: it is a filter a controller can pre-compensate. The part
that *is* a physical limit is what happens when the pre-compensated command
leaves [0, 1] and there is nothing left to ask for.

Separating those two is the point. A team that mistakes the first for the
second buys faster hardware it did not need; a team that mistakes the second
for the first tunes a controller forever.

The inverse
-----------
MuJoCo integrates `da/dt = (u - a) / tau(a, sign(u - a))`, with

    tau = tau_act  * (0.5 + 1.5a)     when rising
    tau = tau_deact / (0.5 + 1.5a)    when falling

The branch depends on the sign of `u - a`, which is not known before solving,
but it equals the sign of the requested rate, which *is* known. So the inverse
is closed-form:

    u = a + a_dot * tau(a, sign(a_dot))

Checked against `mju_muscleDynamics` on a grid rather than derived and trusted:
feed the computed `u` back in and the returned rate must equal the requested
one.

The wall
--------
Commanding `u` outside [0, 1] is not possible. Following a sinusoid of
amplitude A about a midpoint needs a peak rate of A*omega, so there is a
frequency above which the required command saturates and tracking degrades no
matter what the controller does. That frequency is measured here and also
predicted in closed form from the same equation, so the two have to agree.

Run
---
    python -m s9_muscle.latency
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
from s9_muscle.control import activation_rate, invert_activation  # noqa: E402
from s9_muscle.scene import MEDIA_DIR, load  # noqa: E402

SEED = 0
DT = 1e-4                      # integration step, at the converged end of the S9 sweep
CYCLES = 4
MIDPOINT, AMPLITUDE = 0.50, 0.35
FREQUENCIES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def verify_inverse(prm, *, n=40) -> dict:
    """
    Feed the inverse back through the forward model on a grid.

    Rates are scaled by the achievable range at each activation so the grid
    stays inside what the muscle can actually do; asking for a rate no command
    can produce would test the clipping, not the algebra.
    """
    worst = 0.0
    for a in np.linspace(0.0, 1.0, n):
        rise_max = activation_rate(1.0, a, prm)
        fall_max = activation_rate(0.0, a, prm)
        for frac in np.linspace(-0.99, 0.99, n):
            target = frac * (rise_max if frac > 0 else -fall_max)
            if abs(target) < 1e-12:
                continue
            u = invert_activation(a, target, prm)
            worst = max(worst, abs(activation_rate(float(np.clip(u, 0.0, 1.0)), a, prm) - target))
    return {"max_abs_rate_error": float(worst), "grid": n * n}


def reference(t, freq):
    """Sinusoidal activation reference and its exact derivative."""
    omega = 2.0 * np.pi * freq
    return (
        MIDPOINT + AMPLITUDE * np.sin(omega * t),
        AMPLITUDE * omega * np.cos(omega * t),
    )


def track(prm, freq: float, *, compensate: bool) -> dict:
    """
    Follow the reference for a few cycles and report the tracking error.

    Feedforward only. A feedback term would improve both arms and obscure what
    is being compared, which is how much of the lag the inverse alone removes.
    """
    n = int(round(CYCLES / freq / DT))
    t = np.arange(n) * DT
    a_ref, a_dot_ref = reference(t, freq)

    act = np.empty(n)
    cmd = np.empty(n)
    a = float(a_ref[0])
    saturated = 0
    for i in range(n):
        if compensate:
            raw = invert_activation(a_ref[i], a_dot_ref[i], prm)
        else:
            raw = a_ref[i]
        u = float(np.clip(raw, 0.0, 1.0))
        saturated += int(raw < -1e-12 or raw > 1.0 + 1e-12)
        cmd[i] = u
        k1 = activation_rate(u, a, prm)
        k2 = activation_rate(u, a + 0.5 * DT * k1, prm)
        k3 = activation_rate(u, a + 0.5 * DT * k2, prm)
        k4 = activation_rate(u, a + DT * k3, prm)
        a += DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        act[i] = a

    # Score over the final cycle only: the first cycle carries the transient
    # from starting at the reference with no history, which is a start-up
    # artifact rather than a tracking property.
    tail = slice(int(n * (CYCLES - 1) / CYCLES), n)
    err = act[tail] - a_ref[tail]
    return {
        "freq_hz": freq,
        "rms_error": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.abs(err).max()),
        "saturated_fraction": round(saturated / n, 4),
        "_trace": {"t": t, "ref": a_ref, "act": act, "cmd": cmd},
    }


def saturation_frequency(prm) -> float:
    """
    Lowest frequency at which the ideal command leaves [0, 1], in closed form.

    Scans the reference cycle rather than solving analytically for the worst
    phase: the required command is a product of a sinusoid and an
    activation-dependent time constant, so its extremum is not at a phase that
    can be written down in one line, and a fine scan is exact enough and
    obviously correct.
    """
    phase = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    for freq in np.arange(0.1, 60.0, 0.05):
        omega = 2.0 * np.pi * freq
        a = MIDPOINT + AMPLITUDE * np.sin(phase)
        a_dot = AMPLITUDE * omega * np.cos(phase)
        u = np.array([invert_activation(ai, di, prm) for ai, di in zip(a, a_dot)])
        if u.max() > 1.0 or u.min() < 0.0:
            return round(float(freq), 2)
    return float("nan")


def make_figure(results, traces, f_sat, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))

    ax = axes[0]
    freqs = [r["freq_hz"] for r in results["naive"]]
    ax.plot(freqs, [r["rms_error"] for r in results["naive"]], "o-",
            color=PALETTE[1], lw=2, ms=4, label="naive (command = reference)")
    ax.plot(freqs, [r["rms_error"] for r in results["compensated"]], "s-",
            color=PALETTE[0], lw=2, ms=4, label="inverse-compensated")
    ax.axvline(f_sat, color="0.5", lw=1.2, ls="--")
    ax.text(f_sat, ax.get_ylim()[1] * 0.55, f"  command saturates\n  at {f_sat:g} Hz",
            fontsize=7.5, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference frequency (Hz)")
    ax.set_ylabel("RMS activation error")
    ax.set_title("the lag is a filter, until it is not")
    ax.legend(fontsize=8)

    for ax, freq in zip(axes[1:], traces):
        naive, comp = traces[freq]
        t = naive["_trace"]["t"] * 1e3
        ax.plot(t, naive["_trace"]["ref"], color="0.55", lw=2.4, label="reference")
        ax.plot(t, naive["_trace"]["act"], color=PALETTE[1], lw=1.6, label="naive")
        ax.plot(t, comp["_trace"]["act"], color=PALETTE[0], lw=1.6, ls="--", label="compensated")
        ax.plot(t, comp["_trace"]["cmd"], color=PALETTE[2], lw=1.0, alpha=0.7,
                label="compensated command")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("activation")
        ax.set_title(f"{freq:g} Hz")
        ax.legend(fontsize=7)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    seed_everything(SEED)
    model, _ = load("full", with_floor=False)
    prm = model.actuator_dynprm[0][:3].copy()

    config = {"dt": DT, "cycles": CYCLES, "midpoint": MIDPOINT,
              "amplitude": AMPLITUDE, "frequencies": list(FREQUENCIES)}
    with RunLogger("s9_latency", config=config, seed=SEED,
                   extra_manifest={"model_hash": hash_mj_model(model),
                                   "tau_act": float(prm[0]), "tau_deact": float(prm[1])}) as log:
        inverse_check = verify_inverse(prm)
        log.event("inverse_check", **inverse_check)

        results = {"naive": [], "compensated": []}
        for freq in FREQUENCIES:
            results["naive"].append(track(prm, freq, compensate=False))
            results["compensated"].append(track(prm, freq, compensate=True))
        for kind in results:
            for r in results[kind]:
                log.event("track", mode=kind, **{k: v for k, v in r.items()
                                                 if not k.startswith("_")})

        f_sat = saturation_frequency(prm)
        log.event("saturation", predicted_hz=f_sat)

        show = (1.0, 12.0)
        traces = {f: (results["naive"][FREQUENCIES.index(f)],
                      results["compensated"][FREQUENCIES.index(f)]) for f in show}
        figure = make_figure(results, traces, f_sat, MEDIA_DIR / "latency.png")
        run_dir = log.path

    below = [r for r in results["compensated"] if r["freq_hz"] < f_sat]
    above = [r for r in results["compensated"] if r["freq_hz"] >= f_sat]
    naive_by_f = {r["freq_hz"]: r["rms_error"] for r in results["naive"]}
    summary = {
        "inverse_max_abs_rate_error": inverse_check["max_abs_rate_error"],
        "saturation_frequency_hz": f_sat,
        "compensated_rms_below_saturation_max": (
            round(max(r["rms_error"] for r in below), 8) if below else None
        ),
        "compensated_rms_above_saturation_max": (
            round(max(r["rms_error"] for r in above), 6) if above else None
        ),
        "no_saturation_below_predicted": all(
            r["saturated_fraction"] == 0.0 for r in below
        ),
        "saturation_above_predicted": all(r["saturated_fraction"] > 0.0 for r in above),
        "error_reduction": {
            f"{r['freq_hz']:g} Hz": round(naive_by_f[r["freq_hz"]] / max(r["rms_error"], 1e-15), 1)
            for r in results["compensated"]
        },
        "per_frequency": {
            kind: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
            for kind, rows in results.items()
        },
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frequency"}, indent=2))
    print(f"\nfigure: {figure}")


if __name__ == "__main__":
    main()
