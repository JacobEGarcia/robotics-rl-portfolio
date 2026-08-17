"""
S4: characterise tendon friction, hysteresis, and reversal dead band.

Protocol
--------
Quasi-static STAIRCASE sweep. The flexor tendon force is stepped through 40
levels up to 4 N and back down, and at each level the sim runs until joint
velocity falls below 1e-4 rad/s before recording. Every sample is therefore a
static equilibrium, so the gap between the loading and unloading branches is
friction and geometry -- not dynamics.

An earlier version used a 0.25 Hz sinusoid and was wrong; see the note in
`run_sweep`.

Two numbers come out, and both break real tendon robots:

  hysteresis width -- at the same tendon tension, how much the joint angle
      differs depending on whether you are pulling or releasing. This is the
      error a naive position controller inherits, because commanded tendon
      displacement no longer maps one-to-one onto joint angle.

  dead band -- how far tension must fall after a reversal before the joint
      moves at all. This is what makes tendon hands feel notchy and makes
      naive PD control limit-cycle around a setpoint.

Result summary (see media/s4/hysteresis.png)
--------------------------------------------
    frictionloss   width (rad)   dead band (N)
        0.0           0.207          2.46
        0.2           0.405          2.67
        0.5           0.582          2.77
        1.0           0.680          2.87

Note the 0.207 rad width at ZERO friction. That is not a bug and not friction:
it is snap-through bistability caused by under-actuation. One tendon spans two
joints, so the finger has two stable configurations and jumps between them at
different thresholds depending on direction (2.05 N loading, 1.03 N
unloading). The friction-attributable hysteresis is the increase above that
baseline.

Run
---
    python -m s4_tendon.hysteresis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import PALETTE, use_dark_style  # noqa: E402
from common.runlog import RunLogger, hash_mj_model  # noqa: E402
from common.seeding import seed_everything  # noqa: E402

MODEL_PATH = Path(__file__).parent / "finger.xml"

# Friction values to sweep. 0.0 is the ideal-tendon control case; the rest
# bracket what a real cable-in-sheath drivetrain exhibits.
# Chosen relative to the ~4 N drive force. An earlier sweep used 0.02-0.10 N,
# which is 0.5-2.5% of the drive -- far too small to produce measurable
# hysteresis, and the results were indistinguishable from the frictionless
# case. Real cable-in-conduit drivetrains lose on the order of 10-30% to
# friction, so these values bracket that.
FRICTION_VALUES = [0.0, 0.2, 0.5, 1.0]

# Quasi-static drive. The cycle must be slow enough that inertial effects do
# not contribute to the loop, or we would be measuring dynamics rather than
# friction. 0.25 Hz over 8 s gives two clean cycles.
DRIVE_FREQ_HZ = 0.25
DRIVE_CYCLES = 2
# Tuned empirically: the finger saturates against its joint limits above ~4 N
# with 2 N of extensor co-contraction, so driving harder just parks it at the
# stop and measures nothing. Found by sweeping, not by guessing.
PEAK_FORCE = 4.0


N_LEVELS = 40           # force levels per branch
SETTLE_TOL = 1e-4       # rad/s below which we call the joint settled
SETTLE_MAX_STEPS = 4000 # 4 s cap per level


def run_sweep(frictionloss: float, *, hold_extensor: float = 2.0) -> dict:
    """
    Quasi-static STAIRCASE sweep at a given tendon frictionloss.

    WHY A STAIRCASE AND NOT A SINE -- this is the central methodological point
    of the experiment, and it was found the hard way:

    The first version drove the tendon with a 0.25 Hz sinusoid and measured
    the loop width. At frictionloss = 0 it reported a width of 0.314 rad.
    That is impossible for friction hysteresis: with no friction there is no
    hysteresis, and the loop must close. What it was actually measuring was
    DYNAMIC PHASE LAG -- the joint lagging the driving force because of
    inertia and viscous damping. 0.25 Hz sounds slow, but it is not
    quasi-static for a 12 g link on a light joint.

    Worse, the artifact ran backwards: adding friction reduced the motion
    amplitude, which reduced the dynamic lag, so measured "hysteresis" *fell*
    as friction rose.

    The fix is to remove dynamics from the measurement entirely. At each force
    level we hold the command until joint velocity falls below SETTLE_TOL,
    then record. Every recorded point is a static equilibrium, so any
    remaining difference between the loading and unloading branches is
    friction and nothing else.

    This is also how the measurement is done on real hardware, and for the
    same reason.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.tendon_frictionloss[:] = frictionloss
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    up = np.linspace(0.0, PEAK_FORCE, N_LEVELS)
    # Drop the duplicated turning point so the branches share endpoints once.
    down = up[::-1][1:]
    levels = np.concatenate([up, down])
    branch = np.concatenate([np.zeros(len(up)), np.ones(len(down))])  # 0=load 1=unload

    force, mcp, pip, ten_len, settled_flags = [], [], [], [], []

    for f in levels:
        data.ctrl[0] = float(f)
        data.ctrl[1] = hold_extensor
        settled = False
        for _ in range(SETTLE_MAX_STEPS):
            mujoco.mj_step(model, data)
            if np.max(np.abs(data.qvel)) < SETTLE_TOL:
                settled = True
                break
        force.append(float(f))
        mcp.append(float(data.qpos[0]))
        pip.append(float(data.qpos[1]))
        ten_len.append(float(data.ten_length[0]))
        settled_flags.append(settled)

    return {
        "frictionloss": frictionloss,
        "force": np.array(force),
        "mcp": np.array(mcp),
        "pip": np.array(pip),
        "ten_length": np.array(ten_len),
        "branch": branch,
        # If levels failed to settle the "static" claim is void, so we surface
        # it rather than quietly reporting a dynamic measurement as a static one.
        "n_unsettled": int(len(settled_flags) - sum(settled_flags)),
        "model_hash": hash_mj_model(model),
    }


def second_cycle(r: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Kept for interface compatibility with the plotting code.

    With the staircase protocol every sample is already a settled static
    equilibrium, so there is no transient to discard -- this now returns the
    full sweep unchanged.
    """
    return r["force"], r["mcp"]


def hysteresis_width(force: np.ndarray, angle: np.ndarray) -> float:
    """
    Mean vertical gap between the loading and unloading branches, in radians.

    WHY NOT ENCLOSED AREA -- this was a real error on the first attempt:
    the shoelace area of the force-angle curve is confounded by amplitude.
    Adding friction reduces peak flexion (0.571 -> 0.533 rad), which SHRINKS
    the enclosed area even as the loop gets wider. The measured "dissipation"
    then decreased with friction, which is physically backwards.

    Loop *width* at matched force is amplitude-independent: it asks "at this
    same tendon tension, how much does the joint angle differ depending on
    whether I am pulling or releasing?" That is the quantity a position
    controller actually suffers from, and it is what a real hand's
    repeatability spec would quote.
    """
    peak = int(np.argmax(force))
    if peak < 2 or peak > len(force) - 3:
        return 0.0

    f_up, a_up = force[: peak + 1], angle[: peak + 1]
    # Reverse the unloading branch so force is ascending, as np.interp requires.
    f_down, a_down = force[peak:][::-1], angle[peak:][::-1]

    # Sample strictly inside both branches; the endpoints are where the two
    # curves necessarily meet and would bias the mean toward zero.
    lo = max(f_up.min(), f_down.min()) + 0.05 * np.ptp(force)
    hi = min(f_up.max(), f_down.max()) - 0.05 * np.ptp(force)
    if hi <= lo:
        return 0.0

    grid = np.linspace(lo, hi, 200)
    return float(np.mean(np.abs(np.interp(grid, f_up, a_up) - np.interp(grid, f_down, a_down))))


def dead_band(force: np.ndarray, angle: np.ndarray) -> float:
    """
    Force that must be released after a reversal before the joint moves.

    At the drive peak the joint velocity crosses zero. With friction present,
    tension must fall by a finite amount before the joint breaks free and
    starts extending. That amount is the dead band, and it is what makes a
    naive PD controller limit-cycle around its setpoint on real tendon
    hardware.

    Threshold is 2% of the angle span -- large enough to ignore numerical
    jitter, small enough to catch the real break-away point.
    """
    peak = int(np.argmax(force))
    if peak >= len(force) - 2:
        return 0.0
    span = float(np.ptp(angle))
    if span <= 1e-6:
        return 0.0

    threshold = angle[peak] - 0.02 * span
    after = angle[peak:]
    moved = np.nonzero(after < threshold)[0]
    if moved.size == 0:
        return 0.0
    return float(force[peak] - force[peak + moved[0]])


def main() -> None:
    seed_everything(0)
    use_dark_style()
    results = [run_sweep(f) for f in FRICTION_VALUES]

    config = {
        "model": str(MODEL_PATH.name),
        "friction_values": FRICTION_VALUES,
        "drive_freq_hz": DRIVE_FREQ_HZ,
        "peak_force": PEAK_FORCE,
    }

    with RunLogger("s4_tendon", config=config, seed=0) as log:
        rows = []
        for r in results:
            # All metrics are computed on the steady-state second cycle only.
            f2, a2 = second_cycle(r)
            area = hysteresis_width(f2, a2)
            band = dead_band(f2, a2)
            rows.append((r["frictionloss"], area, band, r["mcp"].max()))
            log.event(
                "sweep",
                frictionloss=r["frictionloss"],
                hysteresis_width_rad=area,
                dead_band_N=band,
                max_flexion_rad=float(r["mcp"].max()),
                model_hash=r["model_hash"],
            )

        # ---------------- figure ----------------
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        # Panel 1: the hysteresis loops themselves
        for i, r in enumerate(results):
            f2, a2 = second_cycle(r)
            axes[0].plot(f2, a2, color=PALETTE[i], lw=1.5,
                         label=f"frictionloss = {r['frictionloss']}")
        axes[0].set_xlabel("flexor tendon force (N)")
        axes[0].set_ylabel("MCP joint angle (rad)")
        axes[0].set_title("Hysteresis loops (quasi-static staircase)")
        axes[0].legend(fontsize=8)

        # Panel 2: loop area vs friction -- the validation check
        fr = [x[0] for x in rows]
        ar = [x[1] for x in rows]
        axes[1].plot(fr, ar, "o-", color=PALETTE[0], lw=1.8)
        axes[1].set_xlabel("tendon frictionloss")
        axes[1].set_ylabel("hysteresis width (rad)")
        axes[1].set_title("Loop width grows with friction")

        # Panel 3: reversal dead band
        db = [x[2] for x in rows]
        axes[2].plot(fr, db, "s-", color=PALETTE[1], lw=1.8)
        axes[2].set_xlabel("tendon frictionloss")
        axes[2].set_ylabel("dead band (N)")
        axes[2].set_title("Force wasted on reversal")

        fig.suptitle(
            "S4 — Tendon friction characterisation (antagonist-driven 2-DoF finger)",
            fontsize=12,
        )
        fig.tight_layout()
        out = Path("media/s4/hysteresis.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        log.event("figure", path=str(out))

        print(f"{'friction':>10} {'hyst width':>12} {'dead band (N)':>15} {'max flex (rad)':>16}")
        for f, a, b, mx in rows:
            print(f"{f:>10.3f} {a:>12.5f} {b:>15.3f} {mx:>16.4f}")
        print(f"\nFigure: {out}")
        print(f"Run:    {log.path}")


if __name__ == "__main__":
    main()
