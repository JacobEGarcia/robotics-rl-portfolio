"""
Measure the dexterous workspace of the wrist-down Panda, empirically.

The lift family originally sampled goals up to z=0.30 anywhere in the
r in [0.38, 0.60] annulus, on the assumption that "inside the reach sphere"
means "reachable". It does not: with the wrist constrained to point down,
the effective workspace is much smaller, and near its boundary the DLS
controller trades position tracking for orientation error - the hand tilts,
and a gripped cube is levered out of the fingers. That failure mode was
found by tracing carries, not by this script; this script exists so the
boundary is *measured* rather than guessed at.

For each (r, z) on a grid, the TCP is driven to (r, 0, z) and the settled
tracking error and hand tilt are recorded. A cell is "dexterous" if
error < 5 mm and tilt < 3 degrees. The output is the coefficient table for
the conservative envelope z_max(r) that tasks.py uses for goal sampling.

Run:  python -m s8_scaling.tools.measure_workspace
"""

from __future__ import annotations

import numpy as np

from ..control import R_DOWN, orientation_error
from ..env import TabletopEnv


def tilt_deg(r_hand: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(orientation_error(r_hand, R_DOWN))))


def main() -> None:
    env = TabletopEnv()
    rs = np.arange(0.35, 0.66, 0.03)
    zs = np.arange(0.05, 0.44, 0.03)

    print("        " + " ".join(f"z={z:.2f}" for z in zs))
    z_max = {}
    for r in rs:
        row = []
        for z in zs:
            env.reset(seed=0, family="reach")
            target = np.array([r, 0.0, z])
            for _ in range(60):
                sp = env.privileged_state()["setpoint"]
                delta = (target - sp) / 0.04
                env.step(np.array([*np.clip(delta, -1, 1), 0.0, 1.0]))
            s = env.privileged_state()
            err_mm = np.linalg.norm(s["tcp_pos"] - target) * 1000
            tilt = tilt_deg(s["hand_rot"])
            ok = err_mm < 5.0 and tilt < 3.0
            row.append("  ok " if ok else f"{err_mm:3.0f}/{tilt:2.0f}"[:5].rjust(5))
            if ok:
                z_max[round(r, 2)] = round(z, 2)
        print(f"r={r:.2f} " + " ".join(row))

    print("\nmeasured z_max per radius:")
    for r, z in z_max.items():
        print(f"  r={r:.2f}  z_max={z:.2f}")


if __name__ == "__main__":
    main()
