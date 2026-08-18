"""
Grid-search the reset grasp: finger flexion x cube placement.

The reset pose decides whether the whole project is trainable. If the cube
falls out on step one, every episode is a drop penalty and there is no signal
to learn from. If it starts deeply interpenetrating a finger, the constraint
solver ejects it at tens of metres per second and the episode is noise.

Both failure modes are silent in the sense that nothing raises, so this
searches and reports rather than trusting a hand-picked pose.

    python -m s2_inhand.tools.find_grasp

Reports drift over 2 s of settling and the worst cube penetration at reset.
A configuration is only acceptable if both are small: low drift alone is not
enough, because a deeply penetrating pose can be wedged in place and still be
physically meaningless.
"""

import itertools

import mujoco
import numpy as np

SCENE = "s2_inhand/scene.xml"

FLEXIONS = [0.4, 0.6, 0.8, 1.0]
XS = [-0.02, 0.0, 0.02]
ZS = [0.150, 0.165, 0.180]

DRIFT_OK = 0.03      # metres over 2 s
PENETRATION_OK = -0.006   # metres, i.e. 6 mm


def trial(m, d, cube_body, cube_geom, flex, pos, tsec=2.0):
    q = np.zeros(m.nq)
    ctrl = np.zeros(m.nu)
    for j in range(16):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if any(k in name for k in ("mcp", "pip", "dip")):
            q[j] = flex
        if "th_cmc" in name:
            q[j] = 1.1
        if "th_axl" in name:
            q[j] = 0.4
        if "th_ipl" in name:
            q[j] = 0.3
    ctrl[:] = q[:16]
    q[16:19] = pos
    q[19:23] = [1, 0, 0, 0]

    d.qpos[:] = q
    d.qvel[:] = 0
    d.ctrl[:] = ctrl
    mujoco.mj_forward(m, d)

    pen = min((d.contact[c].dist for c in range(d.ncon)
               if cube_geom in (d.contact[c].geom1, d.contact[c].geom2)), default=0.0)
    start = d.xpos[cube_body].copy()
    for _ in range(int(tsec / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(np.linalg.norm(d.xpos[cube_body] - start)), float(pen)


def main() -> None:
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

    print(" flex   cube position              drift(m)   penetration(m)")
    best = None
    for flex, x, z in itertools.product(FLEXIONS, XS, ZS):
        pos = (x, 0.005, z)
        drift, pen = trial(m, d, cube_body, cube_geom, flex, pos)
        held = drift < DRIFT_OK and pen > PENETRATION_OK
        tag = "  <== acceptable" if held else ""
        if held and (best is None or drift < best[0]):
            best = (drift, flex, pos, pen)
        print(f" {flex:4.1f}  ({x:+.3f}, 0.005, {z:.3f})   {drift:8.4f}   {pen:+.5f}{tag}")

    if best is None:
        print("\nNo acceptable configuration found. Widen the grid before "
              "loosening the thresholds.")
    else:
        drift, flex, pos, pen = best
        print(f"\nbest: flex={flex} pos={pos} drift={drift:.4f} m penetration={pen:+.5f} m")
        print("scene.xml currently uses flex=0.6, cube at (0, 0.005, 0.18)")


if __name__ == "__main__":
    main()
