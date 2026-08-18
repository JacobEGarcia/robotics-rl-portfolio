"""
Measure which way the LEAP hand's palm faces, rather than assuming it.

Menagerie fixes the palm at pos="0 0 0.1" quat="0 1 0 0", which does not
obviously correspond to any particular world direction. Reading the quaternion
and reasoning about it is exactly the kind of step where a sign error hides.

So: curl every finger and watch where the fingertips go. Fingers close toward
the palm, so the mean fingertip displacement under flexion *is* the palm
normal. `scene.xml` sets gravity to 9.81 along the negative of this, which is
physically identical to bolting the hand palm-up.

    python -m s2_inhand.tools.measure_palm
"""

import mujoco
import numpy as np

HAND = "assets/menagerie/leap_hand/scene_right.xml"
TIPS = ["if_ds", "mf_ds", "rf_ds", "th_ds"]


def main() -> None:
    m = mujoco.MjModel.from_xml_path(HAND)
    d = mujoco.MjData(m)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIPS]

    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    open_pos = np.array([d.xpos[i].copy() for i in ids])

    q = np.zeros(m.nq)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if any(k in name for k in ("mcp", "pip", "dip", "cmc", "ipl")):
            q[j] = 0.7 * m.jnt_range[j][1]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    curl_pos = np.array([d.xpos[i].copy() for i in ids])

    delta = curl_pos - open_pos
    print("fingertip displacement under flexion:")
    for n, dd in zip(TIPS, delta):
        print(f"  {n:6s} {np.round(dd, 4)}")

    # The thumb opposes the fingers, so it is excluded from the mean; including
    # it would cancel part of the signal we are trying to measure.
    normal = delta[:3].mean(axis=0)
    normal /= np.linalg.norm(normal)
    gravity = -9.81 * normal

    print(f"\npalm normal      {np.round(normal, 4)}")
    print(f"gravity for scene.xml  {np.round(gravity, 3)}")
    print("\nscene.xml currently uses gravity='8.788 0 -4.354'")


if __name__ == "__main__":
    main()
