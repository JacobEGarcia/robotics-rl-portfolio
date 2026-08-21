"""
The tabletop environment: one compiled model, procedurally varied episodes.

Interface notes
---------------
`TabletopEnv` follows the Gymnasium 5-tuple step contract so the S7
evaluation backend (`common.backends.SimClosedLoop`) drives it unmodified.
`reset(seed=...)` maps the seed to a sampled `TaskInstance`, so an
evaluation over seeds [0, N) is an evaluation over N distinct procedurally
generated episodes - reproducible episode-for-episode, which is what lets
two policies be compared on *identical* instance sequences (the paired
"blind A/B" protocol in the write-up).

Action space: 5-D, all in [-1, 1].
    a[0:3]  TCP displacement per control tick, scaled by ACTION_POS_SCALE
    a[3]    wrist yaw rate, scaled by ACTION_YAW_SCALE
    a[4]    gripper: > 0 commands open, <= 0 commands closed
Control runs at 20 Hz over a 2 ms physics step (25 substeps). The wrist is
held pointing down by the controller; yaw is a controlled degree of freedom
because a fixed-yaw gripper closing on a randomly yawed cube grabs the
corner-to-corner diagonal, from which the cube pivots out - grasping is a
5-DoF problem even on a tabletop of boxes.

Observation: 67-D state vector, layout documented in `obs_layout()`.
State-based observation is a deliberate scope decision: the object of study
is scaling *phenomenology* (power laws, ossification, transfer), which
needs hundreds of training runs on a CPU. Pixels would burn the compute
budget on a perception problem this study is not asking about.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .control import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    CartesianController,
)
from .tasks import FAMILIES, PARK, TaskInstance, is_success, sample_instance

SCENE_PATH = Path(__file__).parent / "scene.xml"

CONTROL_HZ = 20
SUBSTEPS = 25  # 25 x 2 ms = 50 ms per control tick
ACTION_POS_SCALE = 0.04  # metres of TCP setpoint motion per tick at |a|=1
ACTION_YAW_SCALE = 0.15  # radians of wrist yaw per tick at |a|=1
YAW_LIMIT = 0.9  # setpoint clamp; face-aligned grasps only ever need +/- pi/4

OBJECT_NAMES = ("cube_a", "cube_b", "pillar")
OBS_DIM = 69
ACT_DIM = 5

# TCP setpoint bounds: the reachable box above the table. The setpoint is
# clamped here so a policy (or expert) saturating an action cannot integrate
# the target into an unreachable region and wind up the controller.
SETPOINT_LO = np.array([0.15, -0.55, 0.005])
SETPOINT_HI = np.array([0.70, 0.55, 0.55])


def obs_layout() -> dict[str, slice]:
    """Named slices into the 69-D observation vector."""
    return {
        "qpos_arm": slice(0, 7),
        "qvel_arm": slice(7, 14),
        "finger_width": slice(14, 15),
        "tcp_pos": slice(15, 18),
        "tcp_vel": slice(18, 21),
        "tcp_yaw": slice(21, 22),
        "objects": slice(22, 55),  # 3 slots x (present, pos3, quat4, linvel3)
        "goal": slice(55, 58),
        "family_onehot": slice(58, 64),
        "prev_action": slice(64, 69),
    }


class TabletopEnv:
    """One tabletop, six task families, fresh instance every reset."""

    # Menagerie's Panda gripper servo produces ~2.3 N of tendon force, which
    # lets a 100 g cube creep through the fingers during any vertical carry.
    # The real Franka Hand is specified at up to 70 N continuous grasp force,
    # so the servo gain is scaled here toward hardware truth (~23 N). Done in
    # code, once, at load - never per-episode - so every run shares one model.
    GRIP_KP_SCALE = 10.0

    def __init__(self, families: tuple[str, ...] | list[str] = FAMILIES) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.model.actuator_gainprm[aid, 0] *= self.GRIP_KP_SCALE
        self.model.actuator_biasprm[aid, 1] *= self.GRIP_KP_SCALE
        self.model.actuator_biasprm[aid, 2] *= self.GRIP_KP_SCALE
        self.data = mujoco.MjData(self.model)
        self.controller = CartesianController(self.model)
        self.families = tuple(families)

        self._home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "s8_home"
        )
        self._body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in OBJECT_NAMES
        }
        self._geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            for name in OBJECT_NAMES
        }
        self._qpos_adr = {}
        for name in OBJECT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            self._qpos_adr[name] = self.model.jnt_qposadr[jid]
        self._site_point = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_point")
        self._site_zone = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_zone")

        self.instance: TaskInstance | None = None
        self._setpoint = np.zeros(3)
        self._prev_action = np.zeros(ACT_DIM)
        self._steps = 0
        self._succeeded = False

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed: int, family: str | None = None):
        """
        Reset to a fresh instance. The family is derived from the seed by
        cycling the env's family tuple unless given explicitly, so a block
        of consecutive seeds covers every family evenly.
        """
        if family is None:
            family = self.families[seed % len(self.families)]
        return self.reset_to(sample_instance(family, seed))

    def reset_to(self, inst: TaskInstance):
        """Reset the scene to an explicit TaskInstance."""
        self.instance = inst
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self._home_key)

        # Write per-instance object parameters into the model. This is the
        # S5 pattern: vary physics through model fields, never recompile.
        for name in OBJECT_NAMES:
            adr = self._qpos_adr[name]
            if name in inst.objects:
                p = inst.objects[name]
                half = np.asarray(p["half"], dtype=float)
                gid, bid = self._geom_ids[name], self._body_ids[name]
                m.geom_size[gid] = half
                m.body_mass[bid] = p["mass"]
                # Box inertia about its centre, from mass and half-sizes.
                a, b, c = half
                m.body_inertia[bid] = (
                    p["mass"] / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
                )
                m.geom_friction[gid, 0] = p["friction"]
                yaw = p["yaw"]
                d.qpos[adr : adr + 3] = [p["xy"][0], p["xy"][1], half[2] + 0.001]
                d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            else:
                d.qpos[adr : adr + 3] = [*PARK[name], 0.05]
                d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]

        # Goal markers: a floating point for 3-D goals, a zone for table goals.
        in_air = inst.family in ("reach", "lift")
        m.site_pos[self._site_point] = inst.goal if in_air else [0, 0, -1]
        m.site_pos[self._site_zone] = (
            [inst.goal[0], inst.goal[1], 0.001] if not in_air else [0, 0, -1]
        )

        # Let the objects settle onto the table under the home servo.
        mujoco.mj_forward(m, d)
        for _ in range(60):
            mujoco.mj_step(m, d)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        tcp, _ = self.controller.tcp_pose(d)
        self._setpoint = tcp.copy()
        self._yaw = 0.0
        self._prev_action = np.zeros(ACT_DIM)
        self._steps = 0
        self._succeeded = False
        return self._obs(), {"instance": inst}

    # ------------------------------------------------------------------- step

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self._setpoint = np.clip(
            self._setpoint + action[:3] * ACTION_POS_SCALE, SETPOINT_LO, SETPOINT_HI
        )
        self._yaw = float(
            np.clip(self._yaw + action[3] * ACTION_YAW_SCALE, -YAW_LIMIT, YAW_LIMIT)
        )
        grip = GRIPPER_OPEN if action[4] > 0 else GRIPPER_CLOSED

        for _ in range(SUBSTEPS):
            self.data.ctrl[:7] = self.controller.servo_targets(
                self.data, self._setpoint, self._yaw
            )
            self.data.ctrl[7] = grip
            mujoco.mj_step(self.model, self.data)

        self._prev_action = action
        self._steps += 1

        success = is_success(self.instance, self.privileged_state())
        self._succeeded = self._succeeded or success
        terminated = success
        truncated = self._steps >= self.instance.horizon and not terminated
        reward = 1.0 if success else 0.0
        info = {"success": self._succeeded, "steps": self._steps}
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------ observation

    def privileged_state(self) -> dict:
        """Ground-truth state for experts and success predicates only."""
        d = self.data
        tcp_pos, r_hand = self.controller.tcp_pose(d)
        objects = {}
        for name in OBJECT_NAMES:
            adr = self._qpos_adr[name]
            bid = self._body_ids[name]
            vadr = self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            ]
            objects[name] = {
                "pos": d.qpos[adr : adr + 3].copy(),
                "quat": d.qpos[adr + 3 : adr + 7].copy(),
                "linvel": d.qvel[vadr : vadr + 3].copy(),
                "z_axis": d.xmat[bid].reshape(3, 3)[:, 2].copy(),
            }
        return {
            "tcp_pos": tcp_pos,
            "hand_rot": r_hand,
            # Measured hand yaw: with the hand pointing down, column 0 of the
            # rotation is (cos yaw, sin yaw, ~0).
            "tcp_yaw": float(np.arctan2(r_hand[1, 0], r_hand[0, 0])),
            "finger_width": float(d.qpos[7] + d.qpos[8]),
            "setpoint": self._setpoint.copy(),
            "yaw_setpoint": self._yaw,
            "objects": objects,
        }

    def _obs(self) -> np.ndarray:
        d = self.data
        inst = self.instance
        tcp_pos, _ = self.controller.tcp_pose(d)
        tcp_vel = self.controller.tcp_velocity(d)

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        lay = obs_layout()
        obs[lay["qpos_arm"]] = d.qpos[:7]
        obs[lay["qvel_arm"]] = d.qvel[:7]
        obs[lay["finger_width"]] = d.qpos[7] + d.qpos[8]
        obs[lay["tcp_pos"]] = tcp_pos
        obs[lay["tcp_vel"]] = tcp_vel

        slot = lay["objects"].start
        state = self.privileged_state()
        obs[lay["tcp_yaw"]] = state["tcp_yaw"]
        for name in OBJECT_NAMES:
            if name in inst.objects:
                o = state["objects"][name]
                obs[slot] = 1.0
                obs[slot + 1 : slot + 4] = o["pos"]
                obs[slot + 4 : slot + 8] = o["quat"]
                obs[slot + 8 : slot + 11] = o["linvel"]
            # Parked objects leave their slot zeroed: absent means absent,
            # not "present at coordinates far away".
            slot += 11

        obs[lay["goal"]] = inst.goal
        onehot = np.zeros(len(FAMILIES), dtype=np.float32)
        onehot[inst.family_index] = 1.0
        obs[lay["family_onehot"]] = onehot
        obs[lay["prev_action"]] = self._prev_action
        return obs

    # ---------------------------------------------------------------- render

    def render(self, camera: str = "main", width: int = 1280, height: int = 720):
        """Offscreen RGB render of the current state."""
        renderer = getattr(self, "_renderer", None)
        if renderer is None or renderer.width != width or renderer.height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()
