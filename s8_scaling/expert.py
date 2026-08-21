"""
Scripted feedback experts: the data engine's demonstrators.

These are feedback controllers, not open-loop scripts. Every control tick
each expert recomputes its waypoint from the *current* privileged state, so
a cube that slides, a grasp that slips, or a pillar that scoots instead of
tipping produces corrective behaviour - and therefore demonstrations whose
action distribution actually covers the states a learned policy will visit.
Open-loop scripts produce beautiful demos and policies that cannot recover;
the difference is the whole reason DAgger exists. Feedback experts get most
of that benefit for free, and injected action noise (sigma=0.05) widens the
visited-state tube further.

Every expert emits actions in the environment's own 5-D action space by
inverting the setpoint integration: a = (waypoint - setpoint) / scale,
clipped. The recorded action is exactly what the policy will be asked to
reproduce - there is no privileged action channel to leak through.

Grasp yaw: the wrist is rotated so the fingers close on a face pair of the
target cube, not its diagonal. The target yaw is chosen as the multiple of
pi/2 of the cube's yaw nearest the *current* wrist yaw, which makes the
choice hysteresis-free - physics jitter near the pi/4 boundary cannot make
the wrist flip between equivalent grasps mid-approach.
"""

from __future__ import annotations

import numpy as np

from .env import ACTION_POS_SCALE, ACTION_YAW_SCALE, YAW_LIMIT, TabletopEnv
from .tasks import TaskInstance

OPEN, CLOSE = 1.0, -1.0


def cube_yaw(quat: np.ndarray) -> float:
    """Yaw of a (nearly) flat-sitting box from its quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def face_yaw(quat: np.ndarray, reference: float) -> float:
    """
    The wrist yaw that closes the fingers on a face pair of the cube:
    cube yaw modulo pi/2, resolved to the candidate nearest `reference`
    and inside the wrist's yaw clamp.
    """
    base = cube_yaw(quat)
    candidates = base + np.arange(-4, 5) * (np.pi / 2)
    candidates = candidates[np.abs(candidates) <= YAW_LIMIT]
    if candidates.size == 0:
        return 0.0
    return float(candidates[np.argmin(np.abs(candidates - reference))])


class ScriptedExpert:
    """Dispatches to one per-family policy, holding small phase state."""

    def __init__(self, noise_scale: float = 0.05, rng: np.random.Generator | None = None):
        self.noise_scale = noise_scale
        self.rng = rng or np.random.default_rng(0)
        self.phase = 0
        self.hold = 0
        self.yaw_des = 0.0

    def reset(self) -> None:
        self.phase = 0
        self.hold = 0
        self.yaw_des = 0.0

    def act(self, env: TabletopEnv) -> np.ndarray:
        inst = env.instance
        state = env.privileged_state()
        fn = getattr(self, f"_{inst.family}")
        waypoint, grip, speed = fn(inst, state)

        action = np.zeros(5)
        action[:3] = np.clip(
            (waypoint - state["setpoint"]) / ACTION_POS_SCALE, -speed, speed
        )
        action[3] = np.clip(
            (self.yaw_des - state["yaw_setpoint"]) / ACTION_YAW_SCALE, -1.0, 1.0
        )
        action[4] = grip
        if self.noise_scale > 0:
            action[:3] += self.rng.normal(0.0, self.noise_scale, size=3)
            action[3] += self.rng.normal(0.0, 0.5 * self.noise_scale)
        return np.clip(action, -1.0, 1.0)

    # ------------------------------------------------------------------ reach

    def _reach(self, inst: TaskInstance, s: dict):
        self.yaw_des = 0.0
        return inst.goal, OPEN, 1.0

    # ------------------------------------------------------------------- push

    def _push(self, inst: TaskInstance, s: dict):
        cube = s["objects"]["cube_a"]
        half = inst.objects["cube_a"]["half"][0]
        c, g = cube["pos"][:2], inst.goal[:2]
        to_goal = g - c
        dist = np.linalg.norm(to_goal)
        direction = to_goal / max(dist, 1e-6)
        push_z = max(half, 0.015)
        behind = c - direction * (half + 0.035)
        tcp = s["tcp_pos"]
        self.yaw_des = 0.0

        if self.phase == 0:  # travel above the approach point
            wp = np.array([behind[0], behind[1], 0.13])
            if np.linalg.norm(tcp[:2] - behind) < 0.02:
                self.phase = 1
            return wp, CLOSE, 1.0
        if self.phase == 1:  # descend behind the cube
            wp = np.array([behind[0], behind[1], push_z])
            if abs(tcp[2] - push_z) < 0.015:
                self.phase = 2
            return wp, CLOSE, 0.8
        if self.phase == 2:  # push through, tracking the live cube position
            if dist < 0.040:
                self.phase = 3
            # Aim just short of the goal so momentum does not overshoot it.
            wp = np.array([*(c + direction * min(dist, 0.05)), push_z])
            # Lost the line? The cube squirted sideways - re-approach.
            lateral = abs(np.cross(direction, tcp[:2] - c))
            if lateral > half + 0.05:
                self.phase = 0
            return wp, CLOSE, 0.35
        wp = np.array([tcp[0], tcp[1], 0.18])  # retreat, leave the cube be
        return wp, CLOSE, 0.6

    # ------------------------------------------------------------------- lift

    def _lift(self, inst: TaskInstance, s: dict):
        # Carry 5 cm past the goal height: the cube rides ~1.5 cm below the
        # TCP, so targeting the goal exactly parks it just under threshold.
        return self._grasp_and(inst, s, carry_to=inst.goal + np.array([0, 0, 0.05]))

    # ------------------------------------------------------------------ place

    def _place(self, inst: TaskInstance, s: dict):
        half = inst.objects["cube_a"]["half"][2]
        return self._pick_place(
            inst, s,
            place_xy=inst.goal[:2],
            place_z=half + 0.002,
            carry_z=0.16,
        )

    # ------------------------------------------------------------------ stack

    def _stack(self, inst: TaskInstance, s: dict):
        b = s["objects"]["cube_b"]
        a_half = inst.objects["cube_a"]["half"][2]
        b_half = inst.objects["cube_b"]["half"][2]
        return self._pick_place(
            inst, s,
            place_xy=b["pos"][:2],
            place_z=b["pos"][2] + b_half + a_half + 0.003,
            carry_z=b["pos"][2] + b_half + a_half + 0.10,
        )

    # ----------------------------------------------------------------- topple

    def _topple(self, inst: TaskInstance, s: dict):
        pillar = s["objects"]["pillar"]
        p = pillar["pos"][:2]
        half_h = inst.objects["pillar"]["half"][2]
        # Push tangentially (perpendicular to the base->pillar radial), so
        # the pillar is driven neither out of reach nor into the robot.
        radial = p / max(np.linalg.norm(p), 1e-6)
        direction = np.array([-radial[1], radial[0]])
        push_z = 2 * half_h * 0.75  # high contact point: torque, not sliding
        behind = p - direction * 0.055
        tcp = s["tcp_pos"]
        self.yaw_des = 0.0

        if pillar["z_axis"][2] < 0.5:  # already toppled - retreat
            return np.array([tcp[0], tcp[1], 0.20]), CLOSE, 0.6
        if self.phase == 0:
            wp = np.array([behind[0], behind[1], 2 * half_h + 0.06])
            if np.linalg.norm(tcp[:2] - behind) < 0.02:
                self.phase = 1
            return wp, CLOSE, 1.0
        if self.phase == 1:
            wp = np.array([behind[0], behind[1], push_z])
            if abs(tcp[2] - push_z) < 0.015:
                self.phase = 2
            return wp, CLOSE, 0.8
        wp = np.array([*(p + direction * 0.05), push_z])
        return wp, CLOSE, 0.45

    # ------------------------------------------------------- shared grasping

    def _grasp_and(self, inst: TaskInstance, s: dict, carry_to: np.ndarray):
        """Phases 0-3: approach, descend, close, carry to `carry_to`."""
        cube = s["objects"]["cube_a"]
        half = inst.objects["cube_a"]["half"][2]
        c = cube["pos"]
        tcp = s["tcp_pos"]
        width = s["finger_width"]

        if self.phase == 0:  # above the cube, rotating to a face-pair yaw
            self.yaw_des = face_yaw(cube["quat"], s["yaw_setpoint"])
            wp = np.array([c[0], c[1], half + 0.10])
            aligned = abs(s["tcp_yaw"] - self.yaw_des) < 0.08
            if np.linalg.norm(tcp[:2] - c[:2]) < 0.008 and tcp[2] < half + 0.13 and aligned:
                self.phase = 1
            return wp, OPEN, 1.0
        if self.phase == 1:  # descend around it, tracking yaw drift
            self.yaw_des = face_yaw(cube["quat"], s["yaw_setpoint"])
            wp = np.array([c[0], c[1], max(half * 0.9, 0.012)])
            if np.linalg.norm(tcp - wp) < 0.008:
                self.phase = 2
                self.hold = 0
            return wp, OPEN, 0.6
        if self.phase == 2:  # close and confirm; yaw frozen from here on
            self.hold += 1
            if self.hold >= 8:
                if width < 0.006:  # closed on air - missed; try again
                    self.phase = 0
                else:
                    self.phase = 3
            return np.array([c[0], c[1], max(half * 0.9, 0.012)]), CLOSE, 0.3
        # phase 3: carry - slowly. A 0.7-speed carry (0.56 m/s) yanks the
        # cube out of a friction grasp; 0.4 keeps every mass in the sampled
        # range held. If the cube slipped out anyway, start over.
        if width < 0.006 and c[2] < 0.05:
            self.phase = 0
            return np.array([c[0], c[1], half + 0.10]), OPEN, 1.0
        return np.asarray(carry_to, dtype=float), CLOSE, 0.4

    def _pick_place(
        self, inst: TaskInstance, s: dict,
        place_xy: np.ndarray, place_z: float, carry_z: float,
    ):
        """Phases 0-3 grasp (shared), 4 travel, 5 descend, 6 release, 7 retreat."""
        cube = s["objects"]["cube_a"]
        tcp = s["tcp_pos"]

        if self.phase <= 3:
            carry = np.array([*place_xy, carry_z])
            wp, grip, speed = self._grasp_and(inst, s, carry_to=carry)
            if self.phase == 3 and np.linalg.norm(tcp[:2] - place_xy) < 0.006:
                self.phase = 5
                self.hold = 0
            return wp, grip, speed
        if self.phase == 5:  # descend to release height
            wp = np.array([*place_xy, place_z])
            if abs(tcp[2] - place_z) < 0.006:
                self.phase = 6
                self.hold = 0
            # Dropped it on the way down? Regrasp from scratch.
            if s["finger_width"] < 0.006 and cube["pos"][2] < 0.05:
                self.phase = 0
            return wp, CLOSE, 0.35
        if self.phase == 6:  # open and let it settle
            self.hold += 1
            if self.hold >= 6:
                self.phase = 7
            return np.array([*place_xy, place_z]), OPEN, 0.2
        # phase 7: retreat straight up, leaving the placement undisturbed.
        return np.array([tcp[0], tcp[1], tcp[2] + 0.10]), OPEN, 0.6
