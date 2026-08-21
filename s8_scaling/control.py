"""
Resolved-rate Cartesian control for the Panda via damped least squares.

Why resolved-rate rather than a full IK solve per step
------------------------------------------------------
The policy (and the scripted experts) command small end-effector
displacements at 20 Hz. At that rate the tracking problem is differential:
one damped-least-squares step per control tick against the *measured*
Jacobian converges to the moving setpoint faster than the setpoint moves,
which is the textbook resolved-rate controller. A full iterative IK solve
per tick would triple the compute for no measurable tracking gain at these
speeds - and this study needs to simulate hundreds of hours of experience
on a CPU.

The damping term (lambda^2 I) is what makes the pseudo-inverse safe near
singularities: at the workspace boundary the raw pseudo-inverse requests
unbounded joint velocity, while DLS trades tracking error for boundedness.

Orientation is weighted 4x position (ori_gain 8 vs pos_gain 2). This was
tuned against a measured failure, not aesthetics: during fast carries the
6-D error is dominated by position, orientation correction is starved of
the shared max_dq budget, the hand accumulates 6-9 degrees of tilt, and a
gripped cube is levered out of the fingers. At 4x weighting the tilt stays
under ~2 degrees and lift reliability goes from 50% to 90%. Raising max_dq
instead makes it worse (joint-level jerk shakes the grasp).
"""

from __future__ import annotations

import mujoco
import numpy as np

# TCP offset in the hand frame: Franka hand flange-to-grasp-centre.
TCP_OFFSET = np.array([0.0, 0.0, 0.103])

# Desired hand orientation: z axis pointing straight down at the table.
# Rotation matrix columns are the hand frame axes expressed in world frame.
R_DOWN = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


def orientation_error(r_current: np.ndarray, r_desired: np.ndarray) -> np.ndarray:
    """
    Rotation error as an axis-angle vector in world frame.

    Uses the vee of the skew-symmetric part of (R_d R_c^T - R_c R_d^T)/2,
    exact for small errors and well-behaved up to 90 degrees - far beyond
    anything this controller sees, since the wrist starts and stays pointed
    down.
    """
    r_rel = r_desired @ r_current.T
    return 0.5 * np.array(
        [
            r_rel[2, 1] - r_rel[1, 2],
            r_rel[0, 2] - r_rel[2, 0],
            r_rel[1, 0] - r_rel[0, 1],
        ]
    )


class CartesianController:
    """
    Tracks a Cartesian TCP position setpoint with fixed downward orientation.

    The setpoint is integrated by the caller (the environment); this class
    only turns (setpoint - measured TCP) into position-servo targets for the
    seven arm actuators, plus the open/closed gripper command.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        damping: float = 0.05,
        pos_gain: float = 2.0,
        ori_gain: float = 8.0,
        max_dq: float = 0.15,
    ) -> None:
        self.model = model
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise ValueError("scene has no body named 'hand'")
        self.damping = damping
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.max_dq = max_dq
        # The seven arm joints are the first seven DoF in this scene because
        # the Panda include precedes every free object in the worldbody.
        self.arm_dofs = np.arange(7)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        # Joint limits with a small margin so the servo target never parks
        # exactly on a limit, where the position actuator fights the stop.
        jr = model.jnt_range[:7].copy()
        margin = 0.02 * (jr[:, 1] - jr[:, 0])
        self.q_lo = jr[:, 0] + margin
        self.q_hi = jr[:, 1] - margin

    def tcp_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """World TCP position and hand rotation matrix."""
        r_hand = data.xmat[self.hand_id].reshape(3, 3)
        pos = data.xpos[self.hand_id] + r_hand @ TCP_OFFSET
        return pos, r_hand

    def tcp_velocity(self, data: mujoco.MjData) -> np.ndarray:
        """World linear velocity of the TCP, exact via the Jacobian."""
        pos, _ = self.tcp_pose(data)
        mujoco.mj_jac(self.model, data, self.jacp, self.jacr, pos, self.hand_id)
        return self.jacp[:, self.arm_dofs] @ data.qvel[self.arm_dofs]

    def servo_targets(
        self, data: mujoco.MjData, target_pos: np.ndarray, yaw: float = 0.0
    ) -> np.ndarray:
        """
        One DLS step: joint-position targets that move the TCP toward
        `target_pos` while holding a downward orientation with wrist yaw
        `yaw`. Yaw control exists because a two-finger gripper closing on a
        randomly yawed cube otherwise grabs its corner-to-corner diagonal,
        from which the cube pivots out under gravity torque.
        """
        c, s = np.cos(yaw), np.sin(yaw)
        r_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        r_desired = r_z @ R_DOWN

        pos, r_hand = self.tcp_pose(data)
        mujoco.mj_jac(self.model, data, self.jacp, self.jacr, pos, self.hand_id)
        jac = np.vstack(
            [self.jacp[:, self.arm_dofs], self.jacr[:, self.arm_dofs]]
        )  # (6, 7)

        err = np.concatenate(
            [
                self.pos_gain * (target_pos - pos),
                self.ori_gain * orientation_error(r_hand, r_desired),
            ]
        )

        # dq = J^T (J J^T + lambda^2 I)^-1 err  - damped least squares.
        jjt = jac @ jac.T
        jjt[np.diag_indices_from(jjt)] += self.damping**2
        dq = jac.T @ np.linalg.solve(jjt, err)

        dq = np.clip(dq, -self.max_dq, self.max_dq)
        q_target = data.qpos[: len(self.arm_dofs)] + dq
        return np.clip(q_target, self.q_lo, self.q_hi)
