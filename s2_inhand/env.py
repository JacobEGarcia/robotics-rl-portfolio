"""
S2: in-hand cube reorientation on the LEAP hand, as an MJX environment.

Task
----
A 5 cm cube starts held in a partially closed LEAP right hand. A target
orientation is sampled each episode. The policy commands the 16 finger joints
and must rotate the cube to within a tolerance of the target without dropping
it.

Observation is privileged state only: joint positions and velocities, cube
position and quaternion, target quaternion, and the previous action. No vision.
Vision is a separate project and would eat the schedule.

Reward is dense, because the sparse version is unlearnable here: the chance of
a random policy hitting a target orientation is effectively zero, so there is
no gradient to follow.

    r = -w_rot * theta                  angular distance to target
        + w_success * 1[theta < tol]    bonus for landing it
        - w_drop * 1[dropped]           penalty for losing the cube
        - w_action * |a|^2              small effort penalty

`theta = 2 * arccos(|q_target . q_cube|)` is the geodesic angle between the two
orientations. The absolute value matters: q and -q are the same rotation, and
omitting it makes the reward discontinuous at the antipode, which shows up as a
policy that mysteriously refuses to rotate past 180 degrees.

Curriculum
----------
`target_angle_max` scales how far the sampled target is from the cube's current
orientation. Starting wide makes the reward almost flat, because every action
looks equally bad from 180 degrees away. Training starts narrow and the range
widens as success rate crosses a threshold. This is the mechanical form of the
task ladder and without it nothing learns.

Why MJX
-------
This needs on the order of 1e8 environment steps. CPU MuJoCo with a handful of
envs is weeks of wall clock. MJX runs thousands of copies in parallel on one
GPU, which is what makes the run a matter of hours. See `s6_brax_jax/` for the
throughput measurement that motivated this.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

SCENE = Path(__file__).parent / "scene.xml"

N_JOINTS = 16          # LEAP right hand actuated DOF
CUBE_QPOS_START = 16   # free joint: 3 position + 4 quaternion
CUBE_QVEL_START = 16   # free joint: 3 linear + 3 angular


# --------------------------------------------------------------------------
# Quaternion helpers
# --------------------------------------------------------------------------

def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_conj(q: jax.Array) -> jax.Array:
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_angle(qa: jax.Array, qb: jax.Array) -> jax.Array:
    """
    Geodesic angle between two unit quaternions, in radians, on [0, pi].

    The abs() on the dot product folds q and -q together. They represent the
    same rotation, and without the fold the metric is discontinuous at the
    antipode.
    """
    dot = jnp.abs(jnp.sum(qa * qb, axis=-1))
    return 2.0 * jnp.arccos(jnp.clip(dot, 0.0, 1.0))


def random_quat_within(key: jax.Array, ref: jax.Array, max_angle: jax.Array) -> jax.Array:
    """Sample a unit quaternion at most `max_angle` away from `ref`."""
    k_axis, k_ang = jax.random.split(key)
    axis = jax.random.normal(k_axis, (3,))
    axis = axis / (jnp.linalg.norm(axis) + 1e-8)
    # cube-root keeps the sampled angle uniform in solid angle rather than
    # bunched near the maximum
    angle = max_angle * jax.random.uniform(k_ang) ** (1.0 / 3.0)
    half = 0.5 * angle
    delta = jnp.concatenate([jnp.cos(half)[None], jnp.sin(half) * axis])
    return quat_mul(ref, delta)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InHandConfig:
    ctrl_dt: float = 0.02          # policy runs at 50 Hz
    sim_dt: float = 0.002          # matches the model's own timestep
    episode_len: int = 300         # 6 s at 50 Hz

    success_tol: float = 0.1       # radians, per the project spec
    drop_dist: float = 0.10        # cube this far from its start counts as dropped

    w_rot: float = 1.0
    w_success: float = 5.0
    w_drop: float = 20.0
    w_action: float = 0.001

    action_scale: float = 0.3      # fraction of joint range per step

    target_angle_max: float = 0.4  # curriculum, widened by the trainer

    @property
    def n_substeps(self) -> int:
        return max(1, int(round(self.ctrl_dt / self.sim_dt)))


class InHandState(object):
    """Container so the scan carries one pytree rather than a tuple soup."""

    __slots__ = ("data", "target_quat", "prev_action", "step_count", "key",
                 "cube_start", "done", "reward", "success")

    def __init__(self, data, target_quat, prev_action, step_count, key,
                 cube_start, done, reward, success):
        self.data = data
        self.target_quat = target_quat
        self.prev_action = prev_action
        self.step_count = step_count
        self.key = key
        self.cube_start = cube_start
        self.done = done
        self.reward = reward
        self.success = success


jax.tree_util.register_pytree_node(
    InHandState,
    lambda s: ((s.data, s.target_quat, s.prev_action, s.step_count, s.key,
                s.cube_start, s.done, s.reward, s.success), None),
    lambda _, c: InHandState(*c),
)


class InHandEnv:
    """
    Stateless, functional in-hand reorientation environment on MJX.

    Every method takes state and returns new state. Nothing is stored on the
    instance except the compiled model and static config, so the whole thing
    vmaps and scans cleanly.
    """

    def __init__(self, cfg: InHandConfig | None = None):
        self.cfg = cfg or InHandConfig()

        mj_model = mujoco.MjModel.from_xml_path(str(SCENE))
        mj_model.opt.timestep = self.cfg.sim_dt
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)

        # reset pose comes from the `grasp` keyframe, which was grid-searched
        # for stability rather than chosen by eye. See scene.xml.
        self.qpos_init = jnp.array(mj_model.key_qpos[0])
        self.ctrl_init = jnp.array(mj_model.key_ctrl[0])

        lo = mj_model.actuator_ctrlrange[:, 0]
        hi = mj_model.actuator_ctrlrange[:, 1]
        self.ctrl_lo = jnp.array(lo)
        self.ctrl_hi = jnp.array(hi)
        self.ctrl_mid = jnp.array(0.5 * (lo + hi))
        self.ctrl_half_range = jnp.array(0.5 * (hi - lo))

        self.cube_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    # ---- accessors -------------------------------------------------------

    @property
    def obs_size(self) -> int:
        # 16 qpos + 16 qvel + 3 cube pos + 4 cube quat + 4 target quat
        # + 4 relative quat + 16 prev action
        return N_JOINTS * 2 + 3 + 4 + 4 + 4 + N_JOINTS

    @property
    def action_size(self) -> int:
        return N_JOINTS

    # ---- core ------------------------------------------------------------

    def _cube_quat(self, data) -> jax.Array:
        return data.qpos[CUBE_QPOS_START + 3: CUBE_QPOS_START + 7]

    def _cube_pos(self, data) -> jax.Array:
        return data.qpos[CUBE_QPOS_START: CUBE_QPOS_START + 3]

    def _obs(self, state: InHandState) -> jax.Array:
        d = state.data
        cube_q = self._cube_quat(d)
        rel = quat_mul(quat_conj(cube_q), state.target_quat)
        return jnp.concatenate([
            d.qpos[:N_JOINTS],
            d.qvel[:N_JOINTS],
            self._cube_pos(d),
            cube_q,
            state.target_quat,
            rel,
            state.prev_action,
        ])

    def reset(self, key: jax.Array, target_angle_max: jax.Array,
              model=None) -> InHandState:
        # `model` overrides the instance's own compiled model. It exists for
        # s3_domain_rand, which needs a *different* model per environment and
        # therefore has to pass one in through vmap's in_axes rather than
        # closing over a single shared one. Defaulting to None keeps every S2
        # call site and every S2 test unchanged.
        model = self.model if model is None else model
        key, k_target = jax.random.split(key)

        data = mjx.make_data(model)
        data = data.replace(qpos=self.qpos_init, ctrl=self.ctrl_init)
        data = mjx.forward(model, data)

        cube_q = self._cube_quat(data)
        target = random_quat_within(k_target, cube_q, target_angle_max)

        return InHandState(
            data=data,
            target_quat=target,
            prev_action=jnp.zeros(N_JOINTS),
            step_count=jnp.int32(0),
            key=key,
            cube_start=self._cube_pos(data),
            done=jnp.float32(0.0),
            reward=jnp.float32(0.0),
            success=jnp.float32(0.0),
        )

    def step(self, state: InHandState, action: jax.Array,
             model=None) -> InHandState:
        model = self.model if model is None else model
        cfg = self.cfg
        action = jnp.clip(action, -1.0, 1.0)

        # Action is a delta on the current commanded position, not an absolute
        # target. Absolute targets let the policy teleport the fingers between
        # consecutive steps, which trains fine in sim and is meaningless on
        # hardware.
        ctrl = state.data.ctrl + action * cfg.action_scale * self.ctrl_half_range
        ctrl = jnp.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

        def one(d, _):
            return mjx.step(model, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(one, state.data, None, length=cfg.n_substeps)

        cube_q = self._cube_quat(data)
        cube_p = self._cube_pos(data)
        theta = quat_angle(cube_q, state.target_quat)

        success = (theta < cfg.success_tol).astype(jnp.float32)
        dropped = (jnp.linalg.norm(cube_p - state.cube_start) > cfg.drop_dist).astype(jnp.float32)

        reward = (
            -cfg.w_rot * theta
            + cfg.w_success * success
            - cfg.w_drop * dropped
            - cfg.w_action * jnp.sum(jnp.square(action))
        )

        step_count = state.step_count + 1
        timeout = (step_count >= cfg.episode_len).astype(jnp.float32)
        done = jnp.clip(dropped + timeout, 0.0, 1.0)

        return InHandState(
            data=data,
            target_quat=state.target_quat,
            prev_action=action,
            step_count=step_count,
            key=state.key,
            cube_start=state.cube_start,
            done=done,
            reward=reward,
            success=success,
        )

    def obs(self, state: InHandState) -> jax.Array:
        return self._obs(state)
