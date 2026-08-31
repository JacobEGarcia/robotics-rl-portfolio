"""
S3: domain randomization over the S2 in-hand env, plus a held-out evaluation.

The claim this project can honestly make is *sim-to-sim transfer under domain
randomization*. There is no real LEAP hand here. Calling it sim-to-real would
be a lie, and the methodology is the point regardless: the question "does
randomizing training parameters improve performance on parameter values the
policy never saw" is answerable rigorously in simulation.

Design decisions that determine whether the result means anything
----------------------------------------------------------------

1.  **The held-out set is defined before training, in this file, and is
    disjoint from the training range by construction.** Training samples the
    interior of each range; evaluation samples a band outside it. Choosing the
    split after seeing results is how people accidentally report a training
    number and believe it.

2.  **Both arms are trained.** A no-DR baseline and a DR arm, identical in
    every other respect, both evaluated on the same held-out set. Reporting
    only the DR arm proves nothing: it would be consistent with DR helping,
    doing nothing, or hurting.

3.  **Action latency is randomized.** It is the parameter most often omitted
    and the one that most reliably breaks real deployments. s5_sysid measured
    3.30 frames of latency on a real Franka, which is where the range below
    comes from rather than from a guess.

Randomized parameters
---------------------
    cube mass           object inertia the policy must compensate for
    cube friction       slip behaviour, the dominant contact parameter
    actuator gain (kp)  how hard the position servo pulls
    joint damping       velocity-dependent resistance
    observation noise   sensing error on joint angles and object pose
    action latency      integer-frame delay on commanded control

Everything is expressed as a multiplier on the nominal model value, so the
ranges stay meaningful if the underlying model is ever updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Range:
    """A multiplicative range, plus the disjoint band used for evaluation."""
    train_lo: float
    train_hi: float
    test_lo: float
    test_hi: float

    def sample_train(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(key, (), minval=self.train_lo, maxval=self.train_hi)

    def sample_test(self, key: jax.Array) -> jax.Array:
        """
        Sample strictly outside the training range.

        Half the draws land below train_lo and half above train_hi, so the
        held-out set probes both directions rather than only the easy side.
        """
        k_side, k_val = jax.random.split(key)
        low = jax.random.uniform(k_val, (), minval=self.test_lo, maxval=self.train_lo)
        high = jax.random.uniform(k_val, (), minval=self.train_hi, maxval=self.test_hi)
        return jnp.where(jax.random.bernoulli(k_side), high, low)


@dataclass(frozen=True)
class DomainRandConfig:
    # Training band is the interior; test bands sit outside it on both sides.
    cube_mass: Range = Range(0.7, 1.3, 0.5, 1.6)
    cube_friction: Range = Range(0.7, 1.4, 0.5, 1.8)
    actuator_gain: Range = Range(0.8, 1.25, 0.65, 1.5)
    joint_damping: Range = Range(0.6, 1.6, 0.4, 2.2)

    # Additive, in the observation's own units rather than as a multiplier.
    obs_noise_joint: float = 0.01     # rad, roughly encoder-grade
    obs_noise_pos: float = 0.002      # m
    obs_noise_quat: float = 0.01      # unitless, pre-renormalisation

    # Integer control frames. s5_sysid identified 3.30 frames on a real Franka
    # at 15 Hz; this env runs at 50 Hz, so the band is widened accordingly and
    # the test band extends past it.
    latency_train: tuple[int, int] = (0, 4)
    latency_test: tuple[int, int] = (5, 7)

    def sample(self, key: jax.Array, held_out: bool) -> dict:
        keys = jax.random.split(key, 5)
        pick = (lambda r, k: r.sample_test(k)) if held_out else (lambda r, k: r.sample_train(k))
        lat_lo, lat_hi = self.latency_test if held_out else self.latency_train
        return {
            "cube_mass": pick(self.cube_mass, keys[0]),
            "cube_friction": pick(self.cube_friction, keys[1]),
            "actuator_gain": pick(self.actuator_gain, keys[2]),
            "joint_damping": pick(self.joint_damping, keys[3]),
            "latency": jax.random.randint(keys[4], (), lat_lo, lat_hi + 1),
        }


def randomized_fields(params: dict, nominal: dict) -> dict:
    """
    Just the four model arrays the randomizer touches, as a plain dict.

    Separated from `apply_to_model` so it can be `vmap`ped on its own. That
    separation is not stylistic, it is the difference between a batch that
    fits on a T4 and one that does not.

    `jax.vmap` adds a leading axis to *every* leaf of whatever its function
    returns. Vmapping a function that returns the whole model therefore
    replicates the entire model per environment: mesh vertices, convex hulls,
    body trees, everything. On the LEAP scene that is a few megabytes times
    `n_envs`, and at the batch sizes MJX is worth using at, it is an
    out-of-memory error whose message says nothing about domain
    randomization. Vmapping *this* function and folding the four results back
    into one shared model keeps the per-environment cost at four small
    arrays.

    Takes `nominal` explicitly rather than reading the current model values,
    so repeated application does not compound multipliers. Randomizing a model
    that was already randomized is a subtle and very easy mistake: the
    distribution silently drifts wider every episode and the "held-out" set
    stops being held out.
    """
    return {
        "body_mass": nominal["body_mass"].at[nominal["cube_body"]].set(
            nominal["cube_mass"] * params["cube_mass"]),
        "geom_friction": nominal["geom_friction"].at[nominal["cube_geom"], 0].set(
            nominal["cube_friction"] * params["cube_friction"]),
        "actuator_gainprm": nominal["actuator_gainprm"].at[:, 0].set(
            nominal["gain0"] * params["actuator_gain"]),
        "dof_damping": nominal["dof_damping"].at[:16].set(
            nominal["damping16"] * params["joint_damping"]),
    }


def apply_to_model(mjx_model, params: dict, nominal: dict):
    """One randomized model, for single-environment use (tests, debugging)."""
    return mjx_model.tree_replace(randomized_fields(params, nominal))


def add_obs_noise(key: jax.Array, obs: jax.Array, cfg: DomainRandConfig) -> jax.Array:
    """
    Corrupt the observation the way real sensing would.

    Layout must match `InHandEnv._obs`:
        [0:16]  joint positions
        [16:32] joint velocities
        [32:35] cube position
        [35:39] cube quaternion
        [39:43] target quaternion   (commanded, not sensed: left clean)
        [43:47] relative quaternion (derived: left clean)
        [47:63] previous action     (known exactly: left clean)

    Only the sensed quantities are corrupted. Adding noise to the target or to
    the previous action would model nothing physical and would just make the
    task harder for no reason.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    joint_pos = obs[0:16] + cfg.obs_noise_joint * jax.random.normal(k1, (16,))
    joint_vel = obs[16:32] + (cfg.obs_noise_joint * 10.0) * jax.random.normal(k2, (16,))
    cube_pos = obs[32:35] + cfg.obs_noise_pos * jax.random.normal(k3, (3,))

    quat = obs[35:39] + cfg.obs_noise_quat * jax.random.normal(k4, (4,))
    quat = quat / (jnp.linalg.norm(quat) + 1e-8)   # stays a unit quaternion

    return jnp.concatenate([joint_pos, joint_vel, cube_pos, quat, obs[39:]])


class LatencyBuffer:
    """
    Integer-frame action delay.

    Held as a rolling buffer of the last `max_latency + 1` commands; the action
    actually applied is the one from `latency` frames ago. Implemented with a
    fixed-size buffer and a gather rather than a Python-side deque so it stays
    jittable and the shape does not depend on the sampled latency.
    """

    def __init__(self, max_latency: int, action_dim: int):
        self.max_latency = max_latency
        self.action_dim = action_dim

    def init(self) -> jax.Array:
        return jnp.zeros((self.max_latency + 1, self.action_dim))

    def push(self, buf: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.concatenate([action[None], buf[:-1]], axis=0)

    def get(self, buf: jax.Array, latency: jax.Array) -> jax.Array:
        return buf[jnp.clip(latency, 0, self.max_latency)]
