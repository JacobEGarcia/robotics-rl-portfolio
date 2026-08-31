"""
The S2 in-hand env with a different physics model in every parallel copy.

This is the piece that makes domain randomization real on a GPU, and it is
the piece people usually get wrong, in one of two ways.

**The wrong way that is slow.** Rebuild the MJX model between episodes.
`mjx.put_model` is a host-side conversion and every call invalidates the
compiled step function, so a run that re-randomizes each episode spends its
life in XLA rather than in physics.

**The wrong way that is silently not randomized.** Randomize once, share the
resulting model across the whole batch, and re-draw it every few thousand
steps. Every environment then sees the same physics at the same moment. The
policy can still overfit to the current draw and the batch carries a single
sample of the distribution instead of `n_envs` of them, so the gradient's
variance over physics parameters is not reduced at all. It trains, it looks
like DR, and its held-out performance is barely different from the baseline.

The right way is to keep one compiled step function and give it a *batched
model*: the randomized fields carry a leading environment axis and everything
else is shared. `jax.vmap` is told which is which through `in_axes`, so the
kernel is compiled once and each environment reads its own row.

    model_axes = tree of None, with 0 on exactly the randomized leaves
    batched    = the nominal model, with those leaves stacked over n_envs
    step_v     = jax.vmap(env.step, in_axes=(model_axes, 0, 0))

Two further pieces of realism sit on top of the physics, both from
`randomize.py`, and both matter more than the mass and friction ranges do:

*   **Action latency**, an integer-frame delay, per environment. It is the
    parameter most often left out and the one that most reliably breaks a
    deployment. s5_sysid measured 3.30 frames on a real Franka.
*   **Observation noise** on the sensed quantities only. The target
    orientation is commanded, the previous action is known exactly, and the
    relative quaternion is derived; corrupting those models nothing physical
    and just makes the task harder.

The held-out set is defined in `randomize.py`, before training, and is
disjoint from the training band by construction. That ordering is the whole
methodological claim of S3 and it is the reason the split lives in a
different file from the trainer that would be tempted to tune it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from s2_inhand.env import InHandConfig, InHandEnv, N_JOINTS
from s3_domain_rand.randomize import (
    DomainRandConfig,
    LatencyBuffer,
    add_obs_noise,
)

# Fields the randomizer writes. Anything not on this list is shared across the
# batch, which is what keeps the memory cost of `n_envs` models bounded: a
# LEAP model is a few megabytes and 4096 independent copies would not fit on a
# T4, while 4096 copies of four arrays is a few hundred kilobytes.
RANDOMIZED_FIELDS = ("body_mass", "geom_friction", "actuator_gainprm", "dof_damping")


@dataclass
class DRState:
    """S2's state, plus the per-environment things S2 has no concept of."""
    inner: object              # InHandState
    action_buf: jax.Array      # rolling command history, for latency
    latency: jax.Array         # integer frames, fixed for the episode
    obs_key: jax.Array         # observation-noise stream, separate from physics


jax.tree_util.register_pytree_node(
    DRState,
    lambda s: ((s.inner, s.action_buf, s.latency, s.obs_key), None),
    lambda _, c: DRState(*c),
)


class DomainRandEnv:
    """
    S2's task, with per-environment physics.

    Not a subclass. The randomized model has to be threaded through `vmap` as
    an argument, and a subclass that stored it on the instance would be
    closing over a traced value, which is the bug this whole file exists to
    avoid.
    """

    def __init__(self, cfg: InHandConfig | None = None,
                 dr: DomainRandConfig | None = None,
                 *, enabled: bool = True):
        self.base = InHandEnv(cfg)
        self.cfg = self.base.cfg
        self.dr = dr or DomainRandConfig()
        # `enabled=False` is the control arm. It runs this same code path with
        # every multiplier pinned to 1.0 and zero latency, rather than running
        # S2's trainer instead. The two arms must differ in the randomization
        # and in nothing else; comparing against a different script would also
        # be comparing against its buffer handling, its observation ordering
        # and its RNG consumption.
        self.enabled = enabled

        self.max_latency = max(self.dr.latency_test[1], self.dr.latency_train[1])
        self.latency_buf = LatencyBuffer(self.max_latency, N_JOINTS)

        m = self.base.mj_model
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        if self.cube_body < 0 or self.cube_geom < 0:
            raise RuntimeError(
                "scene.xml must define a body named 'cube' and a geom named "
                "'cube_geom'; randomization silently targets index -1 without "
                "this check, which writes the LAST body's mass instead."
            )

        mx = self.base.model
        # Nominal values, captured once. `randomize.apply_to_model` takes them
        # explicitly for the same reason: reading current values and
        # multiplying compounds the multiplier every time it is applied, and
        # the distribution drifts wider on every episode until the "held-out"
        # band is inside the training band.
        self.nominal = {
            "body_mass": mx.body_mass,
            "geom_friction": mx.geom_friction,
            "actuator_gainprm": mx.actuator_gainprm,
            "dof_damping": mx.dof_damping,
            "cube_body": self.cube_body,
            "cube_geom": self.cube_geom,
            "cube_mass": float(mx.body_mass[self.cube_body]),
            "cube_friction": float(mx.geom_friction[self.cube_geom, 0]),
            "gain0": mx.actuator_gainprm[:, 0],
            "damping16": mx.dof_damping[:N_JOINTS],
        }

    # ---- passthrough -----------------------------------------------------

    @property
    def obs_size(self) -> int:
        return self.base.obs_size

    @property
    def action_size(self) -> int:
        return self.base.action_size

    # ---- model batching --------------------------------------------------

    def sample_params(self, keys: jax.Array, held_out: bool) -> dict:
        """Per-environment physics parameters. `keys` is (n_envs, 2)."""
        if not self.enabled:
            n = keys.shape[0]
            return {
                "cube_mass": jnp.ones(n),
                "cube_friction": jnp.ones(n),
                "actuator_gain": jnp.ones(n),
                "joint_damping": jnp.ones(n),
                "latency": jnp.zeros(n, dtype=jnp.int32),
            }
        return jax.vmap(self.dr.sample, in_axes=(0, None))(keys, held_out)

    def batched_model(self, params: dict):
        """
        The nominal model with the randomized leaves stacked over the batch.

        Returns `(model, in_axes)`. Passing the axes tree around with the
        model rather than reconstructing it at each call site is deliberate:
        an `in_axes` tree that disagrees with the batched model by one leaf
        produces a shape error deep inside vmap whose message names an
        internal MJX field, and tracking it back to the wrong axes tree takes
        far longer than it should.
        """
        from s3_domain_rand.randomize import randomized_fields

        mx = self.base.model
        # vmap only over the four arrays. See `randomized_fields` for why
        # vmapping over the whole model is the wrong shape of correct.
        fields = jax.vmap(randomized_fields, in_axes=(0, None))(params, self.nominal)
        batched = mx.tree_replace(fields)

        axes = jax.tree_util.tree_map(lambda _: None, mx)
        axes = axes.tree_replace({f: 0 for f in RANDOMIZED_FIELDS})

        # The axes tree and the batched model must agree leaf for leaf. A
        # mismatch surfaces as a shape error inside vmap naming an internal
        # MJX field, which is a long way from the line that caused it.
        for f in RANDOMIZED_FIELDS:
            got = getattr(batched, f).shape
            base = getattr(mx, f).shape
            if got != (params["cube_mass"].shape[0], *base):
                raise RuntimeError(
                    f"{f} batched to {got}, expected "
                    f"{(params['cube_mass'].shape[0], *base)}")
        return batched, axes

    # ---- rollout ---------------------------------------------------------

    def reset(self, key: jax.Array, target_angle_max, latency, model) -> DRState:
        k_inner, k_obs = jax.random.split(key)
        inner = self.base.reset(k_inner, target_angle_max, model=model)
        return DRState(
            inner=inner,
            action_buf=self.latency_buf.init(),
            latency=latency.astype(jnp.int32),
            obs_key=k_obs,
        )

    def step(self, state: DRState, action: jax.Array, model) -> DRState:
        """
        One control step, with the command the hand receives delayed.

        The delay is applied to the command, not to the observation. Those are
        not the same thing and the difference shows up on hardware: delaying
        the observation and acting immediately models a slow sensor, while
        delaying the command models a slow actuator or a full control-loop
        round trip, which is what s5_sysid actually measured. The policy still
        sees the action it *chose* in its observation, because that is what it
        knows; it does not get told which stale command is currently executing.
        """
        buf = self.latency_buf.push(state.action_buf, action)
        applied = self.latency_buf.get(buf, state.latency)
        inner = self.base.step(state.inner, applied, model=model)

        # `prev_action` in the observation must be the action the policy just
        # chose, not the delayed command the hand is currently executing.
        # base.step wrote the delayed one, so it is replaced here. Rebuilt
        # rather than assigned in place: InHandState is a registered pytree,
        # and mutating a field of one that a transformation may already hold a
        # reference to is the kind of thing that works until it does not.
        from s2_inhand.env import InHandState
        inner = InHandState(
            data=inner.data, target_quat=inner.target_quat,
            prev_action=jnp.clip(action, -1.0, 1.0),
            step_count=inner.step_count, key=inner.key,
            cube_start=inner.cube_start, done=inner.done,
            reward=inner.reward, success=inner.success,
        )
        return DRState(inner=inner, action_buf=buf, latency=state.latency,
                       obs_key=state.obs_key)

    def obs(self, state: DRState) -> jax.Array:
        clean = self.base.obs(state.inner)
        if not self.enabled:
            return clean
        # Fold the step count into the noise key so consecutive steps get
        # independent noise. Reusing one key per episode would give a constant
        # offset, which a policy can and will learn to subtract out, turning
        # sensor noise into a free per-episode calibration signal.
        key = jax.random.fold_in(state.obs_key, state.inner.step_count)
        return add_obs_noise(key, clean, self.dr)


# --------------------------------------------------------------------------
# Vectorised entry points
# --------------------------------------------------------------------------

def make_vectorised(env: DomainRandEnv, axes):
    """
    `(reset_v, step_v, obs_v)` for a batch that carries its own models.

    `axes` is the `in_axes` tree returned alongside the batched model: `0` on
    the randomized leaves, `None` everywhere else.

    Built once and jitted here rather than at the call site, so the trainer
    cannot re-wrap inside its rollout loop. That mistake is invisible, `jax.jit(f)(x)` inside a loop is valid Python and correct JAX, and it
    recompiles the model on every iteration, which on a model whose compile
    costs minutes means the run never gets past the first few thousand steps.
    S2's trainer has a comment about this for the same reason.
    """
    reset_v = jax.jit(jax.vmap(env.reset, in_axes=(0, None, 0, axes)))
    step_v = jax.jit(jax.vmap(env.step, in_axes=(0, 0, axes)))
    obs_v = jax.jit(jax.vmap(env.obs))
    return reset_v, step_v, obs_v
