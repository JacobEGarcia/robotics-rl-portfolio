"""
S6: PPO reimplemented in JAX, plus a vectorised Pendulum environment.

Why this exists
---------------
The PyTorch PPO in `s1_ppo/` is a *loop*: step the env, append to a buffer,
then update. That structure is fundamentally sequential, and it is why
CPU-bound RL is slow — the Python interpreter is in the inner loop.

JAX inverts this. The rollout becomes a `jax.lax.scan` (a compiled loop that
never returns to Python), the environment is `jax.vmap`ped across thousands of
parallel copies, and the entire update is one `jax.jit`ed function. Nothing in
the hot path is interpreted.

That is not a micro-optimisation, it is a different program shape, and being
able to explain the difference is the point of writing both.

Validation strategy
-------------------
This trains on the SAME task as `s1_ppo` (Pendulum, same dynamics, same
reward, same -200 solved threshold), so the JAX port can be validated against
a result that already exists rather than against nothing. If the two
implementations disagree, one of them is wrong.

No flax
-------
Networks are plain JAX pytrees of arrays rather than a neural-network library.
`flax` cannot be installed on this machine (its `orbax` dependency ships test
fixtures whose paths exceed the Windows MAX_PATH limit), and for a two-layer
MLP the library buys nothing anyway. Optax handles the optimiser.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

# --------------------------------------------------------------------------
# Environment: Pendulum, vectorised, in pure JAX
# --------------------------------------------------------------------------
# Dynamics match Gymnasium's Pendulum-v1 exactly so results are comparable to
# the PyTorch run. Constants are from the Gymnasium source.

MAX_SPEED = 8.0
MAX_TORQUE = 2.0
DT = 0.05
GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0
EPISODE_LEN = 200


class PendulumJax:
    """
    Stateless, functional Pendulum. State is (theta, theta_dot).

    Everything is a pure function of (state, action, key) so JAX can trace,
    vmap, and jit it. There is no `self` mutation anywhere — that is the price
    of admission for XLA compilation, and it is also why these environments
    are trivially parallel.
    """

    obs_dim = 3
    act_dim = 1

    @staticmethod
    def reset(key: jax.Array) -> jax.Array:
        """Uniform reset over the full state space, as Gymnasium does."""
        k1, k2 = jax.random.split(key)
        theta = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
        theta_dot = jax.random.uniform(k2, (), minval=-1.0, maxval=1.0)
        return jnp.array([theta, theta_dot])

    @staticmethod
    def observe(state: jax.Array) -> jax.Array:
        """Gymnasium exposes (cos θ, sin θ, θ̇), not θ directly."""
        theta, theta_dot = state[0], state[1]
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])

    @staticmethod
    def step(state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One physics step. Returns (next_state, reward)."""
        theta, theta_dot = state[0], state[1]
        u = jnp.clip(action[0], -MAX_TORQUE, MAX_TORQUE)

        # Angle normalised to [-pi, pi] for the cost, so the reward is
        # symmetric about upright.
        angle_norm = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        reward = -(angle_norm**2 + 0.1 * theta_dot**2 + 0.001 * u**2)

        new_theta_dot = theta_dot + (
            3.0 * GRAVITY / (2 * LENGTH) * jnp.sin(theta)
            + 3.0 / (MASS * LENGTH**2) * u
        ) * DT
        new_theta_dot = jnp.clip(new_theta_dot, -MAX_SPEED, MAX_SPEED)
        # Gymnasium integrates theta using the UPDATED velocity (semi-implicit
        # Euler). Using the old velocity gives subtly different dynamics and
        # would make the comparison against the PyTorch run invalid.
        new_theta = theta + new_theta_dot * DT

        return jnp.array([new_theta, new_theta_dot]), reward


# --------------------------------------------------------------------------
# Network: plain pytrees
# --------------------------------------------------------------------------


def init_network(key: jax.Array, obs_dim: int, act_dim: int, hidden: int = 64) -> dict:
    """
    Orthogonal init with the same per-layer gains as the PyTorch version:
    sqrt(2) hidden, 0.01 policy head, 1.0 value head.
    """
    keys = jax.random.split(key, 6)

    def orth(k, shape, gain):
        # jax.nn.initializers.orthogonal matches torch.nn.init.orthogonal_
        return jax.nn.initializers.orthogonal(gain)(k, shape)

    return {
        "pi_w1": orth(keys[0], (obs_dim, hidden), np.sqrt(2)),
        "pi_b1": jnp.zeros(hidden),
        "pi_w2": orth(keys[1], (hidden, hidden), np.sqrt(2)),
        "pi_b2": jnp.zeros(hidden),
        "pi_w3": orth(keys[2], (hidden, act_dim), 0.01),
        "pi_b3": jnp.zeros(act_dim),
        # State-independent log_std, exactly as in the PyTorch implementation.
        "log_std": jnp.zeros(act_dim),
        "v_w1": orth(keys[3], (obs_dim, hidden), np.sqrt(2)),
        "v_b1": jnp.zeros(hidden),
        "v_w2": orth(keys[4], (hidden, hidden), np.sqrt(2)),
        "v_b2": jnp.zeros(hidden),
        "v_w3": orth(keys[5], (hidden, 1), 1.0),
        "v_b3": jnp.zeros(1),
    }


def policy_mean(p: dict, obs: jax.Array) -> jax.Array:
    h = jnp.tanh(obs @ p["pi_w1"] + p["pi_b1"])
    h = jnp.tanh(h @ p["pi_w2"] + p["pi_b2"])
    return h @ p["pi_w3"] + p["pi_b3"]


def value(p: dict, obs: jax.Array) -> jax.Array:
    h = jnp.tanh(obs @ p["v_w1"] + p["v_b1"])
    h = jnp.tanh(h @ p["v_w2"] + p["v_b2"])
    return (h @ p["v_w3"] + p["v_b3"]).squeeze(-1)


def log_prob(p: dict, obs: jax.Array, action: jax.Array) -> jax.Array:
    mean = policy_mean(p, obs)
    std = jnp.exp(p["log_std"])
    # Diagonal Gaussian log-density, summed over action dims.
    return jnp.sum(
        -0.5 * ((action - mean) / std) ** 2 - p["log_std"] - 0.5 * jnp.log(2 * jnp.pi),
        axis=-1,
    )


@dataclass
class PPOJaxConfig:
    num_envs: int = 512
    rollout_steps: int = 32
    total_steps: int = 3_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 8
    num_minibatches: int = 8
    hidden: int = 64


def make_train(cfg: PPOJaxConfig):
    """
    Build a fully-jitted training step.

    Returns a function `train_step(params, opt_state, env_state, key)` that
    performs one rollout + one update entirely inside XLA. Python sees one
    call per iteration rather than per environment step, which is where the
    speedup comes from.
    """
    env = PendulumJax
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    observe_v = jax.vmap(env.observe)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate, eps=1e-5),
    )

    def rollout(params, env_state, step_count, key, running_return):
        """
        Collect one rollout with lax.scan — a compiled loop, no Python.

        `running_return` is the per-env discounted-return accumulator, carried
        so reward scaling can be applied. See the note in `train.py`: the JAX
        port reproduced the PyTorch implementation's reward-scaling bug
        exactly (flat at ~-1200, 196k steps/s of learning nothing), which is
        itself evidence the two implementations agree.
        """

        def one_step(carry, _):
            state, count, k, run_ret = carry
            k, k_act = jax.random.split(k)

            obs = observe_v(state)
            mean = jax.vmap(policy_mean, in_axes=(None, 0))(params, obs)
            std = jnp.exp(params["log_std"])
            action = mean + std * jax.random.normal(k_act, mean.shape)
            lp = jax.vmap(log_prob, in_axes=(None, 0, 0))(params, obs, action)
            val = jax.vmap(value, in_axes=(None, 0))(params, obs)

            next_state, reward = step_v(state, action)
            count = count + 1

            # Value of the TRUE next state, computed BEFORE autoreset.
            # On a truncated step the stored next state is a freshly reset
            # episode, so values[t+1] refers to a different trajectory. Using
            # it as the bootstrap is the exact truncation bug this codebase
            # calls out in the PyTorch GAE -- it has to be handled here too,
            # and it is invisible without deliberately capturing this value.
            boot_val = jax.vmap(value, in_axes=(None, 0))(params, observe_v(next_state))

            # Track the discounted-return accumulator per env. This is the
            # quantity whose scale the critic must fit, so it — not the raw
            # reward — is what the scaler should be derived from.
            run_ret = run_ret * cfg.gamma + reward

            # Pendulum has no termination, only a 200-step time limit. Every
            # episode end is a TRUNCATION, so the value must be bootstrapped —
            # the same distinction the PyTorch implementation handles in GAE.
            truncated = count >= EPISODE_LEN
            k, k_reset = jax.random.split(k)
            reset_states = reset_v(jax.random.split(k_reset, state.shape[0]))
            next_state = jnp.where(truncated[:, None], reset_states, next_state)
            count = jnp.where(truncated, 0, count)
            # Reset the accumulator at episode boundaries, same as PyTorch.
            run_ret = jnp.where(truncated, 0.0, run_ret)

            return (next_state, count, k, run_ret), (
                obs, action, lp, val, reward, truncated, run_ret, boot_val,
            )

        (final_state, final_count, key, final_run_ret), traj = jax.lax.scan(
            one_step,
            (env_state, step_count, key, running_return),
            None,
            length=cfg.rollout_steps,
        )
        return final_state, final_count, key, final_run_ret, traj

    def compute_gae(rewards, values, truncated, last_value, boot_values):
        """
        GAE with truncation-aware bootstrapping.

        Because every episode end here is a truncation rather than a true
        termination, the value is ALWAYS bootstrapped; what resets is the
        advantage accumulator, since step t+1 belongs to a different episode.
        """

        def scan_fn(carry, x):
            gae, next_value = carry
            reward, value_t, trunc, boot = x
            # On truncation bootstrap the TRUE next state's value (the episode
            # would have continued); otherwise use the next step's value.
            nv = jnp.where(trunc, boot, next_value)
            delta = reward + cfg.gamma * nv - value_t
            gae = delta + cfg.gamma * cfg.gae_lambda * gae
            # The accumulator still resets: step t+1 is a different episode.
            gae = jnp.where(trunc, delta, gae)
            return (gae, value_t), gae

        _, advantages = jax.lax.scan(
            scan_fn,
            (jnp.zeros_like(last_value), last_value),
            (rewards, values, truncated, boot_values),
            reverse=True,
        )
        return advantages, advantages + values

    def update(params, opt_state, batch):
        obs, actions, old_lp, advantages, returns = batch

        def loss_fn(p, mb):
            o, a, olp, adv, ret = mb
            new_lp = jax.vmap(log_prob, in_axes=(None, 0, 0))(p, o, a)
            ratio = jnp.exp(new_lp - olp)
            # Per-minibatch advantage normalisation, matching PyTorch.
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
            ).mean()
            v = jax.vmap(value, in_axes=(None, 0))(p, o)
            vloss = 0.5 * ((v - ret) ** 2).mean()
            ent = (p["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
            return pg + cfg.value_coef * vloss - cfg.entropy_coef * ent

        batch_size = obs.shape[0]
        mb_size = batch_size // cfg.num_minibatches

        def epoch(carry, _):
            p, os_, k = carry
            k, ks = jax.random.split(k)
            perm = jax.random.permutation(ks, batch_size)
            shuffled = tuple(x[perm] for x in (obs, actions, old_lp, advantages, returns))

            def minibatch(c, i):
                p_, os__ = c
                sl = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, i * mb_size, mb_size), shuffled
                )
                loss, grads = jax.value_and_grad(loss_fn)(p_, sl)
                updates, os__ = optimizer.update(grads, os__)
                p_ = optax.apply_updates(p_, updates)
                return (p_, os__), loss

            (p, os_), losses = jax.lax.scan(
                minibatch, (p, os_), jnp.arange(cfg.num_minibatches)
            )
            return (p, os_, k), losses.mean()

        return epoch

    return env, reset_v, observe_v, optimizer, rollout, compute_gae, update
