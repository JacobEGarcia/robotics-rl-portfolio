"""
Train the JAX PPO and compare against the PyTorch implementation.

    python -m s6_brax_jax.train --total-steps 3000000

What is being validated
-----------------------
Same task, same algorithm, two completely different implementations. If the
JAX port reaches the same performance as the PyTorch one, the port is correct.
If it does not, one of them has a bug — and finding out which is the entire
value of writing both.

The throughput number is reported honestly: this machine has no GPU, so what
is measured is JAX-on-CPU versus PyTorch-on-CPU. That still isolates the
benefit of the *program shape* (compiled scan over vectorised envs, versus an
interpreted Python loop) from the benefit of the hardware. The GPU number
would be strictly larger and is not claimed here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common.runlog import RunLogger
from common.seeding import seed_everything
from s6_brax_jax.ppo_jax import (
    EPISODE_LEN,
    PPOJaxConfig,
    PendulumJax,
    init_network,
    make_train,
    policy_mean,
    value,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO in JAX")
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = PPOJaxConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
    )
    env, reset_v, observe_v, optimizer, rollout, compute_gae, make_epoch = make_train(cfg)

    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_reset = jax.random.split(key, 3)

    params = init_network(k_init, env.obs_dim, env.act_dim, cfg.hidden)
    opt_state = optimizer.init(params)
    state = reset_v(jax.random.split(k_reset, cfg.num_envs))
    step_count = jnp.zeros(cfg.num_envs, dtype=jnp.int32)
    running_return = jnp.zeros(cfg.num_envs)

    steps_per_iter = cfg.num_envs * cfg.rollout_steps
    n_iters = max(1, args.total_steps // steps_per_iter)

    @jax.jit
    def train_iter(params, opt_state, state, step_count, key, running_return):
        """One rollout + one full update, entirely inside XLA."""
        state, step_count, key, running_return, traj = rollout(
            params, state, step_count, key, running_return
        )
        obs, actions, lps, values, rewards, truncated, run_rets, boot_vals = traj
        raw_reward_mean = rewards.mean()   # captured BEFORE scaling, for logging

        # REWARD SCALING — the fix for the bug this port reproduced from the
        # PyTorch implementation. Divide by the std of the discounted-return
        # accumulator so the value target is near unit variance and the value
        # loss stops monopolising the clipped global gradient norm.
        # Divide only; never subtract the mean, which would change the optimal
        # policy on variable-length episodes.
        reward_scale = jnp.std(run_rets) + 1e-8
        rewards = jnp.clip(rewards / reward_scale, -10.0, 10.0)

        last_obs = observe_v(state)
        last_value = jax.vmap(value, in_axes=(None, 0))(params, last_obs)
        advantages, returns = compute_gae(
            rewards, values, truncated, last_value, boot_vals / reward_scale
        )

        # Flatten (rollout_steps, num_envs, ...) -> (batch, ...)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        batch = (flat(obs), flat(actions), flat(lps), flat(advantages), flat(returns))

        epoch = make_epoch(params, opt_state, batch)
        key, k_ep = jax.random.split(key)
        (params, opt_state, _), losses = jax.lax.scan(
            epoch, (params, opt_state, k_ep), None, length=cfg.update_epochs
        )
        return (params, opt_state, state, step_count, key, running_return,
                raw_reward_mean, losses.mean())

    config_record = {
        "algo": "ppo_jax",
        "env": "PendulumJax",
        "num_envs": cfg.num_envs,
        "rollout_steps": cfg.rollout_steps,
        "total_steps": args.total_steps,
        "device": str(jax.devices()[0]),
    }

    with RunLogger("s6_ppo_jax", config=config_record, seed=args.seed) as log:
        # Compile once, outside the timed region. JIT compilation is a
        # one-off cost and including it in a throughput number would be
        # dishonest for any run longer than a few seconds.
        t_compile = time.time()
        out = train_iter(params, opt_state, state, step_count, key, running_return)
        jax.block_until_ready(out[0]["pi_w1"])
        compile_time = time.time() - t_compile
        print(f"JIT compile: {compile_time:.1f}s")
        log.event("compile", seconds=compile_time)

        params, opt_state, state, step_count, key, running_return = out[:6]

        t0 = time.time()
        env_steps = steps_per_iter
        for it in range(1, n_iters):
            (params, opt_state, state, step_count, key, running_return,
             mean_r, loss) = train_iter(
                params, opt_state, state, step_count, key, running_return
            )
            env_steps += steps_per_iter

            if it % 20 == 0 or it == n_iters - 1:
                # block_until_ready before reading: JAX is async, and without
                # this the timing would measure dispatch rather than compute.
                mean_r = float(jax.block_until_ready(mean_r))
                elapsed = time.time() - t0
                sps = env_steps / elapsed
                # Mean per-step reward -> approximate episode return.
                est_return = mean_r * EPISODE_LEN
                print(
                    f"  iter {it:5d}  steps {env_steps:9d}  "
                    f"est.return {est_return:8.1f}  {sps:,.0f} steps/s",
                    flush=True,
                )
                log.event(
                    "episode",
                    step=env_steps,
                    ret=est_return,
                    mean_step_reward=mean_r,
                    steps_per_sec=sps,
                )

        total_time = time.time() - t0
        throughput = env_steps / total_time

        # ---- final deterministic evaluation, mean action, fresh episodes ----
        eval_key = jax.random.PRNGKey(12345)
        n_eval = 256
        eval_state = reset_v(jax.random.split(eval_key, n_eval))
        total_r = jnp.zeros(n_eval)
        for _ in range(EPISODE_LEN):
            obs = observe_v(eval_state)
            act = jax.vmap(policy_mean, in_axes=(None, 0))(params, obs)
            eval_state, r = jax.vmap(PendulumJax.step)(eval_state, act)
            total_r = total_r + r
        final_return = float(jnp.mean(total_r))

        print(f"\nFinal deterministic return over {n_eval} episodes: {final_return:.1f}")
        print(f"Throughput: {throughput:,.0f} env steps/s on {jax.devices()[0]}")

        log.event(
            "result",
            final_return=final_return,
            throughput_steps_per_sec=throughput,
            total_env_steps=env_steps,
            wall_seconds=total_time,
            compile_seconds=compile_time,
            device=str(jax.devices()[0]),
        )
        print(f"\nRun: {log.path}")


if __name__ == "__main__":
    main()
