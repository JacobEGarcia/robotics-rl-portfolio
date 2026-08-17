"""
Train PPO on a Gymnasium continuous-control task.

Usage
-----
    python -m s1_ppo.train --env Pendulum-v1 --total-steps 150000 --seed 0
    python -m s1_ppo.train --env HalfCheetah-v5 --total-steps 1000000 --seed 0

Validation targets (what "correct" looks like)
----------------------------------------------
    Pendulum-v1     ~ -200 or better within ~150k steps
    HalfCheetah-v5  ~ 2000-3000 return at 1M steps

These are compared against CleanRL's published openrlbenchmark curves. A
number next to a reference number is validation; "the curve went up" is not.
If this implementation lands far below those, there is a bug -- and finding it
is the point of writing PPO by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from common.runlog import RunLogger
from common.seeding import seed_everything
from s1_ppo.ppo import PPO, PPOConfig, compute_gae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO from scratch")
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--runs-root", default="runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        learning_rate=args.lr,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )

    env = gym.make(args.env)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    # Actions are clipped to the env's bounds at step time rather than squashed
    # in the policy. PPO's Gaussian is unbounded; clipping is the standard
    # treatment and keeps the log-prob exact (a tanh squash would require a
    # Jacobian correction -- see the SAC implementation, where we do exactly
    # that because SAC's objective needs the corrected density).
    act_low, act_high = env.action_space.low, env.action_space.high

    agent = PPO(obs_dim, act_dim, cfg)

    config_record = {
        "env": args.env,
        "algo": "ppo_from_scratch",
        **cfg.to_dict(),
    }

    with RunLogger("s1_ppo", config=config_record, seed=args.seed, root=args.runs_root) as log:
        obs, _ = env.reset(seed=args.seed)
        agent.obs_norm.update(obs)
        norm_obs = agent.obs_norm.normalize(obs)

        global_step = 0
        episode_return, episode_length, episode_index = 0.0, 0, 0
        n_iterations = args.total_steps // cfg.rollout_steps

        for iteration in range(n_iterations):
            # ---------------- rollout ----------------
            buf_obs = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
            buf_act = np.zeros((cfg.rollout_steps, act_dim), dtype=np.float32)
            buf_logp = np.zeros(cfg.rollout_steps, dtype=np.float32)
            buf_rew = np.zeros(cfg.rollout_steps, dtype=np.float32)
            buf_val = np.zeros(cfg.rollout_steps, dtype=np.float32)
            buf_term = np.zeros(cfg.rollout_steps, dtype=bool)
            buf_trunc = np.zeros(cfg.rollout_steps, dtype=bool)

            for t in range(cfg.rollout_steps):
                buf_obs[t] = norm_obs
                with torch.no_grad():
                    action, logp, _, value = agent.net.get_action_and_value(
                        torch.as_tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
                    )
                a = action.squeeze(0).numpy()
                buf_act[t], buf_logp[t], buf_val[t] = a, logp.item(), value.item()

                next_obs, reward, terminated, truncated, _ = env.step(
                    np.clip(a, act_low, act_high)
                )
                # The agent learns from the SCALED reward, but we log the RAW
                # return. Reporting scaled returns would make numbers
                # incomparable to published baselines -- and to our own runs
                # once the scaler's statistics change.
                buf_rew[t] = agent.reward_scaler.scale(float(reward), terminated or truncated)
                buf_term[t], buf_trunc[t] = terminated, truncated

                episode_return += float(reward)
                episode_length += 1
                global_step += 1

                if terminated or truncated:
                    log.event(
                        "episode",
                        episode=episode_index,
                        step=global_step,
                        ret=episode_return,
                        length=episode_length,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    episode_index += 1
                    episode_return, episode_length = 0.0, 0
                    next_obs, _ = env.reset()

                agent.obs_norm.update(next_obs)
                norm_obs = agent.obs_norm.normalize(next_obs)

            # Bootstrap value for the final state of the rollout. Without this
            # the last transitions get a truncated return and the value
            # function is biased low near rollout boundaries.
            with torch.no_grad():
                next_value = agent.net.get_value(
                    torch.as_tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
                ).item()

            advantages, returns = compute_gae(
                buf_rew, buf_val, buf_term, buf_trunc, next_value, cfg.gamma, cfg.gae_lambda
            )

            stats = agent.update(
                buf_obs, buf_act, buf_logp, advantages, returns, buf_val,
                progress=iteration / max(1, n_iterations),
            )
            log.event("update", iteration=iteration, step=global_step, **stats)

            if iteration % 5 == 0 or iteration == n_iterations - 1:
                print(
                    f"iter {iteration:4d}  step {global_step:8d}  "
                    f"ev {stats['explained_variance']:+.3f}  "
                    f"kl {stats['approx_kl']:.4f}  lr {stats['lr']:.2e}",
                    flush=True,
                )

        # Save the policy plus the observation normaliser. Saving the network
        # alone is a classic error: the normaliser is part of the policy, and
        # without it the checkpoint produces garbage at evaluation time.
        ckpt_path = Path(log.path) / "policy.pt"
        torch.save(
            {
                "net": agent.net.state_dict(),
                "obs_norm": agent.obs_norm.state_dict(),
                "config": config_record,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            },
            ckpt_path,
        )
        log.event("checkpoint", path=str(ckpt_path))
        print(f"\nRun complete: {log.path}")

    env.close()


if __name__ == "__main__":
    main()
