"""
Train SAC on a Gymnasium continuous-control task.

Usage
-----
    python -m sac_scratch.train --env Pendulum-v1 --total-steps 30000 --seed 0

The headline comparison
-----------------------
SAC should reach comparable return to PPO in roughly an order of magnitude
fewer environment steps. On Pendulum, PPO needed ~80k steps to cross -200;
SAC should get there in well under 15k.

That ratio -- not either number alone -- is the deliverable, because it is the
empirical basis for "which algorithm would you put on a real robot".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from common.runlog import RunLogger
from common.seeding import seed_everything
from sac_scratch.sac import SAC, ReplayBuffer, SACConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC from scratch")
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--total-steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--runs-root", default="runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = SACConfig(
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        batch_size=args.batch_size,
        # Cap the buffer at the run length: allocating 10^6 rows for a 30k-step
        # run wastes hundreds of MB for no benefit.
        buffer_size=min(1_000_000, max(10_000, args.total_steps)),
    )

    env = gym.make(args.env)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low, act_high = env.action_space.low, env.action_space.high

    agent = SAC(obs_dim, act_dim, act_low, act_high, cfg, seed=args.seed)
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim)

    config_record = {"env": args.env, "algo": "sac_from_scratch", **cfg.to_dict()}

    with RunLogger("sac_scratch", config=config_record, seed=args.seed, root=args.runs_root) as log:
        obs, _ = env.reset(seed=args.seed)
        episode_return, episode_length, episode_index = 0.0, 0, 0

        for step in range(cfg.total_steps):
            # Uniform random actions during warmup. SAC is off-policy, so this
            # costs nothing in correctness and substantially improves early
            # state coverage in the buffer.
            if step < cfg.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.actor.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1

            # Store `terminated` only. On truncation the true next state is
            # the one we just observed and the episode would have continued,
            # so the transition must still bootstrap. Gymnasium exposes the
            # real final observation in info when autoreset occurs.
            real_next_obs = next_obs
            if truncated and "final_observation" in info:
                real_next_obs = info["final_observation"]
            buffer.add(obs, action, reward, real_next_obs, terminated)

            obs = next_obs

            if terminated or truncated:
                log.event("episode", episode=episode_index, step=step,
                          ret=episode_return, length=episode_length,
                          terminated=bool(terminated), truncated=bool(truncated))
                episode_index += 1
                episode_return, episode_length = 0.0, 0
                obs, _ = env.reset()

            if step >= cfg.learning_starts and buffer.size >= cfg.batch_size:
                stats = agent.update(buffer, step)
                if step % 1000 == 0:
                    log.event("update", step=step, **stats)
                    print(
                        f"step {step:7d}  critic {stats['critic_loss']:8.3f}  "
                        f"alpha {stats['alpha']:.4f}  q1 {stats['q1_mean']:+8.2f}",
                        flush=True,
                    )

        ckpt = Path(log.path) / "policy.pt"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "config": config_record,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "act_low": act_low,
                "act_high": act_high,
            },
            ckpt,
        )
        log.event("checkpoint", path=str(ckpt))
        print(f"\nRun complete: {log.path}")

    env.close()


if __name__ == "__main__":
    main()
