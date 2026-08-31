"""
S3 trainer: the same PPO as S2, run twice, differing only in randomization.

    python -m s3_domain_rand.train --arm dr       --total-steps 30_000_000
    python -m s3_domain_rand.train --arm baseline --total-steps 30_000_000

Both arms share this file, this optimiser, this network, this seed handling
and this curriculum. That is the point. A comparison whose two halves came
from different scripts is comparing the scripts.

Two decisions that determine whether the S3 result means anything
----------------------------------------------------------------

**The curriculum is a fixed function of step count, not of success rate.**
S2 widens its target-angle range when the policy starts landing the current
one, which is the right thing for S2 and wrong here. Under an adaptive
curriculum the DR arm, which is solving a harder problem, advances more
slowly, so at any given step the two arms are training on *different task
distributions*, and the final held-out comparison silently mixes "DR helps
generalisation" with "one arm spent longer on easy targets". A schedule keyed
to step count costs some sample efficiency and buys the only thing that
matters here: at every step, both arms face the same task.

**Physics is resampled per episode, not per step.** A policy cannot adapt to
a mass that changes underneath it mid-episode, and randomizing per step would
train it to ignore object properties rather than to infer them. Per episode
is also what a real deployment looks like: you get one robot, with one set of
parameters, for the length of one attempt.

The network, PPO update, GAE and reward scaler are imported from
`s2_inhand.train` rather than copied. The reward-scaling bug that cost S1 a
full rebuild is fixed in exactly one place, and a copy here would be a second
place for it to come back.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from s2_inhand.env import InHandConfig
from s2_inhand.train import (
    RewardScaler,
    init_network,
    load_latest,
    log_prob,
    policy_mean,
    save_ckpt,
    value,
)
from s3_domain_rand.env import DomainRandEnv
from s3_domain_rand.randomize import DomainRandConfig

CKPT_ROOT = Path(__file__).parent / "checkpoints"


def curriculum_at(step: int, total: int, *, start: float, end: float,
                  warm_frac: float = 0.1, ramp_frac: float = 0.6) -> float:
    """
    Target-angle range as a fixed function of progress through the run.

    Flat for the first `warm_frac` of the budget, linear to `end` by
    `ramp_frac`, flat after. Deterministic given (step, total), so both arms
    and every resume see the identical schedule, including a resume that
    happens to land mid-ramp, which an adaptive rule would have to
    reconstruct from a checkpoint and would get subtly wrong.
    """
    frac = step / max(total, 1)
    if frac <= warm_frac:
        return start
    if frac >= ramp_frac:
        return end
    t = (frac - warm_frac) / (ramp_frac - warm_frac)
    return start + t * (end - start)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3: domain randomization, two arms")
    p.add_argument("--arm", choices=("dr", "baseline"), required=True)
    p.add_argument("--total-steps", type=int, default=30_000_000)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rollout-steps", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=2_000_000)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--ckpt-mirror", type=str, default=None)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--curr-start", type=float, default=0.4)
    p.add_argument("--curr-end", type=float, default=3.1)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_steps, args.num_envs, args.rollout_steps = 32, 4, 4
        args.ckpt_every, args.minibatches, args.epochs = 16, 2, 1

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_ROOT / args.arm
    mirror = Path(args.ckpt_mirror) if args.ckpt_mirror else None

    env = DomainRandEnv(InHandConfig(), DomainRandConfig(),
                        enabled=(args.arm == "dr"))
    print(f"arm={args.arm}  randomization={'on' if env.enabled else 'OFF (control)'}  "
          f"obs {env.obs_size}  act {env.action_size}  envs {args.num_envs}",
          flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, k_net = jax.random.split(key)
    params = init_network(k_net, env.obs_size, env.action_size)

    optim = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = optim.init(params)
    start_step = 0

    if args.resume:
        ck = load_latest(ckpt_dir, mirror=mirror)
        if ck is None:
            print("no checkpoint found, starting fresh", flush=True)
        else:
            # Refuse to resume the other arm's checkpoint. The two runs share
            # a network shape, so loading the wrong one succeeds silently and
            # produces a "baseline" that was pretrained with randomization, # exactly the contamination the experiment is designed to exclude.
            got = ck.get("args", {}).get("arm")
            if got and got != args.arm:
                raise SystemExit(
                    f"checkpoint at {ck.get('_from')} was written by arm "
                    f"{got!r}, not {args.arm!r}. Use a separate --ckpt-dir "
                    f"per arm."
                )
            params, opt_state = ck["params"], ck["opt_state"]
            start_step, key = ck["step"], ck["key"]
            print(f"resumed from step {start_step:,} ({ck.get('_from')})", flush=True)

    @jax.jit
    def act(p, obs, k):
        mean = policy_mean(p, obs)
        std = jnp.exp(p["log_std"])
        a = mean + std * jax.random.normal(k, mean.shape)
        return a, log_prob(p, obs, a), value(p, obs)

    value_jit = jax.jit(value)

    @jax.jit
    def update(p, opt_s, obs, act_b, logp_old, adv, ret):
        def loss_fn(pp):
            logp = log_prob(pp, obs, act_b)
            ratio = jnp.exp(logp - logp_old)
            a = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.minimum(ratio * a,
                              jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
            v_loss = jnp.mean((value(pp, obs) - ret) ** 2)
            ent = jnp.sum(pp["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
            return pg + 0.5 * v_loss - args.entropy * ent
        g = jax.grad(loss_fn)(p)
        upd, opt_s = optim.update(g, opt_s)
        return optax.apply_updates(p, upd), opt_s

    scaler = RewardScaler(args.gamma)
    steps_per_iter = args.num_envs * args.rollout_steps
    step_count = start_step
    next_ckpt = start_step + args.ckpt_every
    iters = 0
    t0 = time.time()

    # The in_axes tree depends only on which model fields are randomized, so
    # it is fixed for the whole run. Build it, and the three compiled
    # functions, once here. Rebuilding them per episode would be correct and
    # would recompile the model every reset.
    from s3_domain_rand.env import make_vectorised

    key, k_probe = jax.random.split(key)
    probe = env.sample_params(jax.random.split(k_probe, args.num_envs),
                              held_out=False)
    _, axes = env.batched_model(probe)
    reset_v, step_v, obs_v = make_vectorised(env, axes)

    def new_episode(k, curr):
        """Fresh physics draw, fresh batched model, fresh episode."""
        k_phys, k_reset = jax.random.split(k)
        phys = env.sample_params(jax.random.split(k_phys, args.num_envs),
                                 held_out=False)
        model_b, _ = env.batched_model(phys)
        s = reset_v(jax.random.split(k_reset, args.num_envs),
                    curr, phys["latency"], model_b)
        return s, model_b

    curriculum = curriculum_at(step_count, args.total_steps,
                               start=args.curr_start, end=args.curr_end)
    key, k_ep = jax.random.split(key)
    state, model_b = new_episode(k_ep, curriculum)

    while step_count < args.total_steps:
        curriculum = curriculum_at(step_count, args.total_steps,
                                   start=args.curr_start, end=args.curr_end)

        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf, succ_buf = [], [], []

        for _ in range(args.rollout_steps):
            obs = obs_v(state)
            key, k_a = jax.random.split(key)
            a, lp, v = act(params, obs, k_a)
            state = step_v(state, a, model_b)

            obs_buf.append(np.asarray(obs))
            act_buf.append(np.asarray(a))
            logp_buf.append(np.asarray(lp))
            val_buf.append(np.asarray(v))
            rew_buf.append(np.asarray(state.inner.reward))
            done_buf.append(np.asarray(state.inner.done))
            succ_buf.append(np.asarray(state.inner.success))

            if float(np.mean(state.inner.done)) > 0.5:
                key, k_ep = jax.random.split(key)
                state, model_b = new_episode(k_ep, curriculum)

        rewards = scaler(np.stack(rew_buf), np.stack(done_buf))
        dones = np.stack(done_buf)
        values = np.stack(val_buf)
        last_v = np.asarray(value_jit(params, obs_v(state)))

        adv = np.zeros_like(rewards)
        gae = np.zeros(args.num_envs, dtype=np.float32)
        for t in reversed(range(args.rollout_steps)):
            nv = last_v if t == args.rollout_steps - 1 else values[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + args.gamma * nv * nonterm - values[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + values

        b_obs = jnp.asarray(np.concatenate(obs_buf))
        b_act = jnp.asarray(np.concatenate(act_buf))
        b_logp = jnp.asarray(np.concatenate(logp_buf))
        b_adv = jnp.asarray(adv.reshape(-1))
        b_ret = jnp.asarray(ret.reshape(-1))

        n = b_obs.shape[0]
        mb = n // args.minibatches
        for _ in range(args.epochs):
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n)
            for i in range(args.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                params, opt_state = update(params, opt_state, b_obs[idx],
                                           b_act[idx], b_logp[idx],
                                           b_adv[idx], b_ret[idx])

        step_count += steps_per_iter
        succ_rate = float(np.mean(np.stack(succ_buf)))
        iters += 1
        if iters == 1 or iters % 10 == 0:
            el = time.time() - t0
            sps = (step_count - start_step) / max(el, 1e-6)
            print(f"[{args.arm}] step {step_count:>12,}  "
                  f"reward {float(np.mean(rew_buf[-1])):+8.3f}  "
                  f"success {succ_rate:5.2%}  curr {curriculum:4.2f}  "
                  f"{sps:,.0f} steps/s", flush=True)

        out_of_time = (args.max_hours is not None
                       and (time.time() - t0) > args.max_hours * 3600.0)

        if step_count >= next_ckpt or out_of_time:
            save_ckpt(ckpt_dir / f"ckpt_{step_count:012d}.pkl", {
                "params": params, "opt_state": opt_state, "step": step_count,
                "curriculum": curriculum, "key": key, "args": vars(args),
            }, mirror=mirror, keep_last=args.keep_last)
            meta = json.dumps({"arm": args.arm, "step": step_count,
                               "curriculum": curriculum, "success": succ_rate,
                               "wall_s": time.time() - t0}, indent=2)
            (ckpt_dir / "latest.json").write_text(meta)
            if mirror is not None:
                try:
                    mirror.mkdir(parents=True, exist_ok=True)
                    (mirror / "latest.json").write_text(meta)
                except OSError:
                    # save_ckpt already warned if the mirror is unreachable;
                    # the progress file is the least important thing on it.
                    pass
            print(f"  checkpoint @ {step_count:,}", flush=True)
            next_ckpt += args.ckpt_every

        if out_of_time:
            print(f"reached --max-hours at step {step_count:,}; exiting cleanly",
                  flush=True)
            return

    save_ckpt(ckpt_dir / f"ckpt_{step_count:012d}.pkl", {
        "params": params, "opt_state": opt_state, "step": step_count,
        "curriculum": curriculum, "key": key, "args": vars(args),
    }, mirror=mirror, keep_last=args.keep_last)
    print(f"[{args.arm}] done at {step_count:,} steps in {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
