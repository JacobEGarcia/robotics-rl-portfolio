"""
S2 trainer: PPO on the MJX in-hand reorientation env, with checkpoint-resume.

Checkpointing is not a nicety here, it is the reason the run is possible.
Kaggle's free GPU tier caps a session at roughly 12 hours and then destroys the
machine. A 1e8-step run plus reward-shaping iterations does not fit in one
session, so the trainer is written to be killed at any moment and resumed from
the newest checkpoint.

What gets saved, and why each piece matters:

    params        the policy and value networks. Obvious.
    opt_state     Adam's moment estimates. Dropping these restarts the
                  optimiser cold and produces a visible loss spike on every
                  resume, which is easy to misread as instability in the
                  algorithm.
    step          so the learning-rate schedule and the curriculum continue
                  from where they were rather than restarting.
    key           the PRNG key. Without it a resumed run re-draws the same
                  target orientations it already trained on, and the run is
                  no longer reproducible from its seed.
    curriculum    the current target-angle range.

Usage
-----
    python -m s2_inhand.train --total-steps 100_000_000 --num-envs 2048
    python -m s2_inhand.train --resume                      # picks newest ckpt

Run the same command again after a session dies. It resumes automatically.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from s2_inhand.env import InHandConfig, InHandEnv

CKPT_DIR = Path(__file__).parent / "checkpoints"


# --------------------------------------------------------------------------
# Network: plain pytrees, no flax
# --------------------------------------------------------------------------
# flax cannot be installed on the development machine (orbax ships test
# fixtures whose paths exceed the Windows MAX_PATH limit). For an MLP it buys
# nothing anyway. Same choice as s6_brax_jax/.

def init_network(key, obs_dim: int, act_dim: int, hidden: int = 256) -> dict:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    def dense(k, fan_in, fan_out, scale=None):
        s = scale if scale is not None else np.sqrt(2.0 / fan_in)
        return {"w": jax.random.normal(k, (fan_in, fan_out)) * s,
                "b": jnp.zeros(fan_out)}

    return {
        "pi1": dense(k1, obs_dim, hidden),
        "pi2": dense(k2, hidden, hidden),
        # small final layer so the initial policy is near-neutral rather than
        # saturating the tanh on step one
        "pi3": dense(k3, hidden, act_dim, scale=0.01),
        "log_std": jnp.full((act_dim,), -0.5),
        "v1": dense(k4, obs_dim, hidden),
        "v2": dense(k5, hidden, hidden),
        "v3": dense(k6, hidden, 1, scale=1.0),
    }


def _mlp(p, obs, keys):
    h = jnp.tanh(obs @ p[keys[0]]["w"] + p[keys[0]]["b"])
    h = jnp.tanh(h @ p[keys[1]]["w"] + p[keys[1]]["b"])
    return h @ p[keys[2]]["w"] + p[keys[2]]["b"]


def policy_mean(p, obs):
    return _mlp(p, obs, ("pi1", "pi2", "pi3"))


def value(p, obs):
    return jnp.squeeze(_mlp(p, obs, ("v1", "v2", "v3")), axis=-1)


def log_prob(p, obs, action):
    mean = policy_mean(p, obs)
    log_std = p["log_std"]
    std = jnp.exp(log_std)
    z = (action - mean) / std
    return jnp.sum(-0.5 * z ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


# --------------------------------------------------------------------------
# Reward scaling
# --------------------------------------------------------------------------
# Carried over from s1_ppo deliberately. Without it, returns on the order of
# -300 drive the value loss to ~1e5 while the policy loss sits near 1e-2, and
# global gradient clipping then hands essentially the entire update to the
# critic. Nothing errors, nothing looks wrong, and the policy simply never
# moves. That bug cost a full rebuild in s1 and was independently reproduced in
# s6, so it is fixed here from the start.
#
# Divide by the running std of the discounted return. Never subtract the mean:
# on variable-length episodes that changes which policy is optimal.

class RewardScaler:
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.ret = None
        self.var = 1.0
        self.count = 1e-4

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        if self.ret is None:
            self.ret = np.zeros(rewards.shape[1], dtype=np.float64)
        out = np.empty_like(rewards)
        for t in range(rewards.shape[0]):
            self.ret = self.ret * self.gamma * (1.0 - dones[t]) + rewards[t]
            self.count += 1
            delta = self.ret.mean() - 0.0
            self.var += (self.ret.var() + delta ** 2 - self.var) / self.count
            out[t] = rewards[t] / (np.sqrt(self.var) + 1e-8)
        return out


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename: a session killed mid-write must not leave a truncated
    # checkpoint that the next session then tries to resume from
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def load_latest(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("ckpt_*.pkl"))
    if not cands:
        return None
    with open(cands[-1], "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2: PPO on MJX in-hand reorientation")
    p.add_argument("--total-steps", type=int, default=100_000_000)
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
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="minimal run proving the loop closes, including a "
                        "checkpoint write and reload. Sized for CPU, where a "
                        "single mjx.step costs seconds; this is a correctness "
                        "check, never a performance measurement.")
    # curriculum
    p.add_argument("--curr-start", type=float, default=0.4)
    p.add_argument("--curr-max", type=float, default=3.1)
    p.add_argument("--curr-step", type=float, default=0.1)
    p.add_argument("--curr-success", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        # Deliberately tiny. On CPU one mjx.step of this model costs seconds,
        # so this is sized to close the loop twice and write a checkpoint, not
        # to learn anything.
        args.total_steps, args.num_envs, args.rollout_steps = 32, 4, 4
        args.ckpt_every, args.minibatches, args.epochs = 16, 2, 1

    ckpt_dir = Path(args.ckpt_dir)
    cfg = InHandConfig()
    env = InHandEnv(cfg)
    print(f"obs {env.obs_size}  act {env.action_size}  "
          f"substeps {cfg.n_substeps}  envs {args.num_envs}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, k_net = jax.random.split(key)
    params = init_network(k_net, env.obs_size, env.action_size)

    optim = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = optim.init(params)
    start_step = 0
    curriculum = args.curr_start

    if args.resume:
        ck = load_latest(ckpt_dir)
        if ck is None:
            print("no checkpoint found, starting fresh", flush=True)
        else:
            params = ck["params"]
            opt_state = ck["opt_state"]
            start_step = ck["step"]
            curriculum = ck["curriculum"]
            key = ck["key"]
            print(f"resumed from step {start_step:,} curriculum {curriculum:.2f}", flush=True)

    reset_v = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_v = jax.jit(jax.vmap(env.step))
    obs_v = jax.jit(jax.vmap(env.obs))

    key, k_reset = jax.random.split(key)
    state = reset_v(jax.random.split(k_reset, args.num_envs), curriculum)

    scaler = RewardScaler(args.gamma)
    steps_per_iter = args.num_envs * args.rollout_steps
    step_count = start_step
    next_ckpt = start_step + args.ckpt_every
    iters = 0
    t0 = time.time()

    @jax.jit
    def act(p, obs, k):
        mean = policy_mean(p, obs)
        std = jnp.exp(p["log_std"])
        a = mean + std * jax.random.normal(k, mean.shape)
        return a, log_prob(p, obs, a), value(p, obs)

    # Hoisted deliberately. Calling jax.jit(value)(...) inside the rollout loop
    # builds a fresh wrapper every iteration, so nothing ever hits the
    # compilation cache and the model recompiles once per iteration. On a model
    # whose compile costs minutes that would dominate the entire run.
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

    while step_count < args.total_steps:
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf, succ_buf = [], [], []

        for _ in range(args.rollout_steps):
            obs = obs_v(state)
            key, k_a = jax.random.split(key)
            a, lp, v = act(params, obs, k_a)
            state = step_v(state, a)

            obs_buf.append(np.asarray(obs))
            act_buf.append(np.asarray(a))
            logp_buf.append(np.asarray(lp))
            val_buf.append(np.asarray(v))
            rew_buf.append(np.asarray(state.reward))
            done_buf.append(np.asarray(state.done))
            succ_buf.append(np.asarray(state.success))

            # reset finished envs. Done as a whole-batch reset because a
            # partial reset under vmap needs a select over the entire MJX
            # Data pytree, and the wasted steps are cheap next to the
            # readability cost.
            if float(np.mean(state.done)) > 0.5:
                key, k_r = jax.random.split(key)
                state = reset_v(jax.random.split(k_r, args.num_envs), curriculum)

        rewards = np.stack(rew_buf)
        dones = np.stack(done_buf)
        rewards = scaler(rewards, dones)
        values = np.stack(val_buf)

        obs_last = obs_v(state)
        last_v = np.asarray(value_jit(params, obs_last))

        # GAE. The truncation-vs-termination distinction from s6 applies here
        # too: a timeout is not a terminal state and its value must still be
        # bootstrapped, otherwise the agent learns that time running out is
        # intrinsically bad.
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

        # widen the curriculum once the policy is reliably landing the current
        # range. Never narrow it: oscillating the task distribution makes the
        # value function chase a moving target.
        if succ_rate > args.curr_success and curriculum < args.curr_max:
            curriculum = min(args.curr_max, curriculum + args.curr_step)
            print(f"  curriculum -> {curriculum:.2f} rad", flush=True)

        # Always print the first iteration, then every tenth. The first one is
        # the only evidence that the loop closed at all, and a modulo-only
        # condition silently prints nothing on short runs, which makes the
        # smoke gate unfalsifiable.
        iters += 1
        if iters == 1 or iters % 10 == 0:
            el = time.time() - t0
            sps = (step_count - start_step) / max(el, 1e-6)
            print(f"step {step_count:>12,}  reward {float(np.mean(rew_buf[-1])):+8.3f}  "
                  f"success {succ_rate:5.2%}  curr {curriculum:4.2f}  "
                  f"{sps:,.0f} steps/s", flush=True)

        if step_count >= next_ckpt:
            save_ckpt(ckpt_dir / f"ckpt_{step_count:012d}.pkl", {
                "params": params, "opt_state": opt_state, "step": step_count,
                "curriculum": curriculum, "key": key,
                "args": vars(args),
            })
            (ckpt_dir / "latest.json").write_text(json.dumps({
                "step": step_count, "curriculum": curriculum,
                "success": succ_rate, "wall_s": time.time() - t0,
            }, indent=2))
            print(f"  checkpoint @ {step_count:,}", flush=True)
            next_ckpt += args.ckpt_every

    save_ckpt(ckpt_dir / f"ckpt_{step_count:012d}.pkl", {
        "params": params, "opt_state": opt_state, "step": step_count,
        "curriculum": curriculum, "key": key, "args": vars(args),
    })
    print(f"done at {step_count:,} steps in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
