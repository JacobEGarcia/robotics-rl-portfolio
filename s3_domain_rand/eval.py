"""
Evaluate both S3 arms on physics they never trained on.

    python -m s3_domain_rand.eval \
        --dr s3_domain_rand/checkpoints/dr \
        --baseline s3_domain_rand/checkpoints/baseline \
        --episodes 512

The claim S3 is allowed to make is *sim-to-sim transfer under domain
randomization*. There is no real LEAP hand in this repository, and calling
this sim-to-real would be a lie. The methodology is the point either way: the
question "does randomizing training parameters improve performance on
parameter values the policy never saw" is answerable rigorously, and most
published sim-to-real results answer a weaker version of it.

What this file is careful about
-------------------------------

**Both arms see identical held-out draws.** The physics parameters are sampled
once, from a seed that has nothing to do with either training seed, and the
same batch is handed to both policies. Sampling per arm would put an
independent-sample difference on top of the effect, and with a few hundred
episodes that difference is the same size as the effect.

**Held-out means outside the training band by construction**, from
`randomize.py`, which was written before either arm trained. Not a re-split
chosen after seeing results.

**Three conditions, not two.** Nominal physics, in-distribution physics, and
held-out physics. Without the first two, a DR arm that is simply worse
everywhere and a DR arm that trades in-distribution performance for
robustness look identical. That trade is the interesting result and it is
invisible in a held-out number alone.

**Wilson intervals, and a paired difference.** A success rate of 0/100 has a
bootstrap interval of exactly zero width, which is not a claim about the world
but an artefact of resampling a constant. This bug was found in S8 and fixed
in `common/metrics.py`; the paired difference is what the headline should
quote, because the arms are evaluated on the same episodes and the pairing
removes the episode-difficulty variance that dominates the unpaired numbers.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common.metrics import paired_difference, wilson_interval
from s2_inhand.env import InHandConfig
from s2_inhand.train import policy_mean
from s3_domain_rand.env import DomainRandEnv, make_vectorised
from s3_domain_rand.randomize import DomainRandConfig

# Seed for the evaluation draws. Deliberately far from any training seed, and
# fixed in the source rather than passed in, so "which held-out set" is not a
# knob anyone can turn after seeing a result.
EVAL_SEED = 987_654_321


def load_params(ckpt_dir: Path) -> tuple[dict, dict]:
    cands = sorted(Path(ckpt_dir).glob("ckpt_*.pkl"))
    if not cands:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")
    with open(cands[-1], "rb") as f:
        ck = pickle.load(f)
    return ck["params"], {"step": ck["step"], "file": cands[-1].name,
                          "arm": ck.get("args", {}).get("arm")}


def rollout(env: DomainRandEnv, params: dict, phys: dict, model_b, axes,
            *, n_envs: int, target_angle: float, seed: int) -> dict:
    """
    One deterministic evaluation batch.

    The policy acts at its **mean**, with no exploration noise. Sampling from
    the Gaussian during evaluation measures the exploration schedule as much
    as the policy, and the two arms do not necessarily end with the same
    log_std.
    """
    reset_v, step_v, obs_v = make_vectorised(env, axes)
    act = jax.jit(lambda p, o: jnp.clip(policy_mean(p, o), -1.0, 1.0))

    key = jax.random.PRNGKey(seed)
    state = reset_v(jax.random.split(key, n_envs), target_angle,
                    phys["latency"], model_b)

    from s2_inhand.env import CUBE_QPOS_START, quat_angle

    ever_success = np.zeros(n_envs, dtype=bool)
    ever_dropped = np.zeros(n_envs, dtype=bool)

    cube_q = jax.jit(lambda s: s.inner.data.qpos[:, CUBE_QPOS_START + 3:
                                                 CUBE_QPOS_START + 7])
    cube_p = jax.jit(lambda s: s.inner.data.qpos[:, CUBE_QPOS_START:
                                                 CUBE_QPOS_START + 3])

    for _ in range(env.cfg.episode_len):
        a = act(params, obs_v(state))
        state = step_v(state, a, model_b)

        # Success is "reached the tolerance at any point in the episode", not
        # "is inside it on the last step". A hand that reaches the target and
        # then drifts out has solved the task; scoring only the final frame
        # measures how well it holds still afterwards, which is a different
        # and easier-to-game quantity.
        ever_success |= np.asarray(state.inner.success) > 0.5

        # Drop is computed from the cube's displacement, NOT from `done`.
        # `done` is `dropped OR timeout`, and every episode times out on its
        # last step, so counting `done` reports a 100% drop rate on a run
        # where nothing was ever dropped.
        disp = np.linalg.norm(np.asarray(cube_p(state))
                              - np.asarray(state.inner.cube_start), axis=-1)
        ever_dropped |= disp > env.cfg.drop_dist

    theta = np.asarray(jax.jit(jax.vmap(quat_angle))(
        jnp.asarray(cube_q(state)), state.inner.target_quat))

    return {
        "success_rate": float(ever_success.mean()),
        "drop_rate": float(ever_dropped.mean()),
        "final_theta_mean": float(np.mean(theta)),
        "final_theta_median": float(np.median(theta)),
        "per_episode_success": ever_success.astype(int).tolist(),
    }


def condition(env: DomainRandEnv, n_envs: int, *, kind: str, seed: int):
    """
    Physics for one evaluation condition.

    `nominal` pins every multiplier to 1.0 and latency to 0, the model as
    Menagerie ships it. `train` and `heldout` draw from the two disjoint bands
    defined in randomize.py.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    if kind == "nominal":
        phys = {
            "cube_mass": jnp.ones(n_envs), "cube_friction": jnp.ones(n_envs),
            "actuator_gain": jnp.ones(n_envs), "joint_damping": jnp.ones(n_envs),
            "latency": jnp.zeros(n_envs, dtype=jnp.int32),
        }
    else:
        phys = jax.vmap(env.dr.sample, in_axes=(0, None))(
            keys, kind == "heldout")
    model_b, axes = env.batched_model(phys)
    return phys, model_b, axes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", required=True, help="checkpoint dir for the DR arm")
    ap.add_argument("--baseline", required=True, help="checkpoint dir for the control arm")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--target-angle", type=float, default=3.1)
    ap.add_argument("--out", default="s3_domain_rand/results/eval.json")
    args = ap.parse_args()

    # One env object, randomization ON, for every condition and both arms.
    # The `enabled` flag belongs to training; at evaluation time the physics
    # is supplied explicitly, and a baseline evaluated through an env with
    # noise disabled would be evaluated without observation noise, which is
    # not the same test.
    env = DomainRandEnv(InHandConfig(), DomainRandConfig(), enabled=True)

    arms = {}
    for name, path in (("dr", args.dr), ("baseline", args.baseline)):
        p, meta = load_params(Path(path))
        if meta["arm"] and meta["arm"] != name:
            raise SystemExit(
                f"--{name} points at a checkpoint written by arm {meta['arm']!r}")
        arms[name] = (p, meta)
        print(f"{name}: {meta['file']} at step {meta['step']:,}", flush=True)

    results: dict = {"episodes": args.episodes, "target_angle": args.target_angle,
                     "eval_seed": EVAL_SEED, "arms": {k: v[1] for k, v in arms.items()},
                     "conditions": {}}

    # Fixed offsets, not `hash(kind)`: Python randomises string hashes per
    # process unless PYTHONHASHSEED is pinned, so a hash-derived seed makes
    # the held-out set different on every invocation and the evaluation
    # irreproducible in a way that leaves no trace in the output.
    OFFSET = {"nominal": 0, "train": 101, "heldout": 202}

    for kind in ("nominal", "train", "heldout"):
        # Same seed per condition, so both arms get the identical draw.
        phys, model_b, axes = condition(env, args.episodes, kind=kind,
                                        seed=EVAL_SEED + OFFSET[kind])
        row = {}
        for name, (p, _) in arms.items():
            r = rollout(env, p, phys, model_b, axes, n_envs=args.episodes,
                        target_angle=args.target_angle, seed=EVAL_SEED)
            n_succ = int(sum(r["per_episode_success"]))
            ci = wilson_interval(n_succ, args.episodes)
            r["wilson"] = {"lo": ci.low, "hi": ci.high}
            row[name] = r
            print(f"  {kind:<8} {name:<9} success {r['success_rate']:6.2%} "
                  f"[{ci.low:.2%}, {ci.high:.2%}]  drop {r['drop_rate']:6.2%}",
                  flush=True)

        a = np.asarray(row["dr"]["per_episode_success"], dtype=float)
        b = np.asarray(row["baseline"]["per_episode_success"], dtype=float)
        d = paired_difference(a, b)
        row["paired_dr_minus_baseline"] = {"point": float((a - b).mean()),
                                           "lo": d.low, "hi": d.high}
        print(f"  {kind:<8} paired DR - baseline "
              f"{(a - b).mean():+.3f} [{d.low:+.3f}, {d.high:+.3f}]", flush=True)
        results["conditions"][kind] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
