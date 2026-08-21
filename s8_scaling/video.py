"""
Montage renders: the data engine and the learned policies, on camera.

Two artifacts:

  data_engine.mp4   scripted experts demonstrating each family, several
                    procedurally varied instances per family - what the
                    corpus actually looks like
  policy.mp4        the pretrained multi-task policy (and the transfer
                    policy on stack) driving the same families closed-loop

Frames are rendered once per control tick (20 Hz) and encoded at 20 fps,
so the videos play in real time. Captions are drawn with PIL directly on
the frames; success/failure is stamped at each episode's end so the video
makes no claim a viewer cannot check against the stamp.

Run:  python -m s8_scaling.video --mode expert --out s8_scaling/media/data_engine.mp4
      python -m s8_scaling.video --mode policy --ckpt <run_dir>/policy.pt ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import TabletopEnv
from .expert import ScriptedExpert
from .tasks import FAMILIES, PRETRAIN_FAMILIES, sample_instance

FPS = 20
W, H = 1280, 720


def _font(size: int):
    for name in ("consola.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_BIG = _font(44)
FONT_SMALL = _font(26)


def caption(frame: np.ndarray, title: str, subtitle: str, stamp: str | None = None) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, H - 92, W, H], fill=(8, 8, 10, 190))
    draw.text((28, H - 82), title, font=FONT_BIG, fill=(235, 235, 230))
    draw.text((30, H - 34), subtitle, font=FONT_SMALL, fill=(150, 150, 146))
    if stamp:
        colour = (60, 200, 110) if stamp == "SUCCESS" else (220, 90, 70)
        draw.rectangle([W - 250, H - 78, W - 28, H - 22], outline=colour, width=3)
        draw.text((W - 232, H - 68), stamp, font=FONT_BIG, fill=colour)
    return np.asarray(img)


def render_episodes(
    writer,
    env: TabletopEnv,
    actor,
    family: str,
    seeds: list[int],
    title: str,
    subtitle: str,
) -> None:
    for seed in seeds:
        inst = sample_instance(family, seed)
        env.reset_to(inst)
        if hasattr(actor, "reset"):
            actor.reset()
        frames_tail = 0
        info = {"success": False}
        for _ in range(inst.horizon):
            action = actor.act(env) if isinstance(actor, ScriptedExpert) else actor.act(env._obs())
            _, _, term, trunc, info = env.step(action)
            writer.append_data(caption(env.render(), title, subtitle))
            if term or trunc:
                break
        stamp = "SUCCESS" if info["success"] else "FAILURE"
        # Hold the final frame for a beat with the outcome stamped.
        last = caption(env.render(), title, subtitle, stamp)
        for _ in range(int(FPS * 0.8)):
            writer.append_data(last)
            frames_tail += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Render S8 montage videos.")
    ap.add_argument("--mode", choices=["expert", "policy"], required=True)
    ap.add_argument("--ckpt", help="policy checkpoint (mode=policy)")
    ap.add_argument("--stack-ckpt", help="optional transfer checkpoint for stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=700_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = TabletopEnv()
    writer = imageio.get_writer(
        str(out), fps=FPS, codec="libx264", quality=6, macro_block_size=1
    )

    if args.mode == "expert":
        expert = ScriptedExpert()
        for family in FAMILIES:
            expert.rng = np.random.default_rng(args.seed_base)
            seeds = [args.seed_base + i for i in range(args.episodes)]
            render_episodes(
                writer, env, expert, family, seeds,
                f"data engine - {family}",
                "scripted feedback expert, procedurally sampled instances",
            )
    else:
        from .model import ChunkedPolicyAdapter, load_checkpoint

        policy, cfg = load_checkpoint(args.ckpt)
        adapter = ChunkedPolicyAdapter(policy)
        n_params = policy.n_params()
        for family in PRETRAIN_FAMILIES:
            seeds = [args.seed_base + i for i in range(args.episodes)]
            render_episodes(
                writer, env, adapter, family, seeds,
                f"learned policy - {family}",
                f"one multi-task policy ({n_params/1e6:.1f}M params), "
                f"{cfg.get('hours', '?')} h of pretraining data",
            )
        if args.stack_ckpt:
            policy2, _ = load_checkpoint(args.stack_ckpt)
            adapter2 = ChunkedPolicyAdapter(policy2)
            seeds = [args.seed_base + i for i in range(args.episodes)]
            render_episodes(
                writer, env, adapter2, "stack", seeds,
                "transfer - stack (never pretrained)",
                "post-trained on 100 demonstrations of a withheld family",
            )

    writer.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
