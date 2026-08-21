"""Reproduce one transfer cell (pretrained -> post-trained on stack) as a saved
checkpoint, for the policy montage. Same recipe and seed as transfer.py.

Run:  python -m s8_scaling.tools.make_transfer_ckpt --size xl --hours 32 --budget 100
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..dataset import load_split
from ..model import load_checkpoint, save_checkpoint
from ..transfer import BATCH, FT_LR, FT_STEPS, find_pretrained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="xl")
    ap.add_argument("--hours", type=float, default=32.0)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="s8_scaling/media/stack_transfer_policy.pt")
    args = ap.parse_args()

    run_dir = find_pretrained(Path("runs"), args.size)[(args.hours, args.seed)]
    policy, cfg = load_checkpoint(run_dir / "policy.pt")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    demos = load_split(
        Path("s8_scaling/corpus/stack_demos"), max_episodes=args.budget, val=False, val_fraction=0.0
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=FT_LR, weight_decay=1e-4)
    for step in range(1, FT_STEPS + 1):
        lr = 1e-5 + 0.5 * (FT_LR - 1e-5) * (1 + math.cos(math.pi * step / FT_STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = demos.batch(rng, BATCH)
        loss = torch.mean((policy(torch.from_numpy(x)) - torch.from_numpy(y)) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    cfg = dict(cfg, transfer_budget=args.budget, transfer_from_hours=args.hours)
    save_checkpoint(args.out, policy, cfg)
    print(f"wrote {args.out} (pretrained {args.size}/{args.hours}h -> {args.budget} stack demos)")


if __name__ == "__main__":
    main()
