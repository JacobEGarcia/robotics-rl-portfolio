"""
Post-hoc re-evaluation of every sweep checkpoint on ONE common held-out set.

During training each cell validates on the held-out episodes of *its own*
prefix, so a 0.5 h cell and a 32 h cell are scored on different validation
sets. The hash split makes every val episode held out from every cell
(a val episode is never in any train prefix), so the union of val episodes
across the whole corpus is a legitimate common test set for all 70 cells.
This script scores each saved checkpoint on that common set, and also
pulls the minimum withheld-family loss seen over training from the log.

Metrics as pure functions of what was saved: no retraining, fully
reproducible, and the original in-training numbers are left untouched in
result.json - this writes `common_eval.json` beside it.

Run:  python -m s8_scaling.reeval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.runlog import read_log  # noqa: E402

from .dataset import load_split  # noqa: E402
from .model import load_checkpoint  # noqa: E402
from .train import EVAL_CAP, evaluate  # noqa: E402


def main() -> None:
    torch.set_num_threads(8)
    corpus = Path("s8_scaling/corpus")
    common_val = load_split(corpus / "pretrain", hours=None, val=True)
    withheld = load_split(corpus / "stack_val", val=False, val_fraction=0.0)
    val_idx = np.random.default_rng(0).choice(
        len(common_val), size=min(len(common_val), EVAL_CAP), replace=False
    )
    wh_idx = np.random.default_rng(1).choice(
        len(withheld), size=min(len(withheld), EVAL_CAP), replace=False
    )
    print(f"common val: {len(common_val)} timesteps ({len(val_idx)} scored); withheld {len(wh_idx)}")

    for run_dir in sorted(Path("runs").glob("*_s8_bc_*")):
        ckpt = run_dir / "policy.pt"
        if not ckpt.exists():
            continue
        policy, cfg = load_checkpoint(ckpt)
        policy.eval()
        v = evaluate(policy, common_val, val_idx)
        w = evaluate(policy, withheld, wh_idx)
        manifest, events = read_log(run_dir)
        evs = [e for e in events if e.get("type") == "eval"]
        wh_curve = [e["withheld_loss"] for e in evs]
        # Withheld loss at a FIXED epoch count (8 epochs, the nearest logged
        # evaluation). A selection rule that never looks at the withheld
        # family: taking the minimum over training would be an oracle
        # choice on the test family whose bias grows with the number of
        # evaluations, i.e. with data scale. Found in review.
        train_samples = manifest["config"]["train_samples"]
        target_step = 8 * train_samples / 256
        nearest = min(evs, key=lambda e: abs(e["step"] - target_step)) if evs else None
        out = {
            "val_common": v,
            "withheld_at_checkpoint": w,
            "withheld_at_8ep": float(nearest["withheld_loss"]) if nearest else float("nan"),
            "withheld_at_8ep_step": int(nearest["step"]) if nearest else -1,
            "withheld_min": float(min(wh_curve)) if wh_curve else float("nan"),
            "withheld_min_step": int(evs[int(np.argmin(wh_curve))]["step"]) if wh_curve else -1,
            "withheld_final": float(wh_curve[-1]) if wh_curve else float("nan"),
            "epochs_trained": float(evs[-1]["step"] * 256 / train_samples) if evs else float("nan"),
            "common_val_timesteps": int(len(val_idx)),
        }
        (run_dir / "common_eval.json").write_text(json.dumps(out, indent=2))
        print(
            f"{cfg['size']:>2}/{cfg['hours']:<4}h val_common {v:.4f}  "
            f"withheld ckpt {w:.4f}  @8ep {out['withheld_at_8ep']:.4f}  min {out['withheld_min']:.4f}"
        )


if __name__ == "__main__":
    main()
