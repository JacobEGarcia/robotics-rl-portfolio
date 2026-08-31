"""
Which term does MJX's `implicitfast` disagree with MuJoCo C about?

    python -m s10_gpu_scaling.tools.which_term --model franka_emika_panda

`run.py integrator` establishes that the disagreement is in `implicitfast` and
not in precision, contacts, or the constraint solver. It does not say which
*part* of implicitfast, and "MJX's implicit integrator differs" is a weaker
claim than the evidence can support.

`implicitfast` is Euler plus an implicit treatment of the velocity-dependent
forces: it solves with `M - dt * dF/dv` instead of `M`. Two terms feed `dF/dv`
in these models, joint damping (`dof_damping`) and the velocity gain of a
position servo (`actuator_biasprm[:, 2]`, the `kv` in `kp*(q*-q) - kv*qdot`).
If MJX and MuJoCo C build that Jacobian differently, the disagreement must
scale with those coefficients and must vanish when they are zero.

So: zero them, one at a time, and re-measure. This is a knockout, not a
correlation. The models that disagree are the ones with large coefficients
(the Panda: `dof_damping` 1.0, `kv` up to 450) and the models that agree are
the ones with small ones (`dynamixel_2r`: `dof_damping` 0, `kv` up to 5.5),
which is suggestive and is not evidence until the term is actually removed.

Zeroing damping changes the physics, obviously. That is fine and it is the
point: both sides simulate the same altered physics, and the question is only
whether they still agree with each other.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from s10_gpu_scaling import models, parity

RESULTS = Path(__file__).resolve().parent.parent / "results"


def variants(model: mujoco.MjModel) -> dict[str, mujoco.MjModel]:
    """The model as shipped, and with each velocity-dependent term removed."""
    out: dict[str, mujoco.MjModel] = {}

    out["as_shipped"] = copy.deepcopy(model)

    m = copy.deepcopy(model)
    m.dof_damping[:] = 0.0
    out["no_joint_damping"] = m

    m = copy.deepcopy(model)
    # biasprm column 2 is kv for an affine bias (biastype=1). Zeroing it turns
    # the position servo into a pure spring with no velocity feedback.
    m.actuator_biasprm[:, 2] = 0.0
    out["no_actuator_kv"] = m

    m = copy.deepcopy(model)
    m.dof_damping[:] = 0.0
    m.actuator_biasprm[:, 2] = 0.0
    out["neither"] = m

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="franka_emika_panda")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="which_term.json")
    args = ap.parse_args()

    spec = next(s for s in models.available() if s.key == args.model)
    base = models.load(spec)
    base.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    qpos0, qvel0 = parity.initial_state(base)
    ctrl = parity.make_controls(base, args.steps, seed=args.seed)

    print(f"{args.model}, implicitfast, {args.steps} steps")
    print(f"{'variant':<20}{'mjx vs mjc':>14}{'float32 floor':>16}{'ratio':>12}")

    rows = []
    for name, m in variants(base).items():
        coarse = parity.rollout_mjc(m, qpos0, qvel0, ctrl, substeps=1)
        floor = parity.roundoff_envelope(m, qpos0, qvel0, ctrl, coarse)
        mx = parity.rollout_mjx(m, qpos0, qvel0, ctrl)
        d = parity.divergence(coarse, mx)
        row = {
            "variant": name,
            "mjx_vs_mjc_m": float(d[-1]),
            "float32_floor_m": float(floor[-1]),
            "ratio_to_floor": float(d[-1] / max(floor[-1], 1e-30)),
            "max_dof_damping": float(np.max(m.dof_damping)),
            "max_actuator_kv": float(np.max(np.abs(m.actuator_biasprm[:, 2]))),
            "unstable": coarse.extra.get("unstable", False),
        }
        rows.append(row)
        print(f"{name:<20}{row['mjx_vs_mjc_m']:>14.3e}"
              f"{row['float32_floor_m']:>16.3e}{row['ratio_to_floor']:>12.5f}",
              flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    doc = json.loads((RESULTS / args.out).read_text(encoding="utf-8")) \
        if (RESULTS / args.out).exists() else {}
    doc[args.model] = {"steps": args.steps, "seed": args.seed, "rows": rows,
                       "model": models.describe(base)}
    (RESULTS / args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", RESULTS / args.out)

    # Report single-term knockouts only. The "neither" row is the both-removed
    # control and always collapses if either single term does, so printing it
    # as a third finding reads like three independent causes.
    shipped = rows[0]["mjx_vs_mjc_m"]
    by = {r["variant"]: r for r in rows}
    culprits = []
    for name in ("no_joint_damping", "no_actuator_kv"):
        r = by.get(name)
        if r and r["mjx_vs_mjc_m"] < shipped / 100:
            culprits.append((name.replace("no_", ""),
                             shipped / max(r["mjx_vs_mjc_m"], 1e-30)))

    print()
    if not culprits:
        print("=> neither term accounts for the disagreement. It is somewhere "
              "else in the implicit step.")
    elif len(culprits) == 1:
        term, factor = culprits[0]
        exonerated = "actuator_kv" if term == "joint_damping" else "joint_damping"
        print(f"=> removing '{term}' collapses the disagreement by {factor:,.0f}x, "
              f"while removing '{exonerated}' does not. That term is where the "
              f"two implementations differ.")
    else:
        print("=> both terms individually collapse it, so they are not "
              "separable by this test alone.")


if __name__ == "__main__":
    main()
