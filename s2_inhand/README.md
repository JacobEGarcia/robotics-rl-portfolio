# S2: In-Hand Cube Reorientation (LEAP hand, MJX)

**Status: built and unit-validated on CPU. Not yet trained, needs a GPU.**

A 5 cm cube starts held in a partially closed LEAP right hand. A target
orientation is sampled each episode. The policy commands 16 finger joints and
must rotate the cube to within 0.1 rad of the target without dropping it.

Observation is privileged state only, joint positions and velocities, cube
pose, target orientation, previous action. No vision.

## Why this is the hard one

Roughly 10⁸ environment steps. On CPU MuJoCo with a handful of environments
that is weeks of wall clock. MJX runs thousands of copies in parallel on one
GPU, which is what makes it hours. **[S6](../s6_brax_jax/) is not a side
quest, it is the prerequisite that makes this runnable at all.**

## What is validated, and what is not

Being precise about this matters more than the code.

| Claim | Evidence |
|---|---|
| Grasp is stable | 7.4 mm drift over 5 s of settling, 1.3 mm max penetration at reset |
| Grasp pose was searched, not guessed | 36-point grid over flexion × placement, `tools/find_grasp.py` |
| Palm direction measured, not assumed | fingertips move along (-0.896, 0, 0.444) when curled |
| Quaternion metric is correct | `angle(q,q)=0`, `angle(I, 180°x)=π`, **`angle(q,−q)=0`** |
| Target sampling respects the curriculum cap | 200 draws, max 0.3997 against a 0.4 cap |
| Model runs in MJX | `put_model` 0.3 s, `mjx.step` compiles and runs |
| **Policy learns the task** | **Not established. No GPU.** |

The last row is the whole project and it is not done. Nothing here claims a
trained policy.

## Findings worth keeping

**Initial penetration is a cliff, not a gradient.** Configurations starting the
cube ≥10 mm inside a finger geom do not settle badly, they eject it at tens of
metres per second on the first step as the solver resolves the overlap. Several
grid points blew up this way, reaching 20 m of drift in 2 s. Any change to cube
size, flexion, or placement must be re-checked for penetration at reset.

**LEAP's collision geometry is mostly boxes.** 68 box geoms against 4 meshes, much friendlier to MJX than the 89-geom count suggests. Mesh collision was the
thing I expected to force a model rewrite, and it did not.

**MuJoCo resolves `meshdir` relative to the top-level file, not the file the
directive appears in.** Including Menagerie's `right_hand.xml` from a different
directory silently looks for `s2_inhand/assets/` and fails. Fixed with a
`<compiler>` override placed *after* the include so the later directive wins.

**The `q ≡ −q` fold is not optional.** Without `abs()` on the quaternion dot
product the reward is discontinuous at the antipode, and the symptom is a
policy that refuses to rotate past 180°. Cheap to get right up front,
expensive to diagnose later.

**Single-env MJX on CPU is slower than MuJoCo CPU**, as expected. `mjx.step`
here compiles in ~151 s and then runs ~4 s per step unbatched. That number is
not a throughput measurement, it is a statement that this box cannot run the
project.

## Reward

```
r = -1.0 · θ                      geodesic angle to target
    + 5.0 · 1[θ < 0.1]            success
    - 20.0 · 1[dropped]           drop
    - 0.001 · |a|²                effort
```

Dense, because sparse is unlearnable here, a random policy's chance of hitting
a target orientation is effectively zero, so there is no gradient.

Action is a **delta** on the current commanded position, not an absolute
target. Absolute targets let the policy teleport fingers between steps, which
trains fine in simulation and means nothing on hardware.

## Curriculum

`target_angle_max` starts at 0.4 rad and widens by 0.1 whenever success passes
50%, up to π. Without it the reward is nearly flat, from 180° away every
action looks equally bad. It never narrows: oscillating the task distribution
makes the value function chase a moving target.

## Running it

```bash
python -m s2_inhand.train --smoke                    # correctness only, CPU
python -m s2_inhand.train --total-steps 100000000 --num-envs 2048
python -m s2_inhand.train --resume                   # after a session dies
```

On Kaggle, use [`kaggle_run.py`](kaggle_run.py).

## Checkpoint-resume is load-bearing

Kaggle's free tier caps a session near 12 hours and then destroys the machine.
The trainer is written to be killed at any moment. It saves:

- **params**: obvious
- **opt_state**: Adam's moments. Dropping them restarts the optimiser cold and
  produces a loss spike on every resume, easily misread as instability
- **step**: so the curriculum continues rather than restarting
- **key**: the PRNG key. Without it a resumed run re-draws target orientations
  it already trained on, and the run stops being reproducible from its seed
- **curriculum**: current target-angle range

Checkpoints are written to a temp file and renamed, so a session killed
mid-write cannot leave a truncated file for the next session to resume from.

## What to watch on the first real run

The **curriculum value**, not the reward. If it is not widening after a few
million steps, the reward is not shaping behaviour and the fix is the reward,
not more steps. That is the expected failure mode for this project, the spec
calls S2 the one project where "written" and "working" are far apart, and
reward shaping is why.

## The known throughput cost, deliberately not yet paid down

The rollout loop in `train.py` transfers seven arrays from device to host on
every one of its 16 rollout steps, and separately reads `state.done` back as a
Python float to decide whether to reset the batch. On a GPU each of those is a
blocking synchronisation: the device pipeline drains, refills, and drains
again, roughly 128 times per iteration.

Batching the transfers to one per iteration is easy and numerically identical.
Removing the per-step `done` read is not: the reset decision is data-dependent,
and moving it to the iteration boundary changes when a dropped cube is
recovered. Doing both is most of the way to a scanned rollout.

**This has deliberately not been changed, because the size of the win is
unknown and this is the one project in the portfolio that has never trained.**
An unverifiable performance change to a first-ever training run is a bad trade.
The measurement to make first, once a GPU session exists:

```bash
python -m s10_gpu_scaling.run throughput --models s2_inhand
```

That reports the steps/s a scanned `jax.lax.scan` rollout achieves on this
exact scene with no host round trips at all. Comparing it against the steps/s
the trainer actually prints gives the size of the prize directly. If the
trainer is within ~20% of it, the syncs are not the bottleneck and the loop
should be left alone; if it is at half, the rewrite is worth the risk and can
be validated against a fixed seed for bit-identical results before adoption.
