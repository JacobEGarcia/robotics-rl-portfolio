# S3: Domain Randomization and Zero-Shot Transfer

**Status: complete and unit-validated on CPU, randomization, per-environment
batched physics, both training arms, and the held-out evaluation. Untrained:
needs GPU sessions (`colab/S3_domain_rand.ipynb`).**

The honest label for this project is **sim-to-sim transfer under domain
randomization**. There is no real LEAP hand here. Calling it sim-to-real would
be a lie. The methodology is the point, and it is testable exactly as stated:
*does randomizing training parameters improve performance on parameter values
the policy never saw?*

## The three decisions that make the result mean anything

**1. The held-out set is defined before training, in code, and is disjoint by
construction.** Training samples the interior of each range; evaluation samples
bands strictly outside it. Deciding the split after seeing results is how
people accidentally report a training number and believe it.

**2. Both arms get trained.** A no-DR baseline and a DR arm, identical in every
other respect, both evaluated on the same held-out set. Reporting only the DR
arm proves nothing, it is equally consistent with DR helping, doing nothing,
or hurting.

**3. Action latency is randomized.** It is the parameter most often omitted and
the one that most reliably breaks real deployments. The range is not a guess:
[S5](../s5_sysid/) measured **3.30 frames** of latency on real Franka data.

## Randomized parameters

All multiplicative on the nominal model value, so ranges stay meaningful if the
model is ever updated.

| Parameter | Train | Held-out |
|---|---|---|
| cube mass | 0.70 to 1.30× | 0.50 to 0.70 and 1.30 to 1.60× |
| cube friction | 0.70 to 1.40× | 0.50 to 0.70 and 1.40 to 1.80× |
| actuator gain (kp) | 0.80 to 1.25× | 0.65 to 0.80 and 1.25 to 1.50× |
| joint damping | 0.60 to 1.60× | 0.40 to 0.60 and 1.60 to 2.20× |
| action latency | 0 to 4 frames | 5 to 7 frames |

Plus additive observation noise: 0.01 rad on joint angles, 2 mm on cube
position, 0.01 on the quaternion before renormalisation.

## Verified

```
param            train range      train samples      test samples       DISJOINT?
cube_mass       [0.70,1.30]      [0.700,1.300]      [0.500,1.600]      YES
cube_friction   [0.70,1.40]      [0.700,1.400]      [0.500,1.799]      YES
actuator_gain   [0.80,1.25]      [0.800,1.250]      [0.650,1.500]      YES
joint_damping   [0.60,1.60]      [0.600,1.600]      [0.400,2.199]      YES
latency         (0, 4)           train=[0,1,2,3,4]  test=[5,6,7]       YES

held-out draws below the training band: 50.7%   (both sides probed, not just the easy one)
target/rel/prev-action left clean: True
noised quaternion is unit: 1.000000
latency buffer, delay 0..4 -> [4.0, 3.0, 2.0, 1.0, 0.0]
```

That 50.7% matters. A held-out sampler that only draws from *above* the
training range tests one failure direction and quietly reports half a result.

## Details that are easy to get wrong

**Only sensed quantities get noise.** The target orientation is commanded, not
measured. The relative quaternion is derived. The previous action is known
exactly. Corrupting those models nothing physical and just makes the task
harder for no reason.

**The noised quaternion is renormalised.** Adding Gaussian noise to a unit
quaternion takes it off the unit sphere, and an unnormalised quaternion is not
a rotation.

**`apply_to_model` takes nominal values explicitly** rather than reading the
current model. Randomizing an already-randomized model compounds the
multipliers, the distribution silently drifts wider every episode, and the
held-out set stops being held out.

**Latency uses a fixed-size buffer and a gather**, not a Python deque, so it
stays jittable and its shape does not depend on the sampled latency.

## Per-environment physics on a GPU, without recompiling

This is the piece that makes DR real under MJX, and there are two ways to get
it wrong.

*The wrong way that is slow:* rebuild the MJX model between episodes.
`mjx.put_model` is a host-side conversion and every call invalidates the
compiled step function, so the run spends its life in XLA rather than in
physics.

*The wrong way that is silently not randomized:* randomize once, share the
model across the whole batch, re-draw every few thousand steps. Every
environment then sees the same physics at the same moment, so the batch carries
one sample of the distribution instead of `n_envs` of them and the gradient's
variance over physics parameters is not reduced at all. It trains, it looks like
DR, and its held-out number comes out barely different from the baseline.

The right way is one compiled step function over a *batched model*: the
randomized fields carry a leading environment axis, everything else is shared,
and `jax.vmap` is told which is which through `in_axes`.

**The trap inside the right way.** `jax.vmap` adds a leading axis to every leaf
of whatever its function returns, so vmapping a function that returns an MJX
model replicates mesh vertices, convex hulls and body trees once per
environment. Measured on this scene: the model is 45 KB shared, and the first
implementation batched *all* of it. At the batch sizes MJX is worth using at,
that is an out-of-memory error whose message says nothing about domain
randomization. `randomize.randomized_fields` exists so the vmap covers only the
four arrays; `env.batched_model` then folds them into one shared model and
asserts the shapes. Verified: 4 leaves batched, 1.3× memory at `n_envs = 8`.

## The curriculum decision

S2 widens its target-angle range when the policy starts landing the current
one. That is right for S2 and wrong here. Under an adaptive curriculum the DR
arm, solving a harder problem, advances more slowly, so at any given step the
two arms are training on **different task distributions**, and the final
held-out comparison silently mixes "DR generalises better" with "one arm spent
longer on easy targets".

`curriculum_at(step, total)` is a fixed function of progress through the run.
It costs some sample efficiency and buys the only thing that matters here: at
every step, both arms face the same task. It is also deterministic across a
resume, which an adaptive rule would have to reconstruct from a checkpoint and
would get subtly wrong.

Physics is resampled **per episode**, not per step. A policy cannot adapt to a
mass that changes underneath it mid-episode, and per-step randomization trains
it to ignore object properties rather than to infer them. Per episode is also
what a deployment looks like: one robot, one set of parameters, one attempt.

## The evaluation

`python -m s3_domain_rand.eval --dr ... --baseline ... --episodes 512`

**Three conditions, not two:** nominal physics, in-distribution physics,
held-out physics. Without the first two, a DR arm that is simply worse
everywhere and a DR arm that trades in-distribution performance for robustness
look identical, and that trade is the interesting result.

**Both arms see identical held-out draws**, from `EVAL_SEED`, which is fixed in
the source and unrelated to either training seed. Sampling per arm would put an
independent-sample difference on top of the effect, and at a few hundred
episodes that difference is the same size as the effect.

**The headline is the paired difference**, with Wilson intervals on the
marginals. A success rate of 0/100 has a bootstrap interval of exactly zero
width, an artefact of resampling a constant, found in S8 and fixed in
`common/metrics.py`. The arms are scored on the same episodes, and pairing
removes the episode-difficulty variance that dominates the unpaired numbers.

**Success is "reached the tolerance at any point in the episode."** Scoring only
the final frame measures how well the hand holds still afterwards, which is a
different and easier-to-game quantity.

An interval that straddles zero is a null result, and will be reported as one.

## Bugs caught while building this, continuing the portfolio's count

16. **`done` is not `dropped`.** `done = dropped OR timeout`, and every episode
    times out on its last step, so the first evaluator reported a 100% drop
    rate on runs where nothing was ever dropped. Drop is now computed from the
    cube's displacement.
17. **`hash(kind)` as a seed.** Python randomises string hashes per process
    unless `PYTHONHASHSEED` is pinned, so the held-out set would have been
    different on every invocation, irreproducible in a way that leaves no
    trace in the output. Fixed offsets now.
18. **The whole-model vmap** described above.
19. **Resuming the other arm's checkpoint.** Both arms share a network shape, so
    loading the wrong one succeeds silently and produces a "baseline" that was
    pretrained with randomization. The trainer now refuses on the `arm` field
    recorded in the checkpoint.
