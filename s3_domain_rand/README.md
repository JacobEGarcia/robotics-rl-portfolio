# S3 — Domain Randomization and Zero-Shot Transfer

**Status: randomization machinery built and validated. Blocked on S2, which is
blocked on a GPU.**

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
arm proves nothing — it is equally consistent with DR helping, doing nothing,
or hurting.

**3. Action latency is randomized.** It is the parameter most often omitted and
the one that most reliably breaks real deployments. The range is not a guess:
[S5](../s5_sysid/) measured **3.30 frames** of latency on real Franka data.

## Randomized parameters

All multiplicative on the nominal model value, so ranges stay meaningful if the
model is ever updated.

| Parameter | Train | Held-out |
|---|---|---|
| cube mass | 0.70 – 1.30× | 0.50 – 0.70 and 1.30 – 1.60× |
| cube friction | 0.70 – 1.40× | 0.50 – 0.70 and 1.40 – 1.80× |
| actuator gain (kp) | 0.80 – 1.25× | 0.65 – 0.80 and 1.25 – 1.50× |
| joint damping | 0.60 – 1.60× | 0.40 – 0.60 and 1.60 – 2.20× |
| action latency | 0 – 4 frames | 5 – 7 frames |

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

## Deliverable when unblocked

The two-curve plot: no-DR and DR arms, both evaluated on the held-out set,
plotted against training steps. Same rigour as S5 — the label is sim-to-sim,
stated plainly, and it demonstrates methodology rather than claiming a
real-robot result.
