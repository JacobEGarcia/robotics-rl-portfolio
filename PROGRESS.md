# Build Progress

| # | Project | Status | Evidence |
|---|---|---|---|
| **S10** | GPU simulation: fidelity, throughput, solver cost | **Built, CPU-validated; GPU numbers pending** | `s10_gpu_scaling/README.md`, `colab/S10_gpu_scaling.ipynb` |
| **S16** | S4's finger with a braid behind it | **Done**; S4 correction validated, Myofiber values are upper bounds | `media/s16/`, `s16_myofiber_finger/README.md` |
| **S15** | Switching-valve cost against a continuous one | **Done, validated** | `media/s15/`, `s15_valve/README.md` |
| **S14** | Pressure proprioception | **Done, validated** | `media/s14/`, `s14_proprioception/README.md` |
| **S13** | Flow-limited control of a hydraulic limb | **Done, validated** | `media/s13/`, `s13_flow_control/README.md` |
| **S12** | Braid vs muscle: excursion, force and pump budgets | **Done, validated** | `media/s12/`, `s12_myofiber_leg/README.md` |
| **S11** | Myofiber-class hydraulic actuator | **Done, validated** (20/20 published claims reproduced from the run log) | `media/s11/`, `s11_myofiber/README.md`, `runs/*_s11_myofiber_*` |
| **S9** | Muscle actuation (MS-Human-700), plant + six control studies | **Done, validated** (every number backed by a committed run log) + six films | `media/s9/`, `media/s9/s9_muscle.mp4`, `s9_muscle/README.md`, `runs/*_s9_*` |
| **S8** | Scaling laws for behaviour cloning | **Done, validated** (70-run grid, paired closed-loop transfer, adversarially reviewed twice) | `s8_scaling/results/`, `s8_scaling/figures/`, `runs/*_s8_bc_*` |
| **S7** | Three-way evaluation harness | **Done** (hardware backend is a deliberate stub) | `common/` |
| **S1** | PPO from scratch | **Done, validated** | `media/s1/` |
| **SAC** | SAC from scratch | **Done, validated** | `media/sac/` |
| **S4** | MuJoCo tendon modeling | **Done, validated** | `media/s4/`, `s4_tendon/README.md` |
| **S5** | Real-to-sim sysID | **Done, validated** | `media/s5/` |
| **S6** | PPO in JAX | **Done, validated** | `media/s6/`, `s6_brax_jax/README.md` |
| **S2** | In-hand reorientation | **Built, unit-validated, untrained**, GPU trainer ready (`colab/S2_inhand.ipynb`) | `s2_inhand/`, see README |
| **S3** | Domain randomization | **Complete and unit-validated, untrained**, both arms + held-out eval (`colab/S3_domain_rand.ipynb`) | `s3_domain_rand/` |

**Fourteen of seventeen complete and validated.** S2, S3 and S10 are written and unit-tested; each needs GPU time, and each ships a Colab notebook that checkpoints through a preempted session (see `colab/README.md`). Site published at `docs/` (GitHub Pages ready).

Every completed project was validated against something external: a published
baseline, an independent implementation, a physical consistency check, or
held-out data the optimiser never saw.

---

## S14 / S15 / S16: sensing, valves, and a correction to S4 (2026-08-31)

**S14 — pressure proprioception.** Closing a valve traps near-incompressible
fluid, which makes the actuator **18x to 246x** stiffer than at constant
pressure: a closed valve is a lock a joint can hold at zero power. The same
stiffness makes pressure a superb sensor over a useless range, the full 0-100
psi span covering **0.215 to 2.44 mm** of travel, under 8% of stroke, at
sub-micron resolution. Pose is fully observable from the pressure vector (rank
5 of 5, condition number 6.35 against the moment-arm matrix's 3.82), and the
open-valve alternative of integrating flow drifts **85% of stroke in 60 s** at
1% flow error.

**S15 — switching valves.** The actuator's own fill time constant is 0.354 ms,
so PWM only averages above about 3 kHz. Below it every cycle fully charges and
discharges: ripple is **full scale** at 50 Hz through 1 kHz, and collapses by
**2565x** across the transition. A cliff, not a 1/f rolloff. A 0.1 ms minimum
on-time, fast for a solenoid, costs **440x** against an ideal valve, because
raising the carrier to fix ripple directly worsens duty resolution. The cheap
fix, adding hydraulic compliance, directly undoes S14's lock.

**S16 — S4's finger with a braid.** The ideal-actuator arm reproduces S4's
table to three decimals (0.2066 / 0.4048 / 0.5822 / 0.6798). Then it finds that
**S4's numbers are not converged in the settling cap**: the zero-friction width
falls 0.207 → 0.117 → 0.055 → 0.032 rad at 4, 10, 30 and 80 s, still falling.
S4's qualitative claim survives; the absolute table and the attribution of
0.207 rad to snap-through bistability do not. Separately, the braid cuts
hysteresis **49-85%** against an ideal force source, because a compliant
actuator cannot hold its command through a snap.

---

## S12 / S13: braided actuators in a body, and the pump that feeds them (2026-08-31)

**S12 — can a braid replace a human muscle?** A braided sleeve's maximum
contraction has a closed-form ceiling of `1 - 1/sqrt(3) = 42.26%` as the braid
angle goes to zero, verified against a 4000-point sweep. MS-Human-700's leg
muscles need a median excursion of 37.3%, and **17 of 50 exceed the ceiling**,
so no braid at any braid angle can match them. They are the glutei, adductors,
sartorius, gracilis and psoas: the long-throw hip muscles. The fix is a
stroke-amplifying transmission (median 1.12, max 1.72), paid for linearly in
fluid volume.

Sizing target matters more than any of it. Peak isometric force for one leg is
50,137 N; the force actually used in 277 stance problems is 18,684 N, **37.3%**.
That is 5.69 L against 2.19 L of working fluid, and 5,111 published fibres
against 1,905, for the same limb doing the same job.

**S13 — flow-limited control.** Tracking the S9 stance cycle with those
actuators demands a mean 25.5 L/min and a peak of 60.0, so the published
40 L/min pump is flow-limited **26.6% of every cycle**. Four allocation
policies, scored on joint torque error: **equal-share water-filling wins at
0.139, 3.0x better than uniform scaling at 0.417**.

Two results that were not the hypothesis. The informed policy (`by_torque`,
weighting by how much each actuator moves the tracked joints) *loses*, at
0.176: at the worst instant equal-share fully satisfies 23 of 27 requests and
concentrates the shortfall on four, while uniform serves none of them fully and
puts everyone at 66.7%. Torque error is a norm, so many small correct
contributions beat everything being slightly wrong. And `by_demand` produces
allocations *identical to uniform* at every timestep, because serving in
proportion to request size is uniform scaling restated: it sounds like a policy
and is not one.

16. **A constraint that silently stopped existing.** S13's first version
    tracked only the bladder compliance volume and missed the braid's geometric
    volume, four orders of magnitude larger. Mean demand read 0.003 L/min,
    nothing was rationed, and all four policies tied at exactly zero. Every
    check passed, because a constraint that never binds is trivially respected.

---

## S11: Myofiber-class hydraulic actuator (2026-08-31)

S9 established that MuJoCo's Hill muscle is the wrong model for a Myofiber. This
builds the right class of model and sizes it from Clone's published figures.
Standalone: no MuJoCo, no Menagerie assets.

Sizing is an inference, not a fit. The published ">30% unloaded contraction"
alone constrains the braid angle to under 34.4 degrees via the closed form
`eps_max = 1 - 1/(sqrt(3) cos(theta0))`; the published "3 g fibre, at least
1 kg" then sets the radius. Every spec comparison is an inequality in the
direction the spec states.

| measurement | result |
|---|---|
| blocking angle | 54.736 deg, matching the classic 54.7356 to 1e-3 |
| force at 100 psi | 43.3 N from 1.25 g of water, against a published "at least 9.81 N" |
| **flow budget** | the published 40 L/min pump drives **34 of ~1000 muscles** at full stroke and full speed, **3.4% of the body** |
| **valve requirement** | the published 50 ms response needs a **2.90 mm** orifice, **5.5x** the naive mean-head sizing, because that estimate ignores the pressure transient |
| fill/vent asymmetry | **exactly 1.000** with symmetric plumbing, 0.864 with 15 psi return back-pressure. A Hill muscle's 4x is baked in; this has to be plumbed in |
| antagonist stiffness | **zero sign changes at any pressure**, against S9-C's 1-2 per joint and ~40% of each range negative. Proportional to pressure within 2% |

Two silent bugs, continuing the count:

14. **A bisection bracket that forbade extension.** A braid pulled past rest
    length pulls harder. Capping the bracket at rest length pinned the
    stretched antagonist at the bound and gave a flat stiffness curve at
    exactly `k_series * r^2`, the answer when the solver has stopped solving.
15. **A pinned solve returns a number, not an error.** One low-pressure point
    landed on the bound with a -11 N residual against a nominal 1e-8 and set
    the reported maximum for its whole pressure level. `solve_length` now
    returns `converged`; statistics exclude and count failures.

---

## S9: Muscle actuation on MS-Human-700 (2026-08-31)

700 Hill-type muscle actuators over 85 DoF, all pull-only, all tendon-routed.
The closest publicly available stand-in for Myofiber-style actuation: not a
model of Protoclone, but it shares the three properties that break a
torque-trained policy.

| measurement | result |
|---|---|
| moment-arm matrix rank | 65 of 85 DoF, identical at all 24 sampled poses |
| activation null space | 635-dimensional (co-contraction is a free channel) |
| unactuated directions | 20, of which the 6 pelvis DoF carry weight exactly 1.0 |
| base force from random activation | **0.0 N exactly** (Newton's third law, not a fit) |
| joint torque across one joint's range at constant full command | 52.0 to 69.3 N·m, **1.33x** |
| simulator vs `mju_muscleGain`/`Bias` closed form | 0.0 N max abs error |
| active element peak | normalised length 0.999 (Hill model defines 1.0) |
| activation rise / fall (10-90%, 90-10%) | 31.8 ms / 91.9 ms, ratio 2.89 |

**The finding: the shipped 2 ms timestep misintegrates activation by 0.150, or
15% of full scale.** Activation is advanced by explicit Euler and the fastest
time constant in the model is 5 ms. Error against RK4 falls 0.150 to 0.0038
from 2 ms to 0.1 ms. At 2 ms the first step jumps 0 to 0.4, so a 10-90% rise
time is not merely inaccurate, it is unbracketed.

Two silent bugs, continuing the count:

11. **A NaN that was telling the truth.** The first crossing detector returned
    NaN for rise time; "fixing" it to fall back on the first sample produced
    25.9 ms, a plausible number sitting neatly beside the converged 31.8 ms and
    entirely an artifact of step size. The NaN was correct output.
12. **Actuator index is not tendon index.** Both are length 700 in this model,
    so colouring tendons by `act[i]` compiles, runs, and renders the wrong
    muscles. The orderings are a non-trivial permutation (actuator 0 drives
    tendon 11); go through `actuator_trnid`. Found because a render highlighted
    a strip of shin while the quadriceps it had activated stayed blue.

13. **Equality-coupled joints do not follow their drivers.** 42 `mjEQ_JOINT`
    couplings encode the knee's rolling contact and the shoulder's rhythm.
    `mj_forward` computes constraint forces but does not project positions, so
    writing a driver joint leaves its dependents stale: `knee_angle_r` at
    0.9 rad alone leaves a 0.60 violation. Caught only because an
    inverse-dynamics round trip blew up. It had already produced a published
    number, a "26.6x torque swing" that is 1.33x once projected.

Related to #9 above: MS-Human-700 has 25 `<include>` elements nested two deep,
and loading it by a relative path containing directories prepends the model's
directory prefix twice. Absolute paths only.

### Six control studies (S9-A through S9-F)

| study | headline |
|---|---|
| A synergies | 5 synergies reach 90% activation VAF (matching the 4-6 from human EMG) but only 70% of the torque; shuffled null never reaches 90% |
| B redundancy | peak activation falls and total rises monotonically with the effort exponent; muscle count is *not* monotone, peaking at p = 3 |
| C impedance | co-contraction stiffness changes sign over ~40% of every joint's range; a single-pose literature comparison can give any answer |
| D latency | closed-form activation inverse cuts tracking error 418-504x below a 3.5 Hz saturation wall predicted analytically and confirmed exactly |
| E transfer | a torque-model command loses 10.6% of required torque at 1 Hz walking cadence, 2.0% compensated; plant matches MuJoCo to 1.4e-14 N·m |
| F failure | capability outlasts torque authority to ~25 dead muscles then falls faster; `soleus_r` alone is 11.6% of authority |

---

## S8, Scaling laws for behaviour cloning (2026-08-21)

A GEN-0-style scaling study on a Panda tabletop with six procedurally varied
task families (stack withheld), 33.8 h of expert demonstrations, 5 model
sizes × 7 data scales × 2 seeds, one recipe, all on CPU.

| measurement | result |
|---|---|
| data-scaling curves, common val set | near-straight in log-log, α 0.13 to 0.18, R² 0.96 to 0.99 (seven points per curve; no "law" claimed) |
| 8→16 h doubling (uniform recipe) | every size gains 8.7 to 11.9%, **no ossification threshold found** |
| capacity | 0.5 h: xl 15% ahead of 160k; 2 to 16 h: nothing past 160k (larger 1 to 9% worse); 32 h: three largest within 3.4% |
| withheld (stack) zero-shot error at 8 epochs | t −33%, xl −40% over 64× data |
| specialisation (xl, seed 0) | withheld error minimum at 7.5 (8 h) / 5.4 (32 h) epochs, sustained rise from 14 / 10, while held-in val trends down |
| transfer, 580k, 100 demos | scratch 11% → 42% at 32 h, paired **+32 pts [+24, +40]**; 20/21 cells p<0.05, 18 Bonferroni |
| transfer, 3.26M, 100 demos | 25% scratch; pretraining −6 to +8, none significant; helps at 10/30 demos |
| multi-task closed loop, xl @ 32 h, seed 0 | reach 100 / push 74 / lift 82 / place 78 / topple 100 % |

Six silent bugs found (three by an adversarial review run *before* the sweep):
stale `geom_rbound` on resized objects, friction masked by MuJoCo's
max-combination rule, parked objects keeping stale sizes, a validation set that
changed with the data scale, bootstrap CIs of zero width at 0%/100%, and a
step floor that doubled the 0.5 h cells' epochs (it had made a "pretraining
hurts" claim look significant; rerun uniformly, it is not).

## S5, Real-to-sim system identification

Fitted MuJoCo's Franka Panda to **632 s of real DROID robot data** (32 episodes,
9,475 frames at 15 Hz), open-loop, with CMA-ES.

| | RMSE (rad) |
|---|---|
| Menagerie defaults, held-out | 0.0431 |
| After sysID, held-out | **0.0187** |
| **Improvement** | **56.7%** |

Split is **by episode**, never by frame, frames within an episode are heavily
correlated and a frame-level split leaks.

| Parameter | Default | Identified | Ratio |
|---|---|---|---|
| damping | 1.000 | 5.027 | 5.03× |
| armature | 0.100 | 0.467 | 4.67× |
| frictionloss | 0.000 | 3.071 | from zero |
| latency | 0 | 3.30 frames | 220 ms |
| kp | 3142.9 | 3218.5 | 1.02× |

**Why the fit is credible:** four parameters moved substantially, one did not.
Controller gain shifted 2%, Menagerie already had it right and the optimiser
left it alone. An optimiser merely minimising error would have moved all five.

**Dataset finding:** `lerobot/nyu_franka_play_dataset` is unusable, its
`action` column is exactly `state[t+1] − state[t]` (correlation 1.0000), and its
EEF pose is pure forward kinematics of the joint angles (FK error 0.000 m). It
has no reality gap at all. Now guarded by `verify_dataset_has_gap()`.

## S6, PPO in JAX

| | PyTorch | JAX |
|---|---|---|
| throughput | 1,934 steps/s | **54,808 steps/s** |
| final return | −168.8 | −197.6 |
| solved | yes | yes |

**28.3× on the same CPU**, no GPU. Isolates program shape (compiled scan over
512 vectorised envs) from hardware.

The port **reproduced the PyTorch reward-scaling bug exactly**, same flat
~−1200, in a different framework. That independent reproduction is evidence the
two implementations agree. A second bug was JAX-specific: autoreset inside the
scan makes `values[t+1]` belong to a different trajectory on truncated steps.

## S1 / SAC / S4

See the site or per-project READMEs. Headlines: PPO −1196 → −168.8;
SAC 9.41× PPO sample efficiency; tendon hysteresis 0.207 → 0.680 rad with
snap-through bistability identified at zero friction.

---

## Seven silent bugs found

None raised an exception. All were found by validation rather than by testing.

1. **Missing reward scaling** (PyTorch PPO), value loss monopolised the clipped gradient
2. **Missing reward scaling** (JAX PPO), same bug, independently reproduced
3. **Tendon sign convention**, routing and gear must agree; wrong combination crept to a joint limit
4. **Sinusoid not quasi-static**, measured dynamic phase lag, ran backwards against friction
5. **Wrap geoms colliding**, ~18% phantom hysteresis; found only because MJX failed loudly
6. **Truncation bootstrap under autoreset**, `values[t+1]` from a different episode
7. **Dataset with no reality gap**, would have produced a perfect, meaningless sysID result

---

## What is blocked

**GPU, S2 and S3.** This machine is `torch 2.13.0+cpu`, no CUDA. S2 needs
~10⁸ environment steps. S6 now makes that tractable *given* a GPU, MJX is
verified working on this machine, so the path is open. Needs the Hugging Face
`jobs` scope.

Not an authoring problem. Compute only.


---

## S2 / S3, built 2026-08-18, not yet trained

Both projects now exist as code with their correctness properties tested. What
is missing in both cases is a GPU, not authoring.

**S2** (`s2_inhand/`): LEAP-hand cube reorientation on MJX. Validated: grasp
stability (7.4 mm drift over 5 s, 1.3 mm max penetration), grasp pose found by
36-point grid search, palm direction measured rather than assumed, quaternion
metric correct including the q = -q fold, curriculum sampling respects its cap,
and the model compiles and steps under MJX. **Not validated: that a policy
learns the task.**

**S3** (`s3_domain_rand/`): domain randomization with a held-out set that is
disjoint by construction and defined before training. Verified disjoint on all
five randomized parameters, with 50.7% of held-out draws below the training
band so both failure directions are probed.

### Three more silent failures found

Continuing the count from the seven above; same character, none raised.

8. **Initial penetration is a cliff.** A cube starting >=10 mm inside a finger
   geom is ejected at tens of m/s when the solver resolves the overlap, drifting
   20 m in 2 s. Adjacent grid points looked fine. Low drift alone is not a valid
   acceptance test; penetration at reset must be checked separately.
9. **`meshdir` resolves against the top-level file, not the file it appears
   in.** Including Menagerie's `right_hand.xml` from another directory silently
   looks for `s2_inhand/assets/` and fails to load.
10. **Held-out sampling that only draws above the training band** tests one
    direction and reports half a result. Fixed by drawing from both sides and
    asserting the ~50/50 split.

### Measured, and worth stating plainly

`mjx.step` on this model compiles in ~151 s and then runs ~4 s per step
unbatched on CPU. That is not a throughput number, it is the reason S2 cannot
run here. Single-env MJX being slower than MuJoCo CPU is expected and is the
honest caveat from S6 restated.

LEAP's collision geometry turned out to be 68 boxes against only 4 meshes,
which is far friendlier to MJX than the 91-geom count suggests. The model
rewrite I expected to need was not necessary.
