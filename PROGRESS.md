# Build Progress

| # | Project | Status | Evidence |
|---|---|---|---|
| **S7** | Three-way evaluation harness | **Done** (hardware backend is a deliberate stub) | `common/` |
| **S1** | PPO from scratch | **Done, validated** | `media/s1/` |
| **SAC** | SAC from scratch | **Done, validated** | `media/sac/` |
| **S4** | MuJoCo tendon modeling | **Done, validated** | `media/s4/`, `s4_tendon/README.md` |
| **S5** | Real-to-sim sysID | **Done, validated** | `media/s5/` |
| **S6** | PPO in JAX | **Done, validated** | `media/s6/`, `s6_brax_jax/README.md` |
| **S2** | In-hand reorientation | **Built, unit-validated, untrained** — needs GPU | `s2_inhand/`, see README |
| **S3** | Domain randomization | **Randomization built and validated**; eval blocked on S2 | `s3_domain_rand/` |

**Six of eight complete and validated. S2 and S3 are now written and unit-tested but untrained.** Site published at `docs/` (GitHub Pages ready).

Every completed project was validated against something external — a published
baseline, an independent implementation, a physical consistency check, or
held-out data the optimiser never saw.

---

## S5 — Real-to-sim system identification

Fitted MuJoCo's Franka Panda to **632 s of real DROID robot data** (32 episodes,
9,475 frames at 15 Hz), open-loop, with CMA-ES.

| | RMSE (rad) |
|---|---|
| Menagerie defaults, held-out | 0.0431 |
| After sysID, held-out | **0.0187** |
| **Improvement** | **56.7%** |

Split is **by episode**, never by frame — frames within an episode are heavily
correlated and a frame-level split leaks.

| Parameter | Default | Identified | Ratio |
|---|---|---|---|
| damping | 1.000 | 5.027 | 5.03× |
| armature | 0.100 | 0.467 | 4.67× |
| frictionloss | 0.000 | 3.071 | from zero |
| latency | 0 | 3.30 frames | 220 ms |
| kp | 3142.9 | 3218.5 | 1.02× |

**Why the fit is credible:** four parameters moved substantially, one did not.
Controller gain shifted 2% — Menagerie already had it right and the optimiser
left it alone. An optimiser merely minimising error would have moved all five.

**Dataset finding:** `lerobot/nyu_franka_play_dataset` is unusable — its
`action` column is exactly `state[t+1] − state[t]` (correlation 1.0000), and its
EEF pose is pure forward kinematics of the joint angles (FK error 0.000 m). It
has no reality gap at all. Now guarded by `verify_dataset_has_gap()`.

## S6 — PPO in JAX

| | PyTorch | JAX |
|---|---|---|
| throughput | 1,934 steps/s | **54,808 steps/s** |
| final return | −168.8 | −197.6 |
| solved | yes | yes |

**28.3× on the same CPU**, no GPU. Isolates program shape (compiled scan over
512 vectorised envs) from hardware.

The port **reproduced the PyTorch reward-scaling bug exactly** — same flat
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

1. **Missing reward scaling** (PyTorch PPO) — value loss monopolised the clipped gradient
2. **Missing reward scaling** (JAX PPO) — same bug, independently reproduced
3. **Tendon sign convention** — routing and gear must agree; wrong combination crept to a joint limit
4. **Sinusoid not quasi-static** — measured dynamic phase lag, ran backwards against friction
5. **Wrap geoms colliding** — ~18% phantom hysteresis; found only because MJX failed loudly
6. **Truncation bootstrap under autoreset** — `values[t+1]` from a different episode
7. **Dataset with no reality gap** — would have produced a perfect, meaningless sysID result

---

## What is blocked

**GPU — S2 and S3.** This machine is `torch 2.13.0+cpu`, no CUDA. S2 needs
~10⁸ environment steps. S6 now makes that tractable *given* a GPU — MJX is
verified working on this machine, so the path is open. Needs the Hugging Face
`jobs` scope.

Not an authoring problem. Compute only.


---

## S2 / S3 — built 2026-08-18, not yet trained

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
