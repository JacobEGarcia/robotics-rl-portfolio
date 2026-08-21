# S8 — Scaling Laws for Behaviour Cloning, at Tabletop Scale

GEN-0 (Generalist AI, 2025) reports that embodied foundation models follow
power-law scaling: next-action validation error falls predictably with
pretraining data, small models ossify while large ones keep absorbing, and
pretraining scale buys downstream post-training performance. Those claims
were measured on 270,000 hours of real-world manipulation and 10B+
parameter models.

This project asks: **does the same phenomenology appear in a manipulation
universe small enough to study end-to-end on one CPU?** Every step of the
pipeline — data engine, pretraining, withheld-task validation, post-training
transfer, closed-loop A/B evaluation — is a working miniature of that
methodology, built on this repo's shared evaluation harness (S7) and run
statistics (Wilson intervals, paired differences, bootstrap CIs).

Nothing here claims parity with a 10B-parameter model. The claim is that
the *experimental design* scales down honestly: the axes, the metrics, the
splits, and the statistics are the real ones.

## The universe

A Franka Panda on a tabletop, six task families, every episode procedurally
resampled (object sizes, masses, frictions, poses, goals — the policy never
sees the same instance twice):

| family | skill | role |
|---|---|---|
| reach  | free-space goal reaching | pretrain |
| push   | non-prehensile push to zone | pretrain |
| lift   | grasp and raise to height | pretrain |
| place  | pick and place into zone | pretrain |
| topple | knock over a pillar | pretrain |
| **stack** | **cube onto cube** | **withheld** |

`stack` never appears in pretraining. It serves two measurements: zero-shot
next-action prediction error during pretraining (GEN-0's Figure-1 metric),
and the post-training transfer target afterwards.

Control is 20 Hz Cartesian: a 5-D action (TCP displacement, wrist yaw rate,
gripper) tracked by a damped-least-squares resolved-rate controller over
Menagerie's Panda. Observations are 69-D state vectors. State-based rather
than pixel-based is a deliberate scope decision: the object of study is
scaling phenomenology, which requires ~200 training runs — a perception
stack would spend the entire compute budget answering a different question.

## The data engine

Scripted *feedback* experts (not open-loop scripts — they re-plan from live
state every tick, recover from slips, and re-grasp after failures)
demonstrate each family with injected action noise. Only successful
episodes are kept; per-family acceptance rates are printed on the data card
and reported, because a data engine's rejection rate is a fact about its
data distribution.

Collection runs 28 simulator processes and produces the corpus as
memory-mapped arrays, interleaved so cumulative hours per family stay balanced, so that *the first H
hours of the corpus is a well-defined, family-balanced dataset*. The
data-scale axis {0.5, 1, 2, 4, 8, 16, 32} hours is therefore a chain of
strict prefixes — nested datasets, not seven independent samples.

## The grid

Five MLP policy sizes (11k → 3.3M parameters, geometric) × seven data
scales × two seeds, one fixed recipe (AdamW, warmup+cosine, 30
epoch-equivalents capped at 150k steps — the cap binds only at 32 h — with
the best held-in-validation checkpoint restored).
Splits are by episode, not timestep — adjacent timesteps are nearly
identical, and a timestep-level split lets validation reward memorisation.
Normalisation statistics are computed per-run on that run's own training
subset, never on the full corpus.

Policies predict 8-step action chunks (standard in manipulation BC), and
closed-loop evaluation executes chunks receding-horizon through the S7
`SimClosedLoop` backend.

## The experiments

1. **Data scaling** — validation error on one common held-out set and
   withheld-family (stack) prediction error vs corpus hours, per model size,
   with power-law fits `L(D) = (Dc/D)^α` (and an irreducible-floor variant,
   both in `results/summary.md`).
2. **Capacity** — the same grid cut the other way: error vs parameters at
   each data scale. The ossification question: where does the smallest
   model stop converting data into loss?
3. **Transfer scaling** — pretrained checkpoints post-trained on {10, 30,
   100} stack demonstrations vs identical training from scratch, evaluated
   closed-loop on 100 paired episodes (identical instance sequences for
   every arm; seeds fixed before results were seen).
4. **Multi-task closed-loop** — success rate of each grid checkpoint on all
   five pretraining families, 50 episodes each, Wilson intervals.

## Results

All numbers below are copied from `results/summary.md`, which `analysis.py`
regenerates from the committed run logs. Figures are in `figures/`.

**Corpus.** 33.8 h of demonstrations, 74,393 episodes, 2.44 M control steps,
equal hours per family at every prefix; collected at >100× realtime on 28
processes. Expert acceptance rates: reach 100%, push 78%, lift 80%,
place 98%, topple 100% (stack, for the withheld corpora: 96%).

**Data scaling (common held-out validation set, mean of 2 seeds).**

| size | params | 0.5 h | 32 h | α (pure) | R² |
|---|---|---|---|---|---|
| t  | 11k   | 0.0230 | 0.0132 | 0.151 | 0.971 |
| s  | 31k   | 0.0203 | 0.0111 | 0.162 | 0.967 |
| m  | 160k  | 0.0168 | 0.0091 | 0.148 | 0.985 |
| l  | 582k  | 0.0168 | 0.0088 | 0.145 | 0.985 |
| xl | 3.26M | 0.0165 | 0.0089 | 0.136 | 0.983 |

One data exponent across a 290× range of model sizes. Over the last doubling
(16 → 32 h) the two smallest models gain 2–3%, the three largest 6–13%. At
≤ 8 h the best model size is 160k parameters and larger is slightly worse; at
32 h the optimum has moved to 582k.

**Zero-shot prediction error on the withheld family (stack), best over
training.** t: 0.121 → 0.091 (α = 0.08). xl: 0.081 → 0.045 (α = 0.14).
Withheld error *rises* late in training once a model has made roughly ten
passes over its data — at ~25k steps for 8 h and ~90k steps for 32 h — while
held-in validation keeps falling: specialisation is a function of epochs, not
hours, and a checkpoint selected by held-in loss is a specialised one.

**Transfer to stack, 580k policy, 100 paired episodes per cell (Wilson 95%).**

| pretraining | 0 demos | 10 demos | 30 demos | 100 demos |
|---|---|---|---|---|
| scratch | — | 0% | 5% | 11% |
| 8 h  | 0% | 10% | 24% | 36% |
| 32 h | 2% | 16% | 22% | 42% |

Paired difference (pretrained − scratch, same instances, same seed) at 100
demos: +25 pts [+18, +32] after 8 h, **+32 pts [+24, +40]** after 32 h. Every
cell of the 7 × 3 grid has a paired interval excluding zero.

For the 3.26M policy the picture is sharper: from scratch it reaches 25% on
100 demos and pretraining adds nothing significant (+4 to +8, intervals span
zero), while 0.5 h of pretraining *hurts* (−10 [−18, −2]); at 10 demos
pretraining is worth +9 to +14, and at 30 demos it lifts success from 4% to
27%. Pretraining pays where data is scarce relative to capacity.

**Multi-task closed loop, one policy, 50 episodes per family.** xl at 32 h:
reach 100%, push 74%, lift 82%, place 78%, topple 100% (mean 87%); t at
0.5 h: mean 48%.

**What this is not.** State-based observations, MLP policies, scripted
demonstrators, one CPU. The claims are about the experimental method and the
shapes it reveals at this scale, not about what a 10B-parameter model does.

## Reproduce

```bash
# 1. collect the corpora (~10 min on 32 cores; ~230x realtime)
python -m s8_scaling.data_engine --out s8_scaling/corpus --hours 33

# 2. the pretraining grid (5 sizes x 7 scales x 2 seeds; resumable)
python -m s8_scaling.sweep

# 3. closed-loop evaluation of every checkpoint
python -m s8_scaling.reeval        # score every checkpoint on the common val set
python -m s8_scaling.eval_sweep

# 4. transfer scaling on the withheld family
python -m s8_scaling.transfer --size l
python -m s8_scaling.transfer --size xl --out s8_scaling/results/transfer_xl.json

# 5. numbers, figures and montage videos
python -m s8_scaling.analysis
python -m s8_scaling.figures --dark
python -m s8_scaling.video --mode expert --out s8_scaling/media/data_engine.mp4
```

Diagnostics that found the bugs documented in the write-up live in
`tools/`: `spike.py` (env sanity, expert success, throughput),
`measure_workspace.py` (the dexterous-workspace measurement).
