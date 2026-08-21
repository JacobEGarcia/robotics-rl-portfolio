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

Five MLP policy sizes (11k → 3.3M parameters, roughly log-spaced) × seven
data scales × two seeds, one fixed recipe (AdamW, warmup+cosine, 30
epoch-equivalents capped at 150k steps — the cap binds only at 32 h, at 17.6
epochs — with the best held-in-validation checkpoint restored).
Splits are by episode, not timestep — adjacent timesteps are nearly
identical, and a timestep-level split lets validation reward memorisation.
Normalisation statistics are computed per-run on that run's own training
subset, never on the full corpus.

Policies predict 8-step action chunks (standard in manipulation BC), and
closed-loop evaluation executes each full chunk and then re-plans from live
state (every 8 control steps, 0.4 s) through the S7 `SimClosedLoop` backend.

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
   closed-loop on 100 paired instances (identical instance sequences for
   every arm; seeds fixed before results were seen), with seed-averaged
   paired differences, exact McNemar tests and a Bonferroni threshold.
4. **Multi-task closed-loop** — success rate of each seed-0 grid checkpoint
   on all five pretraining families, 50 episodes each, Wilson intervals.

## Results

All numbers below are copied from `results/summary.md`, which `analysis.py`
regenerates from the committed run logs and results files. Figures are in
`figures/`.

**Corpus.** 33.8 h of demonstrations, 74,393 episodes, 2.44 M control steps,
equal hours per family at every prefix; collected in 944 s at 129× realtime
on 28 processes. Expert acceptance rates: reach 100%, push 78%, lift 80%,
place 98%, topple 100% (stack, for the withheld corpora: 96%).

**Data scaling (common held-out validation set, mean of 2 seeds).**

| size | params | 0.5 h | 32 h | α (pure) | R² | 8→16 h gain | 16→32 h gain |
|---|---|---|---|---|---|---|---|
| t  | 11k   | 0.0268 | 0.0132 | 0.175 | 0.968 | 9.3% | 2.4% |
| s  | 31k   | 0.0235 | 0.0111 | 0.184 | 0.957 | 8.7% | 2.5% |
| m  | 160k  | 0.0182 | 0.0091 | 0.160 | 0.962 | 9.0% | 6.1% |
| l  | 582k  | 0.0166 | 0.0088 | 0.142 | 0.987 | 9.7% | 11.9% |
| xl | 3.26M | 0.0158 | 0.0089 | 0.128 | 0.979 | 11.9% | 12.6% |

Curves are near-straight in log-log with exponents 0.13–0.18 (standard errors
0.007–0.018); seven points per curve do not establish a power law or a shared
exponent, and the write-up does not claim either. Over the 8→16 h doubling,
where every cell trains the identical recipe, all sizes gain 8.7–11.9%. The
small models' gains shrink only in the 16→32 h doubling, which is the one
where the 150k-step cap binds (17.6 epochs) — so **no ossification threshold
is claimed**. Capacity: xl is clearly best at 0.5 h (15% ahead of the 160k
model, 5% ahead of 582k) and nominally best at 1 h; from 2 to 16 h the 160k
model is best and the larger two are 1–9% worse; at 32 h the three largest
are within 3.4%, a gap xl's own seed spread (3.5%) covers.

**Zero-shot prediction error on the withheld family (stack), at 8 epochs** —
a selection rule fixed without looking at the withheld family (the minimum
over training is an oracle choice whose bias grows with data scale; it is in
`summary.md` for transparency only). t: 0.143 → 0.095 (−33%); xl: 0.082 →
0.049 (−40%); R² 0.80–0.88. Over training (seed 0), xl's withheld error
bottoms out at 16k steps (7.5 epochs) with 8 h of data and at 46k steps (5.4
epochs) with 32 h, begins a sustained rise at ~30k and ~87k steps (14 and 10
epochs), and ends at 0.122 and 0.077 while held-in validation keeps trending
down: 4× the data bought 2.9× the steps before specialisation, within a
factor of 1.4 in epochs (seed 1: 2.3× and 1.7×).

**Transfer to stack, 580k policy.** Success is the mean of 2 seeds (Wilson
95% at n = 100 instances; both seeds see the same 100 instances). Paired
differences are seed-averaged per instance, bootstrapped over the 100
instances, with an exact McNemar test on discordant pairs.

| pretraining | 0 demos | 10 demos | 30 demos | 100 demos | paired Δ @ 100 |
|---|---|---|---|---|---|
| scratch | — | 0% | 5% | 11% | — |
| 0.5 h | 0% | 2% | 11% | 22% | +10 [+3, +18] |
| 8 h  | 2% | 10% | 23% | 34% | +22 [+14, +30] |
| 32 h | 2% | 16% | 22% | 42% | **+32 [+24, +40]** |

20 of 21 paired comparisons are significant at p < 0.05; 18 survive
Bonferroni (p < 0.0024). At 100 demos the gain rises with every doubling of
pretraining (+10 → +32); at 10 and 30 demos it rises broadly but not
monotonically.

**Transfer, 3.26M policy.** From scratch it reaches 25% on 100 demos, and
pretraining does not significantly change that at any scale (paired −6 to
+8, every interval spans zero). At 10 and 30 demos pretraining is worth +2 to
+14 and +5 to +24 points; 12 of 21 comparisons significant, 9 after
Bonferroni. An earlier draft reported 0.5 h of pretraining as significantly
harmful; that cell had trained 60 epochs under a step floor, and after the
uniform rerun the effect is −6 [−14, +2], p = 0.13.

**Multi-task closed loop, seed-0 checkpoints, 50 episodes per family**
(Wilson half-width ≈ 12 pts). xl at 32 h: reach 100%, push 74%, lift 82%,
place 78%, topple 100% (mean 87%); t at 0.5 h: mean 38%.

**What this is not.** State-based observations, MLP policies, scripted
demonstrators, two seeds, one CPU. The claims are about the experimental
method and the shapes it reveals at this scale — and the two it does not:
an ossification threshold, and any return to capacity past 160k parameters
between 2 and 16 h of data.

## Reproduce

```bash
# 1. collect the corpora (~16 min on 32 cores; ~130x realtime)
python -m s8_scaling.data_engine --out s8_scaling/corpus --hours 33

# 2. the pretraining grid (5 sizes x 7 scales x 2 seeds; resumable)
python -m s8_scaling.sweep

# 3. score every checkpoint on the common val set (open loop), then closed loop
python -m s8_scaling.reeval
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
