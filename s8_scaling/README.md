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
statistics (bootstrap CIs, IQM).

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
memory-mapped arrays, interleaved family-round-robin so that *the first H
hours of the corpus is a well-defined, family-balanced dataset*. The
data-scale axis {0.5, 1, 2, 4, 8, 16, 32} hours is therefore a chain of
strict prefixes — nested datasets, not seven independent samples.

## The grid

Five MLP policy sizes (11k → 4.4M parameters, geometric) × seven data
scales × two seeds, one fixed recipe (AdamW, warmup+cosine, 30
epoch-equivalents capped at 60k steps, early stopping on held-in val loss).
Splits are by episode, not timestep — adjacent timesteps are nearly
identical, and a timestep-level split lets validation reward memorisation.
Normalisation statistics are computed per-run on that run's own training
subset, never on the full corpus.

Policies predict 8-step action chunks (standard in manipulation BC), and
closed-loop evaluation executes chunks receding-horizon through the S7
`SimClosedLoop` backend.

## The experiments

1. **Data scaling** — held-in validation error and withheld-family (stack)
   prediction error vs corpus hours, per model size, with power-law fits
   `L(D) = (Dc/D)^α` (and an irreducible-floor variant, both reported).
2. **Capacity** — the same grid cut the other way: error vs parameters at
   each data scale. The ossification question: where does the smallest
   model stop converting data into loss?
3. **Transfer scaling** — pretrained checkpoints post-trained on {10, 30,
   100} stack demonstrations vs identical training from scratch, evaluated
   closed-loop on 100 paired episodes (identical instance sequences for
   every arm; seeds fixed before results were seen).
4. **Multi-task closed-loop** — success rate of each grid checkpoint on all
   five pretraining families, 50 episodes each, bootstrap CIs.

## Results

*(filled in from `s8_scaling/results/` and `runs/` — see the write-up at
the portfolio site for the full figures)*

## Reproduce

```bash
# 1. collect the corpora (~10 min on 32 cores; ~230x realtime)
python -m s8_scaling.data_engine --out s8_scaling/corpus --hours 33

# 2. the pretraining grid (5 sizes x 7 scales x 2 seeds; resumable)
python -m s8_scaling.sweep

# 3. closed-loop evaluation of every checkpoint
python -m s8_scaling.eval_sweep

# 4. transfer scaling on the withheld family
python -m s8_scaling.transfer --size l

# 5. figures and montage videos
python -m s8_scaling.figures --dark
python -m s8_scaling.video --mode expert --out s8_scaling/media/data_engine.mp4
```

Diagnostics that found the bugs documented in the write-up live in
`tools/`: `spike.py` (env sanity, expert success, throughput),
`measure_workspace.py` (the dexterous-workspace measurement).
