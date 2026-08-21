# Robotics RL &amp; Sim-to-Real

Scaling laws for behaviour cloning measured end-to-end on a CPU, reinforcement
learning implemented from scratch, tendon actuator modeling in MuJoCo, and
real-to-sim system identification measured against logged data from a physical
Franka Panda.

**[Read the write-up →](https://jacobegarcia.github.io/robotics-rl-portfolio/)**

Everything here runs on a laptop CPU. No GPU is required for any published
result.

---

## Results

| Project | Result |
|---|---|
| **[S8 — Scaling laws for BC](s8_scaling/)** | 70-run size×data grid on 34 h of procedural manipulation data: near-power-law curves (α 0.13–0.18, R² 0.96–0.99), **+32 pt** paired transfer gain on a withheld task (20/21 cells significant), specialisation after ~10 epochs — and no ossification threshold found. All on CPU |
| **[S5 — Real-to-sim sysID](s5_sysid/)** | **56.7%** reality-gap reduction on held-out real robot episodes |
| **[S6 — PPO in JAX](s6_brax_jax/)** | **28.3×** throughput on the same CPU; both implementations solve the task |
| **[SAC from scratch](sac_scratch/)** | **9.41×** PPO's sample efficiency, identical task and seed |
| **[S1 — PPO from scratch](s1_ppo/)** | Pendulum solved: −1196 → −168.8, past the −200 threshold |
| **[S4 — Tendon modeling](s4_tendon/)** | Friction hysteresis characterised; snap-through bistability identified |
| **[S7 — Evaluation harness](common/)** | Open-loop, closed-loop, and hardware behind one interface |

Two projects remain (in-hand reorientation and domain randomization). Both need
a GPU — roughly 10⁸ environment steps — and are compute-blocked, not
authoring-blocked.

---

## The argument

Every result is validated against something **external**: a published baseline,
an independent implementation, a physical consistency check, or held-out data
the optimiser never saw. A learning curve that rises proves only that a number
went up.

That discipline found **thirteen bugs, none of which raised an exception**:

1. **Missing reward scaling** (PyTorch PPO) — value loss of ~10⁶ against a
   policy loss of ~10⁻² meant `clip_grad_norm_` sent nearly the entire gradient
   to the critic. The policy was starved, not broken.
2. **Missing reward scaling** (JAX PPO) — the same bug, reproduced
   independently in a different framework. That reproduction is itself evidence
   the two implementations agree.
3. **Tendon sign convention** — routing sets `sign(J)`, gear sets force sign,
   and the *product* decides which way the joint moves. Two attempts each got
   one half right; the finger crept to the wrong joint limit while compiling
   cleanly.
4. **A 0.25 Hz sinusoid is not quasi-static** — it measured dynamic phase lag,
   reporting hysteresis that *decreased* as friction rose. Exactly backwards.
5. **Wrap geoms in collision detection** — ~18% phantom hysteresis. Found only
   because MJX refuses cylinder-box collisions and failed loudly where plain
   MuJoCo was silent.
6. **Truncation bootstrap under autoreset** — in a vectorised JAX rollout,
   `values[t+1]` on a truncated step belongs to a different trajectory.
7. **A dataset with no reality gap** — `lerobot/nyu_franka_play_dataset` has
   `action == state[t+1] − state[t]` exactly (correlation 1.0000). Fitting a
   simulator to it would have produced a perfect, meaningless result.
8. **A resized geom with a stale bounding radius** (S8) — writing `geom_size`
   at reset leaves `geom_rbound`/`geom_aabb` compiled for the old size; tall
   pillars settled a centimetre inside the table with flickering contact.
9. **Friction randomisation that was never applied** (S8) — MuJoCo combines
   friction by elementwise max, so a 0.8 table and 1.0 fingertip pads masked
   most of the sampled range. Objects now carry contact `priority`.
10. **Parked objects remembering the previous episode** (S8) — unused objects
    kept the previous family's sizes, so physics depended on episode order.
11. **A validation set that moved with the data scale** (S8) — each grid cell
    validated on its own prefix's held-out episodes; error appeared to rise
    from 16 h to 32 h. Re-scored on one common set, every curve is monotone.
12. **Confidence intervals of zero width** (S8) — percentile bootstrap on
    success flags collapses at 0% and 100%. Replaced with Wilson intervals.
13. **A step floor that doubled one cell's epochs** (S8) — "30 epochs, at
    least 8k steps" gave the 0.5 h cells 60 epochs, at the one scale where
    pretraining looked harmful. Rerun uniformly, the effect is not significant.

---

## Setup

```bash
pip install "gymnasium[mujoco]" torch mujoco jax optax \
            scipy matplotlib cma pandas pyarrow imageio

bash scripts/fetch_assets.sh   # Menagerie + 32 DROID episodes
```

## Run

```bash
python -m s8_scaling.data_engine --hours 33     # ~16 min on 32 cores
python -m s8_scaling.sweep                      # 70-run grid
python -m s8_scaling.reeval && python -m s8_scaling.transfer && python -m s8_scaling.eval_sweep
python -m s8_scaling.analysis && python -m s8_scaling.figures --dark
python -m s1_ppo.train       --env Pendulum-v1 --total-steps 150000
python -m sac_scratch.train  --env Pendulum-v1 --total-steps 30000
python -m s4_tendon.hysteresis
python -m s5_sysid.fit       --iterations 45
python -m s6_brax_jax.train  --total-steps 8000000 --num-envs 512
```

Each command writes a timestamped directory under `runs/` containing a JSONL
log whose first line is a manifest: seed, config, git SHA, library versions,
and a hash of the *compiled* MuJoCo model. Those logs are committed — they are
what the published numbers are computed from.

---

## Layout

```
common/          evaluation harness, logging, metrics, plotting, recording
s8_scaling/      behaviour-cloning scaling study: data engine, grid, transfer
s1_ppo/          PPO in PyTorch
sac_scratch/     SAC in PyTorch
s4_tendon/       tendon-driven finger model and friction characterisation
s5_sysid/        real-to-sim system identification against DROID
s6_brax_jax/     PPO in JAX, MJX verified
runs/            JSONL logs and checkpoints backing every published number
docs/            the site
```

## Data and models

Real robot data from [DROID](https://droid-dataset.github.io/)
(`cadene/droid_1.0.1`). Robot models from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Neither is vendored here; `scripts/fetch_assets.sh` retrieves both.

---

Jacob E. Garcia · B.S. Cybersecurity, DePaul University
