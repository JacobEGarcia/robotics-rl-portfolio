# S10: What a GPU buys a physics simulator, and what it costs

**Status: built and CPU-validated. Throughput and solver cost need a GPU
session (`colab/S10_gpu_scaling.ipynb`). The fidelity result below is complete
on CPU and needs GPU confirmation, not GPU discovery; every table says what it
was measured on.**

Three questions a simulation engineer gets asked and usually cannot answer with
a measurement:

1. **Does the GPU simulator agree with the CPU one?**
2. **How many steps per second, against what baseline, and where does it stop
   scaling?**
3. **What does the solver iteration budget actually cost and buy, on the GPU,
   where the trade is not the one you tuned on CPU?**

No training. The whole thing fits in one free Colab session.

---

## 1. Agreement, and why the obvious version of this measurement is worthless

Step MuJoCo C and MJX from the same state with the same controls, plot the
difference, watch it grow, conclude MJX is inaccurate. Every pair of
implementations produces that plot. A robot in contact is chaotic: nearby
trajectories separate exponentially, so *any* perturbation, down to the last
bit of a float, ends up macroscopic. The plot measures chaos, not correctness.

What makes it answerable is a control with no implementation difference in it, the same binary, at the same settings, started a rounding error apart.

### The two floors, and why using the wrong one inverts the answer

**Floor A, `chaos_envelope`.** Perturb the initial state by one float32 epsilon,
integrate in full float64. Answers "how much does this system amplify a single
rounding error".

**Floor B, `roundoff_envelope`.** Same binary, float64 arithmetic, but the state
carried across each step boundary is truncated to float32 and back. A *lower
bound* on what float32 costs, not a simulation of it.

They are not close:

| model | floor A (one ulp) | floor B (float32 state) | ratio |
|---|---|---|---|
| `dynamixel_2r` | 2.48e-08 m | 8.04e-04 m | 32,000× |
| `franka_emika_panda` | 1.56e-07 m | 3.72e-03 m | 24,000× |
| `unitree_z1` | 9.24e-08 m | 3.86e-03 m | 42,000× |

Floor A is the wrong floor, because **float32 does not make one rounding error.
It makes one on every operation, of every step, forever.** On a system whose
amplification is mild, a single initial perturbation stays small while
continuously re-injected round-off random-walks upward.

This is not hypothetical. The first version of this module had only floor A. It
measured the Panda's MJX float32 divergence at 1.6e-2 m against a floor A of
1.6e-7 m and was about to report a **100,000× discrepancy**, a serious
accusation against MJX. Against floor B the same measurement is **4.3×**, the
same order of magnitude, and an entirely ordinary result. Floor A was wrong
because it answered a question nobody had asked.

Floor B is necessary and it is not sufficient, which is what `mjx_f64` is for:
on this same Panda, running MJX in float64 changed the divergence by nothing at
all (see the results below). So the 4.3× is not precision either. Two controls
disagreed with each other and the tiebreak is the one that removes the
suspected cause entirely rather than bounding it.

### Reading the result

| curve | baseline | what it isolates |
|---|---|---|
| `envelope` (floor A) | MuJoCo C at `dt` | the system's amplification |
| `roundoff` (floor B) | MuJoCo C at `dt` | what float32 storage costs |
| `mjx_f32` | MuJoCo C at `dt` | what S2 actually trains in |
| `mjx_f64` | MuJoCo C at `dt` | the same, precision removed |
| `mjc_dt` | MuJoCo C at `dt/32` | what the production timestep already cost |

`mjx_f32` at or below `roundoff`: precision explains it, MJX is as right as
float32 permits. `mjx_f32` above `roundoff` with `mjx_f64` near zero: still
precision, float32 *arithmetic* is worse than float32 *storage*, and floor B
bounds the second, not the first. Only `mjx_f64` staying large is evidence of an
algorithmic difference, and then the place to look is the constraint solver and
the contact set.

`mjc_dt` is the honest denominator for all of it. If the production timestep
already perturbs the trajectory more than the port does, no fidelity objection
to the GPU survives that.

**The two groups have different baselines and are not comparable to each
other.** `curves_vs_mjc` and `curves_vs_ref` are kept in separate keys in the
results JSON for exactly this reason. Comparing everything against one very fine
reference *seems* tidier and silently caps the resolution of question 1 at the
reference's own residual discretisation error, measured here at 1e-5 m against
a floor A of 1e-9 m, four orders of magnitude the fine reference cannot see
into. Every MJX number would have come back sitting on the reference's error
floor, looking like agreement. `reference_check` reports that floor with every
result so the claim stays falsifiable.

### Preliminary results, read the provenance column

| model | measurement | value | measured on |
|---|---|---|---|
| `dynamixel_2r` | `mjx_f64` vs MuJoCo C, 30 steps | **0.000e+00 m, bit-exact** | CPU |
| `dynamixel_2r` | `mjx_f32` vs MuJoCo C | 1.10e-08 m | CPU |
| `dynamixel_2r` | floor B, same window | 8.04e-04 m | CPU |
| `franka_emika_panda` | `mjx_f32` vs MuJoCo C, 20 steps | 1.610e-02 m | CPU |
| `franka_emika_panda` | **`mjx_f64` vs MuJoCo C, same window** | **1.610e-02 m** | CPU |
| `franka_emika_panda` | floor B, same window | 3.72e-03 m | CPU |
| `franka_emika_panda` | `mjc_dt` vs refined reference | 1.78e-03 m | CPU |

**The Panda's two MJX rows are the same number.** Giving MJX float64 changed
the divergence from MuJoCo C by nothing measurable, on a trajectory with
`ncon = 0` from start to finish. Precision excluded; the contact solver and the
collision pipeline excluded, because there are no contacts. MJX also reports
the same integrator, the same Newton solver and the same iteration counts as
the CPU model, so the flags are not being silently substituted.

### It is the integrator, and only one of them

Re-running the identical comparison under every integrator the model could have
been written with (`run.py integrator`, 40 steps, CPU, `results/integrator.json`):

| model | integrator | MJX vs MuJoCo C | ratio to float32 floor |
|---|---|---|---|
| Panda | `mjINT_EULER` | 1.197e-07 m | 0.00002× |
| Panda | **`mjINT_IMPLICITFAST`** (shipped) | **4.989e-02 m** | **7.28×** |
| Panda | `mjINT_RK4` | 2.017e-07 m | 0.006× |
| Panda | `mjINT_IMPLICIT` | **`NotImplementedError`** | |
| `dynamixel_2r` | `mjINT_EULER` (shipped) | 1.394e-08 m | 0.00001× |
| `dynamixel_2r` | `mjINT_IMPLICITFAST` | 1.969e-08 m | 0.00002× |
| `dynamixel_2r` | `mjINT_RK4` | 7.829e-08 m | 0.402× |
| `dynamixel_2r` | `mjINT_IMPLICIT` | **`NotImplementedError`** | |

On the Panda, **two independent explicit integrators agree with MuJoCo C to
about 1e-07 m**, five orders of magnitude below what float32 storage alone
costs, and the one implicit integrator MJX supports is 50 mm off within 40 ms.
Same model, same state, same controls, same solver, one flag different.

**Those agreeing rows are the positive control this module needed.** They prove
the harness can show agreement, so every large number elsewhere is a result
rather than a bug in the comparison. A fidelity study with no configuration
that comes out clean has not established that it can tell the difference. Same
trap as p47 in the sim-lab: "no violations anywhere" is what a working check
and a dead check both produce.

### But "MJX's implicitfast is wrong" is not what the data says

`implicitfast` agrees to 2e-08 m on `dynamixel_2r`. So it is not generically
broken, and the disagreement is a property of the *model*, not of the
integrator alone.

`implicitfast` is Euler plus an implicit treatment of the velocity-dependent
forces: it solves against `M - dt * dF/dv` rather than `M`. Two terms feed that
Jacobian in these models, joint damping and the velocity gain of a position
servo. They differ by roughly two orders of magnitude between the two:

| | `dof_damping` | actuator `kv` |
|---|---|---|
| `dynamixel_2r` (agrees) | 0.0 | 2.9 to 5.5 |
| `franka_emika_panda` (disagrees) | 1.0 | 10 to 450 |

That is a hypothesis with an obvious knockout, which is what
`tools/which_term.py` runs: zero each term and re-measure. Correlation across
two models is not evidence; removing the term is.

### It is the actuator velocity gain

Panda, `implicitfast`, 40 steps, CPU:

| variant | MJX vs MuJoCo C | ratio to float32 floor |
|---|---|---|
| as shipped | 4.989e-02 m | 7.28× |
| joint damping zeroed | 5.400e-02 m | 7.58× |
| **actuator `kv` zeroed** | **2.177e-07 m** | **0.00003×** |
| both zeroed | 3.027e-07 m | 0.00004× |

Zeroing joint damping changes nothing. Zeroing the position servo's velocity
gain collapses the disagreement by **229,000×**, to five orders of magnitude
below the float32 floor. Both terms feed the same implicit Jacobian and only
one of them matters, which is about as clean as a knockout gets.

So the chain is closed, and every link was tested rather than assumed:

| candidate | ruled out by |
|---|---|
| float32 precision | `mjx_f64` gives the identical number |
| contacts, collision, constraint solver | `ncon = 0` for the whole trajectory |
| silently substituted flags | MJX reports the same integrator, solver, iteration counts |
| integration in general | Euler and RK4 both agree to ~1e-07 m on the same model |
| `implicitfast` in general | it agrees to 2e-08 m on `dynamixel_2r` |
| joint damping | zeroing it changes nothing (5.400e-02 vs 4.989e-02) |
| **actuator velocity gain** | **zeroing it collapses the gap 229,000×** |

What this does *not* say is which side is right, or what MJX does internally.
That would need reading MJX's implicit-step source, and the measurement does
not license it. The claim is exactly: **MJX and MuJoCo C treat a position
servo's velocity gain differently inside `implicitfast`, and the disagreement
scales with that gain.**

### Why it matters regardless of which term it is

Of the nine models in the set, `unitree_z1`, `franka_emika_panda`, `leap_hand`
and **`s2_inhand` itself** ship `implicitfast`; `dynamixel_2r`, `shadow_hand`,
`unitree_go1` and `unitree_h1` ship Euler. The S2 scene is an `implicitfast`
model with stiff position servos, which is the combination that disagrees.

The S2 scene is an `implicitfast` model whose hand is driven by position
servos, which is exactly the combination that disagrees. So a policy trained in
MJX on it is trained against physics that measurably differs from the same
scene stepped in MuJoCo C. Anyone evaluating an MJX-trained policy on the CPU,
which is the normal deployment path, crosses that gap without being told, and
the size of the gap is set by a gain they chose for controller reasons and
never thought of as a fidelity knob.

There is a practical consequence for S2 beyond the write-up: the size of this
disagreement is proportional to actuator `kv`, so it is worth measuring on the
LEAP hand before reading too much into any MuJoCo-C evaluation of an
MJX-trained policy.

The `NotImplementedError` on `mjINT_IMPLICIT` stands on its own: MJX does not
implement the full implicit integrator at all. The sweep records a refusal as
`unsupported` rather than skipping it, because "MJX cannot run this" is a
harder fact than any divergence number.

**Still to check on GPU:** whether the gap is the same size on the CUDA
backend, and how it behaves once contacts are present, which neither of these
two models exercises.

Two things to be careful about before quoting any of this:

- **These are CPU-MJX numbers.** MJX compiles through XLA for whatever backend
  it is on, and float32 reduction order and fused multiply-add differ between
  the CPU and GPU backends. The GPU numbers may not match. The notebook runs
  the same code on the T4 and that is what should be quoted.
- **`dynamixel_2r` has `ncon = 0` over the whole trajectory**, checked, not
  assumed. Its bit-exact float64 agreement is a statement about smooth dynamics
  with the constraint solver idle. The Panda run is also contact-free
  (`ncon = 0`, checked). The contact-rich models, `leap_hand`, `s2_inhand`,
  `unitree_go1`, are where the interesting answer is, and they are the ones
  that need the GPU.

---

## 2. Throughput, with a denominator

The number usually quoted for MJX is a single figure, "two million steps per
second", on an unstated model at an unstated batch size against nothing. Three
things have to be measured together:

**The CPU baseline on the same model.** `mujoco.rollout` threads across cores
and releases the GIL, and on a small model with few contacts it is genuinely
competitive. Measured here on this machine: 1,189,000 steps/s on `dynamixel_2r`
across 32 threads. A GPU quoting two million against *that* is a 1.7× speedup,
not a 100× one.

**Compile time, on its own line.** MJX pays a fixed XLA cost before the first
step, and on a contact-rich model it is minutes. Folded into a short run's
throughput it disappears; a simulation engineer sizing a job needs it separately,
because it decides whether an experiment that takes 90 seconds of compute is
worth running at all. It also scales with batch size, every batch size in the
sweep recompiles.

**Where it stops scaling, and why.** Throughput rises with batch size, flattens,
then the allocation fails. `knee()` reports the *smallest* batch within 10% of
the best observed, and that is the number to run at, not the largest that fits.
Past the knee each extra environment buys under 10% while costing proportional
memory and a longer compile, and on a preemptible free-tier session a longer
compile is a direct loss.

Method notes that change the answer:

- The timed region is a `jax.lax.scan` inside one `jit`. Timing a Python loop
  over `step()` measures dispatch overhead, which on small models is most of the
  wall clock and has nothing to do with the simulator.
- `block_until_ready` once, at the end. JAX dispatch is asynchronous; without it
  the measurement is how fast Python can enqueue work.
- The batch is jittered by 0.01 rad at reset. Identical copies are a best case:
  every environment takes the same branch and resolves the same contact set, so
  warp divergence and constraint count are both unrealistically uniform.
- An OOM is recorded as a data point, not raised. It is the right edge of the
  curve. The sweep stops after **two** consecutive failures, not one, a single
  failure can come from fragmentation left by the previous allocation rather
  than from a real capacity limit, and treating that as the ceiling understates
  the hardware.

---

## 3. The solver iteration budget

Timestep gets all the attention and is the wrong first knob for a contact-rich
model on a GPU. Halving the timestep doubles the work exactly. Halving the
solver iteration budget roughly halves the constraint work, and what it costs in
accuracy depends entirely on whether the solver was converging inside that
budget, on a hand holding a cube it usually is, several iterations before the
default cuts it off.

**The GPU changes the trade.** Under `vmap` the batched solver runs a fixed
iteration count for the whole batch; an environment whose constraints are
already satisfied cannot exit early the way it does on CPU. So the budget is
paid in full by every environment on every step, and a value tuned on CPU is
usually the wrong value on GPU. This is the most common surprise when moving an
environment from MuJoCo to MJX.

Corroborating evidence already in Menagerie, which the sweep surfaces:
`franka_emika_panda/panda.xml` ships `iterations=100`, and DeepMind's own MJX
demo scene in the same directory, `mjx_single_cube.xml`, ships **`iterations=5`
and a 5 ms timestep**. They cut it by 20× for the GPU port and did not write
down why.

Accuracy here is trajectory divergence from the tight reference, **not**
penetration depth. Penetration is tempting, one number, obvious right answer of
zero, and wrong: MuJoCo's contacts are soft by design and the steady-state
penetration is set by `solref`. A solver that drives it to zero is not more
accurate, it is running a different contact model.

`ls_iterations` is swept along with `iterations` rather than held fixed. Holding
the line search at its default while cutting iterations to 1 does not produce a
cheap solver, it produces an expensive one that takes a single very carefully
chosen step, and the throughput curve comes out nearly flat for a reason that
has nothing to do with the variable being swept.

---

## Running it

```bash
python -m s10_gpu_scaling.run parity      --steps 300
python -m s10_gpu_scaling.run parity-x64  --steps 300     # float64, subprocess
python -m s10_gpu_scaling.run integrator  --steps 40      # localises to an integrator
python -m s10_gpu_scaling.tools.which_term --model franka_emika_panda   # to a term
python -m s10_gpu_scaling.run throughput  --max-envs 16384
python -m s10_gpu_scaling.run solver      --n-envs 512
python -m s10_gpu_scaling.figures --dark
```

`--no-mjx` runs every CPU-side measurement without JAX, which is how the
controls were validated on a machine with no GPU.

Results are written to `results/*.json`, one model at a time, write-then-rename.
A free session can be reclaimed at any moment, and a driver that accumulates in
memory and writes at the end turns a session that completed six of eight models
into a session that produced nothing.

## Model set

Spread across contact regimes, because the question is where GPU parallelism
pays, and that is dominated by the constraint solver rather than by DOF count.

| key | regime | why |
|---|---|---|
| `dynamixel_2r` | smooth | `ncon = 0`; the solver has nothing to do |
| `unitree_z1`, `franka_emika_panda` | arm | a few contacts, mostly self-collision |
| `panda_mjx_cube` | arm | DeepMind's MJX-tuned settings, for contrast |
| `leap_hand`, `s2_inhand`, `shadow_hand` | hand | many small persistent contacts, the S2 regime |
| `unitree_go1`, `unitree_h1` | legged | intermittent contacts under a floating base |

Filenames are pinned in `models.py` wherever the most-actuated heuristic picks
something surprising. It resolves `franka_emika_panda` to `mjx_single_cube.xml`,
because that scene adds a free-floating cube and wins the `nv` tiebreak against
`panda.xml`. Perfectly good model; not the arm the label claims. A results file
saying `franka_emika_panda` while measuring an arm-plus-cube scene is the kind of
mislabel nobody catches later.

## Bugs and near-misses, continuing the portfolio's count

13. **The control that answered the wrong question.** Floor A (one ulp, then
    float64) understates the float32 floor by ~30,000× and would have turned an
    ordinary 4× result into a 100,000× indictment of MJX. Caught by asking what
    float32 actually does, not what it does once.
14. **A fine reference cannot resolve below its own error.** Comparing
    everything to MuJoCo C at `dt/32` caps the MJX comparison at 1e-5 m while
    the effect of interest sits at 1e-9 m. Every MJX number would have read as
    agreement. Fixed by splitting the two questions onto two baselines, and by
    reporting `reference_check` with every result.
15. **`ncon = 0` on the model the headline was about.** The Panda's divergence
    was going to be written up as a contact-solver difference. It has no
    contacts. Checked rather than assumed, and the write-up says so.
16. **No positive control.** Every configuration measured disagreed by
    something, and there was no case that came out clean, so nothing
    established that the harness could detect agreement at all. The Euler row
    supplies it after the fact (1.4e-07 m, five orders below the float32
    floor). A fidelity study whose every cell is nonzero has not shown it can
    tell the difference. Same shape as p47 in the sim-lab: "no violations
    found" is the output of a working check and of a dead one.
18. **Reporting the control as a third finding.** `which_term.py` printed
    "removing 'neither' collapses the disagreement" alongside the real
    knockout, because the both-terms-removed control always collapses when
    either single term does. Three lines of output, one cause, and it read
    like three independent ones.
17. **A solver sweep that was not solver-limited.** On `unitree_z1` the error
    is identical to three digits across iteration counts 1 through 100, and
    the first version of `cheapest_within` duly recommended 1 iteration
    against the shipped 100, because 1 scored marginally lowest in what was
    pure scatter. `is_solver_limited` now refuses to answer when the spread
    across the whole grid is under 3x. Refusing is the correct output.
