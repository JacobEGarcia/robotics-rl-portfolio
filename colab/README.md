# Running this portfolio on a free Colab T4

**Open [`Autopilot.ipynb`](Autopilot.ipynb) every session.** It reads what is
already in Drive, works out the next most valuable job, and runs it until the
session budget is spent. The plan is recomputed from Drive each time, so it is
correct after a preemption, after a manual run, and after a session that got
halfway through a job.

| | notebook | needs GPU for | one session? |
|---|---|---|---|
| | [`Autopilot.ipynb`](Autopilot.ipynb) | whatever is next | runs until the budget is spent |
| 1 | [`S10_gpu_scaling.ipynb`](S10_gpu_scaling.ipynb) | measurement only, no training | **yes**, ~1 h |
| 2 | [`S2_inhand.ipynb`](S2_inhand.ipynb) | PPO, ~10^8 steps | no, several |
| 3 | [`S3_domain_rand.ipynb`](S3_domain_rand.ipynb) | PPO x2 arms, ~3x10^7 each | no, several |

The numbered three are for driving one project by hand, or for reading what a
stage actually does. Autopilot runs the same commands with the ordering already
decided:

**S10 first.** It finishes inside one session, it produces a result whether or
not the quota holds out that afternoon, and it measures the batch size S2
should be configured with. Starting with the multi-day training run and
discovering afterwards that the batch size was wrong is the expensive ordering.

**S3's control arm before its DR arm.** Neither arm is worth anything alone. If
the budget runs out after one, an orphaned baseline is at least a usable no-DR
number for S2's task, whereas an orphaned DR arm answers no question at all.

## Running several Google accounts at once

Each account is a separate quota, and each has its own Drive. That second fact
is the trap: **running the unfiltered plan on two accounts duplicates the work
rather than dividing it.** The plan is computed from Drive, so a second account
starts its own smoke gate, its own S10, and its own S2 from step zero, and you
finish with two half-done copies of one thing instead of one finished set.

`ONLY` in the Autopilot notebook (or `--only` on `autopilot.py`) fixes that.
Set it per account:

| account | `ONLY` | what it does | rough cost |
|---|---|---|---|
| A | `'S2'` | the flagship, 10^8 steps | days |
| B | `'S3'` | both arms, 2 x 3x10^7 steps | days |
| C | `'S10'` | fidelity, throughput, solver cost | one session |

`--only` takes comma-separated prefixes and is case-insensitive, so `S3` picks
up `S3:baseline`, `S3:dr` and `S3:eval`. `None` runs everything, which is right
when you have one account.

**The smoke gate always runs, whatever `ONLY` says.** It is a gate, not a job:
two minutes against a day of sessions that silently restart from zero.

**Both S3 arms stay on one account on purpose.** They are a control and a
treatment. Splitting them across machines would put a hardware difference
inside the comparison the experiment exists to make, and the whole point of S3
is that the two arms differ in the randomization and in nothing else.

One thing this does not solve: the S3 evaluation needs both arms in one place,
so whichever account trains them also runs `S3:eval`. If you ever do split
them, copy both checkpoint directories into one Drive first.

## Getting the most out of a lease

The free tier charges wall-clock GPU time, not useful work, so three things
decide how much science a fixed quota buys, and none of them is the learning
rate.

**Compilation is paid once per session, not once per project.** MJX compiles
the LEAP scene through XLA in minutes, and a run preempted four times pays it
four times. Every notebook points JAX's persistent compilation cache at Drive,
as environment variables so that subprocesses inherit it and so the directory
is known before the first compile. Measured on this codebase, two processes
sharing a cache: **6.45 s cold, 0.95 s warm.** The saving scales with compile
cost, so on MJX it is worth far more than that.

**Checkpoint spacing is a bet on how long the lease lasts.** Checkpointing
every 2e6 steps means a session killed 1.9e6 steps later throws away 1.9e6
steps, and step spacing alone cannot bound that because the throughput is not
known until the run is going. `--ckpt-every-min` (default 10) bounds the loss
to ten minutes whatever the throughput turns out to be. Verified against a
simulated 2,000 steps/s run: worst gap 10.1 minutes, and on a fast run the
time trigger never fires and step spacing is unchanged.

**Batch size should be the knee, not the ceiling, and it should be measured
once.** S10's throughput sweep reports the smallest batch within 10% of the
best. Past it, each extra environment buys under 10% while costing
proportional memory and a longer compile, and on a preemptible lease a longer
compile is a direct loss. The measured value is written to
`MyDrive/robotics-rl-portfolio/s2_num_envs.json`; the training cells and the
autopilot read it back, so the sweep runs once rather than once per session,
and a number nobody measured is never used silently.

## A bug worth knowing about, since it is the shape of the others

The S2 notebook used to end by printing "run sections 1 and 4, skip 2 and 3",
and section 4 used a variable that only section 3 defined. Every session after
the first, which is every session that matters, would have raised `NameError`
after mounting Drive and cloning: a wasted lease for a one-line cause.

Nothing catches that by reading the code, because each cell is individually
correct. It is caught by asking a different question, "does the path the
document tells you to take actually work", which is why `colab/` is now
audited by walking each notebook's cells in order and checking that every name
is bound before use, that every flag the autopilot passes exists in the target
CLI, and that every model key it names is in S10's catalogue.

## What free Colab actually gives you

Worth being concrete, because the plan depends on it:

- **A T4, usually.** 16 GB, compute capability 7.5. What the runtime menu says
  and what the driver reports are not always the same thing, the notebooks
  print `nvidia-smi` and the JAX-reported device kind, and every result records
  the reported string. Quote it as "what the driver said", never as the
  hardware. A Kaggle session advertised as `T4 x2` reported a P100.
- **An unpublished, varying quota**, consumed by wall-clock GPU time whether or
  not you are computing. A few hours a day is the realistic planning number.
- **A ~90 minute idle disconnect** and a session cap well below the 12 hours
  Kaggle allows. Keep the tab visible.
- **`/content` does not survive the runtime being reclaimed.** Neither does the
  pip environment. Drive does.

## How the notebooks survive that

Three mechanisms, all in `s2_inhand/train.py` and `s3_domain_rand/train.py`:

- **`--ckpt-mirror`** copies each checkpoint to a second directory after
  writing it locally. Point it at Drive. Writing straight to Drive would be
  the obvious thing and is the wrong thing: the Drive FUSE mount is slow and
  occasionally errors mid-write, so the write-then-rename that protects
  against a truncated checkpoint would be protecting the copy that does not
  matter.
- **`--resume` searches both directories**, newest-step-first, and falls back
  down the list if the newest file is unreadable. On Colab the two diverge in
  the ordinary case: the runtime is reclaimed, `/content` is destroyed, and the
  Drive copy is the only one left. A resume that searched only local disk would
  silently start from scratch, which does not error and shows up hours later as
  a learning curve that begins at zero again.
- **`--max-hours`** exits cleanly with a final checkpoint before the platform
  is likely to cut the session off. Exit code 0, running out of session budget
  is the expected end of a Colab session, not a failure, and a nonzero exit
  would make every chained session look like a crash.

`--keep-last 3` bounds what accumulates in Drive. Three, not one: the newest to
resume from, and two behind it in case the newest was written by a session that
was already going wrong.

## The one thing you have to do first

The notebooks `git clone` the public portfolio repo. Anything committed locally
but not pushed is not in that clone, and the setup cell will stop and say so.

```bash
cd ~/Downloads/hermestes/robotics-rl-portfolio && git push origin main
```

If you would rather not push, tar the repo and drop it in
`MyDrive/robotics-rl-portfolio/portfolio.tgz`; the setup cell unpacks it over
the clone.

```bash
tar czf portfolio.tgz --exclude=assets --exclude=runs --exclude=.git .
```

`assets/menagerie` is excluded from both paths on purpose, it is 2.3 GB of
third-party models and is itself a git repository. The notebooks fetch the
seven directories they need with a sparse checkout, in seconds.

## Editing these notebooks

Don't. They are generated:

```bash
python colab/build.py
```

`.ipynb` is JSON with source split into lines, which does not diff and does not
review. Edit `colab/build.py` and regenerate, the same rule the manga guide's
`index.html` follows.
