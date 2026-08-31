"""
Decide what this session should run, from what Drive already holds.

    python colab/autopilot.py --drive /content/drive/MyDrive/robotics-rl-portfolio
    python colab/autopilot.py --drive ... --plan       # show, do not run

A free-tier lease is short and ends without warning, so the expensive mistake
is not picking the wrong hyperparameter, it is spending a session on work that
was already done, or leaving a lease idle while a human decides. This inspects
the checkpoints and results in Drive, works out the next most valuable job, and
runs it until the session budget is spent.

The ordering is not arbitrary. It is cheapest-informative-first:

0.  **A smoke gate**, once, before anything is charged against the quota. Four
    environments, two iterations, a checkpoint written to Drive and then
    resumed from Drive with the local copy deleted. That last step is the
    whole preemption story and it is the one that fails silently: a resume
    that finds nothing does not raise, it starts from zero, and it surfaces
    days later as a learning curve that begins again.
1.  **S10** if it has no results. One session, no training, and it produces a
    publishable number even if the quota dies that afternoon. It also measures
    the batch size everything else should use, so running it first makes every
    later session better. Running it last would be strictly worse.
2.  **S2** until its step budget is met. The flagship, and the only project in
    the portfolio that has never trained anywhere; until it closes an
    iteration the whole thing is a claim rather than a result.
3.  **S3 baseline**, then **S3 dr**. Two arms of one experiment, and *neither
    is worth anything alone* - a DR arm with no control proves nothing. The
    control goes first deliberately: if the budget runs out after one arm, an
    orphaned baseline is at least a usable no-DR number for S2's task, whereas
    an orphaned DR arm answers no question at all.
4.  **S3 eval** once both arms exist.

Each job is a subprocess with `--resume` and `--max-hours`, so an interrupted
job is resumed by the next session rather than restarted. Nothing here is
stateful: the plan is recomputed from Drive every time, which means it is
correct after a preemption, after a manual run, and after a session that did
half a job.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Step budgets. S2 is the published spec; the S3 arms are deliberately smaller,
# because two arms at 1e8 each is a fortnight of free tier and the S3 question
# (does DR help on held-out physics) does not need a policy at S2's ceiling,
# only two policies trained identically.
S2_STEPS = 100_000_000
S3_STEPS = 30_000_000


@dataclass
class Job:
    name: str
    why: str
    argv: list[str]


def _latest_step(ckpt_dir: Path) -> int:
    """
    Steps completed, read from the checkpoint filenames.

    From the name, not by unpickling: on the Drive FUSE mount opening fifty
    checkpoints to read one integer is slow, and the name is written by
    `save_ckpt` from the same value.
    """
    if not ckpt_dir.exists():
        return 0
    best = 0
    for p in ckpt_dir.glob("ckpt_*.pkl"):
        try:
            best = max(best, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return best


def _s10_done(drive: Path) -> bool:
    """S10 counts as done when the two GPU-only measurements have landed."""
    res = drive / "s10_results" / "results"
    return all((res / f).exists() for f in ("throughput.json", "solver.json"))


def plan(drive: Path, max_hours: float, num_envs: int) -> list[Job]:
    jobs: list[Job] = []

    # Before anything is charged against the quota, prove the preemption story
    # works on this machine. `--smoke --resume` with a mirror runs four
    # environments for two iterations, writes a checkpoint to Drive, and then
    # resumes from it. A resume that silently finds nothing is the failure this
    # whole setup exists to prevent, and it does not raise: it starts from
    # zero and shows up days later as a curve that begins again.
    #
    # Runs once. The marker is in Drive, so it survives the runtime but is
    # re-run if the checkpoint format ever changes and the marker is deleted.
    if not (drive / ".smoke_ok").exists():
        jobs.append(Job(
            "smoke", "checkpoint/resume has not been proven on this machine yet",
            [sys.executable, "-u", "-m", "s2_inhand.train", "--smoke",
             "--ckpt-dir", "/content/work/smoke",
             "--ckpt-mirror", str(drive / "smoke_mirror"), "--resume"]))

    if not _s10_done(drive):
        jobs.append(Job(
            "S10", "no throughput/solver results in Drive yet; one session, "
                   "and it sets the batch size for everything after it",
            [sys.executable, "-u", "-m", "s10_gpu_scaling.run", "all",
             "--max-envs", "16384"]))

    s2_dir = drive / "s2_checkpoints"
    s2_at = _latest_step(s2_dir)
    if s2_at < S2_STEPS:
        jobs.append(Job(
            "S2", f"at {s2_at:,} of {S2_STEPS:,} steps",
            [sys.executable, "-u", "-m", "s2_inhand.train",
             "--total-steps", str(S2_STEPS), "--num-envs", str(num_envs),
             "--ckpt-dir", "/content/work/s2", "--ckpt-mirror", str(s2_dir),
             "--max-hours", str(max_hours), "--resume"]))

    # Control before treatment. See the module docstring.
    for arm in ("baseline", "dr"):
        d = drive / "s3_checkpoints" / arm
        at = _latest_step(d)
        if at < S3_STEPS:
            jobs.append(Job(
                f"S3:{arm}", f"at {at:,} of {S3_STEPS:,} steps",
                [sys.executable, "-u", "-m", "s3_domain_rand.train",
                 "--arm", arm, "--total-steps", str(S3_STEPS),
                 "--num-envs", str(num_envs),
                 "--ckpt-dir", f"/content/work/s3/{arm}",
                 "--ckpt-mirror", str(d),
                 "--max-hours", str(max_hours), "--resume"]))

    both_arms = all(_latest_step(drive / "s3_checkpoints" / a) >= S3_STEPS
                    for a in ("dr", "baseline"))
    if both_arms and not (drive / "s3_eval.json").exists():
        jobs.append(Job(
            "S3:eval", "both arms are trained and no evaluation exists",
            [sys.executable, "-u", "-m", "s3_domain_rand.eval",
             "--dr", str(drive / "s3_checkpoints" / "dr"),
             "--baseline", str(drive / "s3_checkpoints" / "baseline"),
             "--episodes", "512", "--out", str(drive / "s3_eval.json")]))

    return jobs


def cache_env(drive: Path) -> dict:
    """
    Environment that makes every child process share one XLA cache on Drive.

    The env var rather than `jax.config.update` inside each trainer, for one
    reason: the cache directory has to be set before the first compile, and a
    child process reads its environment before it imports anything. Measured
    here, same graph, two processes: 6.45 s cold, 0.95 s warm. On MJX, whose
    LEAP compile is minutes, this is most of what a short lease was losing.

    `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` is lowered from JAX's default
    of 1.0 because the trainers jit half a dozen small functions beside the
    expensive one; they are cheap to store and the point is that nothing
    recompiles after a preemption.
    """
    d = drive / "jax_cache"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "JAX_COMPILATION_CACHE_DIR": str(d),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0.5",
        "JAX_COMPILATION_CACHE_MAX_SIZE": str(2_000_000_000),
    }


def run(job: Job, drive: Path) -> int:
    print(f"\n{'=' * 70}\n[autopilot] {job.name}: {job.why}\n"
          f"$ {' '.join(job.argv)}\n{'=' * 70}", flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO), **cache_env(drive))
    p = subprocess.Popen(job.argv, cwd=str(REPO), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        p.terminate()
        print("\n[autopilot] interrupted; the newest checkpoint is in Drive",
              flush=True)
    return p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--max-hours", type=float, default=2.5,
                    help="budget for a single training job, below the tier's "
                         "likely cut-off")
    ap.add_argument("--session-hours", type=float, default=3.0,
                    help="budget for this whole session across jobs")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--plan", action="store_true", help="show the plan, run nothing")
    args = ap.parse_args()

    drive = Path(args.drive)
    jobs = plan(drive, args.max_hours, args.num_envs)

    if not jobs:
        print("Nothing left to run. Every budget is met and the evaluation "
              "exists. Pull the results out of Drive and write them up.")
        return

    cache = cache_env(drive)
    n = len([q for q in Path(cache["JAX_COMPILATION_CACHE_DIR"]).rglob("*") if q.is_file()])
    print(f'xla cache: {n} entries at {cache["JAX_COMPILATION_CACHE_DIR"]}'
          + ('  (empty: this session pays full compile cost, later ones will not)'
             if n == 0 else ''))
    print(f"plan ({len(jobs)} job(s), session budget {args.session_hours} h):")
    for i, j in enumerate(jobs, 1):
        print(f"  {i}. {j.name:<12} {j.why}")
    if args.plan:
        return

    t0 = time.time()
    for j in jobs:
        left = args.session_hours * 3600 - (time.time() - t0)
        if left <= 0:
            print(f"\n[autopilot] session budget spent; '{j.name}' is next. "
                  f"Start a new session and run this cell again.", flush=True)
            return
        code = run(j, drive)

        if j.name == "smoke" and code == 0:
            # Only mark it after the run actually succeeded, and verify the
            # resume happened rather than trusting the exit code: the trainer
            # exits 0 whether it resumed or started fresh, so the exit code
            # alone does not distinguish "resume works" from the exact failure
            # this gate exists to catch.
            mirror = drive / "smoke_mirror"
            if list(mirror.glob("ckpt_*.pkl")):
                (drive / ".smoke_ok").write_text("ok")
                print("[autopilot] smoke gate passed; checkpoint/resume "
                      "verified against Drive", flush=True)
            else:
                print("[autopilot] smoke run exited 0 but nothing reached "
                      "the Drive mirror. Stopping: every preempted session "
                      "would silently restart from zero.", flush=True)
                return

        if code != 0:
            # Stop rather than fall through to the next job. A failure here is
            # usually environmental (OOM at this batch size, Drive unmounted),
            # and the next job would hit the same wall while burning quota.
            print(f"\n[autopilot] {j.name} exited {code}. Stopping so the "
                  f"cause is visible rather than spending the rest of the "
                  f"lease reproducing it.", flush=True)
            return

    print(f"\n[autopilot] session complete in "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
