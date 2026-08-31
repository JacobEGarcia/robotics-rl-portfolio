"""
Getting the most out of a preemptible free-tier GPU session.

A free Colab session is not a machine, it is a lease. It ends without warning,
it takes `/content` and the pip environment with it, and the quota it consumes
is wall-clock time rather than useful work. Three things then decide how much
science a fixed quota buys, and none of them is the learning rate.

**1. Compilation is paid once per session, not once per project.** MJX compiles
the LEAP scene through XLA in minutes. A run that is preempted four times pays
that four times, and on a two-hour session a five-minute compile is 4% of the
lease burned reproducing a result the previous session already had. JAX has a
persistent compilation cache; pointing it at Drive makes every session after
the first start almost immediately. This is the single largest saving available
and it costs nothing.

**2. Checkpoint spacing is a bet on how long the lease lasts.** Checkpointing
every 2e6 steps means a session killed 1.9e6 steps later throws away 1.9e6
steps. Step-based spacing alone cannot bound that, because the throughput is
not known until the run is going. A wall-clock trigger does: checkpoint at
least every N minutes and the worst case is N minutes of loss, whatever the
throughput turns out to be.

**3. A session spent deciding what to run is a session not spent running.**
See `colab/autopilot.py`.

Nothing here changes any result. It changes how much of the lease reaches the
physics.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def enable_compilation_cache(cache_dir: str | Path, *,
                             min_compile_time_secs: float = 0.5,
                             max_size_bytes: int = 2_000_000_000) -> dict:
    """
    Persist XLA compilation artifacts to `cache_dir`, which should be on Drive.

    Measured on this machine, two processes sharing a cache directory, on a
    deliberately expensive unrolled graph:

        cold   3.81 s
        warm   0.68 s      5.6x

    The saving scales with compile cost, so on MJX (minutes for the LEAP
    scene) it is the difference between paying minutes per session and paying
    seconds. A trivial graph shows almost nothing, because the residual is JAX
    and XLA startup rather than compilation.

    Must be called **before** the first jit is traced. Importing jax is fine;
    compiling anything is not, because the cache directory is read when the
    backend is first used and a later change is silently ignored.

    `min_compile_time_secs` is 0.5 against JAX's default of 1.0. That default
    avoids filling a cache with trivial kernels, which is the right trade on a
    long-lived workstation and the wrong one here: the trainers jit half a
    dozen small functions alongside the expensive one, they are cheap to store,
    and the point is that *nothing* recompiles after a preemption. Verified
    that a 0.33 s compile falls below the 1.0 threshold and is not cached.

    `max_size_bytes` bounds what lands in Drive, whose free tier is 15 GB
    shared with everything else the account stores. JAX's own default is -1,
    unbounded, which would grow for the life of the project.
    """
    import jax

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs",
                      min_compile_time_secs)
    jax.config.update("jax_compilation_cache_max_size", max_size_bytes)
    # Cache entries are keyed on the executable, and a corrupt or truncated one
    # (a session reclaimed mid-write to Drive) must not take the run down with
    # it. Errors become cache misses, which cost a recompile and nothing else.
    jax.config.update("jax_raise_persistent_cache_errors", False)

    return cache_stats(cache_dir)


def cache_stats(cache_dir: str | Path) -> dict:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"dir": str(cache_dir), "entries": 0, "bytes": 0}
    files = [p for p in cache_dir.rglob("*") if p.is_file()]
    return {
        "dir": str(cache_dir),
        "entries": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def describe_device() -> dict:
    """
    What the driver says we got, recorded as a claim about the driver.

    Never write "measured on a T4" because a runtime menu said T4. A Kaggle
    session advertised as `T4 x2` reported a P100, which is compute capability
    6.0 and cannot run several things a T4 can. The label and the hardware are
    different facts and only one of them is observable from here.
    """
    import jax

    d = jax.devices()[0]
    out = {
        "platform": d.platform,
        "device_kind_reported": getattr(d, "device_kind", "?"),
        "n_devices": len(jax.devices()),
        "jax": jax.__version__,
    }
    if shutil.which("nvidia-smi"):
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True)
        out["nvidia_smi"] = r.stdout.strip()
    return out


def start(drive_root: str | Path, *, quiet: bool = False) -> dict:
    """
    One call at the top of a session. Returns what it set up.

    Deliberately does not import jax at module scope: the caller may need to
    set environment variables (JAX_ENABLE_X64, XLA flags) before jax is first
    imported, and a module-level import would have already closed that door.
    """
    drive_root = Path(drive_root)
    cache = enable_compilation_cache(drive_root / "jax_cache")
    dev = describe_device()

    if not quiet:
        print(f"device   {dev['platform']} / {dev['device_kind_reported']} "
              f"(as reported by the driver)")
        if "nvidia_smi" in dev:
            print(f"nvidia-smi {dev['nvidia_smi']}")
        print(f"jax      {dev['jax']}")
        print(f"xla cache {cache['entries']} entries, "
              f"{cache['bytes'] / 1e6:.1f} MB at {cache['dir']}")
        if cache["entries"] == 0:
            print("         (empty: this session pays full compile cost, "
                  "later ones will not)")

    if dev["platform"] != "gpu":
        raise SystemExit(
            "No GPU visible to JAX. Runtime -> Change runtime type -> T4 GPU, "
            "then restart the runtime and re-run. On CPU a single mjx.step of "
            "the LEAP scene costs about 4 seconds: this is not a slower "
            "option, it is not an option."
        )

    return {"cache": cache, "device": dev}
