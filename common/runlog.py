"""
Append-only JSONL run logging.

Design rationale
----------------
Every run writes exactly one file: `runs/<run_id>/log.jsonl`.

Line 1 is always a `manifest` record -- seed, config, git SHA, library
versions, and (for MuJoCo work) a hash of the compiled model. Every
subsequent line is one event: an episode result, a training-step summary, or
a note.

Why JSONL rather than a database, pickle, or CSV:

* Append-only means a crashed run still leaves valid, readable data up to the
  crash. Pickle gives you a corrupt file.
* One JSON object per line means `grep`, `head`, and `wc -l` all work, and a
  diff between two runs is readable in a code review.
* Self-describing, unlike CSV, so adding a field later does not silently
  shift columns in old files.

The manifest-first convention matters: you can read the first line of any log
and know whether the run is even comparable to another, without parsing
megabytes of episode records.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .seeding import capture_environment


def hash_mj_model(model) -> str:
    """
    Fingerprint a compiled MuJoCo model.

    Why: a sim result is only meaningful with respect to the model that
    produced it. MJCF files get edited mid-project -- a mass tweaked, a
    tendon's frictionloss nudged -- and then old and new results get plotted
    on the same axes. Hashing the *compiled* model (not the XML text) catches
    this, because it also catches changes to included files and defaults that
    the XML alone would hide.

    We hash the fields that affect dynamics. Visual-only fields are excluded
    deliberately, so recolouring a geom does not invalidate a comparison.
    """
    h = hashlib.sha256()
    for field_name in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "jnt_range",
        "geom_size",
        "geom_friction",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_gear",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_frictionloss",
        "tendon_lengthspring",
        "opt_timestep",
    ):
        # `opt_timestep` lives on model.opt, the rest directly on model.
        if field_name.startswith("opt_"):
            value = getattr(model.opt, field_name[4:], None)
        else:
            value = getattr(model, field_name, None)
        if value is None:
            continue
        h.update(field_name.encode())
        h.update(str(value).encode() if not hasattr(value, "tobytes") else value.tobytes())
    return h.hexdigest()[:16]


@dataclass
class RunLogger:
    """
    Writes one JSONL file for a single run.

    Usage
    -----
        with RunLogger("s1_ppo", config={"env": "Pendulum-v1"}, seed=0) as log:
            log.event("episode", episode=0, ret=-1200.5, length=200)
    """

    project: str
    config: dict[str, Any]
    seed: int
    root: Path = Path("runs")
    extra_manifest: dict[str, Any] = field(default_factory=dict)

    run_id: str = field(init=False)
    path: Path = field(init=False)
    _fh: Any = field(init=False, default=None)
    _t0: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        # Run id is sortable-by-time and includes a config hash, so two runs
        # that differ only in config never collide in the same second.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        self.run_id = f"{stamp}_{self.project}_seed{self.seed}_{cfg_hash}"
        self.path = Path(self.root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "RunLogger":
        self._fh = open(self.path / "log.jsonl", "w", encoding="utf-8")
        self._t0 = time.time()
        env = capture_environment()
        self._write(
            {
                "type": "manifest",
                "run_id": self.run_id,
                "project": self.project,
                "seed": self.seed,
                "config": self.config,
                "environment": env.to_dict(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **self.extra_manifest,
            }
        )
        # A dirty working tree means the git SHA does not describe the code
        # that actually ran. Warn loudly rather than recording a false claim.
        if env.git_dirty:
            self.event(
                "warning",
                message="Working tree is dirty; git_sha does not fully describe this run.",
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._write(
            {
                "type": "run_end",
                "wall_seconds": round(time.time() - self._t0, 3),
                "failed": exc_type is not None,
                "error": repr(exc) if exc else None,
            }
        )
        self._fh.close()

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        # Flush every record: a run killed by Ctrl-C or an OOM should still
        # leave everything up to that moment on disk.
        self._fh.flush()

    def event(self, type_: str, **fields: Any) -> None:
        """Append one event record."""
        self._write({"type": type_, "t": round(time.time() - self._t0, 3), **fields})


def read_log(path: str | Path) -> tuple[dict, list[dict]]:
    """
    Read a run log back.

    Returns (manifest, events). Raises if the first line is not a manifest,
    because a log without one cannot be compared to anything.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "log.jsonl"
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if not records or records[0].get("type") != "manifest":
        raise ValueError(f"{path} does not begin with a manifest record")
    return records[0], records[1:]


def iter_events(path: str | Path, type_: str) -> Iterator[dict]:
    """Yield only events of a given type from a run log."""
    _, events = read_log(path)
    for e in events:
        if e.get("type") == type_:
            yield e
