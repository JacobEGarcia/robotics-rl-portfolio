"""
The task universe: six manipulation families with procedural variation.

Design
------
Every episode is a fresh *instance* drawn from a family: object dimensions,
masses, frictions, initial poses, and goals are all resampled. The policy
never sees the same tabletop twice, which is what makes a scaling study
meaningful - with a fixed scene, more data is just more copies of one
trajectory.

The split mirrors GEN-0's evaluation design:

  PRETRAIN  reach, push, lift, place, topple   - the pretraining corpus
  WITHHELD  stack                              - never pretrained on

`stack` serves two measurements. During pretraining, next-action prediction
error on stack demonstrations is the zero-shot validation curve (their
Figure-1 metric: prediction error on withheld downstream tasks). Afterwards,
post-training on small stack budgets measures transfer scaling: does more
pretraining data buy better downstream success?

Stack is the right holdout because it composes skills the corpus contains
(reach above an object, grasp, lift, precise release) in a combination it
never contains - transfer has something to transfer, and from-scratch
baselines have something real to fail at.

Workspace conventions: the Panda base is at the origin on the table plane
(z=0); the reachable annulus used for object and goal placement is
r in [0.35, 0.62] biased to the +x half-plane the cameras face.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("reach", "push", "lift", "place", "topple", "stack")
PRETRAIN_FAMILIES = ("reach", "push", "lift", "place", "topple")
WITHHELD_FAMILY = "stack"

# Object parking position for bodies a family does not use: far outside the
# workspace but still on the table, so the compiled model never changes.
PARK = {"cube_a": (2.0, 2.0), "cube_b": (2.0, 2.5), "pillar": (2.0, 3.0)}

# Cube half-size ceiling: a 5 cm cube's worst-case yaw presents a 7.1 cm
# diagonal to the fingers, inside the Panda hand's 8 cm stroke. Above that,
# grasps start failing on geometry rather than on policy quality.
CUBE_HALF_RANGE = (0.015, 0.025)
PILLAR_HALF_HEIGHT_RANGE = (0.06, 0.10)


@dataclass
class TaskInstance:
    """Everything needed to reset the scene to one sampled episode."""

    family: str
    seed: int
    # Per-object: half-sizes (3,), mass, sliding friction, xy position, yaw.
    objects: dict[str, dict] = field(default_factory=dict)
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    horizon: int = 160

    @property
    def family_index(self) -> int:
        return FAMILIES.index(self.family)


def _sample_workspace_xy(rng: np.random.Generator, r_lo=0.38, r_hi=0.60) -> np.ndarray:
    """Uniform over the front annulus, area-corrected so radii aren't biased inward."""
    r = np.sqrt(rng.uniform(r_lo**2, r_hi**2))
    theta = rng.uniform(-0.85, 0.85)  # radians about +x
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def _cube_params(rng: np.random.Generator, xy: np.ndarray) -> dict:
    half = rng.uniform(*CUBE_HALF_RANGE)
    return {
        "half": np.array([half, half, half]),
        "mass": rng.uniform(0.04, 0.15),
        "friction": rng.uniform(0.6, 1.2),
        "xy": xy,
        "yaw": rng.uniform(-np.pi, np.pi),
    }


def sample_instance(family: str, seed: int) -> TaskInstance:
    """Deterministically sample one episode instance from a family."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    rng = np.random.default_rng(seed)
    inst = TaskInstance(family=family, seed=seed)

    if family == "reach":
        # Goal is a free point in the air; no objects on the table.
        xy = _sample_workspace_xy(rng)
        inst.goal = np.array([xy[0], xy[1], rng.uniform(0.06, 0.38)])

    elif family == "push":
        cube_xy = _sample_workspace_xy(rng, 0.40, 0.55)
        # Goal on the table, 10-22 cm away, kept inside the workspace.
        for _ in range(64):
            offset = rng.uniform(0.10, 0.22) * _unit(rng)
            goal_xy = cube_xy + offset
            if 0.38 <= np.linalg.norm(goal_xy) <= 0.62 and abs(np.arctan2(goal_xy[1], goal_xy[0])) < 0.9:
                break
        inst.objects["cube_a"] = _cube_params(rng, cube_xy)
        inst.goal = np.array([goal_xy[0], goal_xy[1], 0.0])

    elif family == "lift":
        cube_xy = _sample_workspace_xy(rng)
        inst.objects["cube_a"] = _cube_params(rng, cube_xy)
        inst.goal = np.array([cube_xy[0], cube_xy[1], rng.uniform(0.18, 0.30)])

    elif family == "place":
        cube_xy = _sample_workspace_xy(rng)
        for _ in range(64):
            goal_xy = _sample_workspace_xy(rng)
            if np.linalg.norm(goal_xy - cube_xy) > 0.12:
                break
        inst.objects["cube_a"] = _cube_params(rng, cube_xy)
        inst.goal = np.array([goal_xy[0], goal_xy[1], 0.0])

    elif family == "topple":
        pillar_xy = _sample_workspace_xy(rng, 0.40, 0.55)
        half_h = rng.uniform(*PILLAR_HALF_HEIGHT_RANGE)
        inst.objects["pillar"] = {
            "half": np.array([0.02, 0.02, half_h]),
            "mass": rng.uniform(0.08, 0.20),
            "friction": rng.uniform(0.6, 1.2),
            "xy": pillar_xy,
            "yaw": rng.uniform(-np.pi, np.pi),
        }
        inst.goal = np.array([pillar_xy[0], pillar_xy[1], 0.0])

    elif family == "stack":
        a_xy = _sample_workspace_xy(rng)
        for _ in range(64):
            b_xy = _sample_workspace_xy(rng)
            if np.linalg.norm(b_xy - a_xy) > 0.12:
                break
        a = _cube_params(rng, a_xy)
        b = _cube_params(rng, b_xy)
        # The base cube must be at least as large as the one placed on it,
        # or the instance is unstable by construction rather than by skill.
        if b["half"][0] < a["half"][0]:
            a, b = b, a
            a["xy"], b["xy"] = a_xy, b_xy
        inst.objects["cube_a"] = a
        inst.objects["cube_b"] = b
        inst.goal = np.array([b_xy[0], b_xy[1], 0.0])
        inst.horizon = 220

    return inst


def _unit(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(theta), np.sin(theta)])


# ---------------------------------------------------------------------------
# Success predicates
# ---------------------------------------------------------------------------
# Pure functions of the environment's privileged state dict, defined here so
# the data engine, the closed-loop evaluator, and any future hardware backend
# share one definition per family. A success predicate that lives inside a
# training loop is a number nobody can compare against.


def is_success(inst: TaskInstance, state: dict) -> bool:
    """Evaluate the family's success predicate against privileged state."""
    fam = inst.family

    if fam == "reach":
        return bool(np.linalg.norm(state["tcp_pos"] - inst.goal) < 0.02)

    if fam == "push":
        cube = state["objects"]["cube_a"]
        return bool(np.linalg.norm(cube["pos"][:2] - inst.goal[:2]) < 0.05)

    if fam == "lift":
        cube = state["objects"]["cube_a"]
        return bool(cube["pos"][2] > inst.goal[2] - 0.02)

    if fam == "place":
        cube = state["objects"]["cube_a"]
        half = inst.objects["cube_a"]["half"][2]
        settled = np.linalg.norm(cube["linvel"]) < 0.10
        on_table = abs(cube["pos"][2] - half) < 0.02
        in_zone = np.linalg.norm(cube["pos"][:2] - inst.goal[:2]) < 0.04
        released = state["finger_width"] > 0.05
        return bool(settled and on_table and in_zone and released)

    if fam == "topple":
        pillar = state["objects"]["pillar"]
        # zmat[2,2] of the pillar: cos(tilt). Fallen means tilted past 60 deg.
        return bool(pillar["z_axis"][2] < 0.5)

    if fam == "stack":
        a = state["objects"]["cube_a"]
        b = state["objects"]["cube_b"]
        a_half = inst.objects["cube_a"]["half"][2]
        b_half = inst.objects["cube_b"]["half"][2]
        aligned = np.linalg.norm(a["pos"][:2] - b["pos"][:2]) < 0.03
        at_height = abs(a["pos"][2] - (b["pos"][2] + b_half + a_half)) < 0.015
        settled = np.linalg.norm(a["linvel"]) < 0.10
        released = state["finger_width"] > 0.05
        return bool(aligned and at_height and settled and released)

    raise ValueError(f"no success predicate for family {fam!r}")
