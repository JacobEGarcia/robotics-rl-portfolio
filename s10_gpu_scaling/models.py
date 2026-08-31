"""
The model set S10 measures, and the rules for picking it.

Three properties decide whether a Menagerie model is usable here, and all
three are checked rather than assumed:

1.  **It must compile.** Several directories hold fragments that only make
    sense when `<include>`d from a scene file.
2.  **It must have DOFs.** `realsense_d435i` is a sensor with no joints. There
    is no timestep question to ask it and no work for a solver to do.
3.  **`mjx.put_model` must accept it.** MJX does not implement every MuJoCo
    feature. A model using an unsupported one raises at conversion, and that
    refusal is itself a finding worth recording rather than a reason to hide
    the model.

The corpus is deliberately spread across contact regimes, because the whole
question S10 asks is where GPU parallelism pays and where it does not, and
that answer is dominated by the constraint solver, not by DOF count:

    pendulum-ish     no contact at all, pure smooth dynamics
    arm              a few contacts, mostly self-collision
    hand             many small persistent contacts, the S2 regime
    quadruped        four intermittent contacts under a floating base

`find_model_file` mirrors the rule already used in
`robotics-sim-lab/p17_menagerie/audit.py`: directories are vendor-named and
files are product-named, and several robots ship an actuated variant that only
includes the base file. Taking the first `.xml` in the directory silently
audits the wrong thing (it picks the Panda's gripper, not the Panda), so the
choice is "compiles, and has the most actuators".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import mujoco

MENAGERIE = Path(__file__).resolve().parent.parent / "assets" / "menagerie"

# Local scenes that are not part of Menagerie.
S2_SCENE = Path(__file__).resolve().parent.parent / "s2_inhand" / "scene.xml"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    regime: str          # smooth | arm | hand | legged
    source: str          # menagerie directory name, or "local"
    path: Path | None = None
    file: str | None = None   # pinned filename inside the directory
    note: str = ""


# Ordered from least to most contact-constrained. That ordering is the
# independent variable of half the plots, so it lives here rather than being
# re-derived per figure.
#
# Filenames are pinned wherever the most-actuated heuristic picks something
# surprising, and the pin is the honest choice rather than a convenience: the
# heuristic resolves `franka_emika_panda` to `mjx_single_cube.xml`, because
# that scene adds a free-floating cube and so wins the nv tiebreak against
# `panda.xml`. It is a perfectly good model, it is just not the arm the label
# claims, and a results file saying "franka_emika_panda" while measuring an
# arm-plus-cube scene is the kind of mislabel nobody catches later.
CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec("dynamixel_2r", "smooth", "dynamixel_2r",
              note="two hinges, no collision geometry: the solver has nothing to do"),
    ModelSpec("unitree_z1", "arm", "unitree_z1", file="z1_gripper.xml"),
    ModelSpec("franka_emika_panda", "arm", "franka_emika_panda", file="panda.xml"),
    ModelSpec("panda_mjx_cube", "arm", "franka_emika_panda", file="mjx_single_cube.xml",
              note="DeepMind's own MJX demo scene: arm plus a free cube"),
    ModelSpec("leap_hand", "hand", "leap_hand", file="right_hand.xml",
              note="16 DOF, 68 box collision geoms; the hand S2 trains on"),
    ModelSpec("s2_inhand", "hand", "local", path=S2_SCENE,
              note="the actual S2 training scene: LEAP hand plus a free cube"),
    ModelSpec("shadow_hand", "hand", "shadow_hand", file="right_hand.xml",
              note="tendon-driven, 4 spatial tendons"),
    ModelSpec("unitree_go1", "legged", "unitree_go1", file="go1.xml"),
    ModelSpec("unitree_h1", "legged", "unitree_h1", file="h1.xml",
              note="humanoid, floating base"),
)


@functools.lru_cache(maxsize=None)
def find_model_file(directory: Path) -> Path | None:
    """
    Pick the representative XML in a Menagerie directory.

    Compiles every candidate and returns the one with the most actuators.
    Without the compile step this returns files that are not standalone
    models; without the actuator count it returns the wrong half of a robot.
    """
    best: tuple[int, Path] | None = None
    for xml in sorted(directory.glob("*.xml")):
        try:
            m = mujoco.MjModel.from_xml_path(str(xml))
        except Exception:
            continue
        if m.nv == 0:
            continue
        score = (m.nu, m.nv)
        if best is None or score > best[0]:
            best = (score, xml)
    return best[1] if best else None


def resolve(spec: ModelSpec) -> Path | None:
    if spec.path is not None:
        return spec.path if spec.path.exists() else None
    d = MENAGERIE / spec.source
    if not d.exists():
        return None
    if spec.file is not None:
        pinned = d / spec.file
        # Fall through to the heuristic rather than failing: Menagerie renames
        # files between releases, and a checkout that is merely newer should
        # still produce a result, with the substitution visible in `describe`.
        if pinned.exists():
            return pinned
    return find_model_file(d)


_LOADED_FROM: dict[int, str] = {}


def load(spec: ModelSpec, *, timestep: float | None = None) -> mujoco.MjModel:
    path = resolve(spec)
    if path is None:
        raise FileNotFoundError(
            f"no compilable model for {spec.key!r}; is assets/menagerie fetched? "
            f"(scripts/fetch_assets.sh)"
        )
    # Absolute path, always. MS-Human-700 nests <include> two deep and loading
    # by a relative path containing directories prepends the model directory
    # prefix twice; the failure is a missing-file error naming a path with a
    # duplicated segment, which reads like a corrupt checkout.
    m = mujoco.MjModel.from_xml_path(str(path.resolve()))
    if timestep is not None:
        m.opt.timestep = timestep
    # Which file this actually is, keyed by object identity, so `describe`
    # can record it. Every results file then names the XML it measured, and a
    # heuristic substitution after a Menagerie update is visible in the data
    # rather than only in a log line nobody kept.
    _LOADED_FROM[id(m)] = str(path.relative_to(path.parents[2])
                              if MENAGERIE in path.parents else path.name)
    return m


def available() -> list[ModelSpec]:
    """The subset of the catalogue this checkout can actually load."""
    return [s for s in CATALOGUE if resolve(s) is not None]


def describe(m: mujoco.MjModel) -> dict:
    """Size facts that explain a throughput number after the fact."""
    return {
        "file": _LOADED_FROM.get(id(m), "?"),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "nbody": int(m.nbody),
        "ngeom": int(m.ngeom),
        "ntendon": int(m.ntendon),
        "n_mesh_geom": int(sum(1 for i in range(m.ngeom)
                               if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)),
        "timestep": float(m.opt.timestep),
        "integrator": int(m.opt.integrator),
        "solver": int(m.opt.solver),
        "iterations": int(m.opt.iterations),
        "ls_iterations": int(m.opt.ls_iterations),
    }
