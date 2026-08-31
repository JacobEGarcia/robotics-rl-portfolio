"""
S9 scene: loading and inspecting the MS-Human-700 musculoskeletal model.

Why this model is in this portfolio
-----------------------------------
Every other humanoid in Menagerie is torque-actuated: one motor per joint,
force commanded directly, available torque independent of pose. Clone
Robotics' Myofiber is none of those things. It pulls only, its force depends
on length and shortening velocity, and there are far more actuators than
degrees of freedom.

MS-Human-700 is the closest publicly available model with those same three
properties: 700 Hill-type muscle actuators over 85 DoF, all pull-only
(ctrlrange [0, 1]), all routed through spatial tendons. It is not a model of
Protoclone and nothing here claims it is. It is a testbed for the control
problems that muscle-like actuation creates, regardless of whether the
contractile element is protein or a pneumatic bladder.

Provenance
----------
MS-Human-700 by the LNS Group (https://lnsgroup.cc/research/MS-Human),
vendored via MuJoCo Menagerie under `assets/menagerie/ms_human_700/`. Fetch it
with `bash scripts/fetch_assets.sh`. Requires MuJoCo >= 3.2.6.

A path gotcha worth knowing
---------------------------
Loading this model by a *relative* path that contains directories fails::

    mujoco.MjModel.from_xml_path("assets/menagerie/ms_human_700/MS-Human-700.xml")
    -> XML Error: Error opening file
       ".../ms_human_700/assets/body_primary/assets/menagerie/ms_human_700/..."

The model's directory prefix is prepended twice when nested `<include>`
elements are resolved. The same file loads fine from an absolute path, or from
a relative path with no directory component (cwd inside the model directory).
This model has 25 includes nested two deep, which is why it trips here and
simpler Menagerie models do not. `load()` always resolves to an absolute path,
so callers never see it.

Run
---
    python -m s9_muscle.scene              # structural report
    python -m s9_muscle.scene --shot       # also write media/s9/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "assets" / "menagerie" / "ms_human_700"
MEDIA_DIR = REPO_ROOT / "media" / "s9"

# The three variants shipped with the model. `full` is the one used for the
# measurements in characterize.py; the other two exist because 700 muscles is
# more than a locomotion or manipulation study needs, and dropping the ones
# you are not studying is cheaper than carrying them.
VARIANTS = {
    "full": ("MS-Human-700.xml", "scene.xml"),
    "locomotion": ("MS-Human-700-Locomotion.xml", "scene_locomotion.xml"),
    "manipulation": ("MS-Human-700-Manipulation.xml", "scene_manipulation.xml"),
}


def model_path(variant: str = "full", *, with_floor: bool = True) -> Path:
    """Absolute path to a variant's MJCF. See the include-path note above."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    bare, scene = VARIANTS[variant]
    path = MODEL_DIR / (scene if with_floor else bare)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. MS-Human-700 is not vendored in this repository; "
            f"run `bash scripts/fetch_assets.sh` to clone MuJoCo Menagerie."
        )
    return path


def load(
    variant: str = "full",
    *,
    with_floor: bool = True,
    keyframe: str | int | None = "init",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Compile a variant and return (model, data) at its reset pose.

    `keyframe` defaults to "init" because qpos0 (all zeros) is not a pose this
    model is meant to be evaluated at: the equality constraints encoding the
    knee's rolling contact and the shoulder's scapulohumeral rhythm are
    satisfied at the keyframe and violated at zero, so a forward pass from
    qpos0 spends its first milliseconds resolving constraint error that has
    nothing to do with whatever is being measured.

    The three variants are not consistent about naming: `full` and
    `locomotion` name their single keyframe "init", `manipulation` leaves it
    unnamed. A name that does not resolve therefore falls back to the sole
    keyframe when there is exactly one, and only raises when the choice would
    be ambiguous.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path(variant, with_floor=with_floor)))
    data = mujoco.MjData(model)
    if keyframe is not None:
        if isinstance(keyframe, int):
            key_id = keyframe
        else:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key_id < 0 and model.nkey == 1:
                key_id = 0
        if not 0 <= key_id < model.nkey:
            raise ValueError(
                f"variant {variant!r} has no keyframe {keyframe!r} "
                f"(it defines {model.nkey})"
            )
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data


def moment_arms(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    The (nu, nv) actuator moment-arm matrix R, densified.

    `data.actuator_moment` is stored sparse for a model this size (a flat
    buffer plus row-index arrays), and reading it as if it were (nu, nv) gives
    a 1-D array of the wrong length. numpy will happily compute `matrix_rank`
    of that and return 1. Densify explicitly.

    Requires mj_forward (or mj_step) to have run: moment arms are
    configuration-dependent, which is half the point of this project.
    """
    dense = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        dense,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    return dense


def joint_name_for_dof(model: mujoco.MjModel, dof: int) -> str:
    """Name of the joint owning a given velocity-space index."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.dof_jntid[dof]))


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def describe(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Structural facts about the compiled model, for the run manifest."""
    R = moment_arms(model, data)
    per_dof = np.count_nonzero(R, axis=0)
    per_muscle = np.count_nonzero(R, axis=1)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "na": int(model.na),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ntendon": int(model.ntendon),
        "neq": int(model.neq),
        "total_mass_kg": round(float(model.body_mass.sum()), 3),
        "timestep": float(model.opt.timestep),
        # Every actuator is a muscle: muscle activation dynamics, muscle
        # force-length-velocity gain, muscle passive bias, tendon transmission.
        "all_actuators_are_muscles": bool(
            np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
            and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
            and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_MUSCLE)
            and np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)
        ),
        # Pull-only. A motor takes signed torque; a muscle takes activation in
        # [0, 1] and can only shorten.
        "ctrl_is_unit_interval": bool(
            np.all(model.actuator_ctrllimited)
            and np.allclose(model.actuator_ctrlrange[:, 0], 0.0)
            and np.allclose(model.actuator_ctrlrange[:, 1], 1.0)
        ),
        "peak_force_N": {
            "min": round(float(model.actuator_gainprm[:, 2].min()), 2),
            "max": round(float(model.actuator_gainprm[:, 2].max()), 2),
            "sum": round(float(model.actuator_gainprm[:, 2].sum()), 1),
        },
        "tau_act_s": float(model.actuator_dynprm[0, 0]),
        "tau_deact_s": float(model.actuator_dynprm[0, 1]),
        "muscles_per_dof": {
            "min": int(per_dof.min()),
            "median": int(np.median(per_dof)),
            "max": int(per_dof.max()),
        },
        "dofs_per_muscle": {
            "min": int(per_muscle.min()),
            "median": int(np.median(per_muscle)),
            "max": int(per_muscle.max()),
        },
        "unactuated_dofs": [joint_name_for_dof(model, i) for i in np.where(per_dof == 0)[0]],
    }


# --------------------------------------------------------------------- shots


def framed_camera(data: mujoco.MjData, *, azimuth=140.0, elevation=-6.0, distance=2.4):
    """
    A free camera framed on the body's centre of mass.

    The model's own `record_camera` sits far enough back that the figure covers
    about a fifth of the frame, which is fine for a motion clip and useless for
    a still that has to show individual tendons.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.subtree_com[0]
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    return cam


def _render(model, data, *, camera, width=540, height=720) -> np.ndarray:
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render()


def _save(path: Path, img: np.ndarray) -> None:
    try:
        import imageio.v3 as iio

        iio.imwrite(path, img)
    except ImportError:  # pragma: no cover
        from PIL import Image

        Image.fromarray(img).save(path)


def colour_tendons(model: mujoco.MjModel, values: np.ndarray) -> np.ndarray:
    """
    Tint each muscle's tendon by a per-actuator value in [0, 1], returning the
    previous colours.

    `values` is usually activation, but any normalised per-muscle quantity
    works: the film module also uses it for moment-arm magnitude.

    MuJoCo does not do this on its own. Neither `mjVIS_ACTUATOR` nor the
    `rgba.actuatorpositive` visual defaults touch muscle tendons: measured on
    this model, toggling either changes zero pixels. The tendons are drawn from
    `model.tendon_rgba` (they carry no material, `tendon_matid` is -1
    throughout), so recolouring means writing that array directly.

    The index mapping matters and is easy to get wrong. `values` is indexed by
    actuator; `tendon_rgba` is indexed by tendon. Both have length 700 here,
    which makes the naive `tendon_rgba[i] = f(act[i])` compile, run, and
    produce a picture that is confidently wrong: the two orderings are a
    non-trivial permutation (actuator 0 drives tendon 11). Go through
    `actuator_trnid`.

    Caller is responsible for restoring the returned array.
    """
    previous = model.tendon_rgba.copy()
    a = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    tendon = model.actuator_trnid[:, 0]
    model.tendon_rgba[tendon, 0] = 0.30 + 0.65 * a
    model.tendon_rgba[tendon, 1] = 0.35 * (1.0 - a) + 0.10
    model.tendon_rgba[tendon, 2] = 0.50 * (1.0 - a) + 0.10
    return previous


def shots(out_dir: Path = MEDIA_DIR) -> list[Path]:
    """
    Write reference images: the rest pose, one recruitment pattern, and the two
    reduced variants.

    The recruitment shot drives the muscles that extend the right knee and
    leaves everything else near zero. A uniform activation would produce a
    prettier picture and show nothing: the point is that a command in this
    model is a pattern over 700 channels, and which channels a joint-space
    intent maps onto is a choice the controller has to make.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    model, data = load("full")
    cam = framed_camera(data)
    _save(out_dir / "pose_init.png", _render(model, data, camera=cam))
    written.append(out_dir / "pose_init.png")

    # Knee extensors: muscles whose moment arm about knee_angle_r has the sign
    # that straightens the joint. Selected by measuring the moment-arm matrix,
    # not by matching muscle names, so the selection stays correct if the
    # model's naming changes. Sign verified against qacc, not read off the XML:
    # at init, vasint_r (r = +0.059) drives qacc[knee] negative, and
    # knee_angle_r ranges [0, 2.4] with 0 fully extended, so positive moment
    # arm means extensor.
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r")
    dof = int(model.jnt_dofadr[knee])
    # Bend the knee before selecting. At the init pose it sits at 0, hard
    # against its own joint limit, where the extensors have nothing to do and
    # the picture shows a straight leg.
    data.qpos[model.jnt_qposadr[knee]] = 1.0
    from s9_muscle.control import project_equality

    project_equality(model, data)
    R = moment_arms(model, data)
    act = np.full(model.nu, 0.02)
    act[R[:, dof] > 1e-4] = 1.0
    data.act[:] = act
    mujoco.mj_forward(model, data)
    previous = colour_tendons(model, act)
    _save(
        out_dir / "recruitment_knee.png",
        _render(model, data, camera=framed_camera(data, azimuth=55.0)),
    )
    written.append(out_dir / "recruitment_knee.png")
    model.tendon_rgba[:] = previous

    for variant in ("locomotion", "manipulation"):
        m, d = load(variant)
        _save(out_dir / f"variant_{variant}.png", _render(m, d, camera=framed_camera(d)))
        written.append(out_dir / f"variant_{variant}.png")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    parser.add_argument("--shot", action="store_true", help="write media/s9/*.png")
    args = parser.parse_args()

    model, data = load(args.variant)
    print(json.dumps(describe(model, data), indent=2))

    if args.shot:
        for path in shots():
            print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
