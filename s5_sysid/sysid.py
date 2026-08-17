"""
S5: Real-to-sim system identification against DROID Franka Panda data.

The claim this project supports
------------------------------
"I have closed a reality gap measured against logged data from a physical
robot." Not a proxy, not sim-to-sim -- real measurements from a real Franka.

The measurement
---------------
DROID logs two *independent* signals at 15 Hz:

    action.joint_position            -- what the controller was COMMANDED
    observation.state.joint_position -- where the robot actually WENT

The robot does not perfectly track its commands. Real joints have friction,
rotor inertia, finite controller gain, gravity loading, and communication
latency. The gap between commanded and measured is the physical signature of
all of that, and it is what a simulator has to reproduce to be useful.

So the protocol is:

    1. Feed the REAL commanded joint positions into a MuJoCo Panda as
       position-control targets.
    2. Record where the simulated robot ends up.
    3. Compare to where the REAL robot ended up.
    4. Optimise MuJoCo parameters to make (3) as small as possible.

Step 1 is open-loop replay -- the simulator never sees the real measured
state, so it cannot cheat by being dragged along by the ground truth. This is
the `OpenLoopReplay` backend from `common/backends.py`.

Why a dataset check comes first
-------------------------------
An earlier attempt used `lerobot/nyu_franka_play_dataset`. It turned out that
its `action` column was exactly `state[t+1] - state[t]` -- correlation 1.0000
on every joint, identical standard deviations. The "actions" were relabeled
state deltas, so the dataset was kinematically self-consistent by
construction and contained NO reality gap at all.

Fitting a simulator to that would have produced a near-perfect result that
meant nothing. `verify_dataset_has_gap()` below encodes that check so the
mistake cannot recur silently.

Parameters identified
---------------------
    damping      -- viscous friction, opposes velocity
    armature     -- reflected rotor inertia. The most commonly ignored
                    parameter and often the largest single lever, because
                    harmonic drives multiply rotor inertia by the square of
                    the gear ratio.
    frictionloss -- Coulomb (dry) friction, velocity-independent
    kp           -- position actuator gain, i.e. controller stiffness
    latency      -- integer-step delay between command and execution
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

PANDA_XML = Path("assets/menagerie/franka_emika_panda/panda.xml")
DROID_GLOB = "assets/data/droid/*.parquet"
DROID_FPS = 15.0
N_JOINTS = 7

# Bounds for the identified parameters. Deliberately wide -- if the optimiser
# pins a parameter to a bound, that is information (the bound is wrong, or the
# parameter is not identifiable from this data), and we report it.
PARAM_BOUNDS = {
    "damping": (0.1, 200.0),
    "armature": (0.001, 2.0),
    "frictionloss": (0.0, 10.0),
    "kp": (10.0, 6000.0),   # Menagerie ships 3142.9; bound must exceed it
    "latency_steps": (0.0, 4.0),
}
PARAM_ORDER = list(PARAM_BOUNDS)


@dataclass
class Episode:
    """One DROID episode reduced to the two signals that matter."""

    name: str
    commanded: np.ndarray  # (T, 7) controller targets
    measured: np.ndarray   # (T, 7) where the robot actually went

    def __len__(self) -> int:
        return len(self.commanded)


def load_episodes(pattern: str = DROID_GLOB, max_episodes: int | None = None) -> list[Episode]:
    """Load DROID episodes, keeping only commanded and measured joint positions."""
    episodes = []
    for f in sorted(glob.glob(pattern))[:max_episodes]:
        df = pd.read_parquet(f)
        cmd = np.stack(df["action.joint_position"].values).astype(np.float64)
        meas = np.stack(df["observation.state.joint_position"].values).astype(np.float64)
        # Very short episodes carry mostly startup transient and destabilise
        # the fit without adding information.
        if len(cmd) < 30:
            continue
        episodes.append(Episode(Path(f).stem, cmd, meas))
    return episodes


def verify_dataset_has_gap(episodes: list[Episode], *, min_gap_rad: float = 1e-3) -> dict:
    """
    Refuse to proceed on a dataset with no reality gap.

    Two failure modes are checked:

      1. commanded == measured        -- the log is trivially self-consistent
      2. commanded[t] == measured[t+1] -- "actions" are relabeled state
                                          deltas (the NYU trap)

    Either means there is nothing to identify, and any fit would be
    meaningless. Better to fail loudly here than to publish a perfect plot.
    """
    same, shifted = [], []
    for ep in episodes:
        same.append(np.abs(ep.commanded - ep.measured).mean())
        shifted.append(np.abs(ep.commanded[:-1] - ep.measured[1:]).mean())

    result = {
        "mean_gap_rad": float(np.mean(same)),
        "mean_gap_one_step_shifted_rad": float(np.mean(shifted)),
        "has_gap": bool(np.mean(same) > min_gap_rad and np.mean(shifted) > min_gap_rad),
    }
    if not result["has_gap"]:
        raise ValueError(
            "Dataset has no measurable reality gap "
            f"(mean {result['mean_gap_rad']:.6f} rad, shifted "
            f"{result['mean_gap_one_step_shifted_rad']:.6f} rad). "
            "Commanded and measured are not independent signals; fitting to "
            "this data would be circular. Use a different dataset."
        )
    return result


def build_model(params: dict[str, float]) -> mujoco.MjModel:
    """
    Compile the Panda with a given parameter set applied to all 7 arm joints.

    We deliberately apply ONE value per parameter across all joints rather
    than fitting 7 independent values. With 632 s of data, 35 free parameters
    would overfit comfortably; 5 shared parameters is a claim the data can
    actually support. Per-joint fitting is the natural extension once there is
    more data, and the code is structured so that is a small change.
    """
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))

    for j in range(N_JOINTS):
        model.dof_damping[j] = params["damping"]
        model.dof_armature[j] = params["armature"]
        model.dof_frictionloss[j] = params["frictionloss"]

    # Panda's Menagerie model uses position actuators; set their gain and the
    # matching bias so the actuator remains a proper PD controller.
    # MuJoCo position actuators encode kp in gainprm[0] and -kp in biasprm[1].
    for a in range(min(N_JOINTS, model.nu)):
        model.actuator_gainprm[a, 0] = params["kp"]
        model.actuator_biasprm[a, 1] = -params["kp"]

    return model


def simulate_open_loop(
    model: mujoco.MjModel, ep: Episode, latency_steps: int
) -> np.ndarray:
    """
    Replay commanded joint positions open-loop and return simulated joints.

    The simulator is initialised from the real robot's FIRST measured state
    and then never sees ground truth again. Errors therefore accumulate
    exactly as they would on hardware -- which is the entire point of an
    open-loop comparison, and why it is a far harder test than closed-loop.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos[:N_JOINTS] = ep.measured[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    steps_per_frame = max(1, int(round((1.0 / DROID_FPS) / model.opt.timestep)))
    out = np.zeros_like(ep.measured)

    for t in range(len(ep)):
        # Latency: the robot executes a command issued `latency_steps` frames
        # ago. Communication and controller delay are real and typically worth
        # more error than any single dynamics parameter.
        src = max(0, t - latency_steps)
        data.ctrl[:N_JOINTS] = ep.commanded[src]

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        out[t] = data.qpos[:N_JOINTS]

        # Guard against divergence: a bad parameter set can send the sim
        # unstable, and NaNs would silently poison the optimiser's objective.
        if not np.all(np.isfinite(out[t])):
            out[t:] = 1e3
            break

    return out


def trajectory_rmse(sim: np.ndarray, real: np.ndarray) -> float:
    """Joint-space RMSE in radians over a whole trajectory."""
    return float(np.sqrt(np.mean((sim - real) ** 2)))


def evaluate(params: dict[str, float], episodes: list[Episode]) -> float:
    """Mean RMSE across episodes for one parameter set."""
    model = build_model(params)
    latency = int(round(params["latency_steps"]))
    errs = [trajectory_rmse(simulate_open_loop(model, ep, latency), ep.measured) for ep in episodes]
    return float(np.mean(errs))


def vector_to_params(x: np.ndarray) -> dict[str, float]:
    """Map an optimiser vector in [0,1]^5 to real parameter values."""
    out = {}
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = PARAM_BOUNDS[name]
        # Clip rather than assume: CMA-ES proposes points outside [0,1].
        out[name] = float(lo + np.clip(x[i], 0.0, 1.0) * (hi - lo))
    return out


def params_to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (params[n] - PARAM_BOUNDS[n][0]) / (PARAM_BOUNDS[n][1] - PARAM_BOUNDS[n][0])
            for n in PARAM_ORDER
        ]
    )


def default_params() -> dict[str, float]:
    """
    Menagerie's shipped Panda values, read from the unmodified model.

    This is the honest "before" baseline: what you get by downloading a
    well-maintained model and using it as-is. Any improvement must be measured
    against this, not against a deliberately broken strawman.
    """
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    return {
        "damping": float(np.mean(model.dof_damping[:N_JOINTS])),
        "armature": float(np.mean(model.dof_armature[:N_JOINTS])),
        "frictionloss": float(np.mean(model.dof_frictionloss[:N_JOINTS])),
        "kp": float(np.mean(model.actuator_gainprm[: min(N_JOINTS, model.nu), 0])),
        "latency_steps": 0.0,
    }
