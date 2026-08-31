"""
Shared machinery for the S9 control studies.

The one fact everything here rests on
-------------------------------------
At a fixed pose and zero velocity, MuJoCo's muscle force is *exactly affine*
in activation:

    actuator_force = gain(pose) * act + bias(pose)
    qfrc           = R(pose)^T @ actuator_force

Both are verified numerically rather than assumed, in `check_affine()` and in
every study's run log. On MS-Human-700 the first holds to 0.0 absolute error
and the second to 1.8e-12 N.

That matters because it turns "which of 700 muscles do you fire to produce
this joint torque" from a nonlinear optimal-control problem into a linear
system with box constraints:

    find a in [0, 1]^n  such that  A a = tau - c

with `A = R^T G` and `c` collecting the passive bias plus whatever the
uncontrolled muscles contribute. Underdetermined by construction, which is the
whole subject: on the right leg alone that is 50 muscles for 5 DoF.

`gain` depends on velocity as well as length, so everything here is isometric
(qvel = 0). The studies that need motion re-linearise at each pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import linprog, minimize

from s9_muscle.scene import actuator_names, moment_arms

# Right leg: three hip DoF, knee, ankle. Subtalar and MTP are excluded because
# the foot's intrinsic muscles push the muscle count up without adding a DoF
# that the gait literature this is compared against measures.
LEG_JOINTS = (
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
)


@dataclass
class LinearPlant:
    """`qfrc[dofs] = A @ act[muscles] + c`, valid at one pose, isometric."""

    A: np.ndarray            # (ndof, nmuscle)
    c: np.ndarray            # (ndof,)
    muscles: np.ndarray      # actuator indices, into the full model
    dofs: np.ndarray         # velocity-space indices
    gain: np.ndarray         # (nmuscle,) force at act=1 minus force at act=0
    bias: np.ndarray         # (nmuscle,) force at act=0

    @property
    def n(self) -> int:
        return self.A.shape[1]

    def torque(self, act: np.ndarray) -> np.ndarray:
        return self.A @ act + self.c


def project_equality(model, data, *, sweeps: int = 4) -> float:
    """
    Put qpos back on the equality-constraint manifold, and return the residual.

    MS-Human-700 encodes its knee's rolling contact and its shoulder's
    scapulohumeral rhythm as 42 `mjEQ_JOINT` couplings: a dependent joint is a
    quartic polynomial in a driver joint. Writing a driver joint's qpos and
    calling `mj_forward` does *not* update the dependents. `mj_forward` computes
    constraint forces, it does not project positions.

    Left alone this is not a rounding error. Setting `knee_angle_r` to 0.9 rad
    and nothing else leaves a maximum constraint violation of 0.60, and the
    muscle paths, moment arms and gains are then all evaluated at a
    configuration the model does not actually admit. Nothing raises; the numbers
    are simply wrong, and plausibly so.

    MuJoCo's convention is `y - y0 = a0 + a1 dx + a2 dx^2 + a3 dx^3 + a4 dx^4`
    with `dx = x - x0`, where y is `eq_obj1id`, x is `eq_obj2id` and the
    reference positions are `qpos0`. Couplings chain (a dependent can drive
    another), so this sweeps a few times.
    """
    for _ in range(sweeps):
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            dep, drv = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            y_adr = int(model.jnt_qposadr[dep])
            coef = model.eq_data[i][:5]
            if drv < 0:
                data.qpos[y_adr] = model.qpos0[y_adr] + coef[0]
                continue
            x_adr = int(model.jnt_qposadr[drv])
            dx = float(data.qpos[x_adr] - model.qpos0[x_adr])
            data.qpos[y_adr] = model.qpos0[y_adr] + float(np.polyval(coef[::-1], dx))
    mujoco.mj_forward(model, data)
    return equality_residual(data)


def equality_residual(data) -> float:
    """Largest absolute equality-constraint violation in the current state."""
    rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    return float(np.abs(data.efc_pos[rows]).max()) if rows.any() else 0.0


def dof_indices(model, joints=LEG_JOINTS) -> np.ndarray:
    out = []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"no joint named {name!r}")
        out.append(int(model.jnt_dofadr[jid]))
    return np.asarray(out, dtype=int)


def spanning_muscles(model, data, dofs: np.ndarray, *, tol=1e-9) -> np.ndarray:
    """Actuators with a non-zero moment arm about at least one of `dofs`."""
    R = moment_arms(model, data)
    return np.where(np.any(np.abs(R[:, dofs]) > tol, axis=1))[0]


def linearise(model, data, muscles: np.ndarray, dofs: np.ndarray, *, baseline=0.0) -> LinearPlant:
    """
    Build the affine map from the selected muscles' activation to joint torque.

    `gain` and `bias` are read off the model by evaluating it at act = 0 and
    act = 1 rather than by re-deriving `mju_muscleGain` from `gainprm` and
    `lengthrange`. Two forward passes are cheaper than replicating MuJoCo's
    length normalisation, and more importantly they cannot drift out of sync
    with it.

    Muscles outside `muscles` are held at `baseline` and folded into `c`.
    """
    saved_act, saved_vel = data.act.copy(), data.qvel.copy()
    data.qvel[:] = 0.0

    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    bias_all = np.array(data.actuator_force)
    R = moment_arms(model, data)          # pose-dependent only, act-independent

    data.act[:] = 1.0
    mujoco.mj_forward(model, data)
    gain_all = np.array(data.actuator_force) - bias_all

    data.act[:] = saved_act
    data.qvel[:] = saved_vel
    mujoco.mj_forward(model, data)

    others = np.setdiff1d(np.arange(model.nu), muscles)
    Rsel = R[np.ix_(muscles, dofs)]       # (nmuscle, ndof)
    A = Rsel.T * gain_all[muscles]        # (ndof, nmuscle), broadcasting over columns
    c = Rsel.T @ bias_all[muscles]
    if others.size:
        f_other = gain_all[others] * baseline + bias_all[others]
        c = c + R[np.ix_(others, dofs)].T @ f_other

    return LinearPlant(A=A, c=c, muscles=muscles, dofs=dofs,
                       gain=gain_all[muscles], bias=bias_all[muscles])


def check_affine(model, data, *, rng, samples=4) -> dict:
    """
    Verify the two identities this module is built on, at the current pose.

    Kept as a callable rather than a comment because it is cheap, and because
    the moment `gain` stops being activation-independent (a velocity term
    creeping in, a different actuator type) every number downstream becomes
    quietly wrong with nothing raising.
    """
    saved = data.act.copy()
    data.qvel[:] = 0.0
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    bias = np.array(data.actuator_force)
    R = moment_arms(model, data)
    data.act[:] = 1.0
    mujoco.mj_forward(model, data)
    gain = np.array(data.actuator_force) - bias

    affine_err = qfrc_err = 0.0
    for _ in range(samples):
        a = rng.uniform(0.0, 1.0, model.na)
        data.act[:] = a
        mujoco.mj_forward(model, data)
        affine_err = max(affine_err, float(np.abs(data.actuator_force - (gain * a + bias)).max()))
        qfrc_err = max(
            qfrc_err, float(np.abs(data.qfrc_actuator - R.T @ np.array(data.actuator_force)).max())
        )
    data.act[:] = saved
    mujoco.mj_forward(model, data)
    return {"affine_max_err_N": affine_err, "qfrc_identity_max_err_Nm": qfrc_err}


# ------------------------------------------------------- redundancy solver


def solve_activation(
    plant: LinearPlant,
    tau: np.ndarray,
    *,
    p: float = 3.0,
    tol: float = 1e-8,
    maxiter: int = 300,
) -> dict:
    """
    Minimum-effort activation reproducing a joint torque.

        minimise  sum(a_i^p)   subject to   A a = tau - c,   0 <= a <= 1

    `p` is the cost exponent argued over in the biomechanics literature.
    p = 1 is a linear program and drives the solution to a vertex: the fewest
    muscles that can do the job, each saturated. p = 2 is least-norm and
    spreads load. p = 3 is Crowninshield and Brand's choice, reported to match
    EMG better than either.

    Returns the solution plus the constraint residual. Infeasible targets are
    not an error here: a target outside the achievable set is a fact about the
    body, and callers filter on `residual` rather than on an exception.
    """
    b = np.asarray(tau, dtype=float) - plant.c
    n = plant.n
    bounds = [(0.0, 1.0)] * n

    if np.isclose(p, 1.0):
        # Exact LP. sum(a) with a >= 0 is already linear, no epigraph needed.
        res = linprog(
            c=np.ones(n), A_eq=plant.A, b_eq=b, bounds=bounds, method="highs"
        )
        a = np.clip(res.x, 0.0, 1.0) if res.x is not None else np.zeros(n)
        success = bool(res.success)
    else:
        # Start from the least-squares solution clipped into the box: a random
        # or zero start makes SLSQP spend its budget finding feasibility rather
        # than optimising, and on infeasible targets it never gets there.
        a0 = np.clip(np.linalg.lstsq(plant.A, b, rcond=None)[0], 0.0, 1.0)

        def cost(a):
            return float(np.sum(np.power(a, p)))

        def cost_jac(a):
            return p * np.power(a, p - 1.0)

        res = minimize(
            cost,
            a0,
            jac=cost_jac,
            bounds=bounds,
            constraints=[{
                "type": "eq",
                "fun": lambda a: plant.A @ a - b,
                "jac": lambda a: plant.A,
            }],
            method="SLSQP",
            options={"maxiter": maxiter, "ftol": tol},
        )
        a = np.clip(res.x, 0.0, 1.0)
        success = bool(res.success)

    residual = float(np.linalg.norm(plant.A @ a - b))
    return {
        "act": a,
        "success": success,
        "residual_Nm": residual,
        "cost": float(np.sum(np.power(a, p))),
        "n_active": int(np.count_nonzero(a > 0.01)),
        "peak_act": float(a.max()),
    }


def achievable_range(plant: LinearPlant, direction: np.ndarray) -> tuple[float, float]:
    """
    Largest and smallest torque obtainable along `direction`, in closed form.

    Maximising d^T (A a + c) over a box has a vertex solution: set a_i to 1
    wherever (A^T d)_i is positive and 0 elsewhere. No solver needed, which is
    what makes the failure study (thousands of muscle subsets) tractable.
    """
    w = plant.A.T @ np.asarray(direction, dtype=float)
    offset = float(direction @ plant.c)
    return offset + float(np.sum(np.minimum(w, 0.0))), offset + float(np.sum(np.maximum(w, 0.0)))


# --------------------------------------------------------- activation dynamics


def activation_rate(ctrl: float, act: float, prm) -> float:
    """MuJoCo's muscle activation ODE. Thin wrapper for readability."""
    return float(mujoco.mju_muscleDynamics(ctrl, act, prm))


def invert_activation(act: float, act_dot: float, prm) -> float:
    """
    The command that would produce `act_dot` at the current `act`.

    Inverting `da/dt = (u - a) / tau(a, sign(u - a))` needs the branch chosen
    before tau is known, but the branch depends only on the sign of the
    requested rate, which the caller already has:

        u = a + act_dot * tau_act  * (0.5 + 1.5a)      rising
        u = a + act_dot * tau_deact / (0.5 + 1.5a)     falling

    The formula is verified against `mju_muscleDynamics` in latency.py rather
    than trusted. The result is *not* clipped here: a caller needs to see that
    it asked for u = 3.4 in order to know it hit a physical limit rather than
    a tuning problem.
    """
    tau_act, tau_deact = float(prm[0]), float(prm[1])
    scale = 0.5 + 1.5 * act
    tau = tau_act * scale if act_dot > 0 else tau_deact / scale
    return act + act_dot * tau


def leg_plant(model, data, *, joints=LEG_JOINTS, baseline=0.0):
    """Convenience: the right-leg plant at the current pose."""
    dofs = dof_indices(model, joints)
    muscles = spanning_muscles(model, data, dofs)
    return linearise(model, data, muscles, dofs, baseline=baseline)


def muscle_labels(model, muscles: np.ndarray) -> list[str]:
    names = actuator_names(model)
    return [names[i] for i in muscles]
