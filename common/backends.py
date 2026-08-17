"""
The three evaluation backends behind one interface.

Clone Robotics' Research Scientist posting asks for benchmarks that evaluate
policies "in open loop, in simulation, and on the real robot -- so progress is
measurable at every stage."

That is three different questions, and conflating them is how sim-to-real
projects fool themselves:

  OpenLoopReplay
      Replay a *recorded action sequence* with no feedback. The policy is not
      involved. This isolates the model: if replaying real robot actions in
      sim diverges from the real recording, the simulator is wrong, and no
      amount of policy training will fix it. This is the measurement that
      system identification (S5) minimises.

  SimClosedLoop
      Policy in the loop, seeded, N episodes. This is what everyone means by
      "evaluation", and on its own it is the weakest of the three, because a
      policy can exploit simulator artifacts to score well here.

  HardwareBackend
      Same interface, same metric definitions, real robot. Deliberately a stub
      here -- there is no hardware attached. It exists so the interface is
      designed for it from day one rather than retrofitted, and so a reader
      can see exactly where hardware would plug in.

Because all three share one interface and one log schema, a sim number and a
hardware number are directly comparable. That comparability is the entire
point; it is not achievable if each stage grows its own ad-hoc script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

import numpy as np


class Policy(Protocol):
    """Minimal policy interface: observation in, action out."""

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


class EvalBackend(ABC):
    """
    Common interface for all three evaluation modes.

    Implementations must be deterministic given a seed, and must not mutate
    global state -- two backends should be constructible in the same process
    without interfering.
    """

    name: str = "abstract"

    @abstractmethod
    def run_episode(self, seed: int) -> dict[str, Any]:
        """
        Run one episode and return a record for the log.

        The record must contain at minimum:
            ret     : float   total (undiscounted) return
            length  : int     number of steps
        and should contain `success: bool` where the task defines success.
        """

    def evaluate(self, n_episodes: int, base_seed: int = 0) -> list[dict[str, Any]]:
        """
        Run N episodes with distinct, derived seeds.

        Seeds are `base_seed + i` rather than random, so an evaluation can be
        replayed exactly -- including a single misbehaving episode, which can
        then be re-run in isolation for debugging.
        """
        return [self.run_episode(base_seed + i) for i in range(n_episodes)]


class OpenLoopReplay(EvalBackend):
    """
    Replay a fixed action sequence with no feedback.

    `reference_states` is optional. When supplied (e.g. joint angles logged
    from a real robot), we report the divergence between simulated and
    reference states -- this is the reality gap, and the number S5 exists to
    shrink.
    """

    name = "open_loop"

    def __init__(
        self,
        step_fn: Callable[[np.ndarray], np.ndarray],
        reset_fn: Callable[[int], np.ndarray],
        actions: np.ndarray,
        reference_states: np.ndarray | None = None,
    ) -> None:
        self.step_fn = step_fn
        self.reset_fn = reset_fn
        self.actions = np.asarray(actions)
        self.reference_states = (
            None if reference_states is None else np.asarray(reference_states)
        )
        if self.reference_states is not None and len(self.reference_states) != len(self.actions):
            raise ValueError(
                f"reference_states has {len(self.reference_states)} rows but "
                f"actions has {len(self.actions)}; they must align step-for-step"
            )

    def run_episode(self, seed: int) -> dict[str, Any]:
        self.reset_fn(seed)
        states = []
        for action in self.actions:
            states.append(self.step_fn(action))
        states = np.asarray(states)

        record: dict[str, Any] = {
            "backend": self.name,
            "seed": seed,
            "length": int(len(self.actions)),
            # Open-loop replay has no reward signal; return is undefined.
            # We report NaN rather than 0.0 so it can never be silently
            # averaged into a performance number.
            "ret": float("nan"),
        }

        if self.reference_states is not None:
            err = states - self.reference_states
            # RMSE over all state dimensions and timesteps: the headline
            # scalar for "how wrong is my simulator".
            record["rmse"] = float(np.sqrt(np.mean(err**2)))
            # Terminal error matters separately: errors that accumulate
            # monotonically indicate a systematic parameter error (wrong
            # damping, wrong mass), whereas noisy-but-bounded error indicates
            # something stochastic.
            record["terminal_error"] = float(np.linalg.norm(err[-1]))
            record["max_error"] = float(np.max(np.linalg.norm(err, axis=-1)))
        return record


class SimClosedLoop(EvalBackend):
    """Policy in the loop against a Gymnasium-style environment."""

    name = "sim_closed_loop"

    def __init__(
        self,
        env,
        policy: Policy,
        *,
        max_steps: int = 1000,
        success_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.success_fn = success_fn

    def run_episode(self, seed: int) -> dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        total, steps = 0.0, 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < self.max_steps:
            action = self.policy.act(np.asarray(obs), deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            steps += 1

        record = {
            "backend": self.name,
            "seed": seed,
            "ret": total,
            "length": steps,
            # Recorded separately because they mean different things: a
            # `terminated` episode hit a real end condition (fell over,
            # reached goal); `truncated` merely ran out of clock. Collapsing
            # them is the same mistake that corrupts GAE bootstrapping.
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if self.success_fn is not None:
            record["success"] = bool(self.success_fn(info))
        return record


class HardwareBackend(EvalBackend):
    """
    Real-robot evaluation. Intentionally unimplemented.

    This class is not dead code. It fixes the interface that hardware must
    satisfy *before* any hardware exists, which is what stops the eventual
    hardware script from growing its own incompatible metric definitions.

    When a physical rig is attached, implement `run_episode` to drive it and
    return the same record shape. Nothing else in the stack changes -- the
    metrics, the logging, and the plots all work unaltered. That property is
    the deliverable.
    """

    name = "hardware"

    def __init__(self, policy: Policy, *, device_uri: str | None = None) -> None:
        self.policy = policy
        self.device_uri = device_uri

    def run_episode(self, seed: int) -> dict[str, Any]:
        raise NotImplementedError(
            "No hardware attached. Implement run_episode() against your rig; "
            "return the same record shape as SimClosedLoop so metrics and "
            "plots continue to work without modification."
        )
