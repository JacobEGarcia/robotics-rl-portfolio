"""
PPO for continuous control, written from scratch in PyTorch.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017),
plus the implementation details catalogued in Engstrom et al., "Implementation
Matters in Deep Policy Gradients" (ICLR 2020) and Huang et al., "The 37
Implementation Details of PPO" (ICLR blog, 2022).

Those two papers exist because PPO's *paper* is not sufficient to reproduce
PPO's *results*. The performance comes substantially from engineering details
that the algorithm box omits. Each of those details is called out below with
a `WHY:` comment, because being able to explain them is the entire point of
writing this by hand rather than importing stable-baselines3.

The four that most commonly break a from-scratch implementation:

  1. State-independent log_std (see `Actor`)
  2. Observation normalisation (see `RunningNormalizer`)
  3. terminated vs truncated in GAE bootstrapping (see `compute_gae`)
  4. Per-minibatch advantage normalisation (see `PPO.update`)

Numbers 3 and 4 fail *silently*: the code runs, the loss decreases, and the
agent simply learns worse than it should. Those are the interesting ones.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters. Defaults are the standard MuJoCo continuous-control set."""

    total_steps: int = 1_000_000
    num_envs: int = 1
    rollout_steps: int = 2048       # steps per env before each update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    # WHY entropy_coef = 0.0: for *continuous* control the policy already has a
    # learned std that provides exploration, and an entropy bonus pushes that
    # std up indefinitely, degrading final performance. The nonzero values
    # people copy (0.01) come from discrete-action Atari work. This is a
    # frequent and invisible source of underperformance on MuJoCo.
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    # Early-stop an update if the policy has moved too far. Cheap insurance
    # against a single catastrophic update; standard in modern PPO.
    target_kl: float | None = 0.015
    hidden_sizes: tuple[int, ...] = (64, 64)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Observation / reward normalisation
# --------------------------------------------------------------------------


class RunningNormalizer:
    """
    Online mean/variance via Welford's algorithm.

    WHY this matters more than any hyperparameter: MuJoCo observations have
    wildly different scales across dimensions (a joint angle in radians next
    to an angular velocity that reaches tens). A network with standard init
    cannot cope, and the value function in particular fails to fit. Omitting
    observation normalisation is the single most common reason a correct-looking
    from-scratch PPO "does not work" on MuJoCo.

    Welford rather than storing all samples: constant memory, numerically
    stable, and it updates per batch.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # tiny nonzero start avoids a divide-by-zero

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None]
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0), x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        # Parallel variance combination (Chan et al.), avoids catastrophic
        # cancellation that a naive sum-of-squares update would suffer.
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total

        self.mean, self.var, self.count = new_mean, m2 / total, total

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        # Clipping bounds the damage from a single outlier observation early
        # in training, when the running statistics are still poor.
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class RewardScaler:
    """
    Scale rewards by a running estimate of the *discounted return* std.

    WHY this exists -- found empirically, not copied from a paper:
    ---------------------------------------------------------------
    The first version of this file normalised observations but not rewards.
    On Pendulum-v1 (returns around -1000) it did not learn at all: mean return
    went -1196 -> -1172 over 20k steps while `v_loss` sat at ~4000 and policy
    entropy did not move (1.418 -> 1.416).

    The mechanism is worth understanding because nothing errors:

      1. Returns of magnitude ~1000 make the squared value error ~10^6.
      2. The policy-gradient loss is order 10^-2.
      3. `loss = pg_loss + 0.5 * v_loss` is therefore ~100% value loss.
      4. `clip_grad_norm_(0.5)` rescales the *whole* gradient to norm 0.5,
         so nearly the entire budget goes to the critic and the actor
         receives almost nothing.

    The policy is not broken; it is being starved. Scaling rewards puts the
    value target near unit variance so the two losses are commensurable.

    Note we divide by the standard deviation but do NOT subtract the mean.
    Subtracting a constant from every reward changes the optimal policy on
    tasks where episode length is not fixed (it adds a survival bonus or
    penalty), so only the scale may be adjusted.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.return_stats = RunningNormalizer(())
        self._running_return = 0.0

    def scale(self, reward: float, episode_over: bool) -> float:
        # Track the discounted return accumulator, not raw rewards: it is the
        # scale of the *value target* we need to normalise, and that is what
        # the critic actually regresses.
        self._running_return = self._running_return * self.gamma + reward
        self.return_stats.update(np.array([self._running_return]))
        scaled = reward / float(np.sqrt(self.return_stats.var + 1e-8))
        if episode_over:
            self._running_return = 0.0
        return float(np.clip(scaled, -10.0, 10.0))

    def state_dict(self) -> dict:
        return {"return_stats": self.return_stats.state_dict(), "gamma": self.gamma}


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Orthogonal initialisation with a per-layer gain.

    WHY specific gains: sqrt(2) for hidden layers preserves activation
    variance through ReLU/Tanh. The *policy output* layer uses gain 0.01 so
    the initial policy is nearly deterministic-at-the-mean rather than
    flailing -- this measurably improves early learning. The value head uses
    gain 1.0 because it regresses unbounded returns.

    Engstrom et al. found this init accounts for a meaningful share of PPO's
    reported advantage over vanilla policy gradient.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_gain))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Separate actor and critic trunks.

    WHY separate rather than shared: with a shared trunk the value loss
    gradient (which is typically far larger) dominates and corrupts the policy
    features. Sharing is worth it for pixel observations where the encoder is
    expensive; for low-dimensional state it is a liability.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.critic = mlp(obs_dim, hidden, 1, out_gain=1.0)
        self.actor_mean = mlp(obs_dim, hidden, act_dim, out_gain=0.01)

        # WHY a bare Parameter instead of a network output:
        # A state-*dependent* std (predicted by the network) is the single most
        # common from-scratch PPO bug on continuous control. It lets the policy
        # drive std toward zero in familiar states, which collapses exploration
        # and produces a policy that looks converged but is stuck. Every
        # reference implementation (CleanRL, SB3, the original paper's code)
        # uses a state-independent learned log_std. Starting at 0 => std = 1.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        # Sum over action dimensions: the joint log-prob of a diagonal
        # Gaussian is the sum of per-dimension log-probs.
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(obs).squeeze(-1),
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Inference helper matching the `Policy` protocol used by the eval harness."""
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            # At evaluation we use the distribution mean, not a sample.
            # Reporting sampled-policy performance understates the policy and
            # adds variance that has nothing to do with the learned behaviour.
            return self.actor_mean(t).squeeze(0).numpy()
        action, _, _, _ = self.get_action_and_value(t)
        return action.squeeze(0).numpy()


# --------------------------------------------------------------------------
# Advantage estimation
# --------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation (Schulman et al., 2016).

    WHY `terminated` and `truncated` are separate arguments -- this is the
    subtle bug that Gymnasium's API change exposed, and it is worth being able
    to explain precisely:

      * `terminated` means the MDP genuinely ended (the pole fell, the goal was
        reached). There is no future reward. The correct bootstrap is 0.

      * `truncated` means we stopped early for a reason *outside* the MDP --
        almost always a time limit. The episode would have continued and earned
        more reward. The correct bootstrap is V(s_next).

    Treating truncation as termination tells the agent that running out of
    clock is as bad as failing. On time-limited MuJoCo tasks that is a large,
    systematic, and completely silent underestimate of value near the horizon.
    Nothing errors; the agent just learns a worse policy.

    Returns (advantages, returns) where returns = advantages + values, the
    standard value-function regression target.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_val = next_value
            # `done` here controls the *value* bootstrap, so only true
            # termination zeroes it.
            next_non_terminal = 1.0 - float(terminated[t])
        else:
            next_val = values[t + 1]
            next_non_terminal = 1.0 - float(terminated[t])

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

        # An episode boundary of *either* kind breaks the GAE recursion,
        # because step t+1 belongs to a different episode. Termination is
        # handled by next_non_terminal above; truncation must additionally
        # reset the accumulator or advantage leaks across the boundary.
        if terminated[t] or truncated[t]:
            last_gae = 0.0

    return advantages, advantages + values


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class PPO:
    """PPO trainer. Call `.collect_rollout()` then `.update()` in a loop."""

    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, act_dim, config.hidden_sizes).to(self.device)
        # eps=1e-5 rather than torch's 1e-8 default: standard in RL
        # implementations, avoids occasional instability with small gradients.
        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate, eps=1e-5)
        self.obs_norm = RunningNormalizer((obs_dim,))
        self.reward_scaler = RewardScaler(config.gamma)

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        values: np.ndarray,
        progress: float,
    ) -> dict[str, float]:
        """
        One PPO update: several epochs of minibatch SGD over the rollout.

        `progress` in [0, 1] drives learning-rate annealing.
        """
        cfg = self.cfg

        if cfg.anneal_lr:
            # WHY anneal: PPO's trust region is enforced by the clip, which is
            # a fixed ratio bound. As the policy improves, the same ratio
            # corresponds to a larger behavioural change, so a constant LR
            # gets progressively more aggressive. Annealing compensates.
            for group in self.opt.param_groups:
                group["lr"] = cfg.learning_rate * (1.0 - progress)

        t_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        t_act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        t_logp = torch.as_tensor(logprobs, dtype=torch.float32, device=self.device)
        t_adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        t_ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = len(obs)
        minibatch_size = max(1, batch_size // cfg.num_minibatches)
        indices = np.arange(batch_size)

        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        n_updates = 0
        stop_early = False

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = indices[start : start + minibatch_size]

                _, new_logp, entropy, new_value = self.net.get_action_and_value(
                    t_obs[mb], t_act[mb]
                )
                log_ratio = new_logp - t_logp[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator (k3). The naive
                    # estimator (-log_ratio).mean() is unbiased but so noisy
                    # that it is useless as an early-stopping signal.
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                # WHY normalise advantages *per minibatch* rather than once
                # over the whole batch: after the first epoch the policy has
                # moved, so batch-level statistics computed before the update
                # are stale. Per-minibatch keeps the gradient scale consistent
                # across epochs. Getting this wrong does not error -- it just
                # makes the effective learning rate drift during the update.
                mb_adv = t_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # The clipped surrogate objective. We take the *minimum* of the
                # unclipped and clipped terms, which makes it a pessimistic
                # lower bound: improvement is only credited when it survives
                # the clip.
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                v_loss = 0.5 * ((new_value - t_ret[mb]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - cfg.entropy_coef * entropy_loss + cfg.value_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                # Global grad-norm clipping: bounds the damage from a single
                # pathological minibatch.
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats["pg_loss"] += pg_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                n_updates += 1

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                # Abandon the rest of the epochs for this rollout. The data is
                # now too off-policy for the importance ratio to be trusted.
                stop_early = True
                break

        for k in stats:
            stats[k] /= max(1, n_updates)
        stats["lr"] = self.opt.param_groups[0]["lr"]
        stats["early_stop"] = float(stop_early)
        # Explained variance answers "is the critic doing anything?".
        # <= 0 means the value function is no better than predicting the mean,
        # which almost always indicates a bug rather than a hard task.
        var_y = np.var(returns)
        stats["explained_variance"] = float(
            np.nan if var_y == 0 else 1 - np.var(returns - values) / var_y
        )
        return stats
