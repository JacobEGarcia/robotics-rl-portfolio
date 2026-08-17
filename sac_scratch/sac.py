"""
Soft Actor-Critic, written from scratch in PyTorch.

Reference: Haarnoja et al., "Soft Actor-Critic" (ICML 2018) and the follow-up
"Soft Actor-Critic Algorithms and Applications" (2019), which introduced the
automatic entropy temperature tuning implemented here.

Why write this after already writing PPO
-----------------------------------------
PPO and SAC fail in different ways, and knowing only one leaves you unable to
answer the question that actually matters for robotics: *which would you put
on a real robot, and why?*

The structural differences worth being able to explain:

    PPO                             SAC
    on-policy                       off-policy (replay buffer)
    discards data after one update  reuses every transition many times
    stochastic policy is a means    stochastic policy is the objective
    ~10^6 steps on HalfCheetah      ~10^5 steps for comparable return
    trivially parallel              sample-efficient but sequential

For a real robot, environment steps cost wall-clock time and actuator wear,
so SAC's ~10x sample efficiency usually wins. For a massively parallel
simulator, PPO's throughput usually wins. That trade is the deliverable of the
S2 comparison, and it is why both exist here.

The three from-scratch SAC bugs
--------------------------------
  1. Missing the tanh log-probability correction (see `SquashedGaussianActor`)
  2. Backpropagating into the target networks (see `SAC.update`)
  3. Optimising `alpha` directly instead of `log_alpha` (see `SAC.__init__`)

All three train without erroring. Number 1 in particular produces a policy
that looks fine and is quietly optimising the wrong objective.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Bounds on the log of the policy's standard deviation. Without these the
# network can drive std to 0 (log_std -> -inf), which makes log-probabilities
# explode and training diverge within a few hundred steps.
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass
class SACConfig:
    total_steps: int = 200_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    # Polyak averaging coefficient for the target critics. Small tau means
    # slow-moving targets, which is what keeps the bootstrapped regression
    # stable; tau=1.0 (hard copy every step) diverges.
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    # Uniform random actions for the first N steps. Without this the replay
    # buffer's early contents are all drawn from a near-identical untrained
    # policy, and the critic overfits that narrow slice of state space.
    start_steps: int = 5_000
    learning_starts: int = 1_000
    policy_frequency: int = 2   # delayed policy updates (from TD3)
    target_frequency: int = 1
    hidden_sizes: tuple[int, ...] = (256, 256)
    autotune_alpha: bool = True
    init_alpha: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of transitions.

    Preallocated numpy arrays rather than a list of tuples: at 10^6
    transitions the per-object overhead of Python tuples dominates memory and
    sampling becomes the bottleneck.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # We store `terminated` only, NOT `terminated or truncated`.
        # Same distinction as PPO's GAE: a time-limit truncation must still
        # bootstrap V(s'), because the episode would have continued. Storing
        # truncation as done teaches the agent that the clock running out is
        # a terminal failure.
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.ptr
        self.obs[i], self.actions[i] = obs, action
        self.rewards[i], self.next_obs[i] = reward, next_obs
        self.terminated[i] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(arr[idx])
            for arr in (self.obs, self.actions, self.rewards, self.next_obs, self.terminated)
        )


def mlp(sizes: list[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """
    Gaussian policy squashed through tanh into the action bounds.

    THE correction term
    -------------------
    We sample u ~ N(mu, sigma), then output a = tanh(u), scaled to the action
    range. Because tanh is a nonlinear change of variables, the density of `a`
    is NOT the density of `u`. The change-of-variables formula requires the
    log-determinant of the Jacobian:

        log p(a) = log p(u) - sum_i log(1 - tanh(u_i)^2)

    Omitting that second term is the single most common from-scratch SAC bug.
    Nothing crashes. The entropy term is simply computed in the pre-squash
    space, so the entropy the algorithm maximises is not the entropy of the
    policy it actually executes -- and with automatic temperature tuning, alpha
    then chases a target that does not correspond to real exploration.

    The `+ 1e-6` inside the log guards against tanh(u)^2 == 1.0 in float32,
    which happens for |u| greater than about 6 and would otherwise produce
    log(0) = -inf.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...],
                 act_low: np.ndarray, act_high: np.ndarray) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden], output_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)

        # Rescaling from tanh's [-1, 1] to the environment's action range.
        # Registered as buffers so they move with .to(device) and are saved in
        # the state_dict -- a checkpoint that loses these produces actions of
        # the wrong magnitude with no warning.
        self.register_buffer("act_scale", torch.tensor((act_high - act_low) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((act_high + act_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        # Squash log_std into a sane range with tanh rather than clamp:
        # clamp has zero gradient outside the range, so a network that drifts
        # out cannot recover. tanh keeps gradient everywhere.
        log_std = self.log_std_head(h)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, deterministic_action)."""
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)

        # rsample() not sample(): SAC backpropagates the actor loss THROUGH
        # the sampling step (the reparameterisation trick). `sample()` detaches
        # and the actor would receive no gradient at all.
        u = dist.rsample()
        y = torch.tanh(u)
        action = y * self.act_scale + self.act_bias

        log_prob = dist.log_prob(u)
        # The Jacobian correction described in the class docstring.
        log_prob -= torch.log(self.act_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        mean_action = torch.tanh(mu) * self.act_scale + self.act_bias
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Inference helper matching the eval harness `Policy` protocol."""
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, mean_action = self.sample(t)
        chosen = mean_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy()


class Critic(nn.Module):
    """
    Twin Q networks in one module.

    Why two: Q-learning with a max (or, here, a bootstrapped target) is
    systematically biased *upward*, because errors that overestimate get
    selected. Taking the minimum of two independently initialised critics is
    a cheap, effective bias correction -- the Clipped Double-Q trick from TD3.
    Single-critic SAC overestimates and plateaus early.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    """SAC trainer."""

    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray,
                 config: SACConfig, device: str = "cpu", seed: int = 0) -> None:
        self.cfg = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, config.hidden_sizes,
                                           act_low, act_high).to(self.device)
        self.critic = Critic(obs_dim, act_dim, config.hidden_sizes).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, config.hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Targets are updated only by Polyak averaging, never by an optimiser.
        # Leaving requires_grad on invites bug #2: a stray backward() call
        # silently trains the targets and destroys the stability they provide.
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        if config.autotune_alpha:
            # Target entropy heuristic from the paper: -dim(A). For a 1-D
            # action space that is -1; for a 6-DoF arm, -6. It encodes
            # "roughly one nat of entropy per action dimension".
            self.target_entropy = -float(act_dim)
            # We optimise LOG alpha, not alpha. Alpha must stay strictly
            # positive; optimising it directly lets a single gradient step
            # push it negative, which flips the sign of the entropy term and
            # makes the policy actively minimise its own entropy. That is
            # bug #3, and it manifests as sudden irreversible collapse.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
        else:
            self.log_alpha = torch.log(torch.tensor(config.init_alpha, device=self.device))
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def update(self, buffer: ReplayBuffer, step: int) -> dict[str, float]:
        cfg = self.cfg
        obs, actions, rewards, next_obs, terminated = (
            t.to(self.device) for t in buffer.sample(cfg.batch_size, self.rng)
        )
        rewards = rewards.unsqueeze(-1)
        terminated = terminated.unsqueeze(-1)

        # ---------------- critic ----------------
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            # The "soft" in Soft Actor-Critic: the bootstrap target is the
            # minimum twin-Q MINUS the entropy penalty, so the value of a
            # state includes how much freedom the policy retains there.
            target_q = torch.min(tq1, tq2) - self.alpha * next_logp
            # (1 - terminated), not (1 - done): see ReplayBuffer.add.
            backup = rewards + cfg.gamma * (1.0 - terminated) * target_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        stats = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "alpha": float(self.alpha.item()),
        }

        # ---------------- actor (delayed) ----------------
        if step % cfg.policy_frequency == 0:
            # Delayed policy updates (TD3): the critic is a moving target, and
            # updating the actor against a poorly-fit critic amplifies its
            # errors. Updating the actor half as often lets the critic settle.
            for _ in range(cfg.policy_frequency):
                pi, logp, _ = self.actor.sample(obs)
                qpi1, qpi2 = self.critic(obs, pi)
                # Maximise Q - alpha*logp  ==  minimise alpha*logp - Q.
                actor_loss = (self.alpha * logp - torch.min(qpi1, qpi2)).mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                if self.alpha_opt is not None:
                    # Temperature objective: drive entropy toward the target.
                    # logp is detached -- we are tuning alpha, not the policy.
                    alpha_loss = (-self.log_alpha.exp() * (logp + self.target_entropy).detach()).mean()
                    self.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    self.alpha_opt.step()
                    stats["alpha_loss"] = alpha_loss.item()

            stats["actor_loss"] = actor_loss.item()
            stats["entropy"] = float(-logp.mean().item())

        # ---------------- target Polyak update ----------------
        if step % cfg.target_frequency == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    # In-place lerp: p_targ = tau*p + (1-tau)*p_targ
                    p_targ.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

        return stats
