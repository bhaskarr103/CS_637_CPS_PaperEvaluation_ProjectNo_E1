"""Lightweight Actor-Critic PPO Agent for Prototype 3.

Implements standard on-policy Proximal Policy Optimization (PPO) with:
- 2-layer MLP Actor-Critic (suitable for fast CPU/RTX 3050 execution)
- Continuous 1D action space in [-1, 1]
- Exact on-policy policy action logging:
    The rollout buffer strictly records u_policy and its true log-probability
    log pi(u_policy | obs). When a CBF filter intervenes, u_safe is applied to the
    environment, but u_policy is preserved in the buffer, adhering to Yang et al.
    Algorithm 1 and preventing off-policy importance-sampling breakdown.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Separate 2-layer MLP Actor and Critic networks."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Policy network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Trainable state-independent log standard deviation
        # Initialized to -0.5 (std ~ 0.606), giving good early exploration
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Value network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns action distribution parameters and state value."""
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float], float]:
        """Samples an action from the policy given a single observation.

        Returns:
            action (np.ndarray): Action clipped to [-1.0, 1.0].
            log_prob (float or None): Log probability of unclipped action.
            value (float): Critic state value estimate.
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, std, value = self.forward(obs_tensor)

            if deterministic:
                action = mean
                log_prob = 0.0
            else:
                dist = Normal(mean, std)
                sampled_action = dist.sample()
                action = sampled_action
                log_prob = float(dist.log_prob(sampled_action).sum(dim=-1).item())

            action_clipped = torch.clamp(action, -1.0, 1.0).squeeze(0).numpy()
            val_scalar = float(value.item())

        return action_clipped, log_prob, val_scalar

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for PPO gradient updates.

        Returns:
            log_probs: Log probabilities under current policy.
            entropy: Policy distribution entropy.
            values: Critic value estimates.
        """
        mean, std, values = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values


class RolloutBuffer:
    """Fixed-capacity rollout buffer storing transitions for on-policy PPO."""

    def __init__(self, capacity: int, obs_dim: int = 4, act_dim: int = 1):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self) -> None:
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Appends one transition to the buffer."""
        assert self.ptr < self.capacity, "Rollout buffer overflow"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE-lambda) and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_val = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_batches(self, batch_size: int):
        """Yields mini-batches of tensors for PPO optimization."""
        indices = np.random.permutation(self.ptr)
        for start_idx in range(0, self.ptr, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                torch.as_tensor(self.obs[batch_idx], dtype=torch.float32),
                torch.as_tensor(self.actions[batch_idx], dtype=torch.float32),
                torch.as_tensor(self.log_probs[batch_idx], dtype=torch.float32),
                torch.as_tensor(self.advantages[batch_idx], dtype=torch.float32),
                torch.as_tensor(self.returns[batch_idx], dtype=torch.float32),
            )


class PPOAgent:
    """Minimal, self-contained PPO agent with clipped surrogate objective."""

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.network = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    @property
    def ac(self) -> ActorCritic:
        """Alias for self.network."""
        return self.network

    def select_action(

        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float], float]:
        """Queries policy for an action."""
        return self.network.get_action(obs, deterministic=deterministic)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Runs PPO training updates over the filled rollout buffer."""
        # Normalize advantages across the rollout
        valid_adv = buffer.advantages[: buffer.ptr]
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std() + 1e-8
        buffer.advantages[: buffer.ptr] = (valid_adv - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for _ in range(self.n_epochs):
            for obs_b, acts_b, old_logp_b, adv_b, ret_b in buffer.get_batches(self.batch_size):
                obs_b = obs_b.to(self.device)
                acts_b = acts_b.to(self.device)
                old_logp_b = old_logp_b.to(self.device)
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)

                # Evaluate actions under updated policy
                new_logp, entropy, values = self.network.evaluate_actions(obs_b, acts_b)

                # Ratio: r(theta) = exp(new_logp - old_logp)
                log_ratio = new_logp - old_logp_b
                ratio = torch.exp(log_ratio)

                # Clipped surrogate loss
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Mean Squared Error)
                critic_loss = 0.5 * ((values - ret_b) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Tracking metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                num_updates += 1

        return {
            "actor_loss": total_actor_loss / max(1, num_updates),
            "critic_loss": total_critic_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "approx_kl": total_approx_kl / max(1, num_updates),
            "std": float(torch.exp(self.network.log_std).item()),
        }
