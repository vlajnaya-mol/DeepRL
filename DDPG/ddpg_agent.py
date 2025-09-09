"""DDPG-style agent with OU exploration noise and replay buffer.

This module defines:
- Agent: interacts with the environment, stores experience, and learns.
- OUNoise: Ornstein–Uhlenbeck noise for temporally correlated exploration.
- ReplayBuffer: experience replay with on-sample observation normalization.
- NormPlaceholder: no-op normalizer (default).
- SimpleNormalizer: running mean/var normalizer (optional).

Notes:
- Observations are normalized inside ReplayBuffer.sample().
- Targets are soft-updated with a standard τ-blend each learning step.
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from DeepRL.DDPG.model import Actor, Critic
from DeepRL.utils import OUNoise, ReplayBuffer, NormPlaceholder, SimpleNormalizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment using DDPG-style updates."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        random_seed: int,
        buffer_size: int = int(1e6),
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        weight_decay: float = 1e-4,
        noise_mu: float = 0.0,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,
        warmup_steps: int = 20_000,
        updates_per_step: int = 1,
        update_each: int = 1,
        
        # MADDPG
        n_agents: int = 1,
        td3_critic: bool = False
    ) -> None:
        """Initialize an Agent.

        Args:
            state_size: Dimension of each state.
            action_size: Dimension of each action.
            random_seed: RNG seed for Python, NumPy, and PyTorch components.
            buffer_size: Replay buffer capacity (number of transitions).
            batch_size: Minibatch size sampled from replay.
            gamma: Discount factor.
            tau: Soft-update interpolation factor for target networks.
            lr_actor: Learning rate for the actor.
            lr_critic: Learning rate for the critic.
            weight_decay: L2 weight decay for the critic optimizer.
            noise_mu: Mean (μ) of OU noise.
            noise_theta: Mean-reversion rate (θ) of OU noise.
            noise_sigma: Diffusion coefficient (σ) of OU noise.
            warmup_steps: Number of env steps before starting gradient updates.
            updates_per_step: Gradient updates performed for each env step
                that triggers learning.
            update_each: Perform updates every `update_each` env steps.

        Notes:
            - Observations are normalized on sampling from the replay buffer.
            - Call `reset(progress)` periodically to adapt noise scale across
              training (e.g., linearly anneal with progress ∈ [0, 1]).
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.update_each = update_each
        self.total_steps = 0

        # Actor networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic networks
        self.critic_local = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_target = Critic(state_size * n_agents, action_size * n_agents, random_seed).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay
        )

        # TD3 critic
        self.td3_critic = td3_critic
        if td3_critic:
            self.td3_critic_local = Critic(state_size * n_agents, action_size * n_agents, random_seed+1).to(device)
            self.td3_critic_target = Critic(state_size * n_agents, action_size * n_agents, random_seed+1).to(device)
            self.td3_critic_target.load_state_dict(self.td3_critic_local.state_dict())
            self.td3_critic_optimizer = optim.Adam(
                self.td3_critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay
            )

        # Exploration noise
        self.noise = OUNoise(
            size=action_size,
            seed=random_seed,
            mu=noise_mu,
            theta=noise_theta,
            sigma=noise_sigma,
        )
        self.init_sigma = noise_sigma
        self.noise_scale = 1.0  # set in reset()
        self.reset(progress=0.0)

        # Observation normalizer + Replay memory
        self.obs_norm = NormPlaceholder(state_size * n_agents)
        self.memory = ReplayBuffer(
            state_size=state_size * n_agents,
            action_size=action_size * n_agents,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=random_seed,
            obs_norm=self.obs_norm,
        )

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save experience and, after warmup, perform learning updates.

        Learning is triggered only if:
        - There are at least `batch_size` items in replay,
        - `total_steps > warmup_steps`, and
        - `total_steps % update_each == 0`.
        """
        self.total_steps += 1
        self.memory.add(state, action, reward, next_state, done)
        self.obs_norm.update(state)

        if (
            len(self.memory) >= self.batch_size
            and self.total_steps > self.warmup_steps
            and self.total_steps % self.update_each == 0
        ):
            for _ in range(self.updates_per_step):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select an action given the current policy.

        Args:
            state: Observation (D,).
            add_noise: If True, add OU exploration noise.

        Returns:
            Action clipped to [-1, 1].
        """
        state = self.obs_norm.normalize(state)
        state_t = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_t).cpu().numpy()
        self.actor_local.train()

        if add_noise:
            noise = self.noise.sample()
            action = action + self.noise_scale * noise

        return np.clip(action, -1.0, 1.0)

    def reset(self, progress: float) -> None:
        """Reset OU process and update noise scale based on training progress.

        Args:
            progress: Value in [0, 1]; higher means later in training.
                Noise std decays linearly from initial OU std to 0.05.
        """
        self.noise.reset()
        self.noise_scale = self._noise_scale_for_progress(
            progress, start_std=self.noise.std_ou, end_std=0.05
        )


    def learn(
        self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Update policy and value networks using a batch of experiences.

        Critic target:
            Q_target = r + γ * Q_target(next_state, actor_target(next_state))
        """
        states, actions, rewards, next_states, dones = experiences

        # --- update critic ---
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            q_targets_next = self.critic_target(next_states, actions_next)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- update actor ---
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- soft update targets ---
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
        """Soft-update model parameters: θ_target ← τθ_local + (1−τ)θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _noise_scale_for_progress(progress: float, start_std: float, end_std: float) -> float:
        """Return factor so that (factor * std_ou) ≈ desired_std(progress)."""
        progress = float(progress)
        desired_std = end_std + (start_std - end_std) * (1.0 - progress)
        start_std = max(1e-8, start_std)
        return desired_std / start_std
