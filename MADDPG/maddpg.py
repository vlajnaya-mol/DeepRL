from typing import List, Sequence, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from DeepRL.DDPG.ddpg_agent import Agent
from DeepRL.utils import ReplayBuffer, NormPlaceholder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """MADDPG with DDPG sub-agents (CTDE)."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        n_agents: int,
        gamma: float = 0.95,
        tau: float = 0.02,
        lr_actor: float = 5e-5,
        lr_critic: float = 5e-4,
        weight_decay: float = 1e-5,

        noise_mu: float = 0.0,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,

        random_seed: int = 42,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        warmup_steps: int = 20_000,
        update_each: int = 1,
        updates_per_step: int = 1,
        
        # TD3
        target_policy_noise: float = 0.2, 
        noise_clip: float = 0.5,
        td3_critic: bool = True,
        td3_delay: int = 2
    ) -> None:
        super().__init__()

        self.N = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_each = update_each
        self.updates_per_step = updates_per_step
        self.total_steps = 0
        self.update_cnt = 0

        self.td3_critic = td3_critic
        self.td3_delay = td3_delay
        self.target_policy_noise = target_policy_noise
        self.noise_clip = noise_clip

        self.agents: List[Agent] = [
            Agent(
                state_size=state_size,
                action_size=action_size,
                random_seed=random_seed + i,
                
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                lr_actor=lr_actor,
                lr_critic=lr_critic,

                weight_decay=weight_decay,
                noise_mu=noise_mu,
                noise_theta=noise_theta,
                noise_sigma=noise_sigma,

                n_agents=self.N,
                td3_critic=td3_critic,
            )
            for i in range(self.N)
        ]

        # Shared replay. Individual replays from agents are not used.
        self.memory = ReplayBuffer(
            state_size=state_size * self.N,
            action_size=action_size * self.N,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=random_seed,
            
            # TODO: make per-agent actor-side normalisation instead of the placeholder below:
            obs_norm=NormPlaceholder(state_size * self.N),
        )

    def _concat(self, xs: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=-1)

    def _split_by_agent(self, X: torch.Tensor, dim_per_agent: int) -> List[torch.Tensor]:
        """
        X: (B, N*D) -> list of N tensors each (B, D)
        """
        chunks = torch.split(X, dim_per_agent, dim=1)
        return list(chunks)

    def act(self, obs_all_agents: Sequence[np.ndarray], add_noise: bool = True) -> List[np.ndarray]:
        """obs_all_agents: list of length N with shapes (state_size,) each."""
        actions = [ag.act(obs, add_noise) for ag, obs in zip(self.agents, obs_all_agents)]
        return actions
    
    def reset(self, progress: float) -> None:
        for agent in self.agents:
            agent.reset(progress=progress)

    def step(
        self,
        states_per_agent: Sequence[np.ndarray],        # list of N (state_size,)
        actions_per_agent: Sequence[np.ndarray],       # list of N (action_size,)
        rewards: Sequence[float],                      # list/array length N
        next_states_per_agent: Sequence[np.ndarray],   # list of N (state_size,)
        dones: Sequence[bool],                         # list/array length N
    ) -> None:
        self.total_steps += 1

        s = self._concat(states_per_agent)                   # (N*state_size,)
        a = self._concat(actions_per_agent)                  # (N*action_size,)
        s2 = self._concat(next_states_per_agent)             # (N*state_size,)
        r = np.asarray(rewards, dtype=np.float32)            # (N,)
        d = np.asarray(dones, dtype=np.uint8)                # (N,)

        self.memory.add(s, a, r, s2, d)

        if (
            len(self.memory) >= self.batch_size
            and self.total_steps > self.warmup_steps
            and self.total_steps % self.update_each == 0
        ):
            for _ in range(self.updates_per_step):
                self.update_cnt += 1
                experiences = self.memory.sample()
                self.learn(experiences, delayed_td3_update=self.update_cnt % self.td3_delay)

    def learn(
        self,
        experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        delayed_td3_update=False
    ) -> None:
        """One MADDPG update over a minibatch for ALL agents."""
        states, actions, rewards, next_states, dones = experiences
        # shapes:
        # states:      (B, N*S)
        # actions:     (B, N*A)
        # rewards:     (B, N)
        # next_states: (B, N*S)
        # dones:       (B, N)

        # Local obs for each agent
        states_i      = self._split_by_agent(states,      self.state_size)
        next_states_i = self._split_by_agent(next_states, self.state_size)


        with torch.no_grad():
            next_actions_list = [ag.actor_target(s_i) for ag, s_i in zip(self.agents, next_states_i)]
            next_actions = torch.cat(next_actions_list, dim=1)  # (B, N*A)

            # TD3 policy smoothing
            if self.td3_critic and self.noise_clip > 0.0:
                noise = torch.normal(torch.zeros_like(next_actions),
                                     torch.zeros_like(next_actions) + self.target_policy_noise**2)
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_actions = torch.clamp(next_actions+noise, -1, 1)

        for i, ag in enumerate(self.agents):
            with torch.no_grad():
                q_next = ag.critic_target(next_states, next_actions)                # (B,1)
                if self.td3_critic:
                    q_next_td3 = ag.td3_critic_target(next_states, next_actions)                # (B,1)
                    q_next = torch.min(q_next, q_next_td3)
                y = rewards[:, i:i+1] + self.gamma * q_next * (1.0 - dones[:, i:i+1])

            q = ag.critic_local(states, actions)                                    # (B,1)
            critic_loss = F.mse_loss(q, y)

            ag.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(ag.critic_local.parameters(), 1.0)
            ag.critic_optimizer.step()
            
            if self.td3_critic:
                q = ag.td3_critic_local(states, actions)                                    # (B,1)
                td3_critic_loss = F.mse_loss(q, y)

                ag.td3_critic_optimizer.zero_grad()
                td3_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(ag.td3_critic_local.parameters(), 1.0)
                ag.td3_critic_optimizer.step()
            
            if not delayed_td3_update:
                actions_pred_list = []
                for j, other in enumerate(self.agents):
                    a_j = other.actor_local(states_i[j])
                    if j != i:
                        a_j = a_j.detach()
                    actions_pred_list.append(a_j)

                actions_pred = torch.cat(actions_pred_list, dim=1)                       # (B, N*A)
                actor_loss = -ag.critic_local(states, actions_pred).mean()

                ag.actor_optimizer.zero_grad()
                actor_loss.backward()
                ag.actor_optimizer.step()

                # --- soft update this agent's targets ---
                ag.soft_update(ag.critic_local, ag.critic_target, self.tau)
                ag.soft_update(ag.actor_local, ag.actor_target, self.tau)
                if self.td3_critic:
                    ag.soft_update(ag.td3_critic_local, ag.td3_critic_target, self.tau)
