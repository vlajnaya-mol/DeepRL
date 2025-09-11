import numpy as np
import random
import torch
from typing import Tuple
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NormPlaceholder:
    """No-op normalizer (for disabling normalization without branching)."""

    def __init__(self, size: int) -> None:
        pass

    def update(self, x: np.ndarray) -> None:
        """No-op."""
        return None

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return inputs unchanged."""
        return x


class SimpleNormalizer:
    """Per-feature running mean/variance normalizer with clipping."""

    def __init__(self, size: int, eps: float = 1e-4, clip: float = 5.0) -> None:
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = float(eps)
        self.clip = float(clip)
        self._eps = 1e-8

    def update(self, x: np.ndarray) -> None:
        """Update running mean/variance from raw env observations.

            x: Shape (D,) or (N, D), raw observations *before* replay.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[0:][None, :]  # ensure (1, D)

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])

        # Parallel/online mean & variance update
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return normalized and clipped observation."""
        x = np.asarray(x, dtype=np.float32)
        x = (x - self.mean) / np.sqrt(self.var + self._eps)
        return np.clip(x, -self.clip, self.clip)


class OUNoise:
    """Ornstein–Uhlenbeck noise via Euler–Maruyama discretization.

    dx = θ(μ − x)dt + σ√(dt) * N(0, I)

    Attributes:
        std_ou: Stationary standard deviation of the OU process (used to scale noise).
    """

    def __init__(
        self,
        size: int,
        seed: int = None,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1.0,
    ) -> None:
        self.mu = np.full(size, mu, dtype=float)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)

        self.rng = np.random.RandomState(seed)

        # Stationary std for the discrete-time OU approximation.
        denom = max(1e-12, 2.0 * self.theta - (self.theta ** 2) * self.dt)
        self.std_ou = self.sigma / np.sqrt(denom)

        self.state = self.mu.copy()

    def reset(self) -> None:
        """Reset internal state to the mean μ (use at episode start)."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        """Advance the OU process one step and return the new state."""
        x = self.state
        noise = self.rng.normal(0.0, 1.0, size=x.shape)
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * noise
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer storing experience tuples for off-policy learning."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        seed: int,
        obs_norm = NormPlaceholder(-1),
    ) -> None:
        """Create a ReplayBuffer.

        Args:
            state_size: Dimension of state vectors (unused but informative).
            action_size: Dimension of action vectors.
            buffer_size: Maximum number of experiences to retain.
            batch_size: Number of samples returned by `sample()`.
            seed: RNG seed for Python's `random`.
            obs_norm: Normalizer instance; called for `update()` and `normalize()`.
        """
        _ = state_size  # intentionally unused; kept for clarity
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        random.seed(seed)
        self.obs_norm = obs_norm

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences (with normalized observations)."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([self.obs_norm.normalize(e.state) for e in experiences])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])
        ).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([self.obs_norm.normalize(e.next_state) for e in experiences])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Current size of the replay buffer."""
        return len(self.memory)

class FenwickTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.bit = [None] + [0]*buffer_size
        self.leafs = [0]*buffer_size

    def add(self, delta, index):
        index += 1
        while index <= self.buffer_size:
            self.bit[index] += delta
            index += index & -index

    def set(self, value, index):
        delta = value - self.leafs[index]
        self.leafs[index] = value
        self.add(delta=delta, index=index)

    def prefix(self, index):
        index += 1
        s = 0
        while index > 0:
            s += self.bit[index]
            index -= index & -index
        return s
    
    def total(self):
        return self.prefix(self.buffer_size-1)
    
    def find_by_prefix(self, u: float) -> int:
        idx = 0
        bit_mask = 1 << (self.buffer_size.bit_length() - 1)
        while bit_mask > 0:
            next_idx = idx + bit_mask
            if next_idx <= self.buffer_size and float(self.bit[next_idx]) < u:
                u -= float(self.bit[next_idx])
                idx = next_idx
            bit_mask >>= 1
        return idx

class PERBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        seed: int,
        alpha: float,
        obs_norm = NormPlaceholder(-1),
    ) -> None:
        _ = state_size  # intentionally unused; kept for clarity
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.epsilon = 1e-5
        random.seed(seed)

        self.head = 0
        self.cnt = 0
        self.max_priority = 1.0
        self.memory = [None]*buffer_size
        self.fw = FenwickTree(buffer_size=buffer_size)

        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.obs_norm = obs_norm

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> int:
        """Add a new experience to memory."""
        ind = self.head
        self.head = (self.head + 1) % self.buffer_size
        self.cnt = min((self.cnt + 1), self.buffer_size)

        e = self.experience(state, action, reward, next_state, done)
        self.memory[ind] = e
        self.fw.set(value=(self.max_priority + self.epsilon) ** self.alpha, index=ind)
        return ind
    
    def _get_by_indices(self, indices):
        experiences = [self.memory[i] for i in indices]

        states = torch.from_numpy(
            np.vstack([self.obs_norm.normalize(e.state) for e in experiences])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences])
        ).float().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([self.obs_norm.normalize(e.next_state) for e in experiences])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def sample(
        self,
        beta
    ) -> Tuple[Tuple, list, np.array]:
        assert self.cnt >= self.batch_size, "PERBuffer not warmed up"
        S = self.fw.prefix(self.cnt - 1)
        mass_sampling = [(np.random.random()+i)/self.batch_size for i in range(self.batch_size)]
        mass_sampling_indices = [min(self.fw.find_by_prefix(u * S), self.cnt-1) for u in mass_sampling]

        IS_weights = torch.tensor([(S / max(1, self.cnt) / max(self.fw.leafs[i], 1e-12))**beta 
                           for i in mass_sampling_indices]).to(device)
        IS_weights = IS_weights / IS_weights.max()

        experiences = self._get_by_indices(mass_sampling_indices)

        return experiences, mass_sampling_indices, IS_weights
    
    def update(self, errors, indices):
        p = np.abs(errors) + self.epsilon
        masses = p ** self.alpha
        for mass, index in zip(masses, indices):
            self.fw.set(mass, index)
        self.max_priority = max(self.max_priority, masses.max())

    def __len__(self) -> int:
        """Current size of the replay buffer."""
        return self.cnt
