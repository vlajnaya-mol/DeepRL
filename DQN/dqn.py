import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DeepRL.utils import PERBuffer
from typing import Tuple


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer: nn.Linear) -> Tuple[float, float]:
    """Return symmetric uniform init bounds based on fan-in of a Linear layer."""
    fan_in = layer.weight.data.size(1)
    lim = 1.0 / np.sqrt(float(fan_in))
    return -lim, lim

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        for layer in (self.fc1, self.fc2):
            lo, hi = hidden_init(layer)
            layer.weight.data.uniform_(lo, hi)
            if layer.bias is not None:
                layer.bias.data.uniform_(lo, hi)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class Agent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 random_seed: int,
                 hidden_size: int = 256,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 buffer_size: int = int(1e6),
                 batch_size: int = 128,
                 tau: float = 1e-3,
                 epsilon_bounds: tuple = (0.05, 1.0),
                 anneal_endpoint: float = 1.0,
                 update_every: int = 1,
                 beta_bounds: tuple = (0.4, 1.0),
                 alpha: float = 0.6
                 ):
        self.action_size = action_size
        self.tau = tau
        self.epsilon_bounds = epsilon_bounds
        self.epsilon = epsilon_bounds[1]
        self.anneal_endpoint = anneal_endpoint
        self.gamma = gamma
        self.batch_size = batch_size
        self.beta_bounds = beta_bounds
        self.beta = beta_bounds[0]
        self.step_cnt = 0
        self.update_every = update_every

        self.qn_local = QNetwork(input_size=state_size,
                                 hidden_size=hidden_size,
                                 output_size=action_size, 
                                 seed=random_seed).to(device)
        self.qn_target = QNetwork(input_size=state_size,
                                  hidden_size=hidden_size,
                                  output_size=action_size, 
                                  seed=random_seed).to(device)
        self.qn_target.load_state_dict(self.qn_local.state_dict())
        self.q_optimizer = optim.Adam(self.qn_local.parameters(), lr=lr)

        self.memory = PERBuffer(state_size=state_size,
                                   action_size=action_size,
                                   buffer_size=buffer_size,
                                   batch_size=batch_size,
                                   seed=random_seed,
                                   alpha=alpha)
        
    def reset(self, progress):
        self.epsilon = (self.epsilon_bounds[1] 
                        - max(1.0, progress / self.anneal_endpoint) * (self.epsilon_bounds[1] - self.epsilon_bounds[0]))
        self.beta = self.beta_bounds[0] + progress * (self.beta_bounds[1] - self.beta_bounds[0])

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            q_values = self.qn_local(state)
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, self.action_size)
            else:
                action = torch.argmax(q_values).item()
        return action
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_cnt += 1
        if (
            len(self.memory) >= self.batch_size
            and self.step_cnt % self.update_every == 0
        ):
            buffer_sample = self.memory.sample(beta=self.beta)
            self.learn(buffer_sample)
    
    def learn(self, buffer_sample):
        experiences, indices, IS_weights = buffer_sample
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            q_next = self.qn_target(next_states)
            y = rewards + self.gamma * (1 - dones) * q_next.max(dim=1).values.unsqueeze(1)

        q_pred = self.qn_local(states).gather(1, actions.long())
        td_error = y - q_pred
        loss = (F.mse_loss(q_pred, y, reduction='none') * IS_weights.unsqueeze(1)).mean()
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.memory.update(errors=td_error.detach().cpu().numpy().squeeze(), indices=indices)
        self.soft_update(self.qn_local, self.qn_target, self.tau)

    @staticmethod
    def soft_update(local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float) -> None:
        """Soft-update model parameters: θ_target ← τθ_local + (1−τ)θ_target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)