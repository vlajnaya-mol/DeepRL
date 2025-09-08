"""Actor–Critic network definitions for DDPG-style agents.

This module provides:
- Actor: maps state -> action (tanh output in [-1, 1]).
- Critic: maps (state, action) -> scalar Q-value.

Weight initialization:
- Hidden layers: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
- Final layers: U(-final_init, +final_init)
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Actor", "Critic"]


def hidden_init(layer: nn.Linear) -> Tuple[float, float]:
    """Return symmetric uniform init bounds based on fan-in of a Linear layer."""
    fan_in = layer.weight.data.size(1)
    lim = 1.0 / np.sqrt(float(fan_in))
    return -lim, lim


class Actor(nn.Module):
    """Actor (policy) network: state -> action in [-1, 1] via tanh."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        fc1_units: int = 256,
        fc2_units: int = 128,
        final_init: float = 3e-3,
    ) -> None:
        """
        Args:
            state_size: Dimension of input state.
            action_size: Dimension of output action.
            seed: Random seed for torch.
            fc1_units: Hidden units in first layer.
            fc2_units: Hidden units in second layer.
            final_init: Uniform bound for final layer init.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.final_init = float(final_init)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        for layer in (self.fc1, self.fc2):
            lo, hi = hidden_init(layer)
            layer.weight.data.uniform_(lo, hi)
            if layer.bias is not None:
                layer.bias.data.uniform_(lo, hi)

        self.fc3.weight.data.uniform_(-self.final_init, self.final_init)
        if self.fc3.bias is not None:
            self.fc3.bias.data.uniform_(-self.final_init, self.final_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute action from state.

        Args:
            state: Tensor of shape (N, state_size) or (state_size,).

        Returns:
            Tensor of shape (N, action_size) in [-1, 1].
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (value) network: (state, action) -> Q-value."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        fcs1_units: int = 256,
        fc2_units: int = 256,
        fc3_units: int = 128,
        final_init: float = 3e-3,
    ) -> None:
        """
        Args:
            state_size: Dimension of input state.
            action_size: Dimension of input action.
            seed: Random seed for torch.
            fcs1_units: Hidden units for state-processing layer.
            fc2_units: Hidden units after state–action concat.
            fc3_units: Hidden units before output.
            final_init: Uniform bound for final layer init.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Process state, then concatenate action
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        self.final_init = float(final_init)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        for layer in (self.fcs1, self.fc2, self.fc3):
            lo, hi = hidden_init(layer)
            layer.weight.data.uniform_(lo, hi)
            if layer.bias is not None:
                layer.bias.data.uniform_(lo, hi)

        self.fc4.weight.data.uniform_(-self.final_init, self.final_init)
        if self.fc4.bias is not None:
            self.fc4.bias.data.uniform_(-self.final_init, self.final_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Estimate Q-value for (state, action).

        Args:
            state: Tensor of shape (N, state_size) or (state_size,).
            action: Tensor of shape (N, action_size) or (action_size,).

        Returns:
            Tensor of shape (N, 1) with Q-values.
        """
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
