"""
PPO implementation for continuous control.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous actions.

    Actor: obs -> Gaussian distribution over actions
    Critic: obs -> value estimate (scalar)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()

        # Build shared backbone dynamically
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor head
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs):
        features = self.shared(obs)

        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        value = self.critic(features)

        return mean, std, value
    
    def get_action(self, obs):

        mean, std, value = self.forward(obs)
        action_dist = Normal(mean, std)
        action = action_dist.sample()
        # need to sum over action dimensions to get total log prob for 1 action
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
        

