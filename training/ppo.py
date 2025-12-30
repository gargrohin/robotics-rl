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
        

class RolloutBuffer:
    def  __init__(self, buffer_size, obs_dim, action_dim, device):
        self.observations = torch.zeros(buffer_size, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)


        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        self.curr_ptr = 0
        self.buffer_size = buffer_size

    def add(self, obs, action, reward, done, log_prob, value):

        self.observations[self.curr_ptr] = obs
        self.actions[self.curr_ptr] = action
        self.rewards[self.curr_ptr] = reward
        self.dones[self.curr_ptr] = done
        self.log_probs[self.curr_ptr] = log_prob
        self.values[self.curr_ptr] = value

        self.curr_ptr += 1
    
    def compute_advantages(self, gamma, lambda_, last_value):

        gae = 0 # running accumulator


        for t in reversed(range(self.curr_ptr)):
            if t == self.curr_ptr - 1:
                next_value = last_value * (1 - self.dones[t])
                delta = self.rewards[t] + gamma*next_value - self.values[t]
            else:
                next_value = self.values[t+1] * (1 - self.dones[t])
                delta = self.rewards[t] + gamma*next_value - self.values[t]
            
            gae = delta + gamma*lambda_*gae * (1 - self.dones[t])
            self.advantages[t] = gae
        
        self.returns = self.advantages + self.values
    
    def clear(self):
        self.curr_ptr = 0
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.advantages.zero_()
        self.returns.zero_()
    
    def get_batches(self, batch_size):
        # Yield mini batches for PPO update
        indices = np.random.permutation(self.curr_ptr)

        for start in range(0, self.curr_ptr, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield (
                self.observations[batch_indices], 
                self.actions[batch_indices], 
                self.rewards[batch_indices], 
                self.dones[batch_indices], 
                self.log_probs[batch_indices], 
                self.advantages[batch_indices], 
                self.returns[batch_indices]
            )

    def __len__(self):
        return self.curr_ptr

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx], self.dones[idx], self.log_probs[idx], self.values[idx]
    

