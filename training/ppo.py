"""
PPO implementation for continuous control.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from configs.ppo_config import PPOConfig
import torch.nn.functional as F
import wandb

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

    def normalize_advantages(self):
      adv = self.advantages[:self.curr_ptr]
      self.advantages[:self.curr_ptr] = (adv - adv.mean()) / (adv.std() + 1e-8)
    

    def __len__(self):
        return self.curr_ptr

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx], self.dones[idx], self.log_probs[idx], self.values[idx]

class PPOTrainer:
    def __init__(self, config, env, device="cuda",):

        self.config = config
        self.env = env
        self.device = device

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.policy = ActorCritic(obs_dim, action_dim, config.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.rollout_buffer = RolloutBuffer(config.rollout_steps, obs_dim, action_dim, device)

    def collect_rollouts(self):
        self.rollout_buffer.clear()

        episode_reward = 0
        episode_rewards = []

        obs, _ = self.env.reset()
        with torch.no_grad():
            for step in range(self.config.rollout_steps):
                
                obs = torch.from_numpy(obs).float().to(self.device)
                action, log_prob, value = self.policy.get_action(obs)
                obs_new, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = float(truncated or terminated)

                episode_reward += reward

                self.rollout_buffer.add(obs, action, reward, done, log_prob, value)

                obs = obs_new
                if done:
                    obs, _ = self.env.reset()
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    continue
        
            _, _, last_value = self.policy.get_action(torch.from_numpy(obs).float().to(self.device))
            if done:
                last_value = 0
            
            self.rollout_buffer.compute_advantages(self.config.gamma, self.config.gae_lambda, last_value)
        
        return episode_rewards
    
    def update(self):
        # ppo update for n epochs on the collected rollouts

        total_policy_loss = 0
        total_value_loss = 0
        num_updates = 0

        # This is non trivial
        self.rollout_buffer.normalize_advantages()

        for epoch in range(self.config.n_epochs):
            for batch in self.rollout_buffer.get_batches(self.config.batch_size):
                obs, actions, rewards, dones, log_probs, advantages, returns = batch

                # current policy outputs
                mean, std, values = self.policy(obs)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)

                # policy loss
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                total_policy_loss += policy_loss.item()
                # value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                total_value_loss += value_loss.item()
                # loss
                loss = policy_loss + self.config.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_updates += 1
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
        }

    def train(self, total_timesteps, project_name="ppo-robosuite"):

        wandb.init(project=project_name, config=self.config)

        num_iterations = total_timesteps // self.config.rollout_steps
        for iteration in range(num_iterations):
            episode_rewards = self.collect_rollouts()
            losses = self.update()

            timesteps = (iteration + 1) * self.config.rollout_steps

            # Log to wandb (handle empty episode_rewards)
            log_dict = {
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "timesteps": timesteps,
            }
            if len(episode_rewards) > 0:
                log_dict["episode_reward_mean"] = np.mean(episode_rewards)
                log_dict["episode_reward_min"] = np.min(episode_rewards)
                log_dict["episode_reward_max"] = np.max(episode_rewards)
                log_dict["episodes_this_iter"] = len(episode_rewards)

            wandb.log(log_dict)

            # Console logging every 10 iterations
            if iteration % 10 == 0:
                if len(episode_rewards) > 0:
                    print(f"Iter {iteration}/{num_iterations} | Timesteps: {timesteps} | "
                          f"Avg reward: {np.mean(episode_rewards):.2f} | "
                          f"Policy loss: {losses['policy_loss']:.4f} | "
                          f"Value loss: {losses['value_loss']:.4f}")
                else:
                    print(f"Iter {iteration}/{num_iterations} | Timesteps: {timesteps} | "
                          f"No episodes completed | "
                          f"Policy loss: {losses['policy_loss']:.4f} | "
                          f"Value loss: {losses['value_loss']:.4f}")

        wandb.finish()

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
