"""
Multi-Agent PPO Trainer for Charging Area Optimization
Uses Centralized Training with Decentralized Execution (CTDE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from collections import deque

from ChargingAreaEnv import MultiAgentChargingEnv


class ActorNetwork(nn.Module):
    """Policy network for each agent (decentralized execution)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize with small weights for stable training
        self.mean_head.weight.data.mul_(0.01)
        self.mean_head.bias.data.mul_(0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = torch.tanh(self.mean_head(features))  # Bound actions to [-1, 1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample action from policy."""
        with torch.no_grad():
            mean, std = self.forward(obs)
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
            
            # Compute log probability
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.cpu().numpy(), log_prob


class CriticNetwork(nn.Module):
    """Centralized value network (uses global state)."""
    
    def __init__(self, total_obs_dim: int, n_agents: int, hidden_dim: int = 512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Output value for each agent
        self.value_head = nn.Linear(hidden_dim // 2, n_agents)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimates for all agents."""
        features = self.net(global_obs)
        values = self.value_head(features)
        return values


class MultiAgentPPO:
    """
    Multi-Agent PPO with CTDE architecture.
    """
    
    def __init__(
        self,
        env: MultiAgentChargingEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.env = env
        self.n_agents = env.n_agents
        
        # Auto-select device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Get dimensions from environment
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        
        if obs_shape is None or len(obs_shape) < 2:
            raise ValueError(f"Invalid observation_space shape: {obs_shape}")
        if action_shape is None or len(action_shape) < 2:
            raise ValueError(f"Invalid action_space shape: {action_shape}")
        
        obs_dim_per_agent = obs_shape[1]  # Per agent obs dim
        action_dim_per_agent = action_shape[1]  # Per agent action dim
        total_obs_dim = obs_shape[0] * obs_shape[1]  # Global obs dim (n_agents * obs_dim)
        
        print(f"Observation dim per agent: {obs_dim_per_agent}")
        print(f"Action dim per agent: {action_dim_per_agent}")
        print(f"Total observation dim: {total_obs_dim}")
        
        # Networks
        self.actors = [
            ActorNetwork(obs_dim_per_agent, action_dim_per_agent).to(self.device)
            for _ in range(self.n_agents)
        ]
        
        self.critic = CriticNetwork(total_obs_dim, self.n_agents).to(self.device)
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Training statistics - use lists instead of deque for slicing support
        self.episode_rewards: List[float] = []
        self.episode_costs: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []
        
        # Keep last N values for moving averages
        self.max_history = 100
        
        # Best model tracking
        self.best_avg_reward = float('-inf')
    
    def _add_to_history(self, value: float, history_list: List[float]):
        """Add value to history list with max length."""
        history_list.append(value)
        if len(history_list) > self.max_history:
            history_list.pop(0)
    
    def collect_trajectories(self, n_episodes: int) -> Dict:
        """
        Collect trajectories by running episodes.
        """
        trajectories = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': [],
        }
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            
            episode_obs = [obs]
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            
            while not done:
                # Get actions from all actors
                actions = []
                log_probs = []
                
                for i, actor in enumerate(self.actors):
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                    action, log_prob = actor.get_action(obs_tensor)
                    actions.append(action[0])
                    log_probs.append(log_prob.item())
                
                actions = np.array(actions)
                
                # Get value estimates from centralized critic
                global_obs = obs.flatten()
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    values = self.critic(global_obs_tensor).cpu().numpy()[0]
                
                # Step environment
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # Store transition
                episode_actions.append(actions)
                episode_rewards.append(rewards)
                episode_values.append(values)
                episode_log_probs.append(log_probs)
                episode_dones.append([done] * self.n_agents)
                
                obs = next_obs
            
            # Store episode data
            trajectories['observations'].append(np.array(episode_obs))
            trajectories['actions'].append(np.array(episode_actions))
            trajectories['rewards'].append(np.array(episode_rewards))
            trajectories['values'].append(np.array(episode_values))
            trajectories['log_probs'].append(np.array(episode_log_probs))
            trajectories['dones'].append(np.array(episode_dones))
            trajectories['next_observations'].append(obs)
            
            # Track statistics
            total_reward = np.sum(episode_rewards)
            self._add_to_history(total_reward, self.episode_rewards)
            self._add_to_history(info.get('total_cost', 0), self.episode_costs)
            
            if (ep + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-min(10, len(self.episode_rewards)):]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {ep+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Cost: {info.get('total_cost', 0):.0f}")
        
        return trajectories
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for agent_idx in range(self.n_agents):
            agent_rewards = rewards[:, agent_idx]
            agent_values = values[:, agent_idx]
            agent_dones = dones[:, agent_idx]
            
            for t in reversed(range(len(agent_rewards))):
                if t == len(agent_rewards) - 1:
                    next_value = 0
                else:
                    next_value = agent_values[t + 1]
                
                delta = agent_rewards[t] + self.gamma * next_value * (1 - agent_dones[t]) - agent_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - agent_dones[t]) * last_gae
                advantages[t, agent_idx] = last_gae
            
            last_gae = 0
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, trajectories: Dict, n_epochs: int = 10, batch_size: int = 32) -> Dict:
        """
        Update policy and value function using PPO.
        """
        # Prepare data
        all_obs = np.concatenate(trajectories['observations'])
        all_actions = np.concatenate(trajectories['actions'])
        all_rewards = np.concatenate(trajectories['rewards'])
        all_values = np.concatenate(trajectories['values'])
        all_log_probs = np.concatenate(trajectories['log_probs'])
        all_dones = np.concatenate(trajectories['dones'])
        
        advantages, returns = self.compute_gae(all_rewards, all_values, all_dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        obs_tensor = torch.FloatTensor(all_obs).to(self.device)
        actions_tensor = torch.FloatTensor(all_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        global_obs = all_obs.reshape(all_obs.shape[0], -1)
        global_obs_tensor = torch.FloatTensor(global_obs).to(self.device)
        
        n_samples = len(all_obs)
        indices = np.arange(n_samples)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_global_obs = global_obs_tensor[batch_idx]
                
                actor_losses = []
                for agent_idx, actor in enumerate(self.actors):
                    mean, std = actor(batch_obs[:, agent_idx])
                    dist = Normal(mean, std)
                    
                    new_log_probs = dist.log_prob(batch_actions[:, agent_idx]).sum(dim=-1)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs[:, agent_idx])
                    
                    surr1 = ratio * batch_advantages[:, agent_idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages[:, agent_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    entropy = dist.entropy().mean()
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    
                    self.actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    self.actor_optimizers[agent_idx].step()
                    
                    actor_losses.append(policy_loss.item())
                    total_entropy += entropy.item()
                
                predicted_values = self.critic(batch_global_obs)
                value_loss = nn.MSELoss()(predicted_values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_policy_loss += np.mean(actor_losses)
                total_value_loss += value_loss.item()
                n_updates += 1
        
        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / (n_updates * self.n_agents),
        }
        
        self.policy_losses.append(metrics['policy_loss'])
        self.value_losses.append(metrics['value_loss'])
        self.entropy_losses.append(metrics['entropy'])
        
        return metrics
    
    def train(
        self,
        total_episodes: int = 1000,
        episodes_per_update: int = 10,
        n_epochs: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50,
    ):
        """
        Main training loop.
        """
        print(f"\nStarting training for {total_episodes} episodes")
        print(f"Agents: {self.n_agents}, Device: {self.device}")
        
        for update_idx in range(0, total_episodes, episodes_per_update):
            print(f"\n--- Update {update_idx // episodes_per_update + 1} ---")
            trajectories = self.collect_trajectories(episodes_per_update)
            
            metrics = self.update(trajectories, n_epochs=n_epochs)
            
            # Calculate averages using list slicing
            recent_rewards = self.episode_rewards[-episodes_per_update:]
            recent_costs = self.episode_costs[-episodes_per_update:]
            
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_cost = np.mean(recent_costs) if recent_costs else 0.0
            
            print(f"Avg Reward: {avg_reward:.2f}, Avg Cost: {avg_cost:.0f}")
            print(f"Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, "
                  f"Entropy: {metrics['entropy']:.4f}")
            
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.save("best_model.pt")
                print(f"New best model saved! Reward: {avg_reward:.2f}")
            
            if (update_idx // episodes_per_update + 1) % max(1, eval_interval // episodes_per_update) == 0:
                self.evaluate()
            
            if (update_idx // episodes_per_update + 1) % max(1, save_interval // episodes_per_update) == 0:
                self.save(f"checkpoint_{update_idx}.pt")
        
        print("\nTraining completed!")
        self.save("final_model.pt")
    
    def evaluate(self, n_episodes: int = 5) -> Dict:
        """
        Evaluate current policy.
        """
        print("\n--- Evaluation ---")
        
        eval_rewards = []
        eval_costs = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                actions = []
                for i, actor in enumerate(self.actors):
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                    action, _ = actor.get_action(obs_tensor, deterministic=True)
                    actions.append(action[0])
                
                actions = np.array(actions)
                obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                episode_reward += np.sum(rewards)
            
            eval_rewards.append(episode_reward)
            eval_costs.append(info.get('total_cost', 0))
        
        avg_reward = np.mean(eval_rewards)
        avg_cost = np.mean(eval_costs)
        
        print(f"Eval Reward: {avg_reward:.2f} (+/- {np.std(eval_rewards):.2f})")
        print(f"Eval Cost: {avg_cost:.0f} (+/- {np.std(eval_costs):.0f})")
        
        best = self.env.get_best_config()
        if best:
            print(f"\nBest configuration found:")
            print(f"  Total cost: ${best['cost']:,.2f}")
            for i in range(self.n_agents):
                print(f"  Station {i+1}: {best['positions'][i]:.1f}km, "
                      f"{best['n_chargers'][i]} chargers, "
                      f"{best['n_waiting'][i]} waiting spots")
        
        return {
            'avg_reward': avg_reward,
            'avg_cost': avg_cost,
            'best_config': best,
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'best_avg_reward': self.best_avg_reward,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
        print(f"Model loaded from {path}")