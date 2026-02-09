"""
PPO Agent Implementation
Proximal Policy Optimization for EV charging operational control
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import gymnasium as gym

from optimization_config import PPOConfig, DEFAULT_PPO_CONFIG


class ActorNetwork(nn.Module):
    """Policy network (actor)"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        mean = torch.tanh(self.mean_head(features))  # Bounded actions
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), log_prob, mean


class CriticNetwork(nn.Module):
    """Value network (critic)"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int], activation: str = "relu"):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PPOAgent:
    """
    PPO Agent for EV charging control
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: PPOConfig = DEFAULT_PPO_CONFIG):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = ActorNetwork(
            state_dim, action_dim, 
            config.actor_hidden_dims, 
            config.activation
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim, 
            config.critic_hidden_dims, 
            config.activation
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.learning_rate_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=config.learning_rate_critic
        )
        
        # Learning rate schedulers
        if config.lr_decay:
            self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
                self.actor_optimizer, gamma=0.995
            )
            self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
                self.critic_optimizer, gamma=0.995
            )
        
        # Memory buffers
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[bool] = []
        
        # Training stats
        self.training_step = 0
    
    def select_action(self, state: Dict, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state dictionary
        
        Returns: (action, value, log_prob)
        """
        # Flatten state dict to vector
        state_vec = self._flatten_state(state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor).item()
        
        return action, value, log_prob.item()
    
    def _flatten_state(self, state: Dict) -> np.ndarray:
        """Flatten state dictionary to vector"""
        # Concatenate all state components
        components = [
            state['time'].flatten(),
            state['stations'].flatten(),
            state['vehicles'].flatten(),
            state['energy'].flatten(),
            state['system'].flatten()
        ]
        return np.concatenate(components)
    
    def store_transition(self, state: Dict, action: np.ndarray, 
                        reward: float, value: float, 
                        log_prob: float, done: bool):
        """Store transition in memory"""
        self.states.append(self._flatten_state(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(torch.tensor(log_prob))
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        """
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + self.config.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return advantages, returns
    
    def update(self, next_state: Optional[Dict] = None) -> Dict:
        """
        Perform PPO update
        """
        if len(self.states) == 0:
            return {}
        
        # Get next value for GAE
        if next_state is not None:
            next_state_vec = self._flatten_state(next_state)
            next_state_tensor = torch.FloatTensor(next_state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).item()
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for _ in range(self.config.epochs_per_update):
            # Generate random minibatches
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Evaluate actions with current policy
                mean, std = self.actor(mb_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.critic(mb_states)
                critic_loss = nn.MSELoss()(values, mb_returns)
                
                # Combined loss
                loss = (actor_loss 
                       + self.config.value_coef * critic_loss 
                       - self.config.entropy_coef * entropy)
                
                # Update
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
        
        # Clear memory
        self.clear_memory()
        
        # Step learning rate schedulers
        if self.config.lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        self.training_step += 1
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'advantage_mean': advantages.mean().item(),
            'return_mean': returns.mean().item()
        }
    
    def clear_memory(self):
        """Clear rollout memory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']