"""
Micro-Level RL Agent for Real-Time Energy Management
Each EnergyManager has its own agent that learns to:
- Charge battery when grid is cheap (night)
- Discharge battery when grid is expensive (peak)
- Maximize renewable usage
- Minimize grid costs through arbitrage
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import random
from collections import deque


@dataclass
class GridPricingSchedule:
    """Time-of-use grid pricing."""
    # Hours: 0-23
    off_peak_hours: Tuple[int, int] = (23, 7)  # 11 PM to 7 AM
    peak_hours: Tuple[int, int] = (17, 21)     # 5 PM to 9 PM
    shoulder_hours: Tuple[int, int] = (7, 17)  # 7 AM to 5 PM
    
    # Prices ($/kWh)
    off_peak_price: float = 0.08
    shoulder_price: float = 0.15
    peak_price: float = 0.35
    
    def get_price(self, hour: int) -> float:
        """Get grid price for given hour."""
        if self._in_range(hour, self.off_peak_hours):
            return self.off_peak_price
        elif self._in_range(hour, self.peak_hours):
            return self.peak_price
        else:
            return self.shoulder_price
    
    def _in_range(self, hour: int, range_tuple: Tuple[int, int]) -> bool:
        """Check if hour is in range (handles overnight ranges)."""
        start, end = range_tuple
        if start > end:  # Overnight range (e.g., 23, 7)
            return hour >= start or hour < end
        else:
            return start <= hour < end


class EnergyManagerNetwork(nn.Module):
    """Neural network for energy management policy and value."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Policy head (mean and log_std)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize policy head with small weights
        self.policy_mean.weight.data.mul_(0.1)
        self.policy_mean.bias.data.mul_(0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        mean = torch.tanh(self.policy_mean(features))  # Bounded to [-1, 1]
        std = torch.exp(self.policy_log_std).expand_as(mean)
        value = self.value(features)
        return mean, std, value


class EnergyManagerAgent:
    """
    RL Agent for real-time energy management within a single charging area.
    
    Observations:
    - Time features (hour, sin/cos encoding)
    - Grid price (current and predicted)
    - Solar output (current)
    - Wind output (current)
    - Battery SOC
    - Current demand
    - Recent grid usage (penalty for overuse)
    
    Actions (3 continuous values, [-1, 1]):
    - Action[0]: Battery target power (-1 = max discharge, +1 = max charge)
    - Action[1]: Grid power adjustment (-1 = minimize, +1 = use freely)
    - Action[2]: Renewable curtailment (-1 = waste excess, +1 = capture all)
    
    Reward components:
    - Negative grid cost (weighted by price)
    - Battery cycle penalty (avoid excessive cycling)
    - Shortage penalty (not meeting demand)
    - Renewable waste penalty (not using available green energy)
    """
    
    def __init__(
        self,
        agent_id: str,
        pricing_schedule: GridPricingSchedule,
        battery_capacity_kwh: float = 500.0,
        battery_max_power_kw: float = 100.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.pricing = pricing_schedule
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_power_kw = battery_max_power_kw
                
        self.device = torch.device(device)
        self.gamma = gamma
        
        # Network
        self.obs_dim = 15  # See _get_observation()
        self.action_dim = 3
        
        self.network = EnergyManagerNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training
        self.trajectory_buffer: List[Dict] = []
        self.value_estimates: List[float] = []
        
        # Statistics
        self.total_episodes = 0
        self.hourly_grid_cost: List[float] = []
        self.hourly_battery_cycles: List[float] = []
        
        # State tracking
        self.current_soc: float = 0.5
        self.last_action: Optional[np.ndarray] = None
        self.cumulative_grid_cost: float = 0.0
        
        # Hyperparameters
        self.grid_cost_weight = 1.0
        self.battery_cycle_weight = 0.1
        self.shortage_weight = 10.0
        self.renewable_waste_weight = 0.5
    
    def reset(self, initial_soc: float = 0.5):
        """Reset agent for new episode (day)."""
        self.current_soc = initial_soc
        self.last_action = None
        self.cumulative_grid_cost = 0.0
        self.trajectory_buffer.clear()
        self.value_estimates.clear()
    
    def _get_observation(
        self,
        timestamp: datetime,
        solar_output: float,
        wind_output: float,
        demand_kw: float,
        grid_price: float,
    ) -> np.ndarray:
        """Construct observation vector."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Time encoding (0-23 hours)
        obs[0] = hour / 24.0
        obs[1] = np.sin(2 * np.pi * hour / 24)  # Daily cycle
        obs[2] = np.cos(2 * np.pi * hour / 24)
        
        # Grid pricing
        obs[3] = grid_price / 0.5  # Normalize (max expected $0.50/kWh)
        obs[4] = self.pricing.get_price(int((hour + 1) % 24)) / 0.5  # Next hour price
        obs[5] = self.pricing.get_price(int((hour + 2) % 24)) / 0.5  # 2-hour ahead
        
        # Renewable generation
        obs[6] = solar_output / 500.0  # Normalize
        obs[7] = wind_output / 200.0   # Normalize
        obs[8] = (solar_output + wind_output) / 600.0  # Total renewable
        
        # Battery state
        obs[9] = self.current_soc
        obs[10] = 1.0 if self.current_soc < 0.2 else 0.0  # Low battery warning
        obs[11] = 1.0 if self.current_soc > 0.9 else 0.0  # Full battery indicator
        
        # Demand
        obs[12] = demand_kw / 1000.0  # Normalize to MW
        
        # Historical context
        obs[13] = min(1.0, self.cumulative_grid_cost / 1000.0)  # Cumulative cost penalty
        
        # Last action (for temporal consistency)
        if self.last_action is not None:
            obs[14] = np.mean(self.last_action)
        else:
            obs[14] = 0.0
        
        return obs
    
    def select_action(
        self,
        timestamp: datetime,
        solar_output: float,
        wind_output: float,
        demand_kw: float,
        grid_price: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given current state.
        
        Returns:
            action: 3D action vector
            log_prob: Log probability of action
            value: Estimated value
        """
        if grid_price is None:
            grid_price = self.pricing.get_price(timestamp.hour)
        
        obs = self._get_observation(timestamp, solar_output, wind_output, demand_kw, grid_price)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std, value = self.network(obs_tensor)
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
            
            # Compute log probability
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum().item()
            action_np = action.cpu().numpy()[0]
            value_np = value.item()
        
        self.last_action = action_np
        return action_np, log_prob, value_np
    
    def interpret_action(
        self,
        action: np.ndarray,
        solar_output: float,
        wind_output: float,
        demand_kw: float,
    ) -> Dict[str, float]:
        """
        Convert neural network action to energy flows.
        
        Action interpretation:
        - action[0]: Battery target (-1 = max discharge, +1 = max charge)
        - action[1]: Grid preference (-1 = minimize grid, +1 = use grid freely)
        - action[2]: Renewable priority (-1 = curtail if excess, +1 = always use)
        """
        # Battery power target
        # -1 -> discharge at max rate
        # 0 -> hold
        # +1 -> charge at max rate
        battery_target_kw = action[0] * self.battery_max_power_kw
        
        # Grid preference affects how much we rely on grid vs battery
        # -1 -> only use grid if battery empty
        # 0 -> balanced
        # +1 -> use grid freely (don't discharge battery)
        grid_preference = action[1]
        
        # Renewable priority
        # -1 -> can curtail if excess
        # +1 -> must use all renewable
        renewable_priority = action[2]
        
        # Calculate flows
        renewable_available = solar_output + wind_output
        
        # Step 1: Use renewable to meet demand
        renewable_to_demand = min(renewable_available, demand_kw)
        remaining_demand = demand_kw - renewable_to_demand
        
        # Step 2: Decide battery action
        battery_power = 0.0  # Positive = charging, Negative = discharging
        
        if battery_target_kw > 0:  # Want to charge
            # Charge from excess renewable first
            excess_renewable = renewable_available - renewable_to_demand
            charge_from_renewable = min(battery_target_kw, excess_renewable)
            battery_power += charge_from_renewable
            
            # Charge remainder from grid if grid_preference allows
            remaining_charge = battery_target_kw - charge_from_renewable
            if remaining_charge > 0 and grid_preference > -0.5:
                # Can charge from grid
                battery_power += remaining_charge
        
        elif battery_target_kw < 0:  # Want to discharge
            # Discharge to meet remaining demand
            discharge_needed = min(-battery_target_kw, remaining_demand)
            battery_power -= discharge_needed
            remaining_demand -= discharge_needed
        
        # Step 3: Grid usage
        # Use grid for remaining demand, adjusted by preference
        if remaining_demand > 0:
            if grid_preference < -0.5 and battery_power >= 0:
                # Try to discharge battery more if available
                extra_discharge = min(
                    remaining_demand,
                    (self.current_soc - 0.1) * self.battery_capacity_kwh,  # Don't go below 10%
                    self.battery_max_power_kw + battery_power  # Respect power limit
                )
                battery_power -= extra_discharge
                remaining_demand -= extra_discharge
            
            grid_power = remaining_demand
        else:
            grid_power = 0.0
        
        # Step 4: Handle excess renewable
        excess_renewable = renewable_available - renewable_to_demand
        if excess_renewable > 0:
            if renewable_priority > 0 or battery_power > 0:
                # Try to store excess in battery
                charge_capacity = self.battery_max_power_kw - max(0, battery_power)
                charge_amount = min(excess_renewable, charge_capacity)
                battery_power += charge_amount
                excess_renewable -= charge_amount
            
            # Any remaining excess is curtailed (wasted)
            curtailed = excess_renewable if renewable_priority < 0 else 0.0
        else:
            curtailed = 0.0
        
        return {
            'renewable_to_demand': renewable_to_demand,
            'grid_power': grid_power,
            'battery_power': battery_power,  # Positive = charging
            'curtailed_renewable': curtailed,
            'total_demand_met': renewable_to_demand + max(0, -battery_power) + grid_power,
            'demand_shortfall': max(0, demand_kw - (renewable_to_demand + max(0, -battery_power) + grid_power)),
        }
    
    def compute_reward(
        self,
        energy_flows: Dict[str, float],
        grid_price: float,
        timestamp: datetime,
    ) -> float:
        """
        Compute reward for energy management decision.
        """
        # Component 1: Grid cost (negative reward)
        grid_cost = energy_flows['grid_power'] * grid_price
        self.cumulative_grid_cost += grid_cost
        grid_reward = -grid_cost * self.grid_cost_weight
        
        # Component 2: Time-of-use arbitrage bonus
        # Extra reward for using battery during peak, charging during off-peak
        hour = timestamp.hour
        battery_power = energy_flows['battery_power']
        
        arbitrage_bonus = 0.0
        if self.pricing._in_range(hour, self.pricing.peak_hours) and battery_power < 0:
            # Discharging during peak - good!
            arbitrage_bonus = abs(battery_power) * 0.05
        elif self.pricing._in_range(hour, self.pricing.off_peak_hours) and battery_power > 0:
            # Charging during off-peak - good!
            arbitrage_bonus = battery_power * 0.03
        
        # Component 3: Battery cycling penalty (avoid excessive cycling)
        cycle_penalty = -abs(battery_power) * self.battery_cycle_weight * 0.01
        
        # Component 4: Shortage penalty (critical)
        shortage_penalty = -energy_flows['demand_shortfall'] * self.shortage_weight
        
        # Component 5: Renewable waste penalty
        waste_penalty = -energy_flows['curtailed_renewable'] * self.renewable_waste_weight * 0.1
        
        total_reward = grid_reward + arbitrage_bonus + cycle_penalty + shortage_penalty + waste_penalty
        
        return total_reward
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store transition for training."""
        self.trajectory_buffer.append({
            'obs': observation,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done,
        })
    
    def update(self, n_epochs: int = 5, batch_size: int = 32) -> Dict[str, float]:
        """
        Update policy using collected trajectories (PPO).
        """
        if len(self.trajectory_buffer) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Prepare data
        observations = torch.FloatTensor([t['obs'] for t in self.trajectory_buffer]).to(self.device)
        actions = torch.FloatTensor([t['action'] for t in self.trajectory_buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.trajectory_buffer]).to(self.device)
        rewards = [t['reward'] for t in self.trajectory_buffer]
        values = [t['value'] for t in self.trajectory_buffer]
        dones = [t['done'] for t in self.trajectory_buffer]
        
        # Compute returns and advantages (simple GAE)
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_updates = 0
        
        for _ in range(n_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(self.trajectory_buffer))
            
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_idx = indices[start:end]
                
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Forward pass
                mean, std, values = self.network(batch_obs)
                dist = Normal(mean, std)
                
                # New log probs
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1
        
        # Clear buffer
        self.trajectory_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cumulative_cost': self.cumulative_grid_cost,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.cumulative_grid_cost = checkpoint.get('cumulative_cost', 0.0)