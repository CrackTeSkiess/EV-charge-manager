"""
Early Warning System
Extracts interpretable warning indicators from trained RL policies
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from ppo_agent import PPOAgent
from rl_environment import EVChargingRLEnv
from optimization_config import EarlyWarningConfig, DEFAULT_EW_CONFIG


@dataclass
class WarningEvent:
    """Early warning event record"""
    timestamp: float  # Simulation step
    warning_level: str  # 'GREEN', 'YELLOW', 'RED'
    rews_score: float  # Resilience Early Warning Score
    primary_indicator: str  # Which indicator triggered
    contributing_factors: Dict[str, float]
    recommended_action: str


class EarlyWarningExtractor:
    """
    Extracts early warning signals from trained RL policies
    """
    
    def __init__(self, 
                 agent: PPOAgent,
                 config: EarlyWarningConfig = DEFAULT_EW_CONFIG):
        self.agent = agent
        self.config = config
        
        # History buffers for gradient calculation
        self.value_history: deque = deque(maxlen=config.gradient_window_size)
        self.action_history: deque = deque(maxlen=config.gradient_window_size)
        self.state_history: deque = deque(maxlen=config.gradient_window_size)
        
        # Attention tracking (if using attention mechanism)
        self.attention_weights: List[np.ndarray] = []
        
        # Warning log
        self.warnings: List[WarningEvent] = []
    
    def analyze_step(self, 
                    state: Dict,
                    action: np.ndarray,
                    value: float) -> Optional[WarningEvent]:
        """
        Analyze current step for warning signs
        
        Returns: WarningEvent if warning triggered, None otherwise
        """
        # Store history
        self.value_history.append(value)
        self.action_history.append(action)
        self.state_history.append(state)
        
        # Calculate indicators
        indicators = {}
        
        # 1. Value gradient indicator
        grad_indicator = self._compute_value_gradient()
        indicators['value_gradient'] = grad_indicator
        
        # 2. Action sensitivity indicator
        action_indicator = self._compute_action_sensitivity()
        indicators['action_change'] = action_indicator
        
        # 3. System stress indicator
        stress_indicator = self._compute_system_stress(state)
        indicators['system_stress'] = stress_indicator
        
        # 4. Attention-based indicator (if available)
        attention_indicator = self._compute_attention_indicator(state)
        indicators['attention'] = attention_indicator
        
        # Compute composite REWS score
        rews_score = (
            self.config.rews_gradient_weight * grad_indicator +
            self.config.rews_action_weight * action_indicator +
            self.config.rews_attention_weight * attention_indicator
        )
        
        # Determine warning level
        if rews_score >= self.config.rews_red_threshold:
            level = 'RED'
        elif rews_score >= self.config.rews_yellow_threshold:
            level = 'YELLOW'
        else:
            level = 'GREEN'
        
        # Determine primary indicator
        primary = max(indicators, key=indicators.get) # pyright: ignore[reportArgumentType, reportCallIssue]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(primary, state)
        
        # Create warning if not green
        if level != 'GREEN':
            warning = WarningEvent(
                timestamp=state.get('system', [0,0,0,0])[3],  # Progress indicator
                warning_level=level,
                rews_score=rews_score,
                primary_indicator=primary,
                contributing_factors=indicators,
                recommended_action=recommendation
            )
            self.warnings.append(warning)
            return warning
        
        return None
    
    def _compute_value_gradient(self) -> float:
        """Compute normalized value function gradient"""
        if len(self.value_history) < 2:
            return 0.0
        
        values = list(self.value_history)
        # Compute rate of change
        gradients = np.diff(values)
        avg_gradient = np.mean(np.abs(gradients))
        
        # Normalize by value magnitude
        normalized = avg_gradient / (np.abs(values[-1]) + 1e-8)
        
        # Apply threshold from config
        indicator = min(normalized / self.config.value_gradient_threshold, 1.0)
        
        return indicator
    
    def _compute_action_sensitivity(self) -> float:
        """Compute action change sensitivity"""
        if len(self.action_history) < 2:
            return 0.0
        
        actions = np.array(self.action_history)
        # Compute L2 distance between consecutive actions
        diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        avg_diff = np.mean(diffs)
        
        # Normalize
        indicator = min(avg_diff / self.config.action_change_threshold, 1.0)
        
        return indicator
    
    def _compute_system_stress(self, state: Dict) -> float:
        """Compute composite system stress indicator"""
        stress_signals = []
        
        # Queue stress
        stations = state.get('stations', np.zeros((5, 6)))
        avg_queue = np.mean(stations[:, 0])  # Queue occupancy
        stress_signals.append(avg_queue)
        
        # Battery stress (low SOC)
        battery_soc = np.mean(stations[:, 1])
        stress_signals.append(1.0 - battery_soc)  # Inverse: low SOC = high stress
        
        # Energy shortage flag
        shortage_flags = stations[:, 4]
        stress_signals.append(np.mean(shortage_flags))
        
        # Demand-supply imbalance
        system = state.get('system', np.zeros(4))
        demand = system[0]
        supply = system[1]
        if supply > 0:
            imbalance = max(0, (demand - supply) / supply)
            stress_signals.append(min(imbalance, 1.0))
        
        return np.mean(stress_signals) # pyright: ignore[reportReturnType]
    
    def _compute_attention_indicator(self, state: Dict) -> float:
        """
        Compute attention-based indicator
        Would extract from attention mechanism if using transformer architecture
        """
        # Placeholder - would implement if using attention-based actor
        # For now, use variance of vehicle state attention
        
        vehicles = state.get('vehicles', np.zeros((100, 4)))
        mask = state.get('mask', np.zeros(100))
        
        # Focus on high-urgency vehicles (in queue with low SOC)
        if np.sum(mask) > 0:
            # Vehicle features: [soc, dist, urgency, status]
            urgency = vehicles[:, 2] * mask
            max_urgency = np.max(urgency)
            return max_urgency
        return 0.0
    
    def _generate_recommendation(self, primary_indicator: str, state: Dict) -> str:
        """Generate human-readable recommendation"""
        recommendations = {
            'value_gradient': "CRITICAL: Rapid value function decline detected. "
                            "Initiate emergency load shedding and activate all battery reserves.",
            
            'action_change': "WARNING: Policy instability detected. "
                           "Grid fluctuations or demand surge likely. "
                           "Monitor renewable output and prepare demand response.",
            
            'system_stress': "ELEVATED: System operating near capacity. "
                           "Consider queue management and dynamic pricing to reduce demand.",
            
            'attention': "CAUTION: High-priority vehicles requiring immediate attention. "
                        "Ensure sufficient charging power allocation to at-risk vehicles."
        }
        
        return recommendations.get(primary_indicator, "Monitor system status closely.")
    
    def extract_policy_insights(self, env: EVChargingRLEnv, n_episodes: int = 10) -> Dict:
        """
        Run analysis episodes and extract policy insights
        """
        all_warnings = []
        state_action_map = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_warnings = []
            
            while True:
                action, value, _ = self.agent.select_action(obs)
                
                # Analyze for warnings
                warning = self.analyze_step(obs, action, value)
                if warning:
                    episode_warnings.append(warning)
                
                # Store state-action pair for analysis
                state_action_map.append({
                    'state_summary': self._summarize_state(obs),
                    'action_summary': self._summarize_action(action),
                    'value': value
                })
                
                next_obs, _, terminated, truncated, info = env.step(action)
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            all_warnings.extend(episode_warnings)
        
        # Compile insights
        insights = {
            'total_warnings': len(all_warnings),
            'warning_breakdown': self._count_warnings_by_level(all_warnings),
            'primary_indicators': self._count_primary_indicators(all_warnings),
            'temporal_pattern': self._analyze_temporal_pattern(all_warnings),
            'state_action_patterns': self._analyze_patterns(state_action_map),
            'recommended_thresholds': self._suggest_thresholds(all_warnings)
        }
        
        return insights
    
    def _summarize_state(self, state: Dict) -> Dict:
        """Create compact state summary"""
        return {
            'hour': np.argmax(state['time'][:2]),  # Rough hour estimate
            'avg_queue': np.mean(state['stations'][:, 0]),
            'avg_battery_soc': np.mean(state['stations'][:, 1]),
            'n_active_vehicles': int(state['system'][2])
        }
    
    def _summarize_action(self, action: np.ndarray) -> Dict:
        """Create compact action summary"""
        n_stations = len(action) // 10  # Approximate
        return {
            'avg_grid_fraction': np.mean(action[::10]),  # Every 10th is grid
            'avg_battery_action': np.mean(action[1::10]),  # Battery actions
            'action_variance': np.var(action)
        }
    
    def _count_warnings_by_level(self, warnings: List[WarningEvent]) -> Dict[str, int]:
        """Count warnings by severity level"""
        counts = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        for w in warnings:
            counts[w.warning_level] = counts.get(w.warning_level, 0) + 1
        return counts
    
    def _count_primary_indicators(self, warnings: List[WarningEvent]) -> Dict[str, int]:
        """Count which indicators trigger most warnings"""
        counts = {}
        for w in warnings:
            counts[w.primary_indicator] = counts.get(w.primary_indicator, 0) + 1
        return counts
    
    def _analyze_temporal_pattern(self, warnings: List[WarningEvent]) -> Dict:
        """Analyze when warnings occur"""
        if not warnings:
            return {}
        
        times = [w.timestamp for w in warnings]
        return {
            'mean_time': np.mean(times),
            'warning_density': len(warnings) / (max(times) - min(times) + 1e-8)
        }
    
    def _analyze_patterns(self, state_action_map: List[Dict]) -> Dict:
        """Analyze state-action patterns"""
        # Simplified pattern analysis
        high_value_states = [s for s in state_action_map if s['value'] > np.percentile([x['value'] for x in state_action_map], 90)]
        
        return {
            'high_value_state_commonalities': self._find_commonalities(high_value_states),
            'action_variance_by_stress': self._correlate_action_variance(state_action_map)
        }
    
    def _find_commonalities(self, states: List[Dict]) -> Dict:
        """Find common features in high-value states"""
        if not states:
            return {}
        
        queues = [s['state_summary']['avg_queue'] for s in states]
        batteries = [s['state_summary']['avg_battery_soc'] for s in states]
        
        return {
            'typical_queue_occupancy': np.mean(queues),
            'typical_battery_soc': np.mean(batteries)
        }
    
    def _correlate_action_variance(self, state_action_map: List[Dict]) -> Dict:
        """Correlate action variance with system stress"""
        # Simplified correlation
        variances = [s['action_summary']['action_variance'] for s in state_action_map]
        return {
            'mean_action_variance': np.mean(variances),
            'high_variance_frequency': np.mean([v > 0.5 for v in variances])
        }
    
    def _suggest_thresholds(self, warnings: List[WarningEvent]) -> Dict:
        """Suggest optimized warning thresholds based on data"""
        if not warnings:
            return {}
        
        rews_scores = [w.rews_score for w in warnings]
        
        return {
            'suggested_yellow': np.percentile(rews_scores, 50),
            'suggested_red': np.percentile(rews_scores, 80),
            'current_config': {
                'yellow': self.config.rews_yellow_threshold,
                'red': self.config.rews_red_threshold
            }
        }
    
    def generate_report(self, insights: Dict) -> str:
        """Generate human-readable report"""
        report = []
        report.append("=" * 60)
        report.append("EARLY WARNING SYSTEM ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"\nTotal Warnings Generated: {insights['total_warnings']}")
        
        report.append("\nWarning Severity Distribution:")
        for level, count in insights['warning_breakdown'].items():
            report.append(f"  {level}: {count}")
        
        report.append("\nPrimary Indicators:")
        for indicator, count in insights['primary_indicators'].items():
            report.append(f"  {indicator}: {count}")
        
        report.append("\nRecommended Threshold Adjustments:")
        suggestions = insights['recommended_thresholds']
        report.append(f"  Suggested YELLOW: {suggestions['suggested_yellow']:.3f}")
        report.append(f"  Suggested RED: {suggestions['suggested_red']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)