"""
Visualization tools for charging area optimization results
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, List
import json
from ev_charge_manager.energy import (
    EnergyManager,
    EnergyManagerConfig, 
    EnergySourceConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)


def plot_training_curves(history: Dict, save_path: str = None): # pyright: ignore[reportArgumentType]
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    ax.plot(history['episodes'], history['rewards'], alpha=0.3, label='Raw')
    # Moving average
    window = 20
    if len(history['rewards']) > window:
        ma = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
        ax.plot(history['episodes'][window-1:], ma, linewidth=2, label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True)
    
    # Episode costs
    ax = axes[0, 1]
    ax.plot(history['episodes'], history['costs'], color='red', alpha=0.3)
    if len(history['costs']) > window:
        ma = np.convolve(history['costs'], np.ones(window)/window, mode='valid')
        ax.plot(history['episodes'][window-1:], ma, color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Optimization Cost')
    ax.grid(True)
    
    # Losses
    ax = axes[1, 0]
    ax.plot(history['policy_losses'], label='Policy Loss')
    ax.plot(history['value_losses'], label='Value Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    # Entropy
    ax = axes[1, 1]
    ax.plot(history['entropy_losses'], color='green')
    ax.set_xlabel('Update')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_highway_layout(config: Dict, highway_length: float = 300.0, save_path: str = None): # pyright: ignore[reportArgumentType]
    """Visualize highway charging area layout."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [1, 2]})
    
    positions = config['positions']
    n_agents = len(positions)
    
    # Top plot: Highway schematic
    ax1.set_xlim(0, highway_length)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Distance (km)')
    ax1.set_title('Highway Charging Area Layout')
    
    # Draw highway
    ax1.plot([0, highway_length], [0.5, 0.5], 'k-', linewidth=4, label='Highway')
    
    # Draw stations
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, n_agents))
    for i, (pos, color) in enumerate(zip(positions, colors)):
        ax1.scatter(pos, 0.5, s=500, c=[color], marker='s', zorder=5, edgecolors='black')
        ax1.annotate(f'Station {i+1}\n{pos:.1f}km', 
                    xy=(pos, 0.5), xytext=(0, 20),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')
        
        # Draw spacing arrows
        if i < n_agents - 1:
            next_pos = positions[i+1]
            mid = (pos + next_pos) / 2
            ax1.annotate('', xy=(next_pos, 0.3), xytext=(pos, 0.3),
                        arrowprops=dict(arrowstyle='<->', color='gray'))
            ax1.text(mid, 0.25, f'{next_pos-pos:.1f}km', ha='center', fontsize=8, color='gray')
    
    ax1.set_yticks([])
    ax1.legend(loc='upper right')
    
    # Bottom plot: Configuration details
    ax2.axis('off')
    
    table_data = []
    headers = ['Station', 'Position (km)', 'Chargers', 'Waiting', 'Grid (kW)', 
               'Solar (kW)', 'Wind (kW)', 'Battery (kW/kWh)']
    
    for i in range(n_agents):
        cfg = config['configs'][i]
        table_data.append([
            f'{i+1}',
            f'{positions[i]:.1f}',
            f"{config['n_chargers'][i]}",
            f"{config['n_waiting'][i]}",
            f'{cfg.grid_capacity_kw:.0f}',
            f'{cfg.solar_capacity_kw:.0f}',
            f'{cfg.wind_capacity_kw:.0f}',
            f'{cfg.battery_capacity_kw:.0f} / {cfg.battery_storage_kwh:.0f}'
        ])
    
    table = ax2.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center',
                     colWidths=[0.08, 0.12, 0.1, 0.1, 0.12, 0.12, 0.12, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_energy_mix(config: Dict, save_path: str = None): # pyright: ignore[reportArgumentType]
    """Plot energy source distribution across stations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_agents = len(config['configs'])
    positions = config['positions']
    
    # Power capacity by source
    ax = axes[0]
    sources = ['Grid', 'Solar', 'Wind', 'Battery']
    x = np.arange(n_agents)
    width = 0.2
    
    grid_caps = [cfg.grid_capacity_kw for cfg in config['configs']]
    solar_caps = [cfg.solar_capacity_kw for cfg in config['configs']]
    wind_caps = [cfg.wind_capacity_kw for cfg in config['configs']]
    battery_caps = [cfg.battery_capacity_kw for cfg in config['configs']]
    
    ax.bar(x - 1.5*width, grid_caps, width, label='Grid', color='gray')
    ax.bar(x - 0.5*width, solar_caps, width, label='Solar', color='orange')
    ax.bar(x + 0.5*width, wind_caps, width, label='Wind', color='skyblue')
    ax.bar(x + 1.5*width, battery_caps, width, label='Battery', color='green')
    
    ax.set_xlabel('Station')
    ax.set_ylabel('Power Capacity (kW)')
    ax.set_title('Power Capacity by Source')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}\n({positions[i]:.0f}km)' for i in range(n_agents)])
    ax.legend()
    ax.grid(True, axis='y')
    
    # Renewable fraction
    ax = axes[1]
    renewable_fracs = []
    for cfg in config['configs']:
        total = cfg.grid_capacity_kw + cfg.solar_capacity_kw + cfg.wind_capacity_kw + cfg.battery_capacity_kw
        renewable = cfg.solar_capacity_kw + cfg.wind_capacity_kw + cfg.battery_capacity_kw
        renewable_fracs.append(renewable / total if total > 0 else 0)
    
    colors = ['green' if rf > 0.5 else 'orange' if rf > 0.25 else 'red' 
              for rf in renewable_fracs]
    bars = ax.bar(range(n_agents), renewable_fracs, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='50% renewable')
    ax.set_xlabel('Station')
    ax.set_ylabel('Renewable Fraction')
    ax.set_title('Renewable Energy Share')
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels([f'{i+1}' for i in range(n_agents)])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y')
    
    # Add value labels
    for bar, frac in zip(bars, renewable_fracs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{frac:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_summary_report(result_path: str, output_path: str = 'report.pdf'):
    """Create comprehensive PDF report."""
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Load results
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    best_config = results.get('best_config')
    
    with PdfPages(output_path) as pdf:
        # Page 1: Summary statistics
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis('off')
        
        summary_text = f"""
        CHARGING AREA OPTIMIZATION REPORT
        
        Performance Metrics:
        -------------------
        Average Reward: {np.mean(results.get('rewards', [])):.2f}
        Average Cost: ${np.mean(results.get('costs', [])):,.2f}
        Average Stranded Vehicles: {np.mean(results.get('stranded', [])):.2f}
        Average Blackout Events: {np.mean(results.get('blackouts', [])):.2f}
        
        Optimal Configuration:
        ---------------------
        Total Cost: ${best_config.get('cost', 0):,.2f}
        Number of Stations: {len(best_config.get('positions', []))}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Layout
        if best_config:
            plot_highway_layout(best_config, save_path=None) # pyright: ignore[reportArgumentType]
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # Page 3: Energy mix
            plot_energy_mix(best_config, save_path=None) # pyright: ignore[reportArgumentType]
            pdf.savefig(bbox_inches='tight')
            plt.close()
    
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <command> [args]")
        print("Commands: curves, layout, energy, report")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'curves':
        # Example: plot training curves
        history = {
            'episodes': list(range(1000)),
            'rewards': np.random.randn(1000).cumsum() + np.linspace(-100, -50, 1000),
            'costs': np.random.randn(1000).cumsum() * 1000 + 500000,
            'policy_losses': np.abs(np.random.randn(100)).cumsum() * 0.01,
            'value_losses': np.abs(np.random.randn(100)).cumsum() * 0.01,
            'entropy_losses': np.abs(np.random.randn(100)) * 0.1 + 0.5,
        }
        plot_training_curves(history, 'training_curves.png')
        
    elif cmd == 'layout':
        # Example layout
        from ev_charge_manager.energy import EnergyManagerConfig
        config = {
            'positions': [75.0, 150.0, 225.0],
            'n_chargers': [6, 8, 6],
            'n_waiting': [8, 12, 8],
            'configs': [
                EnergyManagerConfig([
                    GridSourceConfig(max_power_kw=500),
                    SolarSourceConfig(peak_power_kw=200),
                    WindSourceConfig(base_power_kw=100),
                    BatteryStorageConfig(
                        capacity_kwh=100,
                        max_charge_rate_kw=400,
                        max_discharge_rate_kw=400,
                    )
                ]),                
                EnergyManagerConfig([
                    GridSourceConfig(max_power_kw=800),
                    SolarSourceConfig(peak_power_kw=300),
                    WindSourceConfig(base_power_kw=200),
                    BatteryStorageConfig(
                        capacity_kwh=150,
                        max_charge_rate_kw=600,
                        max_discharge_rate_kw=600,
                    )
                ]),
                EnergyManagerConfig([
                    GridSourceConfig(max_power_kw=500),
                    SolarSourceConfig(peak_power_kw=200),
                    WindSourceConfig(base_power_kw=100),
                    BatteryStorageConfig(
                        capacity_kwh=100,
                        max_charge_rate_kw=400,
                        max_discharge_rate_kw=400,
                    )
                ]),
            ],
            'cost': 450000,
        }
        plot_highway_layout(config, 300, 'layout.png')
        plot_energy_mix(config, 'energy_mix.png')
        
    elif cmd == 'report':
        create_summary_report('results.json', 'optimization_report.pdf')