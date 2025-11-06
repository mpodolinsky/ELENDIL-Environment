"""
Visualization script for obstacle generation.

This script generates and displays multiple random obstacle configurations
to demonstrate the obstacle generation logic for both physical and visual obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent


def visualize_obstacle_generation(num_samples=10, grid_size=20, num_physical=5, num_visual=3):
    """
    Generate and visualize multiple random obstacle configurations.
    
    Args:
        num_samples: Number of random configurations to generate
        grid_size: Size of the grid
        num_physical: Number of physical obstacles
        num_visual: Number of visual obstacles
    """
    
    # Load configurations
    agent_config_path = "configs/agent_config.yaml"
    target_config_path = "configs/target_config.yaml"
    
    with open(target_config_path, "r") as f:
        target_config = yaml.safe_load(f)
    
    with open(agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)
    
    # Create a simple agent for the environment
    agents = [
        FOVAgent(
            name="alpha",
            color=(80, 160, 255),
            env_size=grid_size,
            fov_size=5,
            outline_width=2,
            box_scale=0.9,
            show_target_coords=False
        )
    ]
    
    # Setup agents
    for agent in agents:
        agent.env_size = grid_size
        agent._setup_agent(
            fov_size=agent.fov_size,
            outline_width=agent.outline_width,
            box_scale=agent.box_scale,
            show_target_coords=agent.show_target_coords
        )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Obstacle Generation Examples - 10 Random Instances (Grid: {grid_size}x{grid_size}, Physical: {num_physical}, Visual: {num_visual})',
                 fontsize=13, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # Colors matching the actual rendering
    PHYSICAL_COLOR = np.array([90, 90, 90]) / 255  # Dark gray
    VISUAL_COLOR = np.array([150, 200, 240]) / 255  # Light blue
    GRID_COLOR = (0.9, 0.9, 0.9)
    
    # For legend - create dummy artists
    legend_artists = []
    legend_labels = []
    
    # Generate and visualize each configuration
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Create environment
        env = GridWorldEnvMultiAgent(
            render_mode=None,  # No rendering needed
            size=grid_size,
            agents=agents,
            no_target=False,
            enable_obstacles=True,
            num_obstacles=num_physical,
            num_visual_obstacles=num_visual,
            show_fov_display=False,
            target_config=target_config,
            lambda_fov=0.5
        )
        
        # Reset to generate obstacles (with different seed each time)
        env.reset(seed=idx * 42)
        
        # Setup the plot
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(f'Instance {idx + 1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i, color=GRID_COLOR, linewidth=0.5)
            ax.axvline(i, color=GRID_COLOR, linewidth=0.5)
        
        # Draw physical obstacles (solid dark gray)
        if len(env._obstacles) > 0:
            ox, oy, ow, oh = env._obstacles[0]
            rect = patches.Rectangle(
                (ox, oy), ow, oh,
                linewidth=1.5,
                edgecolor='black',
                facecolor=PHYSICAL_COLOR,
                alpha=1.0
            )
            ax.add_patch(rect)
            if idx == 0:
                legend_artists.append(rect)
                legend_labels.append('Physical Obstacle (blocks movement)')
            
            # Draw rest of physical obstacles
            for (ox, oy, ow, oh) in env._obstacles[1:]:
                rect = patches.Rectangle(
                    (ox, oy), ow, oh,
                    linewidth=1.5,
                    edgecolor='black',
                    facecolor=PHYSICAL_COLOR,
                    alpha=1.0
                )
                ax.add_patch(rect)
        
        # Draw visual obstacles (light blue with hatching)
        if len(env._visual_obstacles) > 0:
            ox, oy, ow, oh = env._visual_obstacles[0]
            rect = patches.Rectangle(
                (ox, oy), ow, oh,
                linewidth=1.5,
                edgecolor=VISUAL_COLOR * 0.7,
                facecolor=VISUAL_COLOR,
                alpha=0.5,
                hatch='///',  # Use matplotlib's built-in hatching
                fill=True
            )
            ax.add_patch(rect)
            if idx == 0:
                legend_artists.append(rect)
                legend_labels.append('Visual Obstacle (blocks view only)')
            
            # Draw rest of visual obstacles
            for (ox, oy, ow, oh) in env._visual_obstacles[1:]:
                rect = patches.Rectangle(
                    (ox, oy), ow, oh,
                    linewidth=1.5,
                    edgecolor=VISUAL_COLOR * 0.7,
                    facecolor=VISUAL_COLOR,
                    alpha=0.5,
                    hatch='///',
                    fill=True
                )
                ax.add_patch(rect)
        
        # Draw agents (small circles)
        if len(env.agent_list) > 0:
            agent = env.agent_list[0]
            circle = patches.Circle(
                (agent.location[0] + 0.5, agent.location[1] + 0.5),
                0.3,
                color=np.array(agent.color) / 255,
                alpha=0.8
            )
            ax.add_patch(circle)
            if idx == 0:
                legend_artists.append(circle)
                legend_labels.append('Agent spawn')
            
            # Draw rest of agents
            for agent in env.agent_list[1:]:
                circle = patches.Circle(
                    (agent.location[0] + 0.5, agent.location[1] + 0.5),
                    0.3,
                    color=np.array(agent.color) / 255,
                    alpha=0.8
                )
                ax.add_patch(circle)
        
        # Draw target (triangle)
        if env.target is not None:
            target_x, target_y = env.target.location
            triangle = patches.Polygon(
                [(target_x + 0.5, target_y + 0.2),
                 (target_x + 0.2, target_y + 0.8),
                 (target_x + 0.8, target_y + 0.8)],
                color=np.array(env.target.color) / 255,
                alpha=0.8
            )
            ax.add_patch(triangle)
            if idx == 0:
                legend_artists.append(triangle)
                legend_labels.append('Target spawn')
        
        # Add statistics text
        stats_text = f'P:{len(env._obstacles)} V:{len(env._visual_obstacles)}'
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Invert y-axis to match grid coordinate system
        ax.invert_yaxis()
        
        env.close()
    
    # Add a single figure-level legend
    fig.legend(legend_artists, legend_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.945),
              ncol=4,
              fontsize=10,
              frameon=True,
              framealpha=0.95,
              edgecolor='black',
              fancybox=True,
              shadow=True)
    
    # Adjust layout to make room for title and legend
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    
    # Save the figure
    output_path = "obstacles_large_5p5v.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Obstacle Generation Visualization".center(80))
    print(f"{'='*80}")
    print(f"Generated {num_samples} random obstacle configurations")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Physical obstacles per instance: {num_physical}")
    print(f"Visual obstacles per instance: {num_visual}")
    print(f"\nVisualization saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Generate visualization with 10 samples
    visualize_obstacle_generation(
        num_samples=10,
        grid_size=20,
        num_physical=5,
        num_visual=5
    )

