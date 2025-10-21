#!/usr/bin/env python3
"""
Test script to demonstrate ObserverAgent rendering with different flight levels.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent
import time

def test_observer_rendering():
    """Test ObserverAgent rendering with different flight levels."""
    
    print("Testing ObserverAgent Rendering")
    print("=" * 50)
    
    # Create agents with different configurations
    agents = [
        # Standard FOV agent (circle)
        FOVAgent(
            name="standard_agent",
            color=(0, 255, 0),
            env_size=8,
            fov_size=3,
            show_target_coords=False
        ),
        
        # Observer agent at ground level (solid square)
        ObserverAgent(
            name="ground_observer",
            color=(255, 0, 255),
            env_size=8,
            fov_base_size=3,  # FOV size 5
            max_altitude=3,
            show_target_coords=False
        ),
        
        # Observer agent at flight level 1 (solid square)
        ObserverAgent(
            name="flight1_observer",
            color=(255, 100, 100),
            env_size=8,
            fov_base_size=5,  # FOV size 7
            max_altitude=3,
            show_target_coords=False
        ),
        
        # Observer agent at flight level 2 (dashed square)
        ObserverAgent(
            name="flight2_observer",
            color=(100, 255, 100),
            env_size=8,
            fov_base_size=4,  # FOV size 6
            max_altitude=3,
            show_target_coords=False
        ),
        
        # Observer agent at flight level 3 (dotted square)
        ObserverAgent(
            name="flight3_observer",
            color=(100, 100, 255),
            env_size=8,
            fov_base_size=3,  # FOV size 5
            max_altitude=3,
            show_target_coords=False
        )
    ]
    
    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.name}: {type(agent).__name__}")
        if isinstance(agent, ObserverAgent):
            print(f"    Current altitude: {agent.altitude}")
            print(f"    FOV size: {agent.fov_size}")
    
    # Create environment
    env = GridWorldEnvMultiAgent(
        agents=agents,
        size=8,
        render_mode="human",  # Enable rendering
        show_target_coords=False,
        intrinsic=False
    )
    
    print(f"\nEnvironment created with {len(env.possible_agents)} agents")
    
    # Reset environment
    env.reset()
    print("Environment reset successfully")
    
    # Set different flight levels for ObserverAgents
    observer_agents = [a for a in env.agent_list if isinstance(a, ObserverAgent)]
    
    if len(observer_agents) >= 4:
        observer_agents[0].altitude = 1  # Flight level 1 - solid
        observer_agents[1].altitude = 1  # Flight level 1 - solid
        observer_agents[2].altitude = 2  # Flight level 2 - dashed
        observer_agents[3].altitude = 3  # Flight level 3 - dotted
        
        print("\nSet flight levels:")
        for i, agent in enumerate(observer_agents):
            print(f"  {agent.name}: Flight level {agent.altitude}")
    
    print("\n" + "=" * 50)
    print("Rendering Test - Check the window to see:")
    print("- Standard agent: Circle")
    print("- Flight 1 observer: Solid square (3x3 border)")
    print("- Flight 1 observer: Solid square (3x3 border)")
    print("- Flight 2 observer: Dashed square (5x5 border)")
    print("- Flight 3 observer: Dotted square (7x7 border)")
    print("=" * 50)
    
    # Run a few steps to show the rendering
    step_count = 0
    max_steps = 10
    
    for agent in env.agent_iter(max_iter=max_steps):
        if step_count >= max_steps:
            break
        
        # For ObserverAgents, try altitude actions to demonstrate line style changes
        agent_obj = env._agents_by_name[agent]
        if isinstance(agent_obj, ObserverAgent):
            # Cycle through altitude actions
            altitude_action = 5 + (step_count % 3)  # Actions 5, 6, 7
            action = altitude_action
            print(f"Step {step_count + 1}: {agent} taking altitude action {altitude_action}")
        else:
            # Regular movement for standard agent
            action = env.action_spaces[agent].sample()
            print(f"Step {step_count + 1}: {agent} taking action {action}")
        
        env.step(action)
        
        # Show current altitude for ObserverAgents
        if isinstance(agent_obj, ObserverAgent):
            print(f"  Current altitude: {agent_obj.altitude}")
        
        step_count += 1
        time.sleep(1.0)  # Slow down to see changes
    
    print("\nRendering test completed!")
    print("Close the window to continue...")
    
    # Keep the window open for a moment
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    test_observer_rendering()
