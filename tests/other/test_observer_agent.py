#!/usr/bin/env python3
"""
Test script to demonstrate the new ObserverAgent class.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent
import numpy as np

def test_observer_agent():
    """Test the new ObserverAgent class."""
    
    print("Testing ObserverAgent Class")
    print("=" * 50)
    
    # Create agents including the new ObserverAgent
    agents = [
        # Standard FOV agent
        FOVAgent(
            name="standard_agent",
            color=(0, 255, 0),
            env_size=8,
            fov_size=3,
            show_target_coords=False
        ),
        
        # Observer agent with base FOV size 3 (actual FOV will be 5)
        ObserverAgent(
            name="observer_agent",
            color=(255, 0, 255),
            env_size=8,
            fov_base_size=3,  # Will result in FOV size 5
            max_altitude=3,  # Flight levels 1, 2, 3
            show_target_coords=False
        ),
        
        # Another observer agent with different FOV
        ObserverAgent(
            name="wide_observer",
            color=(0, 255, 255),
            env_size=8,
            fov_base_size=5,  # Will result in FOV size 7
            max_altitude=3,  # Flight levels 1, 2, 3
            show_target_coords=False
        )
    ]
    
    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.name}: {type(agent).__name__}")
        print(f"    Action space: {agent.get_action_space()}")
        print(f"    Observation space keys: {list(agent.get_observation_space().spaces.keys())}")
        
        # Show special attributes for ObserverAgent
        if isinstance(agent, ObserverAgent):
            print(f"    FOV base size: {agent.fov_base_size}")
            print(f"    Actual FOV size: {agent.fov_size}")
            print(f"    Max altitude: {agent.max_altitude} (Flight levels: 1, 2, 3)")
            print(f"    Current altitude: {agent.altitude} (0=ground, 1-3=flight levels)")
    
    print("\n" + "=" * 50)
    print("Testing ObserverAgent Actions")
    print("=" * 50)
    
    # Test individual agent methods
    observer = agents[1]  # Get the observer agent
    env_state = {
        "agents": agents,
        "target": None,
        "obstacles": [],
        "grid_size": 8
    }
    
    print(f"\nTesting {observer.name}:")
    
    # Test different actions
    test_actions = [
        (0, "Move Right"),
        (1, "Move Up"),
        (5, "Increase Altitude"),
        (6, "Remain at Altitude"),
        (7, "Reduce Altitude"),
        (3, "Move Down"),
    ]
    
    for action, description in test_actions:
        print(f"\n  Action {action} ({description}):")
        
        # Get observation
        obs = observer.observe(env_state)
        print(f"    Observation keys: {list(obs.keys())}")
        print(f"    Current flight level: {obs['flight_level']}")
        print(f"    Flight level: {obs['flight_level']}")
        
        # Execute step
        step_result = observer.step(action, env_state)
        print(f"    Step result: {list(step_result.keys())}")
        print(f"    New altitude: {step_result['new_altitude']}")
        print(f"    Altitude changed: {step_result['altitude_changed']}")
        print(f"    Movement action: {step_result['movement_action']}")
        print(f"    Altitude action: {step_result['altitude_action']}")
    
    print("\n" + "=" * 50)
    print("Testing Environment Integration")
    print("=" * 50)
    
    # Create environment with mixed agent types
    env = GridWorldEnvMultiAgent(
        agents=agents,
        size=8,
        render_mode=None,  # Disable rendering for testing
        show_target_coords=False,
        intrinsic=False
    )
    
    print(f"Environment created with {len(env.possible_agents)} agents")
    print(f"Agent names: {env.possible_agents}")
    
    # Test observation spaces
    print("\nObservation spaces:")
    for agent_name in env.possible_agents:
        agent_obj = env._agents_by_name[agent_name]
        obs_space = env.observation_spaces[agent_name]
        print(f"  {agent_name}: {list(obs_space.spaces.keys())}")
        
        # Show special attributes for ObserverAgent
        if isinstance(agent_obj, ObserverAgent):
            print(f"    Flight level observation: {obs_space['flight_level']}")
            print(f"    Flight level observation: {obs_space['flight_level']}")
    
    # Reset environment
    env.reset()
    print("\nEnvironment reset successfully")
    
    # Test a few steps
    print("\nTesting agent steps:")
    step_count = 0
    max_steps = 8  # Limit to prevent long runs
    
    for agent in env.agent_iter(max_iter=max_steps):
        if step_count >= max_steps:
            break
            
        # Get observation
        obs = env.observe(agent)
        print(f"  {agent} observes: {list(obs.keys())}")
        
        # Show altitude for ObserverAgent
        agent_obj = env._agents_by_name[agent]
        if isinstance(agent_obj, ObserverAgent):
            print(f"    Flight level: {obs['flight_level']}")
            print(f"    Flight Level: {obs['flight_level']}")
        
        # Take random action
        action = env.action_spaces[agent].sample()
        print(f"  {agent} takes action: {action}")
        
        # Step environment
        env.step(action)
        
        # Get last observation, reward, etc.
        last_obs, reward, terminated, truncated, info = env.last()
        print(f"  {agent} reward: {reward:.3f}")
        
        step_count += 1
    
    print("\n" + "=" * 50)
    print("ObserverAgent Test Completed Successfully!")
    print("=" * 50)

if __name__ == "__main__":
    test_observer_agent()
