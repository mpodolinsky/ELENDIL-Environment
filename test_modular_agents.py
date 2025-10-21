#!/usr/bin/env python3
"""
Test script to demonstrate the new modular agent system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent, Agent
from agents.special_agents import GlobalViewAgent, TelepathicAgent
import numpy as np

def test_modular_agents():
    """Test the new modular agent system with different agent types."""
    
    print("Testing Modular Agent System")
    print("=" * 50)
    
    # Create a mix of different agent types
    agents = [
        # Standard FOV agent
        FOVAgent(
            name="fov_agent",
            color=(0, 255, 0),
            env_size=7,
            fov_size=3,
            show_target_coords=True
        ),
        
        # Global view agent
        GlobalViewAgent(
            name="global_agent",
            color=(0, 0, 255),
            env_size=7
        ),
        
        # Telepathic agent
        TelepathicAgent(
            name="telepathic_agent",
            color=(255, 255, 0),
            env_size=7,
            fov_size=5,
            see_other_actions=False
        ),
        
        # Legacy agent for compatibility
        Agent(
            env_size=7,
            name="legacy_agent",
            color=(255, 0, 255),
            fov_size=3,
            show_target_coords=True
        )
    ]
    
    print(f"Created {len(agents)} agents of different types:")
    for agent in agents:
        print(f"  - {agent.name}: {type(agent).__name__}")
        print(f"    Action space: {agent.get_action_space()}")
        print(f"    Observation space keys: {list(agent.get_observation_space().spaces.keys())}")
    
    print("\n" + "=" * 50)
    print("Testing Environment Integration")
    print("=" * 50)
    
    # Create environment with mixed agent types
    env = GridWorldEnvMultiAgent(
        agents=agents,
        size=7,
        render_mode=None,  # Disable rendering for testing
        show_target_coords=True,
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
    
    # Reset environment
    env.reset()
    print("\nEnvironment reset successfully")
    
    # Test a few steps with proper termination
    print("\nTesting agent steps:")
    step_count = 0
    max_steps = 12  # Limit to prevent infinite loop
    
    for agent in env.agent_iter(max_iter=max_steps):
        if step_count >= max_steps:
            break
            
        # Get observation
        obs = env.observe(agent)
        print(f"  {agent} observes: {list(obs.keys())}")
        
        # Take random action
        action = env.action_spaces[agent].sample()
        print(f"  {agent} takes action: {action}")
        
        # Step environment
        env.step(action)
        
        # Get last observation, reward, etc.
        last_obs, reward, terminated, truncated, info = env.last()
        print(f"  {agent} reward: {reward:.3f}")
        
        step_count += 1
        
        # Check if any agent terminated
        if terminated or truncated:
            print(f"  Agent {agent} terminated/truncated")
            break
    
    print("\n" + "=" * 50)
    print("Testing Individual Agent Methods")
    print("=" * 50)
    
    # Test individual agent observation and step methods
    env_state = {
        "agents": env.agent_list,
        "target": env.target,
        "obstacles": env._obstacles,
        "grid_size": env.size
    }
    
    for agent_obj in env.agent_list:
        print(f"\nTesting {agent_obj.name} ({type(agent_obj).__name__}):")
        
        # Test observation method
        obs = agent_obj.observe(env_state)
        print(f"  Observation keys: {list(obs.keys())}")
        
        # Test step method
        action = 0  # Move right
        step_result = agent_obj.step(action, env_state)
        print(f"  Step result: {list(step_result.keys())}")
        print(f"  Proposed location: {step_result['new_location']}")
    
    print("\n" + "=" * 50)
    print("Modular Agent System Test Completed Successfully!")
    print("=" * 50)

if __name__ == "__main__":
    test_modular_agents()
