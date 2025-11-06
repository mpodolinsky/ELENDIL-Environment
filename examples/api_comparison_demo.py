#!/usr/bin/env python3
"""
Comparison script showing AEC vs Parallel API usage for the Grid World Multi-Agent Environment.
This demonstrates the key differences between sequential and parallel agent interactions.
"""

import yaml
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent, GridWorldEnvParallel

def load_agent_configs():
    """Load agent configurations from YAML files."""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(base_dir, "configs", "agent_configs", "ground_agent.yaml"), "r") as f:
        ground_agent_config = yaml.safe_load(f)
    
    with open(os.path.join(base_dir, "configs", "agent_configs", "air_observer_agent.yaml"), "r") as f:
        air_observer_config = yaml.safe_load(f)
    
    with open(os.path.join(base_dir, "configs", "target_configs", "target_config.yaml"), "r") as f:
        target_config = yaml.safe_load(f)
    
    return [ground_agent_config, air_observer_config], target_config

def demo_aec_api():
    """Demonstrate the AEC (Agent-Environment-Cycle) API - Sequential agent interactions."""
    print("\n" + "="*80)
    print("AEC API DEMO - Sequential Agent Interactions".center(80))
    print("="*80)
    
    agent_configs, target_config = load_agent_configs()
    
    # Create AEC environment
    env = GridWorldEnvMultiAgent(
        render_mode=None,  # No rendering for demo
        size=10,
        agents=agent_configs,
        target_config=target_config,
        enable_obstacles=True,
        num_obstacles=2,
        max_steps=20,
        show_fov_display=False
    )
    
    env.reset()
    
    print(f"Active agents: {env.agents}")
    print(f"Agent selection order: {env.possible_agents}")
    print("\nSequential step-by-step execution:")
    
    step_count = 0
    # AEC API: Use agent_iter() to cycle through agents sequentially
    for agent in env.agent_iter():
        step_count += 1
        
        # Get current agent's observation, reward, etc.
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            # Sample action for current agent
            action = env.action_spaces[agent].sample()
        
        # Step only the current agent
        env.step(action)
        
        print(f"  Step {step_count}: Agent '{agent}' takes action {action}, gets reward {reward:.3f}")
        
        if step_count >= 20:  # Limit for demo
            break
    
    env.close()
    print("AEC demo completed!\n")

def demo_parallel_api():
    """Demonstrate the Parallel API - Simultaneous agent interactions."""
    print("\n" + "="*80)
    print("PARALLEL API DEMO - Simultaneous Agent Interactions".center(80))
    print("="*80)
    
    agent_configs, target_config = load_agent_configs()
    
    # Create Parallel environment
    env = GridWorldEnvParallel(
        render_mode=None,  # No rendering for demo
        size=10,
        agents=agent_configs,
        target_config=target_config,
        enable_obstacles=True,
        num_obstacles=2,
        max_steps=20,
        show_fov_display=False
    )
    
    observations, infos = env.reset()
    
    print(f"Active agents: {env.agents}")
    print(f"Possible agents: {env.possible_agents}")
    print("\nParallel step-by-step execution:")
    
    step_count = 0
    # Parallel API: All agents act simultaneously
    while env.agents:  # Continue while there are active agents
        step_count += 1
        
        # Generate actions for ALL active agents simultaneously
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()
        
        # Step ALL agents at once
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"  Step {step_count}:")
        for agent in env.agents:
            print(f"    {agent}: Action={actions[agent]}, Reward={rewards[agent]:.3f}")
        
        if step_count >= 10:  # Limit for demo
            break
    
    env.close()
    print("Parallel demo completed!\n")

def compare_apis():
    """Compare the key differences between AEC and Parallel APIs."""
    print("\n" + "="*80)
    print("API COMPARISON SUMMARY".center(80))
    print("="*80)
    
    print("""
AEC API (Sequential):
├── Agents take turns acting one at a time
├── Use env.agent_iter() to cycle through agents
├── Call env.last() to get current agent's state
├── Call env.step(action) for single agent
├── Target moves after each agent's turn
└── Better for turn-based games, strict ordering

Parallel API (Simultaneous):
├── All agents act at the same time
├── Use while env.agents: loop
├── Generate actions dict for all agents
├── Call env.step(actions_dict) for all agents
├── Target moves after all agents have acted
└── Better for real-time scenarios, natural interactions

Key Benefits of Parallel API:
• More natural agent interactions
• Better performance (no sequential overhead)
• Easier integration with many RL frameworks
• Simpler reward calculation logic
• Agents rewarded before target moves (fair attribution)
""")

if __name__ == "__main__":
    print("Grid World Multi-Agent Environment API Comparison")
    print("This demo shows the difference between AEC and Parallel APIs")
    
    # Run both demos
    demo_aec_api()
    demo_parallel_api()
    
    # Show comparison summary
    compare_apis()
    
    print("\nBoth environments are now available:")
    print("• GridWorldEnvMultiAgent (AEC API)")
    print("• GridWorldEnvParallel (Parallel API)")
    print("\nChoose the API that best fits your use case!")
