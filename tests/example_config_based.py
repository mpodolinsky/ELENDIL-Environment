"""
Example: Using Configuration-Based Agent Initialization

This script demonstrates how to use the flexible agent configuration system
in GridWorldEnvMultiAgent, which allows you to:
1. Pass agent configurations as dictionaries (no need to pre-instantiate)
2. Mix and match FOVAgent and ObserverAgent types
3. Use any number of agents
4. Easily integrate with other repositories
"""

import yaml
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent


def example_1_pure_configs():
    """Example 1: Using only configuration dictionaries (no pre-instantiation)"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Pure Configuration-Based Initialization".center(80))
    print("="*80)
    
    # Load target config from file
    with open("configs/target_config.yaml", "r") as f:
        target_config = yaml.safe_load(f)
    
    # Define agents as configuration dictionaries
    agent_configs = [
        {
            "name": "alpha",
            "type": "FOVAgent",
            "color": (80, 160, 255),
            "fov_size": 5,
            "outline_width": 2,
            "box_scale": 0.9
        },
        {
            "name": "beta",
            "type": "FOVAgent",
            "color": (255, 180, 60),
            "fov_size": 3,
            "outline_width": 2,
            "box_scale": 0.85
        },
        {
            "name": "observer1",
            "type": "ObserverAgent",
            "color": (255, 100, 100),
            "fov_base_size": 3,
            "max_altitude": 3,
            "target_detection_probs": (1.0, 0.66, 0.33),
            "outline_width": 2,
            "box_scale": 0.8
        },
        {
            "name": "observer2",
            "type": "ObserverAgent",
            "color": (100, 255, 100),
            "fov_base_size": 4,
            "max_altitude": 2,
            "target_detection_probs": (1.0, 0.5),
            "outline_width": 2,
            "box_scale": 0.75
        }
    ]
    
    # Create environment - agents will be instantiated automatically!
    env = GridWorldEnvMultiAgent(
        render_mode="human",
        size=15,
        agents=agent_configs,  # Pass configs directly
        target_config=target_config,
        enable_obstacles=True,
        num_obstacles=4,
        num_visual_obstacles=3,
        show_fov_display=True,
        lambda_fov=0.5
    )
    
    env.reset(seed=42)
    
    print(f"Environment created with {len(env.agent_list)} agents:")
    for agent in env.agent_list:
        agent_type = type(agent).__name__
        print(f"  - {agent.name} ({agent_type})")
    
    # Run a few steps
    step = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        # Handle target turn
        if agent == "_target":
            env.step(0)  # Dummy action for target
            step += 1
            continue
        if termination or truncation:
            action = None
        else:
            action = env.action_spaces[agent].sample()
        env.step(action)
        step += 1
        if step >= 20:
            break
    
    env.close()
    print("✓ Example 1 complete!\n")


def example_2_auto_type_detection():
    """Example 2: Automatic type detection (no need to specify 'type' field)"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Automatic Agent Type Detection".center(80))
    print("="*80)
    
    with open("configs/target_config.yaml", "r") as f:
        target_config = yaml.safe_load(f)
    
    # The environment can auto-detect agent type based on parameters
    agent_configs = [
        {
            "name": "fov_agent",
            # No "type" field - will be detected as FOVAgent due to fov_size
            "color": (80, 160, 255),
            "fov_size": 5
        },
        {
            "name": "observer_agent",
            # No "type" field - will be detected as ObserverAgent due to fov_base_size
            "color": (255, 100, 100),
            "fov_base_size": 3,
            "max_altitude": 3
        }
    ]
    
    env = GridWorldEnvMultiAgent(
        render_mode=None,
        size=10,
        agents=agent_configs,
        target_config=target_config,
        enable_obstacles=True,
        num_obstacles=3,
        num_visual_obstacles=2
    )
    
    env.reset()
    
    print("Agents created with automatic type detection:")
    for agent in env.agent_list:
        print(f"  - {agent.name}: {type(agent).__name__}")
    
    env.close()
    print("✓ Example 2 complete!\n")


def example_3_minimal_config():
    """Example 3: Minimal configuration with defaults"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Minimal Configuration (Using Defaults)".center(80))
    print("="*80)
    
    # Minimal configs - environment will use default values
    agent_configs = [
        {"name": "agent1"},  # Will be FOVAgent with default settings
        {"name": "agent2", "color": (255, 100, 100)},  # Custom color only
        {"name": "observer", "fov_base_size": 3}  # Auto-detected as ObserverAgent
    ]
    
    env = GridWorldEnvMultiAgent(
        size=10,
        agents=agent_configs,
        enable_obstacles=True,
        num_obstacles=2
    )
    
    env.reset()
    
    print("Agents created with minimal configuration:")
    for agent in env.agent_list:
        print(f"  - {agent.name}: {type(agent).__name__}, color={agent.color}")
    
    env.close()
    print("✓ Example 3 complete!\n")


def example_4_loading_from_yaml():
    """Example 4: Loading agent configurations from YAML file"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Loading Configs from YAML".center(80))
    print("="*80)
    
    # You can store agent configs in YAML and load them
    agent_configs_yaml = """
agents:
  - name: scout
    type: FOVAgent
    color: [80, 160, 255]
    fov_size: 7
    outline_width: 2
    box_scale: 0.9
    
  - name: recon
    type: ObserverAgent
    color: [255, 100, 100]
    fov_base_size: 3
    max_altitude: 3
    target_detection_probs: [1.0, 0.66, 0.33]
    outline_width: 2
    box_scale: 0.8
"""
    
    # Parse YAML
    config = yaml.safe_load(agent_configs_yaml)
    agent_configs = config["agents"]
    
    # Convert color lists to tuples
    for cfg in agent_configs:
        if "color" in cfg:
            cfg["color"] = tuple(cfg["color"])
        if "target_detection_probs" in cfg:
            cfg["target_detection_probs"] = tuple(cfg["target_detection_probs"])
    
    env = GridWorldEnvMultiAgent(
        size=12,
        agents=agent_configs,
        enable_obstacles=True,
        num_obstacles=3
    )
    
    env.reset()
    
    print("Agents created from YAML configuration:")
    for agent in env.agent_list:
        print(f"  - {agent.name}: {type(agent).__name__}")
    
    env.close()
    print("✓ Example 4 complete!\n")


def example_5_no_agents_specified():
    """Example 5: Default agents when none specified"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Default Agents (No Config Provided)".center(80))
    print("="*80)
    
    # If agents=None, environment creates 2 default FOVAgents
    env = GridWorldEnvMultiAgent(
        size=10,
        agents=None,  # Will create default agents
        enable_obstacles=True,
        num_obstacles=2
    )
    
    env.reset()
    
    print("Default agents created:")
    for agent in env.agent_list:
        print(f"  - {agent.name}: {type(agent).__name__}")
    
    env.close()
    print("✓ Example 5 complete!\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FLEXIBLE AGENT CONFIGURATION EXAMPLES".center(80))
    print("="*80)
    print("\nThis demonstrates the new config-based agent initialization system")
    print("that makes it easy to use GridWorldEnvMultiAgent in other repositories.")
    print()
    
    # Run all examples
    example_1_pure_configs()
    example_2_auto_type_detection()
    example_3_minimal_config()
    example_4_loading_from_yaml()
    example_5_no_agents_specified()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED".center(80))
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ Pass agent configs as dicts - no need to pre-instantiate")
    print("  ✓ Mix FOVAgent and ObserverAgent freely")
    print("  ✓ Auto-detection of agent types based on parameters")
    print("  ✓ Support for any number of agents")
    print("  ✓ Easy to load from YAML or JSON files")
    print("  ✓ Minimal configs use sensible defaults")
    print("="*80 + "\n")

