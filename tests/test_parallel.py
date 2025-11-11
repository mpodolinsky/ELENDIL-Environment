#!/usr/bin/env python3

import os
import gymnasium as gym
from gymnasium import spaces
from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel
from gymnasium.wrappers import RecordVideo
from agents.observer_agent import ObserverAgent  # Only for isinstance checks
import time
import yaml

if __name__ == "__main__":

    # Configuration file paths (works from both root and tests directory)
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ground_agent_config_path = os.path.join(base_dir, "configs", "agent_configs", "ground_agent.yaml")
    air_observer_config_path = os.path.join(base_dir, "configs", "agent_configs", "air_observer_agent.yaml")
    target_config_path = os.path.join(base_dir, "configs", "target_configs", "target_config.yaml")

    # Load target configuration(s)
    # Option 1: Single target (backward compatible)
    # with open(target_config_path, "r") as f:
    #     target_config = yaml.safe_load(f)
    
    # Option 2: Multiple targets - create a list of target configs
    target_config = [
        {
            "name": "target_1",
            "color": [255, 100, 100],  # Red
            "movement_speed": 0.3,
            "movement_range": 1,
            "smooth_movement": True,
            "box_cells": 3,
            "outline_width": 3,
            "box_scale": 1.0
        },
        # {
        #     "name": "target_2",
        #     "color": [100, 255, 100],  # Green
        #     "movement_speed": 0.5,
        #     "movement_range": 1,
        #     "smooth_movement": True,
        #     "box_cells": 3,
        #     "outline_width": 3,
        #     "box_scale": 1.0
        # },
        # {
        #     "name": "target_3",
        #     "color": [100, 100, 255],  # Blue
        #     "movement_speed": 0.2,
        #     "movement_range": 1,
        #     "smooth_movement": True,
        #     "box_cells": 3,
        #     "outline_width": 3,
        #     "box_scale": 1.0
        # }
    ]
    
    # Option 3: Load from YAML and convert to list (if you want to use YAML)
    # with open(target_config_path, "r") as f:
    #     single_target = yaml.safe_load(f)
    # target_config = [single_target]  # Wrap in list for single target
    # Or load multiple YAML files:
    # target_config = []
    # for i in range(1, 4):  # Load target_1.yaml, target_2.yaml, etc.
    #     with open(f"configs/target_configs/target_{i}.yaml", "r") as f:
    #         target_config.append(yaml.safe_load(f))

    # Load agent configurations from YAML files
    with open(ground_agent_config_path, "r") as f:
        ground_agent_config = yaml.safe_load(f)

    with open(air_observer_config_path, "r") as f:
        air_observer_config = yaml.safe_load(f)
    
    # Create list of agent configurations
    # The environment will automatically instantiate agents from these configs!
    agent_configs = [
        ground_agent_config,
        air_observer_config
    ]

    size = int(10)

    render_mode = "human"

    env = GridWorldEnvParallel(render_mode=render_mode,
    size=size,
    agents=agent_configs,  # Pass configs - agents will be auto-instantiated!
    no_target=False,
    enable_obstacles=True,
    target_velocity=True,
    num_obstacles=0,
    num_visual_obstacles=0,  # Visual obstacles for ObserverAgent (block view but not movement)
    show_fov_display=True,
    target_config=target_config,
    max_steps=500,
    lambda_fov=0.5,
    goal_enabled=True,
    death_on_sight=True,
    show_target_coords=False)

    observations, infos = env.reset()
    
    print("\n" + "="*80)
    print("PARALLEL ENVIRONMENT SETUP".center(80))
    print("="*80)
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Physical obstacles: {len(env._obstacles)} (block movement)")
    print(f"Visual obstacles: {len(env._visual_obstacles)} (block ObserverAgent view only)")
    print(f"Targets: {len(env.targets)}")
    for idx, target in enumerate(env.targets):
        print(f"  Target {idx+1}: {target.name} at {target.location}, color={target.color}")
    print(f"Active agents: {env.agents}")
    print(f"Possible agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")

    step_idx = 0
    
    # Use Parallel API pattern
    while env.agents:  # Continue while there are active agents
        step_idx += 1
        
        # Generate actions for all active agents
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()
        
        # Step all agents simultaneously
        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"Observations: {observations}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        print(f"Infos: {infos}")

        # Print step information
        print(f"Step {step_idx}:")
        for agent in env.agents:
            print(f"  {agent}: Action={actions[agent]}, Reward={rewards[agent]:.3f}")
            # Show target distances from info
            if agent in infos:
                for key, value in infos[agent].items():
                    if 'distance' in key:
                        print(f"    {key}: {value:.2f}")
        
        if step_idx >= 100:  # Limit for demo
            break
            
        # time.sleep(0.5)  # Uncomment to slow down for observation

    env.close()
    print("\nParallel episode completed!")
