#!/usr/bin/env python3

import os
import gymnasium as gym
from gymnasium import spaces
from elendil.envs.grid_world_multi_agent import GridWorldEnvParallelExploration
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

    # Load target configuration(s) - these are moving targets (hidden from observations)
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
        {
            "name": "target_2",
            "color": [100, 255, 100],  # Green
            "movement_speed": 0.5,
            "movement_range": 1,
            "smooth_movement": True,
            "box_cells": 3,
            "outline_width": 3,
            "box_scale": 1.0
        },
        {
            "name": "target_3",
            "color": [100, 100, 255],  # Blue
            "movement_speed": 0.2,
            "movement_range": 1,
            "smooth_movement": True,
            "box_cells": 3,
            "outline_width": 3,
            "box_scale": 1.0
        }
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

    size = int(20)
    record = False

    # Ensure video folder exists
    os.makedirs("./videos", exist_ok=True)

    if record:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    # Create exploration environment with static goal
    env = GridWorldEnvParallelExploration(
        render_mode=render_mode,
        size=size,
        agents=agent_configs,  # Pass configs - agents will be auto-instantiated!
        no_target=False,
        enable_obstacles=True,
        num_obstacles=3,
        num_visual_obstacles=3,  # Visual obstacles for ObserverAgent (block view but not movement)
        show_fov_display=False,
        target_config=target_config,  # Moving targets (hidden from observations)
        max_steps=500,
        lambda_fov=0.5,
        death_on_sight=True,
        goal_color=(0, 255, 0)  # Green goal
    )

    if record: 
        # Wrap with RecordVideo (requires rgb_array render_mode)
        env = RecordVideo(
            env,
            video_folder="./videos",
            name_prefix="multi_agent_exploration_run",
            episode_trigger=lambda ep_id: True,  # record every episode
        )

    observations, infos = env.reset()
    
    print("\n" + "="*80)
    print("EXPLORATION ENVIRONMENT SETUP".center(80))
    print("="*80)
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Physical obstacles: {len(env._obstacles)} (block movement)")
    print(f"Visual obstacles: {len(env._visual_obstacles)} (block ObserverAgent view only)")
    print(f"\n🎯 STATIC GOAL:")
    print(f"  Goal location: {env.goal_location}")
    print(f"  Goal color: {env.goal_color}")
    print(f"  Reward for reaching goal: +5.0")
    print(f"\n⚠️  MOVING TARGETS (hidden from observations):")
    print(f"  Number of targets: {len(env.targets)}")
    for idx, target in enumerate(env.targets):
        print(f"  Target {idx+1}: {target.name} at {target.location}, color={target.color}")
    print(f"\n🤖 AGENTS:")
    print(f"  Active agents: {env.agents}")
    print(f"  Possible agents: {env.possible_agents}")
    print(f"  Action spaces: {env.action_spaces}")
    print(f"  Observation spaces: {env.observation_spaces}")
    
    # Print observation space in a nice ASCII format
    print("\n" + "="*80)
    print("OBSERVATION SPACE DETAILS".center(80))
    print("="*80)
    
    for agent_name in env.possible_agents:
        agent_obs_space = env.observation_spaces[agent_name]
        agent_obj = env._agents_by_name[agent_name]
        
        print(f"\n┌─ {agent_name.upper()} AGENT OBSERVATION SPACE ─┐")
        print(f"│ Agent Position: {agent_obs_space['agent']}")
        if 'goal' in agent_obs_space:
            print(f"│ Goal Position: {agent_obs_space['goal']}")
        print(f"│ FOV Obstacles:   {agent_obs_space['obstacles_fov']}")
        
        # Show flight level for ObserverAgent
        if isinstance(agent_obj, ObserverAgent):
            print(f"│ Flight Level:    {agent_obs_space['flight_level']}")
        
        print("└" + "─" * 40 + "┘")
    
    # Print FOV encoding legend
    print("\n" + "="*80)
    print("FOV ENCODING LEGEND".center(80))
    print("="*80)
    print("  -10 = Masked area (ObserverAgent flight level masking)")
    print("   0  = Empty space")
    print("   1  = Physical obstacle (blocks movement)")
    print("   2  = Other agent")
    print("   5  = Static Goal (agents must reach this)")
    print("   4  = Visual obstacle (blocks ObserverAgent view, not movement)")
    print("\n  NOTE: Moving targets are NOT visible in FOV (hidden from observations)")
    print("="*80)
    
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
        
        # Print step information
        print(f"\nStep {step_idx}:")
        for agent in env.agents:
            reached_goal = terminations.get(agent, False) and agent in env._agents_by_name
            if reached_goal:
                print(f"  🎉 {agent}: Reached GOAL! Reward={rewards[agent]:.3f}")
            else:
                print(f"  {agent}: Action={actions[agent]}, Reward={rewards[agent]:.3f}")
            
            # Show goal distance from info
            if agent in infos:
                if 'distance_to_goal' in infos[agent]:
                    print(f"    Distance to goal: {infos[agent]['distance_to_goal']:.2f}")
                # Also show other distance info if available
                for key, value in infos[agent].items():
                    if 'distance' in key and key != 'distance_to_goal':
                        print(f"    {key}: {value:.2f}")
        
        if step_idx >= 100:  # Limit for demo
            break
            
        # time.sleep(0.5)  # Uncomment to slow down for observation

    env.close()
    print("\n" + "="*80)
    print("Exploration episode completed!".center(80))
    print("="*80)
    print("\nSummary:")
    print(f"  - Agents tried to reach the static goal at {env.goal_location}")
    print(f"  - Moving targets were present but hidden from observations")
    print(f"  - Agents received +5.0 reward for reaching the goal")

