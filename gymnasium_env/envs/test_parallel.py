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

    # Configuration file paths
    ground_agent_config_path = "configs/agent_configs/ground_agent.yaml"
    air_observer_config_path = "configs/agent_configs/air_observer_agent.yaml"
    target_config_path = "configs/target_configs/target_config.yaml"

    # Load target configuration
    with open(target_config_path, "r") as f:
        target_config = yaml.safe_load(f)

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

    env = GridWorldEnvParallel(render_mode=render_mode,
    size=size,
    agents=agent_configs,  # Pass configs - agents will be auto-instantiated!
    no_target=False,
    enable_obstacles=True,
    num_obstacles=3,
    num_visual_obstacles=3,  # Visual obstacles for ObserverAgent (block view but not movement)
    show_fov_display=False,
    target_config=target_config,
    max_steps=500,
    lambda_fov=0.5)

    if record: 
        # Wrap with RecordVideo (requires rgb_array render_mode)
        env = RecordVideo(
            env,
            video_folder="./videos",
            name_prefix="multi_agent_parallel_run",
            episode_trigger=lambda ep_id: True,  # record every episode
        )

    observations, infos = env.reset()
    
    print("\n" + "="*80)
    print("PARALLEL ENVIRONMENT SETUP".center(80))
    print("="*80)
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Physical obstacles: {len(env._obstacles)} (block movement)")
    print(f"Visual obstacles: {len(env._visual_obstacles)} (block ObserverAgent view only)")
    print(f"Active agents: {env.agents}")
    print(f"Possible agents: {env.possible_agents}")
    print(f"Action spaces: {env.action_spaces}")
    print(f"Observation spaces: {env.observation_spaces}")
    
    # Print observation space in a nice ASCII format
    print("\n" + "="*80)
    print("OBSERVATION SPACE DETAILS".center(80))
    print("="*80)
    
    for agent_name in env.possible_agents:
        agent_obs_space = env.observation_spaces[agent_name]
        agent_obj = env._agents_by_name[agent_name]
        
        print(f"\n┌─ {agent_name.upper()} AGENT OBSERVATION SPACE ─┐")
        print(f"│ Agent Position: {agent_obs_space['agent']}")
        if 'target' in agent_obs_space:
            print(f"│ Target Position: {agent_obs_space['target']}")
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
    print("   3  = Target")
    print("   4  = Visual obstacle (blocks ObserverAgent view, not movement)")
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
        print(f"Step {step_idx}:")
        for agent in env.agents:
            print(f"  {agent}: Action={actions[agent]}, Reward={rewards[agent]:.3f}")
        
        if step_idx >= 100:  # Limit for demo
            break
            
        # time.sleep(0.5)  # Uncomment to slow down for observation

    env.close()
    print("\nParallel episode completed!")
