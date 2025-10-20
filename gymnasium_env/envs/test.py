import os
from pickle import FALSE
import gymnasium as gym
from gymnasium import spaces
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from gymnasium.wrappers import RecordVideo
import time
import yaml

if __name__ == "__main__":

    agent_config_path = "configs/agent_config.yaml"
    target_config_path = "configs/target_config.yaml"

    agents: list[dict] = []
    target_config: dict = {}

    with open(agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)
        agents.append({**agent_config})

    with open(target_config_path, "r") as f:
        target_config = yaml.safe_load(f)
        target_config = {**target_config}

    agents.append({"name": "beta",
    "color": [0, 255, 200],
    "outline_width": 2,
    "box_scale": 0.8,
    "fov_size": 5})

    agents.append({"name": "gamma",
    "color": [255, 200, 20],
    "outline_width": 2,
    "box_scale": 0.8,
    "fov_size": 5})

    size = int(10)
    record = False

    # Ensure video folder exists
    os.makedirs("./videos", exist_ok=True)

    if record:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        render_mode = "rgb_array"
    else:
        render_mode = "human"

    env = GridWorldEnvMultiAgent(render_mode=render_mode,
    size=size,
    agents=agents,
    no_target=False,
    enable_obstacles=True,
    num_obstacles=3,
    show_fov_display=False,
    target_config=target_config,
    lambda_fov=0.5)

    if record: 

        # # Wrap with RecordVideo (requires rgb_array render_mode)
        env = RecordVideo(
            env,
            video_folder="./videos",
            name_prefix="multi_agent_run_new",
            episode_trigger=lambda ep_id: True,  # record every episode
        )

        # Access the underlying environment through the wrapper
        base_env = env.env
    
    else:
        base_env = env

    env.reset()
    
    
    print("Possible agents:", base_env.possible_agents)
    print("Action spaces:", base_env.action_spaces)
    print("Observation spaces:", base_env.observation_spaces)
    
    # Print observation space in a nice ASCII format
    print("\n" + "="*80)
    print("OBSERVATION SPACE DETAILS".center(80))
    print("="*80)
    
    for agent_name in base_env.possible_agents:
        agent_obs_space = base_env.observation_spaces[agent_name]
        print(f"\n┌─ {agent_name.upper()} AGENT OBSERVATION SPACE ─┐")
        print(f"│ Agent Position: {agent_obs_space['agent']}")
        if 'target' in agent_obs_space:
            print(f"│ Target Position: {agent_obs_space['target']}")
        print(f"│ FOV Obstacles:   {agent_obs_space['obstacles_fov']}")
        print("└" + "─" * 40 + "┘")
    
    print("\n" + "="*80)
    
    # Print FOV for each agent (only if FOV display is enabled)
    if base_env.show_fov_display:
        for agent_name in base_env.possible_agents:
            obs = base_env.observe(agent_name)
            print(f"\n{agent_name} FOV obstacles:")
            print(obs['obstacles_fov'])

    step_idx = 0
    
    # Use AEC agent_iter() pattern
    for agent in env.agent_iter():
        step_idx += 1
        
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            action = env.action_spaces[agent].sample()
            
        env.step(action)
        
        print(f"Step {step_idx}, Agent {agent}, Action: {action}, Reward: {reward}")
        
        if step_idx >= 500:
            break
            
        time.sleep(0.2)

    env.close()