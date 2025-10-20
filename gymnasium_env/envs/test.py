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

    size = int(15)
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

    obs, info = env.reset()
    
    
    print("Agents:", base_env.agents)
    print("Action space:", base_env.action_space)
    print("Observation space:", base_env.observation_space)
    print("Initial Observation:", obs)
    
    # Print observation space in a nice ASCII format
    print("\n" + "="*80)
    print("OBSERVATION SPACE DETAILS".center(80))
    print("="*80)
    
    for agent_name in base_env.agents:
        agent_obs_space = base_env.observation_space[agent_name]
        print(f"\n┌─ {agent_name.upper()} AGENT OBSERVATION SPACE ─┐")
        print(f"│ Agent Position: {agent_obs_space['agent']}")
        print(f"│ Target Position: {agent_obs_space['target']}")
        print(f"│ FOV Obstacles:   {agent_obs_space['obstacles_fov']}")
        print("└" + "─" * 40 + "┘")
    
    print("\n" + "="*80)
    
    # Print FOV for each agent (only if FOV display is enabled)
    if base_env.show_fov_display:
        for agent_name in base_env.agents:
            print(f"\n{agent_name} FOV obstacles:")
            print(obs[agent_name]['obstacles_fov'])

    done = False
    step_idx = 0
    
    agent_cycle = iter(base_env.agents)
    while not done and step_idx < 500:
        step_idx += 1

        try:
            current_agent = next(agent_cycle)
        except StopIteration:
            # Restart the cycle if at end
            agent_cycle = iter(base_env.agents)
            current_agent = next(agent_cycle)
        
        action = base_env.action_space[current_agent].sample()
        step_action = {"agent": current_agent, "action": action}
        obs, reward, terminated, truncated, info = env.step(step_action)
        print(f"Agent {current_agent}, Action: {action}, Reward: {reward}")
        
        if terminated or truncated:
            done = True
            break

        time.sleep(0.2)

    env.close()