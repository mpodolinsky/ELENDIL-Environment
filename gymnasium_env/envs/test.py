import os
from pickle import FALSE
import gymnasium as gym
from gymnasium import spaces
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from gymnasium.wrappers import RecordVideo
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent
import time
import yaml

if __name__ == "__main__":

    agent_config_path = "configs/agent_config.yaml"
    target_config_path = "configs/target_config.yaml"

    # Load target configuration
    target_config: dict = {}
    with open(target_config_path, "r") as f:
        target_config = yaml.safe_load(f)
        target_config = {**target_config}

    # Load agent configuration for first agent
    with open(agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)

    # Create FOVAgent instances instead of dictionary configurations
    agents = [
        # FOVAgent(
        #     name=agent_config.get("name", "alpha"),
        #     color=tuple(agent_config.get("color", [80, 160, 255])),
        #     env_size=10,  # Will be updated with actual size
        #     fov_size=agent_config.get("fov_size", 3),
        #     outline_width=agent_config.get("outline_width", 1),
        #     box_scale=agent_config.get("box_scale", 1.0),
        #     show_target_coords=False
        # ),
        # FOVAgent(
        #     name="beta",
        #     color=(0, 255, 200),
        #     env_size=10,  # Will be updated with actual size
        #     fov_size=5,
        #     outline_width=2,
        #     box_scale=0.8,
        #     show_target_coords=False
        # ),
        # FOVAgent(
        #     name="gamma",
        #     color=(255, 200, 20),
        #     env_size=10,  # Will be updated with actual size
        #     fov_size=5,
        #     outline_width=2,
        #     box_scale=0.8,
        #     show_target_coords=False
        # ),
        ObserverAgent(
            name="epsilon",
            color=(100, 100, 255),
            env_size=10,  # Will be updated with actual size
            fov_base_size=3,
            outline_width=2,
            box_scale=0.7,
            show_target_coords=False,
            target_detection_probs= (1, 0.66, 0.33),
            max_altitude=3
        ),
        # ObserverAgent(
        #     name="sigma",
        #     color=(100, 255, 100),
        #     env_size=10,  # Will be updated with actual size
        #     fov_base_size=3,
        #     outline_width=2,
        #     box_scale=0.7,
        #     show_target_coords=False,
        #     target_detection_probs= (1, 0.66, 0.33),
        #     max_altitude=3
        # )
    ]

    size = int(10)
    record = False

    # Update agents' env_size with the actual size
    for agent in agents:
        agent.env_size = size
        # Recreate observation space with correct size
        if isinstance(agent, ObserverAgent):
            # ObserverAgent uses fov_base_size and has additional parameters
            agent._setup_agent(
                fov_base_size=agent.fov_base_size,
                outline_width=agent.outline_width,
                box_scale=agent.box_scale,
                show_target_coords=agent.show_target_coords,
                max_altitude=agent.max_altitude,
                target_detection_probs=agent.target_detection_probs
            )
        else:
            # FOVAgent uses fov_size
            agent._setup_agent(
                fov_size=agent.fov_size,
                outline_width=agent.outline_width,
                box_scale=agent.box_scale,
                show_target_coords=agent.show_target_coords
            )

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
    num_visual_obstacles=2,  # Visual obstacles for ObserverAgent (block view but not movement)
    show_fov_display=True,
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
    
    print("\n" + "="*80)
    print("ENVIRONMENT SETUP".center(80))
    print("="*80)
    print(f"Grid size: {base_env.size}x{base_env.size}")
    print(f"Physical obstacles: {len(base_env._obstacles)} (block movement)")
    print(f"Visual obstacles: {len(base_env._visual_obstacles)} (block ObserverAgent view only)")
    print(f"Possible agents: {base_env.possible_agents}")
    print(f"Action spaces: {base_env.action_spaces}")
    print(f"Observation spaces: {base_env.observation_spaces}")
    
    # Print observation space in a nice ASCII format
    print("\n" + "="*80)
    print("OBSERVATION SPACE DETAILS".center(80))
    print("="*80)
    
    for agent_name in base_env.possible_agents:
        agent_obs_space = base_env.observation_spaces[agent_name]
        agent_obj = base_env._agents_by_name[agent_name]
        
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
            
        # time.sleep(1)

    env.close()