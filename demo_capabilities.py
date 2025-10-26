"""
Demonstration script showcasing the capabilities of FOVAgent and ObserverAgent
in the multi-agent grid world environment.

This script demonstrates:
1. FOVAgent navigation and target detection
2. ObserverAgent altitude control and dynamic FOV
3. Visual obstacles (block view) vs Physical obstacles (block movement)
4. Target detection and reward mechanics
5. Agent coordination and exploration
"""

import os
import yaml
import time
import imageio
import numpy as np
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent


def create_demo_scenario():
    """Create and run a scripted demonstration scenario"""
    
    # Load configurations
    agent_config_path = "configs/agent_config.yaml"
    target_config_path = "configs/target_config.yaml"
    
    with open(target_config_path, "r") as f:
        target_config = yaml.safe_load(f)
    
    with open(agent_config_path, "r") as f:
        agent_config = yaml.safe_load(f)
    
    # Create agents with specific configurations for demo
    agents = [
        FOVAgent(
            name="alpha",
            color=(80, 160, 255),  # Blue
            env_size=20,
            fov_size=5,
            outline_width=2,
            box_scale=0.9,
            show_target_coords=False
        ),
        ObserverAgent(
            name="epsilon",
            color=(255, 100, 100),  # Red
            env_size=20,
            fov_base_size=3,
            outline_width=2,
            box_scale=0.8,
            show_target_coords=False,
            target_detection_probs=(1.0, 0.66, 0.33),  # Detection prob by altitude
            max_altitude=3
        ),
    ]
    
    size = 20
    
    # Update agents' env_size
    for agent in agents:
        agent.env_size = size
        if isinstance(agent, ObserverAgent):
            agent._setup_agent(
                fov_base_size=agent.fov_base_size,
                outline_width=agent.outline_width,
                box_scale=agent.box_scale,
                show_target_coords=agent.show_target_coords,
                max_altitude=agent.max_altitude,
                target_detection_probs=agent.target_detection_probs
            )
        else:
            agent._setup_agent(
                fov_size=agent.fov_size,
                outline_width=agent.outline_width,
                box_scale=agent.box_scale,
                show_target_coords=agent.show_target_coords
            )
    
    # Setup recording
    record = True
    os.makedirs("./videos", exist_ok=True)
    
    if record:
        render_mode = "rgb_array"
        frames = []
    else:
        render_mode = "human"
    
    # Create environment
    env = GridWorldEnvMultiAgent(
        render_mode=render_mode,
        size=size,
        agents=agents,
        no_target=False,
        enable_obstacles=True,
        num_obstacles=4,
        num_visual_obstacles=3,
        show_fov_display=True,
        target_config=target_config,
        lambda_fov=0.5
    )
    
    # Reset with fixed seed for reproducibility
    env.reset(seed=42)
    
    # Print demo information
    print("\n" + "="*100)
    print("MULTI-AGENT CAPABILITIES DEMONSTRATION".center(100))
    print("="*100)
    print("\nThis demonstration showcases:")
    print("  ✓ FOVAgent (alpha - BLUE):")
    print("    - Fixed field-of-view (FOV) size")
    print("    - Navigates to search for target")
    print("    - Blocked by physical obstacles (solid dark gray)")
    print()
    print("  ✓ ObserverAgent (epsilon - RED):")
    print("    - Dynamic FOV based on altitude (higher = larger FOV)")
    print("    - Altitude control: INCREASE (5) / REMAIN (6) / REDUCE (7)")
    print("    - Starts at altitude 1, can climb to max altitude 3")
    print("    - Detection probability varies with altitude:")
    print("      • Altitude 1: 100% detection (small FOV)")
    print("      • Altitude 2: 66% detection (medium FOV)")
    print("      • Altitude 3: 33% detection (large FOV)")
    print("    - Can move through visual obstacles (but view is blocked)")
    print()
    print("  ✓ Obstacles:")
    print("    - Physical obstacles (solid dark gray): Block all movement")
    print("    - Visual obstacles (light blue with stripes): Block ObserverAgent view only")
    print()
    print("  ✓ Rewards:")
    print("    - Based on target detection in FOV")
    print("    - Weighted by FOV coverage (lambda_fov = 0.5)")
    print("="*100)
    
    print(f"\nEnvironment Setup:")
    print(f"  Grid size: {env.size}x{env.size}")
    print(f"  Physical obstacles: {len(env._obstacles)}")
    print(f"  Visual obstacles: {len(env._visual_obstacles)}")
    print(f"  Target location: {env.target.location if env.target else 'None'}")
    print(f"  Agents: {env.possible_agents}")
    print()
    
    # Hardcoded action sequence to demonstrate capabilities
    # Actions: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN, 4=NO_OP
    # ObserverAgent also has: 5=INCREASE_ALT, 6=REMAIN_ALT, 7=REDUCE_ALT
    
    demo_script = {
        "alpha": [
            # Phase 1: Initial exploration (10 steps)
            0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
            
            # Phase 2: Navigate around obstacles (10 steps)
            0, 0, 3, 3, 0, 0, 1, 1, 0, 0,
            
            # Phase 3: Search pattern (10 steps)
            1, 1, 2, 2, 1, 1, 0, 0, 1, 1,
            
            # Phase 4: Continue searching (10 steps)
            3, 3, 0, 0, 1, 1, 2, 2, 3, 3,
            
            # Phase 5: Final exploration (10 steps)
            0, 0, 1, 1, 0, 0, 3, 3, 0, 1,
        ],
        "epsilon": [
            # Phase 1: Start at altitude 1, then climb to max
            5, 5,  # Climb to altitude 3 (starts at 1)
            0, 0, 0, 1, 1, 1, 0, 0,  # Move at high altitude
            
            # Phase 2: Move at high altitude (large FOV, low detection)
            0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
            
            # Phase 3: Drop to medium altitude
            7, 7,  # Reduce altitude to 1
            0, 0, 1, 1, 0, 0, 1, 1,
            
            # Phase 4: Navigate at low altitude for best detection
            0, 0, 0, 1, 1, 1, 0, 0,
            
            # Phase 5: Navigate through visual obstacles area
            1, 1, 0, 0, 3, 3, 0, 0, 1, 1,
            
            # Phase 6: Climb and descend demo
            5, 5,  # Climb up to max
            2, 2, 2,  # Move left
            7, 7,  # Drop down
            1, 1, 0, 0,  # Move up and right
        ]
    }
    
    print("="*100)
    print("SIMULATION START".center(100))
    print("="*100)
    print()
    
    step_idx = 0
    phase_markers = [0, 10, 20, 30, 40, 50]
    
    # Run simulation
    for agent in env.agent_iter():
        step_idx += 1
        
        observation, reward, termination, truncation, info = env.last()
        
        # Handle target turn (it moves automatically)
        if agent == "_target":
            env.step(0)  # Dummy action for target
            # Capture frame if recording
            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            continue
        
        if termination or truncation:
            action = None
        else:
            # Get scripted action or random
            agent_actions = demo_script.get(agent, [])
            agent_step = (step_idx - 1) // len(env.possible_agents)
            
            if agent_step < len(agent_actions):
                action = agent_actions[agent_step]
            else:
                # Fall back to random actions after script ends
                action = env.action_spaces[agent].sample()
        
        env.step(action)
        
        # Capture frame if recording
        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Enhanced logging
        agent_obj = env._agents_by_name[agent]
        
        # Build log message
        log_msg = f"[Step {step_idx:3d}] {agent:8s} @ {str(agent_obj.location):8s}"
        
        # Add altitude info for ObserverAgent
        if isinstance(agent_obj, ObserverAgent):
            fov_size = agent_obj.fov_base_size + 2 * agent_obj.altitude
            detection_prob = agent_obj.target_detection_probs[min(agent_obj.altitude - 1, 
                                                                   len(agent_obj.target_detection_probs)-1)]
            log_msg += f" Alt:{agent_obj.altitude} (FOV:{fov_size}x{fov_size}, DetProb:{detection_prob:.0%})"
        else:
            log_msg += " " * 35  # Spacing for alignment
        
        # Action name
        action_names = {0: "RIGHT", 1: "UP", 2: "LEFT", 3: "DOWN", 4: "NO_OP", 
                       5: "ALT↑", 6: "ALT=", 7: "ALT↓"}
        action_str = action_names.get(action, "None") if action is not None else "None"
        log_msg += f" | Act:{action_str:6s}"
        
        # Reward
        log_msg += f" | Rew:{reward:6.3f}"
        
        # Check for target detection
        if env.target is not None and observation is not None:
            if 'obstacles_fov' in observation:
                fov_has_target = 3 in observation['obstacles_fov']
                if fov_has_target:
                    log_msg += " | 🎯 TARGET DETECTED!"
        
        print(log_msg)
        
        # Print phase markers
        if agent_step + 1 in phase_markers and agent == env.possible_agents[-1]:
            phase_num = phase_markers.index(agent_step + 1)
            print(f"\n--- Phase {phase_num} Complete ---\n")
        
        # Stop after 50 full rounds (100 total steps with 2 agents)
        if step_idx >= 100:
            break
        
        if not record:
            time.sleep(0.15)
    
    # Save video if recording
    if record and len(frames) > 0:
        video_path = "./videos/demo_capabilities.mp4"
        print(f"\n{'='*100}")
        print(f"Saving video with {len(frames)} frames to {video_path}...")
        imageio.mimsave(video_path, frames, fps=10)
        print(f"✓ Video saved successfully!")
        print(f"{'='*100}\n")
    
    env.close()
    
    print("\n" + "="*100)
    print("DEMONSTRATION COMPLETE".center(100))
    print("="*100)
    print("\nKey Observations:")
    print("  • FOVAgent maintained constant FOV size throughout")
    print("  • ObserverAgent dynamically adjusted altitude and FOV")
    print("  • Physical obstacles (dark gray) blocked movement for both agents")
    print("  • Visual obstacles (light blue striped) only blocked ObserverAgent's view")
    print("  • Rewards varied based on target detection and FOV coverage")
    print("="*100 + "\n")


if __name__ == "__main__":
    create_demo_scenario()

