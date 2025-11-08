#!/usr/bin/env python3
"""
Simple demonstration of radar system being optional.

This script shows the difference between:
1. Simple 1-cell detection (radar_enabled: false) - DEFAULT
2. Advanced radar system (radar_enabled: true) - OPTIONAL
"""

import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel


def test_detection_mode(radar_enabled, mode_name):
    """Test detection with radar enabled or disabled."""
    print(f"\n{mode_name}")
    print("=" * 40)
    
    # Load agent configs
    with open('configs/agent_configs/ground_agent.yaml') as f:
        ground_config = yaml.safe_load(f)
    with open('configs/agent_configs/air_observer_agent.yaml') as f:
        air_config = yaml.safe_load(f)
    
    # Create target config
    target_config = {
        'name': 'test_target',
        'color': [255, 100, 100],
        'movement_speed': 0.3,
        'movement_range': 1,
        'smooth_movement': True,
        'box_cells': 3,
        'outline_width': 3,
        'box_scale': 1,
        'radar_enabled': radar_enabled,
        'radar_range': 4,
        'radar_sensitivity': 0.6,
        'radar_altitude_factor': 0.4,
        'radar_noise_level': 0.05
    }
    
    # Create environment
    env = GridWorldEnvParallel(
        agents=[ground_config, air_config],
        size=10,
        target_config=target_config,
        max_steps=50,
        render_mode=None
    )
    
    print(f"Radar enabled: {env.target.radar_enabled}")
    
    # Test different scenarios
    env.reset()
    ground_agent = env._agents_by_name['alpha_ground']
    observer_agent = env._agents_by_name['epsilon_air']
    target_pos = env.target.location
    
    scenarios = [
        {"name": "Ground Agent (1 cell)", "agent": ground_agent, "pos": target_pos + [1, 0], "altitude": 0},
        {"name": "Ground Agent (2 cells)", "agent": ground_agent, "pos": target_pos + [2, 0], "altitude": 0},
        {"name": "ObserverAgent FL1 (2 cells)", "agent": observer_agent, "pos": target_pos + [2, 0], "altitude": 1},
        {"name": "ObserverAgent FL3 (2 cells)", "agent": observer_agent, "pos": target_pos + [2, 0], "altitude": 3},
    ]
    
    for scenario in scenarios:
        scenario["agent"].location = scenario["pos"]
        if hasattr(scenario["agent"], 'altitude'):
            scenario["agent"].altitude = scenario["altitude"]
        
        # Test detection
        detected = env._is_agent_in_target_fov(scenario["agent"])
        print(f"  {scenario['name']}: {'DETECTED' if detected else 'NOT DETECTED'}")
    
    env.close()


def main():
    """Demonstrate both detection modes."""
    print("ELENDIL Target Detection Modes")
    print("=" * 50)
    
    # Test simple mode (default)
    test_detection_mode(False, "SIMPLE MODE (Default)")
    
    # Test radar mode (optional)
    test_detection_mode(True, "RADAR MODE (Optional)")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("- Simple mode: Only 1-cell FOV detection")
    print("- Radar mode: FOV + radar detection with altitude effects")
    print("- Default: Simple mode (radar_enabled: false)")
    print("- To enable radar: Set radar_enabled: true in target config")


if __name__ == "__main__":
    main()
