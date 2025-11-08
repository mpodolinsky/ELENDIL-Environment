#!/usr/bin/env python3
"""
Test script for the new radar detection system.

This script tests the radar system to ensure it creates the intended balance
between altitude and detection risk for ObserverAgents.
"""

import os
import sys
import yaml
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel
from agents.observer_agent import ObserverAgent


def test_radar_detection():
    """Test the radar detection system with different scenarios."""
    print("Testing Radar Detection System")
    print("="*50)
    
    # Load configurations
    with open('configs/agent_configs/ground_agent.yaml') as f:
        ground_config = yaml.safe_load(f)
    
    with open('configs/agent_configs/air_observer_agent.yaml') as f:
        air_config = yaml.safe_load(f)
    
    with open('configs/target_configs/target_config.yaml') as f:
        target_config = yaml.safe_load(f)
    
    # Create environment
    env = GridWorldEnvParallel(
        agents=[ground_config, air_config],
        size=15,
        target_config=target_config,
        max_steps=100,
        render_mode=None
    )
    
    print(f"Target radar settings:")
    print(f"  - Radar enabled: {env.target.radar_enabled}")
    print(f"  - Radar range: {env.target.radar_range}")
    print(f"  - Radar sensitivity: {env.target.radar_sensitivity}")
    print(f"  - Altitude factor: {env.target.radar_altitude_factor}")
    print(f"  - Noise level: {env.target.radar_noise_level}")
    print()
    
    # Test different scenarios
    scenarios = [
        {"name": "Ground Agent Close", "agent_type": "ground", "distance": 1, "altitude": 0},
        {"name": "Ground Agent Far", "agent_type": "ground", "distance": 3, "altitude": 0},
        {"name": "ObserverAgent FL1 Close", "agent_type": "observer", "distance": 1, "altitude": 1},
        {"name": "ObserverAgent FL1 Far", "agent_type": "observer", "distance": 3, "altitude": 1},
        {"name": "ObserverAgent FL2 Close", "agent_type": "observer", "distance": 1, "altitude": 2},
        {"name": "ObserverAgent FL2 Far", "agent_type": "observer", "distance": 3, "altitude": 2},
        {"name": "ObserverAgent FL3 Close", "agent_type": "observer", "distance": 1, "altitude": 3},
        {"name": "ObserverAgent FL3 Far", "agent_type": "observer", "distance": 3, "altitude": 3},
    ]
    
    # Run detection tests
    detection_results = defaultdict(list)
    
    for scenario in scenarios:
        print(f"Testing: {scenario['name']}")
        
        # Reset environment
        env.reset()
        
        # Get agents
        ground_agent = env._agents_by_name['alpha_ground']
        observer_agent = env._agents_by_name['epsilon_air']
        
        # Select agent based on scenario
        if scenario['agent_type'] == 'ground':
            test_agent = ground_agent
        else:
            test_agent = observer_agent
            test_agent.altitude = scenario['altitude']
        
        # Position agent at specific distance from target
        target_pos = env.target.location
        if scenario['distance'] == 1:
            # Adjacent to target
            test_agent.location = target_pos + np.array([1, 0])
        elif scenario['distance'] == 3:
            # 3 cells away
            test_agent.location = target_pos + np.array([3, 0])
        
        # Test detection multiple times
        detections = []
        for _ in range(100):  # 100 trials
            detected = env._is_agent_in_target_fov(test_agent)
            detections.append(detected)
        
        detection_rate = sum(detections) / len(detections)
        detection_results[scenario['name']] = detection_rate
        
        print(f"  Detection rate: {detection_rate:.2%}")
        print(f"  Agent altitude: {scenario['altitude']}")
        print(f"  Distance: {scenario['distance']}")
        print()
    
    # Analyze results
    print("Detection Rate Analysis:")
    print("-" * 30)
    
    # Group by altitude
    altitude_groups = defaultdict(list)
    for scenario_name, rate in detection_results.items():
        if "FL1" in scenario_name:
            altitude_groups["FL1"].append(rate)
        elif "FL2" in scenario_name:
            altitude_groups["FL2"].append(rate)
        elif "FL3" in scenario_name:
            altitude_groups["FL3"].append(rate)
        elif "Ground" in scenario_name:
            altitude_groups["Ground"].append(rate)
    
    for altitude, rates in altitude_groups.items():
        avg_rate = np.mean(rates)
        print(f"{altitude}: {avg_rate:.2%} average detection rate")
    
    print()
    
    # Check if radar creates intended balance
    print("Balance Analysis:")
    print("-" * 20)
    
    # Higher altitude should have higher detection rates
    fl1_avg = np.mean(altitude_groups["FL1"])
    fl2_avg = np.mean(altitude_groups["FL2"])
    fl3_avg = np.mean(altitude_groups["FL3"])
    ground_avg = np.mean(altitude_groups["Ground"])
    
    print(f"Ground agents: {ground_avg:.2%}")
    print(f"FL1 (altitude 1): {fl1_avg:.2%}")
    print(f"FL2 (altitude 2): {fl2_avg:.2%}")
    print(f"FL3 (altitude 3): {fl3_avg:.2%}")
    
    # Check if altitude increases detection (as intended)
    altitude_increases_detection = fl1_avg < fl2_avg < fl3_avg
    print(f"\nAltitude increases detection: {altitude_increases_detection}")
    
    if altitude_increases_detection:
        print("✅ Radar system working as intended!")
        print("   Higher altitude = higher detection risk")
    else:
        print("⚠️  Radar system may need tuning")
        print("   Consider adjusting radar_altitude_factor")
    
    env.close()
    return detection_results


def test_radar_parameters():
    """Test different radar parameter configurations."""
    print("\nTesting Radar Parameter Sensitivity")
    print("="*40)
    
    # Test different altitude factors
    altitude_factors = [0.1, 0.3, 0.5, 0.7]
    
    for factor in altitude_factors:
        print(f"\nTesting altitude_factor = {factor}")
        
        # Create custom target config
        target_config = {
            'name': 'test_target',
            'color': [255, 100, 100],
            'movement_speed': 0.3,
            'movement_range': 1,
            'smooth_movement': True,
            'box_cells': 3,
            'outline_width': 3,
            'box_scale': 1,
            'radar_enabled': True,
            'radar_range': 4,
            'radar_sensitivity': 0.8,
            'radar_altitude_factor': factor,
            'radar_noise_level': 0.1
        }
        
        # Create environment with custom config
        env = GridWorldEnvParallel(
            agents=[{"name": "test_observer", "type": "ObserverAgent", "fov_base_size": 3}],
            size=10,
            target_config=target_config,
            max_steps=50,
            render_mode=None
        )
        
        # Test ObserverAgent at different altitudes
        observer = env._agents_by_name['test_observer']
        detection_rates = []
        
        for altitude in [1, 2, 3]:
            observer.altitude = altitude
            observer.location = env.target.location + np.array([2, 0])  # 2 cells away
            
            detections = []
            for _ in range(50):  # 50 trials
                detected = env.target.detect_agent_with_radar(observer)
                detections.append(detected)
            
            detection_rate = sum(detections) / len(detections)
            detection_rates.append(detection_rate)
            print(f"  FL{altitude}: {detection_rate:.2%}")
        
        # Check if altitude increases detection
        increasing = detection_rates[0] < detection_rates[1] < detection_rates[2]
        print(f"  Increasing with altitude: {increasing}")
        
        env.close()


if __name__ == "__main__":
    # Run tests
    detection_results = test_radar_detection()
    test_radar_parameters()
    
    print("\n" + "="*50)
    print("Radar System Test Complete!")
    print("="*50)
