#!/usr/bin/env python3
"""
Simple PettingZoo Compliance Test Runner for ELENDIL Parallel Environment

This script runs the essential PettingZoo compliance tests for the Parallel environment.
"""

import os
import sys
import time
import yaml
import numpy as np
import pygame
from pettingzoo.test import parallel_api_test, parallel_seed_test

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel


def load_configs():
    """Load agent and target configurations."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ground_agent_path = os.path.join(base_dir, "configs", "agent_configs", "ground_agent.yaml")
    air_observer_path = os.path.join(base_dir, "configs", "agent_configs", "air_observer_agent.yaml")
    target_path = os.path.join(base_dir, "configs", "target_configs", "target_config.yaml")
    
    with open(ground_agent_path, "r") as f:
        ground_agent_config = yaml.safe_load(f)
    
    with open(air_observer_path, "r") as f:
        air_observer_config = yaml.safe_load(f)
    
    with open(target_path, "r") as f:
        target_config = yaml.safe_load(f)
    
    return [ground_agent_config, air_observer_config], target_config


def create_env(**kwargs):
    """Create a GridWorldEnvParallel instance."""
    agent_configs, target_config = load_configs()
    
    default_params = {
        "agents": agent_configs,
        "size": 15,
        "render_mode": None,
        "enable_obstacles": True,
        "num_obstacles": 3,
        "num_visual_obstacles": 2,
        "target_config": target_config,
        "max_steps": 100,
        "lambda_fov": 0.5,
        "show_fov_display": False
    }
    
    default_params.update(kwargs)
    return GridWorldEnvParallel(**default_params)


def test_parallel_api():
    """Test Parallel API compliance."""
    print("Testing Parallel API compliance...")
    try:
        env = create_env()
        parallel_api_test(env, num_cycles=100)
        print("✅ Parallel API test PASSED")
        return True
    except Exception as e:
        print(f"❌ Parallel API test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def test_seed_determinism():
    """Test seed determinism."""
    print("Testing seed determinism...")
    try:
        def env_fn():
            return create_env()
        
        parallel_seed_test(env_fn, num_cycles=20)
        print("✅ Seed determinism test PASSED")
        return True
    except Exception as e:
        print(f"❌ Seed determinism test FAILED: {e}")
        return False


def test_performance():
    """Test performance benchmark."""
    print("Testing performance benchmark...")
    try:
        env = create_env()
        observations, infos = env.reset()
        
        start_time = time.time()
        steps = 0
        
        while time.time() - start_time < 5.0:  # Run for 5 seconds
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_spaces[agent].sample()
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            steps += 1
            
            # Break if all agents are done
            if all(terminations.values()) or all(truncations.values()):
                observations, infos = env.reset()
        
        elapsed = time.time() - start_time
        steps_per_second = steps / elapsed
        
        print(f"Performance: {steps} steps in {elapsed:.2f} seconds ({steps_per_second:.2f} steps/sec)")
        print("✅ Performance benchmark test PASSED")
        return True
    except Exception as e:
        print(f"❌ Performance benchmark test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def test_render_modes():
    """Test render modes."""
    print("Testing render modes...")
    try:
        # Test rgb_array mode
        env = create_env(render_mode="rgb_array")
        env.reset()
        rgb_array = env.render()
        assert isinstance(rgb_array, np.ndarray), f"Render should return numpy array, got {type(rgb_array)}"
        assert len(rgb_array.shape) == 3, f"Render should return 3D array, got shape {rgb_array.shape}"
        env.close()
        
        # Test human mode (if display available)
        try:
            env = create_env(render_mode="human")
            env.reset()
            env.render()
            env.close()
        except pygame.error:
            print("⚠️  Human render mode skipped (no display available)")
        
        print("✅ Render modes test PASSED")
        return True
    except Exception as e:
        print(f"❌ Render modes test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("ELENDIL Parallel Environment - PettingZoo Compliance Tests")
    print("="*60)
    
    tests = [
        ("Parallel API", test_parallel_api),
        ("Seed Determinism", test_seed_determinism),
        ("Performance", test_performance),
        ("Render Modes", test_render_modes),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Environment is PettingZoo compliant!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
