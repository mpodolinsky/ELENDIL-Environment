#!/usr/bin/env python3
"""
Final PettingZoo Compliance Test for ELENDIL Parallel Environment

This script runs the essential PettingZoo compliance tests for the Parallel environment.
Focuses on the core requirements: API compliance, basic functionality, and performance.
"""

import os
import sys
import time
import yaml
import numpy as np
from pettingzoo.test import parallel_api_test

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
    """Test Parallel API compliance using PettingZoo's official test."""
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


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic functionality...")
    try:
        env = create_env()
        
        # Test reset
        observations, infos = env.reset()
        assert isinstance(observations, dict), "Reset should return observations dict"
        assert isinstance(infos, dict), "Reset should return infos dict"
        
        # Test step
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Check return types
        assert isinstance(observations, dict), "Step should return observations dict"
        assert isinstance(rewards, dict), "Step should return rewards dict"
        assert isinstance(terminations, dict), "Step should return terminations dict"
        assert isinstance(truncations, dict), "Step should return truncations dict"
        assert isinstance(infos, dict), "Step should return infos dict"
        
        # Check that all agents have results
        for agent in env.agents:
            assert agent in observations, f"Agent {agent} missing from observations"
            assert agent in rewards, f"Agent {agent} missing from rewards"
            assert agent in terminations, f"Agent {agent} missing from terminations"
            assert agent in truncations, f"Agent {agent} missing from truncations"
            assert agent in infos, f"Agent {agent} missing from infos"
        
        print("✅ Basic functionality test PASSED")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def test_observation_spaces():
    """Test observation space consistency."""
    print("Testing observation spaces...")
    try:
        env = create_env()
        observations, infos = env.reset()
        
        # Check observation space consistency
        for agent in env.agents:
            obs_space = env.observation_spaces[agent]
            obs = observations[agent]
            
            # Check that observation matches space
            assert obs_space.contains(obs), f"Observation for {agent} doesn't match space"
            
            # Check specific observation components
            if 'agent' in obs_space.spaces:
                agent_pos = obs['agent']
                assert isinstance(agent_pos, np.ndarray), "Agent position should be numpy array"
                assert agent_pos.shape == (2,), "Agent position should be 2D coordinates"
            
            if 'obstacles_fov' in obs_space.spaces:
                fov = obs['obstacles_fov']
                assert isinstance(fov, np.ndarray), "FOV should be numpy array"
                assert len(fov.shape) == 2, "FOV should be 2D"
        
        print("✅ Observation spaces test PASSED")
        return True
    except Exception as e:
        print(f"❌ Observation spaces test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def test_action_spaces():
    """Test action space consistency."""
    print("Testing action spaces...")
    try:
        env = create_env()
        observations, infos = env.reset()
        
        # Test action sampling and validation
        actions = {}
        for agent in env.agents:
            action_space = env.action_spaces[agent]
            action = action_space.sample()
            actions[agent] = action
            
            # Check that action is valid
            assert action_space.contains(action), f"Action {action} invalid for {agent}"
        
        # Test step with sampled actions
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print("✅ Action spaces test PASSED")
        return True
    except Exception as e:
        print(f"❌ Action spaces test FAILED: {e}")
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
            result = env.render()
            assert result is None, f"Human render should return None, got {type(result)}"
            env.close()
        except Exception as e:
            print(f"⚠️  Human render mode skipped: {e}")
        
        print("✅ Render modes test PASSED")
        return True
    except Exception as e:
        print(f"❌ Render modes test FAILED: {e}")
        return False


def test_performance():
    """Test environment performance."""
    print("Testing performance...")
    try:
        env = create_env()
        observations, infos = env.reset()
        
        start_time = time.time()
        steps = 0
        
        while time.time() - start_time < 3.0:  # Run for 3 seconds
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
        
        # Performance should be reasonable (at least 100 steps/sec)
        assert steps_per_second > 100, f"Performance too slow: {steps_per_second:.2f} steps/sec"
        
        print("✅ Performance test PASSED")
        return True
    except Exception as e:
        print(f"❌ Performance test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def test_environment_properties():
    """Test environment properties and metadata."""
    print("Testing environment properties...")
    try:
        env = create_env()
        
        # Test required properties
        assert hasattr(env, 'agents'), "Environment should have 'agents' property"
        assert hasattr(env, 'possible_agents'), "Environment should have 'possible_agents' property"
        assert hasattr(env, 'action_spaces'), "Environment should have 'action_spaces' property"
        assert hasattr(env, 'observation_spaces'), "Environment should have 'observation_spaces' property"
        assert hasattr(env, 'metadata'), "Environment should have 'metadata' property"
        
        # Test metadata
        assert 'render_modes' in env.metadata, "Metadata should include 'render_modes'"
        assert 'render_fps' in env.metadata, "Metadata should include 'render_fps'"
        
        # Test agent lists
        assert isinstance(env.agents, list), "agents should be a list"
        assert isinstance(env.possible_agents, list), "possible_agents should be a list"
        assert len(env.agents) <= len(env.possible_agents), "agents should be subset of possible_agents"
        
        # Test action_space and observation_space methods
        for agent in env.agents:
            action_space = env.action_space(agent)
            obs_space = env.observation_space(agent)
            assert action_space == env.action_spaces[agent], "action_space method should match action_spaces dict"
            assert obs_space == env.observation_spaces[agent], "observation_space method should match observation_spaces dict"
        
        print("✅ Environment properties test PASSED")
        return True
    except Exception as e:
        print(f"❌ Environment properties test FAILED: {e}")
        return False
    finally:
        if 'env' in locals():
            env.close()


def main():
    """Run all compliance tests."""
    print("ELENDIL Parallel Environment - PettingZoo Compliance Tests")
    print("="*70)
    
    tests = [
        ("Parallel API", test_parallel_api),
        ("Basic Functionality", test_basic_functionality),
        ("Observation Spaces", test_observation_spaces),
        ("Action Spaces", test_action_spaces),
        ("Render Modes", test_render_modes),
        ("Performance", test_performance),
        ("Environment Properties", test_environment_properties),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Environment is PettingZoo compliant!")
        print("\nYour ELENDIL Parallel environment meets PettingZoo standards and is ready for use!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
