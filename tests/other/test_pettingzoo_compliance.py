#!/usr/bin/env python3
"""
PettingZoo Compliance Tests for ELENDIL Parallel Environment

This module implements comprehensive compliance tests following PettingZoo's testing framework
to ensure the GridWorldEnvParallel environment meets all API standards.

Based on: https://pettingzoo.farama.org/content/environment_tests/
"""

import os
import sys
import time
import yaml
import numpy as np
import pygame
from typing import Dict, List, Any, Optional, Callable
from pettingzoo.test import parallel_api_test, parallel_seed_test, render_test, performance_benchmark

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel
from agents.observer_agent import ObserverAgent


class PettingZooComplianceTester:
    """Comprehensive compliance tester for ELENDIL Parallel Environment."""
    
    def __init__(self):
        """Initialize the compliance tester."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_results = {}
        
    def load_configs(self) -> tuple:
        """Load agent and target configurations."""
        ground_agent_path = os.path.join(self.base_dir, "configs", "agent_configs", "ground_agent.yaml")
        air_observer_path = os.path.join(self.base_dir, "configs", "agent_configs", "air_observer_agent.yaml")
        target_path = os.path.join(self.base_dir, "configs", "target_configs", "target_config.yaml")
        
        with open(ground_agent_path, "r") as f:
            ground_agent_config = yaml.safe_load(f)
        
        with open(air_observer_path, "r") as f:
            air_observer_config = yaml.safe_load(f)
        
        with open(target_path, "r") as f:
            target_config = yaml.safe_load(f)
        
        return [ground_agent_config, air_observer_config], target_config
    
    def create_env(self, **kwargs) -> GridWorldEnvParallel:
        """Create a GridWorldEnvParallel instance with default parameters."""
        agent_configs, target_config = self.load_configs()
        
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
        
        # Override with any provided parameters
        default_params.update(kwargs)
        
        return GridWorldEnvParallel(**default_params)
    
    def test_parallel_api_compliance(self, num_cycles: int = 1000) -> bool:
        """Test Parallel API compliance using PettingZoo's official test."""
        print("="*80)
        print("TESTING PARALLEL API COMPLIANCE".center(80))
        print("="*80)
        
        try:
            env = self.create_env()
            parallel_api_test(env, num_cycles=num_cycles)
            print("✅ Parallel API compliance test PASSED")
            self.test_results["parallel_api"] = True
            return True
        except Exception as e:
            print(f"❌ Parallel API compliance test FAILED: {e}")
            self.test_results["parallel_api"] = False
            return False
        finally:
            if 'env' in locals():
                env.close()
    
    def test_seed_determinism(self, num_cycles: int = 50) -> bool:
        """Test seed determinism for reproducible behavior."""
        print("\n" + "="*80)
        print("TESTING SEED DETERMINISM".center(80))
        print("="*80)
        
        try:
            # Create environment factory function
            def env_fn():
                return self.create_env()
            
            parallel_seed_test(env_fn, num_cycles=num_cycles)
            print("✅ Seed determinism test PASSED")
            self.test_results["seed_determinism"] = True
            return True
        except Exception as e:
            print(f"❌ Seed determinism test FAILED: {e}")
            self.test_results["seed_determinism"] = False
            return False
    
    def test_render_modes(self) -> bool:
        """Test all supported render modes."""
        print("\n" + "="*80)
        print("TESTING RENDER MODES".center(80))
        print("="*80)
        
        try:
            # Test human mode
            print("Testing 'human' render mode...")
            env_human = self.create_env(render_mode="human")
            env_human.reset()
            env_human.render()
            env_human.close()
            print("✅ 'human' render mode PASSED")
            
            # Test rgb_array mode
            print("Testing 'rgb_array' render mode...")
            env_rgb = self.create_env(render_mode="rgb_array")
            env_rgb.reset()
            rgb_array = env_rgb.render()
            assert isinstance(rgb_array, np.ndarray), "rgb_array should return numpy array"
            assert len(rgb_array.shape) == 3, "rgb_array should be 3D (height, width, channels)"
            env_rgb.close()
            print("✅ 'rgb_array' render mode PASSED")
            
            # Test None mode (no rendering)
            print("Testing None render mode...")
            env_none = self.create_env(render_mode=None)
            env_none.reset()
            env_none.close()
            print("✅ None render mode PASSED")
            
            self.test_results["render_modes"] = True
            return True
            
        except Exception as e:
            print(f"❌ Render modes test FAILED: {e}")
            self.test_results["render_modes"] = False
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Test environment performance."""
        print("\n" + "="*80)
        print("TESTING PERFORMANCE BENCHMARK".center(80))
        print("="*80)
        
        try:
            env = self.create_env()
            print("Running 5-second performance benchmark...")
            performance_benchmark(env)
            print("✅ Performance benchmark test PASSED")
            self.test_results["performance"] = True
            return True
        except Exception as e:
            print(f"❌ Performance benchmark test FAILED: {e}")
            self.test_results["performance"] = False
            return False
        finally:
            if 'env' in locals():
                env.close()
    
    def test_observation_spaces(self) -> bool:
        """Test observation space consistency."""
        print("\n" + "="*80)
        print("TESTING OBSERVATION SPACES".center(80))
        print("="*80)
        
        try:
            env = self.create_env()
            observations, infos = env.reset()
            
            # Check that all agents have observations
            for agent in env.agents:
                assert agent in observations, f"Agent {agent} missing from observations"
                assert agent in infos, f"Agent {agent} missing from infos"
            
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
            self.test_results["observation_spaces"] = True
            return True
            
        except Exception as e:
            print(f"❌ Observation spaces test FAILED: {e}")
            self.test_results["observation_spaces"] = False
            return False
        finally:
            if 'env' in locals():
                env.close()
    
    def test_action_spaces(self) -> bool:
        """Test action space consistency."""
        print("\n" + "="*80)
        print("TESTING ACTION SPACES".center(80))
        print("="*80)
        
        try:
            env = self.create_env()
            
            # Test action sampling
            observations, infos = env.reset()
            actions = {}
            
            for agent in env.agents:
                action_space = env.action_spaces[agent]
                action = action_space.sample()
                actions[agent] = action
                
                # Check that action is valid
                assert action_space.contains(action), f"Action {action} invalid for {agent}"
            
            # Test step with sampled actions
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check that all agents have results
            for agent in env.agents:
                assert agent in rewards, f"Agent {agent} missing from rewards"
                assert agent in terminations, f"Agent {agent} missing from terminations"
                assert agent in truncations, f"Agent {agent} missing from truncations"
                assert agent in infos, f"Agent {agent} missing from infos"
            
            print("✅ Action spaces test PASSED")
            self.test_results["action_spaces"] = True
            return True
            
        except Exception as e:
            print(f"❌ Action spaces test FAILED: {e}")
            self.test_results["action_spaces"] = False
            return False
        finally:
            if 'env' in locals():
                env.close()
    
    def test_agent_types(self) -> bool:
        """Test different agent type configurations."""
        print("\n" + "="*80)
        print("TESTING AGENT TYPE CONFIGURATIONS".center(80))
        print("="*80)
        
        try:
            agent_configs, target_config = self.load_configs()
            
            # Test with only ground agents
            print("Testing with only ground agents...")
            ground_only_env = GridWorldEnvParallel(
                agents=[agent_configs[0]],  # Only ground agent
                size=10,
                target_config=target_config,
                max_steps=50
            )
            ground_only_env.reset()
            ground_only_env.close()
            print("✅ Ground agents only test PASSED")
            
            # Test with only observer agents
            print("Testing with only observer agents...")
            observer_only_env = GridWorldEnvParallel(
                agents=[agent_configs[1]],  # Only observer agent
                size=10,
                target_config=target_config,
                max_steps=50
            )
            observer_only_env.reset()
            observer_only_env.close()
            print("✅ Observer agents only test PASSED")
            
            # Test with mixed agents
            print("Testing with mixed agents...")
            mixed_env = GridWorldEnvParallel(
                agents=agent_configs,  # Both agent types
                size=10,
                target_config=target_config,
                max_steps=50
            )
            mixed_env.reset()
            mixed_env.close()
            print("✅ Mixed agents test PASSED")
            
            self.test_results["agent_types"] = True
            return True
            
        except Exception as e:
            print(f"❌ Agent types test FAILED: {e}")
            self.test_results["agent_types"] = False
            return False
    
    def test_environment_properties(self) -> bool:
        """Test environment properties and metadata."""
        print("\n" + "="*80)
        print("TESTING ENVIRONMENT PROPERTIES".center(80))
        print("="*80)
        
        try:
            env = self.create_env()
            
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
            
            print("✅ Environment properties test PASSED")
            self.test_results["environment_properties"] = True
            return True
            
        except Exception as e:
            print(f"❌ Environment properties test FAILED: {e}")
            self.test_results["environment_properties"] = False
            return False
        finally:
            if 'env' in locals():
                env.close()
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all compliance tests."""
        print("ELENDIL PARALLEL ENVIRONMENT - PETTINGZOO COMPLIANCE TESTS")
        print("="*80)
        
        # Run all tests
        self.test_parallel_api_compliance(num_cycles=100)
        self.test_seed_determinism(num_cycles=20)
        self.test_render_modes()
        self.test_performance_benchmark()
        self.test_observation_spaces()
        self.test_action_spaces()
        self.test_agent_types()
        self.test_environment_properties()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY".center(80))
        print("="*80)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Environment is PettingZoo compliant!")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return self.test_results


def main():
    """Main function to run compliance tests."""
    tester = PettingZooComplianceTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
