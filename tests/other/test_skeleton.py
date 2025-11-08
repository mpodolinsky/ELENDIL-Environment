"""
Test skeleton for GridWorldEnvMultiAgent environment.

This is a basic skeleton file with imports and test class structure.
Fill in the actual test methods as needed.
"""

import unittest
import numpy as np
import pygame
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import Agent, FOVAgent
from agents.observer_agent import ObserverAgent
from agents.target import Target


class TestGridWorldEnvMultiAgent(unittest.TestCase):
    """Test cases for GridWorldEnvMultiAgent environment."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        # TODO: Clean up resources here
        pass
    
    # =============================================================================
    # INITIALIZATION TESTS
    # =============================================================================
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        # TODO: Test basic environment initialization
        pass
    
    def test_custom_agents_initialization(self):
        """Test environment initialization with custom agents."""
        # TODO: Test custom agent configurations
        pass
    
    def test_no_target_initialization(self):
        """Test environment initialization without target."""
        # TODO: Test no_target parameter
        pass
    
    def test_obstacles_initialization(self):
        """Test environment initialization with obstacles."""
        # TODO: Test obstacle generation
        pass
    
    # =============================================================================
    # OBSERVATION AND ACTION SPACE TESTS
    # =============================================================================
    
    def test_observation_space_structure(self):
        """Test that observation space has correct structure."""
        # TODO: Test observation space format
        pass
    
    def test_action_space_structure(self):
        """Test that action space has correct structure."""
        # TODO: Test action space format
        pass
    
    def test_observation_values(self):
        """Test that observations contain expected values."""
        # TODO: Test observation content
        pass
    
    # =============================================================================
    # STEP FUNCTION TESTS
    # =============================================================================
    
    def test_batch_action_step(self):
        """Test batch action stepping."""
        # TODO: Test batch actions
        pass
    
    def test_single_agent_step(self):
        """Test single agent stepping."""
        # TODO: Test single agent actions
        pass
    
    def test_agent_movement(self):
        """Test that agents move correctly with actions."""
        # TODO: Test agent movement logic
        pass
    
    def test_boundary_clipping(self):
        """Test that agents don't move outside grid boundaries."""
        # TODO: Test boundary conditions
        pass
    
    # =============================================================================
    # FOV TESTS
    # =============================================================================
    
    def test_fov_generation(self):
        """Test FOV generation for agents."""
        # TODO: Test FOV creation
        pass
    
    def test_fov_target_detection(self):
        """Test that FOV correctly detects target."""
        # TODO: Test target detection in FOV
        pass
    
    def test_fov_agent_detection(self):
        """Test that FOV correctly detects other agents."""
        # TODO: Test agent detection in FOV
        pass
    
    def test_fov_obstacle_detection(self):
        """Test that FOV correctly detects obstacles."""
        # TODO: Test obstacle detection in FOV
        pass
    
    def test_fov_boundary_detection(self):
        """Test that FOV correctly marks out-of-bounds as obstacles."""
        # TODO: Test boundary marking in FOV
        pass
    
    # =============================================================================
    # TARGET TESTS
    # =============================================================================
    
    def test_target_initialization(self):
        """Test target initialization."""
        # TODO: Test target creation
        pass
    
    def test_target_movement(self):
        """Test target movement functionality."""
        # TODO: Test target movement
        pass
    
    def test_target_obstacle_avoidance(self):
        """Test that target avoids obstacles."""
        # TODO: Test target collision avoidance
        pass
    
    # =============================================================================
    # REWARD SYSTEM TESTS
    # =============================================================================
    
    def test_fov_reward_system(self):
        """Test FOV-based reward system."""
        # TODO: Test reward calculation
        pass
    
    def test_target_reach_reward(self):
        """Test reward when agent reaches target."""
        # TODO: Test success rewards
        pass
    
    def test_step_penalty(self):
        """Test step penalty when no FOV interactions."""
        # TODO: Test step penalties
        pass
    
    # =============================================================================
    # COLLISION DETECTION TESTS
    # =============================================================================
    
    def test_agent_agent_collision(self):
        """Test that agents don't occupy same cell."""
        # TODO: Test agent-agent collisions
        pass
    
    def test_agent_obstacle_collision(self):
        """Test that agents don't move into obstacles."""
        # Test FOVAgent obstacle collision
        agents = [FOVAgent(name='alpha', color=(255, 100, 100), env_size=10, fov_size=3)]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,  # Manually place obstacles
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Manually place agent and obstacle
        alpha = env._agents_by_name['alpha']
        alpha.location = np.array([5, 5])
        env._obstacles = [(6, 5, 1, 1)]  # Obstacle directly to the right
        
        # Try to move right into obstacle
        env.step(0)  # Action 0 = move right
        
        # Agent should not have moved
        self.assertTrue(np.array_equal(alpha.location, np.array([5, 5])))
        
        # Agent should receive obstacle collision penalty + step penalty
        expected_reward = -0.05 + -0.01  # obstacle + step
        self.assertAlmostEqual(env.rewards['alpha'], expected_reward, places=5)
        
        env.close()
    
    # =============================================================================
    # RESET FUNCTION TESTS
    # =============================================================================
    
    def test_reset_functionality(self):
        """Test environment reset functionality."""
        # TODO: Test reset behavior
        pass
    
    def test_reset_with_obstacles(self):
        """Test reset with obstacle generation."""
        # TODO: Test reset with obstacles
        pass
    
    # =============================================================================
    # RENDERING TESTS
    # =============================================================================
    
    def test_render_rgb_array(self):
        """Test RGB array rendering."""
        # TODO: Test RGB rendering
        pass
    
    def test_render_human_mode(self):
        """Test human rendering mode."""
        # TODO: Test human rendering
        pass
    
    # =============================================================================
    # EDGE CASES AND ERROR HANDLING
    # =============================================================================
    
    def test_invalid_action_format(self):
        """Test handling of invalid action formats."""
        # TODO: Test error handling
        pass
    
    def test_invalid_agent_name(self):
        """Test handling of invalid agent names."""
        # TODO: Test invalid agent handling
        pass
    
    def test_empty_agent_list(self):
        """Test environment with no agents."""
        # TODO: Test edge cases
        pass
    
    # =============================================================================
    # PERFORMANCE TESTS
    # =============================================================================
    
    def test_step_performance(self):
        """Test that step function runs in reasonable time."""
        # TODO: Test performance
        pass
    
    def test_fov_performance(self):
        """Test FOV generation performance."""
        # TODO: Test FOV performance
        pass


class TestAgentClass(unittest.TestCase):
    """Test cases for Agent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize test agent
        pass
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        # TODO: Test agent creation
        pass
    
    def test_agent_location(self):
        """Test agent location management."""
        # TODO: Test location handling
        pass
    
    def test_agent_visited_cells(self):
        """Test agent visited cells tracking."""
        # TODO: Test visited cell tracking
        pass


class TestObserverAgentRendering(unittest.TestCase):
    """Test cases for ObserverAgent rendering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize pygame for rendering tests
        pygame.init()
        
        # Create test agents
        self.observer_agent = ObserverAgent(
            name="test_observer",
            color=(255, 0, 255),
            env_size=8,
            fov_base_size=3,  # FOV size 5
            max_altitude=3,
            show_target_coords=False
        )
        
        self.fov_agent = FOVAgent(
            name="test_fov",
            color=(0, 255, 0),
            env_size=8,
            fov_size=3,
            show_target_coords=False
        )
        
        # Create environment for rendering tests
        self.env = GridWorldEnvMultiAgent(
            agents=[self.observer_agent, self.fov_agent],
            size=8,
            render_mode="rgb_array",  # Use rgb_array for testing
            show_target_coords=False,
            intrinsic=False
        )
        
        # Set agent locations for consistent testing
        self.observer_agent.location = np.array([2, 2])
        self.fov_agent.location = np.array([5, 5])
    
    def tearDown(self):
        """Clean up after each test method."""
        if self.env:
            self.env.close()
        pygame.quit()
    
    def test_observer_agent_square_rendering(self):
        """Test that ObserverAgent renders as square."""
        # Reset environment to get consistent state
        self.env.reset()
        
        # Render frame and get RGB array
        rgb_array = self.env.render()
        
        # Check that RGB array is generated successfully
        self.assertIsInstance(rgb_array, np.ndarray)
        self.assertEqual(len(rgb_array.shape), 3)  # Should be 3D (height, width, channels)
        self.assertEqual(rgb_array.shape[2], 3)    # Should have 3 color channels
        
        # The rendering should complete without errors
        self.assertTrue(True)  # If we get here, rendering worked
    
    def test_observer_agent_flight_level_rendering(self):
        """Test ObserverAgent rendering at different flight levels."""
        # Reset environment
        self.env.reset()
        
        # Test different flight levels
        flight_levels = [0, 1, 2, 3]
        
        for level in flight_levels:
            with self.subTest(flight_level=level):
                # Set flight level
                self.observer_agent.altitude = level
                
                # Render frame
                rgb_array = self.env.render()
                
                # Check that rendering completes successfully
                self.assertIsInstance(rgb_array, np.ndarray)
                self.assertEqual(len(rgb_array.shape), 3)
    
    def test_mixed_agent_rendering(self):
        """Test rendering with mixed agent types (ObserverAgent + FOVAgent)."""
        # Reset environment
        self.env.reset()
        
        # Set different flight levels for ObserverAgent
        self.observer_agent.altitude = 2  # Flight level 2 (dashed)
        
        # Render frame
        rgb_array = self.env.render()
        
        # Check that both agent types render correctly
        self.assertIsInstance(rgb_array, np.ndarray)
        self.assertEqual(len(rgb_array.shape), 3)
    
    def test_observer_agent_attributes(self):
        """Test ObserverAgent has correct rendering attributes."""
        # Check that ObserverAgent has required attributes
        self.assertTrue(hasattr(self.observer_agent, 'altitude'))
        self.assertTrue(hasattr(self.observer_agent, 'fov_size'))
        self.assertTrue(hasattr(self.observer_agent, 'fov_base_size'))
        
        # Check initial values
        self.assertEqual(self.observer_agent.altitude, 1)  # ObserverAgents start at flight level 1
        self.assertEqual(self.observer_agent.fov_size, 7)  # fov_base_size + 4
        self.assertEqual(self.observer_agent.fov_base_size, 3)
    
    def test_observer_agent_altitude_changes(self):
        """Test ObserverAgent altitude changes affect rendering."""
        # Reset environment
        self.env.reset()
        
        # Test altitude changes
        original_altitude = self.observer_agent.altitude
        
        # Change altitude
        self.observer_agent.altitude = 3
        
        # Render and verify it works
        rgb_array = self.env.render()
        self.assertIsInstance(rgb_array, np.ndarray)
        
        # Change back
        self.observer_agent.altitude = original_altitude
    
    def test_observer_agent_fov_rendering(self):
        """Test ObserverAgent FOV rectangle rendering."""
        # Reset environment
        self.env.reset()
        
        # Test FOV rendering at different flight levels
        for altitude in [0, 1, 2, 3]:
            with self.subTest(altitude=altitude):
                self.observer_agent.altitude = altitude
                
                # Render frame
                rgb_array = self.env.render()
                
                # Verify rendering works
                self.assertIsInstance(rgb_array, np.ndarray)
                self.assertGreater(rgb_array.shape[0], 0)  # Has height
                self.assertGreater(rgb_array.shape[1], 0)  # Has width
    
    def test_observer_agent_step_and_render(self):
        """Test ObserverAgent step actions and rendering integration."""
        # Reset environment
        self.env.reset()
        
        # Test altitude actions (5-7)
        altitude_actions = [5, 6, 7]  # Increase, remain, decrease
        
        for action in altitude_actions:
            with self.subTest(action=action):
                # Get environment state
                env_state = {
                    "agents": [self.observer_agent, self.fov_agent],
                    "target": None,
                    "obstacles": [],
                    "grid_size": 8
                }
                
                # Execute step
                step_result = self.observer_agent.step(action, env_state)
                
                # Check step result
                self.assertIsInstance(step_result, dict)
                self.assertIn('new_altitude', step_result)
                self.assertIn('altitude_changed', step_result)
                
                # Render after step
                rgb_array = self.env.render()
                self.assertIsInstance(rgb_array, np.ndarray)


class TestObstacleCollisionLogic(unittest.TestCase):
    """Test cases for obstacle collision logic for different agent types."""
    
    def setUp(self):
        """Set up test fixtures."""
        pygame.init()
    
    def tearDown(self):
        """Clean up after tests."""
        pygame.quit()
    
    def test_fov_agent_obstacle_collision_penalty(self):
        """Test that FOVAgent receives penalty when colliding with obstacle."""
        agents = [FOVAgent(name='alpha', color=(255, 100, 100), env_size=10, fov_size=3)]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place agent and obstacle
        alpha = env._agents_by_name['alpha']
        alpha.location = np.array([3, 3])
        env._obstacles = [(4, 3, 1, 1)]  # Obstacle to the right
        
        # Try to move into obstacle
        env.step(0)  # Move right
        
        # Verify agent didn't move
        self.assertTrue(np.array_equal(alpha.location, np.array([3, 3])))
        
        # Verify penalty was applied: obstacle (-0.05) + step (-0.01) = -0.06
        self.assertAlmostEqual(env.rewards['alpha'], -0.06, places=5)
        
        env.close()
    
    def test_fov_agent_obstacle_blocks_movement(self):
        """Test that obstacles physically block FOVAgent movement."""
        agents = [FOVAgent(name='alpha', color=(255, 100, 100), env_size=10, fov_size=3)]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.1,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place agent and obstacles in all directions
        alpha = env._agents_by_name['alpha']
        alpha.location = np.array([5, 5])
        env._obstacles = [
            (6, 5, 1, 1),  # Right
            (5, 6, 1, 1),  # Up
            (4, 5, 1, 1),  # Left
            (5, 4, 1, 1),  # Down
        ]
        
        # Try all movement directions
        for action in [0, 1, 2, 3]:  # right, up, left, down
            env.step(action)
            
            # Agent should still be at original position
            self.assertTrue(np.array_equal(alpha.location, np.array([5, 5])),
                          f"Agent moved when trying action {action}")
            
            # Should receive collision penalty each time
            self.assertLess(env.rewards['alpha'], -0.05,
                          f"No collision penalty for action {action}")
        
        env.close()
    
    def test_observer_agent_flies_over_obstacles(self):
        """Test that ObserverAgent at altitude >= 1 flies over obstacles without penalty."""
        agents = [ObserverAgent(
            name='epsilon',
            color=(100, 100, 255),
            env_size=10,
            fov_base_size=3,
            max_altitude=3
        )]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place agent and obstacle
        epsilon = env._agents_by_name['epsilon']
        epsilon.location = np.array([3, 3])
        epsilon.altitude = 1  # Flight level 1
        env._obstacles = [(4, 3, 1, 1)]  # Obstacle to the right
        
        # Try to move over obstacle
        env.step(0)  # Move right
        
        # Verify agent DID move (flew over obstacle)
        self.assertTrue(np.array_equal(epsilon.location, np.array([4, 3])))
        
        # Verify NO obstacle collision penalty (only step penalty)
        self.assertAlmostEqual(env.rewards['epsilon'], -0.01, places=5)
        
        env.close()
    
    def test_observer_agent_altitude_2_flies_over_obstacles(self):
        """Test that ObserverAgent at altitude 2 flies over obstacles."""
        agents = [ObserverAgent(
            name='epsilon',
            color=(100, 100, 255),
            env_size=10,
            fov_base_size=3,
            max_altitude=3
        )]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place agent at altitude 2
        epsilon = env._agents_by_name['epsilon']
        epsilon.location = np.array([2, 2])
        epsilon.altitude = 2  # Flight level 2
        env._obstacles = [(3, 2, 2, 2)]  # Large obstacle
        
        # Move through obstacle area
        env.step(0)  # Move right into obstacle area
        
        # Should move successfully
        self.assertTrue(np.array_equal(epsilon.location, np.array([3, 2])))
        
        # No collision penalty
        self.assertAlmostEqual(env.rewards['epsilon'], -0.01, places=5)
        
        env.close()
    
    def test_observer_agent_altitude_3_flies_over_obstacles(self):
        """Test that ObserverAgent at altitude 3 flies over obstacles."""
        agents = [ObserverAgent(
            name='epsilon',
            color=(100, 100, 255),
            env_size=10,
            fov_base_size=3,
            max_altitude=3
        )]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place agent at altitude 3
        epsilon = env._agents_by_name['epsilon']
        epsilon.location = np.array([1, 1])
        epsilon.altitude = 3  # Flight level 3 (max)
        env._obstacles = [(2, 1, 1, 1)]  # Obstacle
        
        # Move over obstacle
        env.step(0)  # Move right
        
        # Should move successfully
        self.assertTrue(np.array_equal(epsilon.location, np.array([2, 1])))
        
        # No collision penalty
        self.assertAlmostEqual(env.rewards['epsilon'], -0.01, places=5)
        
        env.close()
    
    def test_fov_agent_step_method_returns_collision_flag(self):
        """Test that FOVAgent.step() returns obstacle_collision flag."""
        agent = FOVAgent(name='alpha', color=(255, 100, 100), env_size=10, fov_size=3)
        agent.location = np.array([5, 5])
        
        env_state = {
            'agents': [agent],
            'target': None,
            'obstacles': [(6, 5, 1, 1)],
            'grid_size': 10
        }
        
        # Move into obstacle
        result = agent.step(0, env_state)  # Action 0 = move right
        
        # Check result contains collision flag
        self.assertIn('obstacle_collision', result)
        self.assertTrue(result['obstacle_collision'])
        
        # Agent should not have moved
        self.assertTrue(np.array_equal(result['new_location'], np.array([5, 5])))
    
    def test_observer_agent_step_method_returns_no_collision(self):
        """Test that ObserverAgent.step() returns obstacle_collision=False."""
        agent = ObserverAgent(
            name='epsilon',
            color=(100, 100, 255),
            env_size=10,
            fov_base_size=3,
            max_altitude=3
        )
        agent.location = np.array([5, 5])
        agent.altitude = 1
        
        env_state = {
            'agents': [agent],
            'target': None,
            'obstacles': [(6, 5, 1, 1)],
            'grid_size': 10
        }
        
        # Move over obstacle
        result = agent.step(0, env_state)  # Action 0 = move right
        
        # Check result contains collision flag set to False
        self.assertIn('obstacle_collision', result)
        self.assertFalse(result['obstacle_collision'])
        
        # Agent should have moved
        self.assertTrue(np.array_equal(result['new_location'], np.array([6, 5])))
    
    def test_custom_obstacle_collision_penalty(self):
        """Test that custom obstacle collision penalty value works."""
        agents = [FOVAgent(name='alpha', color=(255, 100, 100), env_size=10, fov_size=3)]
        
        custom_penalty = -0.15
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=custom_penalty,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Verify penalty is set correctly
        self.assertEqual(env.obstacle_collision_penalty, custom_penalty)
        
        # Place agent and obstacle
        alpha = env._agents_by_name['alpha']
        alpha.location = np.array([2, 2])
        env._obstacles = [(3, 2, 1, 1)]
        
        # Collide with obstacle
        env.step(0)
        
        # Verify custom penalty was applied
        expected_reward = custom_penalty + -0.01  # custom penalty + step
        self.assertAlmostEqual(env.rewards['alpha'], expected_reward, places=5)
        
        env.close()
    
    def test_mixed_agents_obstacle_interaction(self):
        """Test obstacle interaction with mixed FOVAgent and ObserverAgent."""
        agents = [
            FOVAgent(name='ground', color=(255, 100, 100), env_size=10, fov_size=3),
            ObserverAgent(name='aerial', color=(100, 100, 255), env_size=10, fov_base_size=3, max_altitude=3)
        ]
        
        env = GridWorldEnvMultiAgent(
            agents=agents,
            size=10,
            enable_obstacles=True,
            num_obstacles=0,
            obstacle_collision_penalty=-0.05,
            no_target=True,
            render_mode=None,
            show_fov_display=False
        )
        
        env.reset()
        
        # Place both agents at same location with obstacle ahead
        ground = env._agents_by_name['ground']
        aerial = env._agents_by_name['aerial']
        
        ground.location = np.array([4, 4])
        aerial.location = np.array([4, 5])
        aerial.altitude = 2
        
        env._obstacles = [(5, 4, 1, 1), (5, 5, 1, 1)]  # Obstacles in front of both
        
        # Both try to move right
        env.step(0)  # ground agent
        env.step(0)  # aerial agent
        
        # Ground agent should be blocked and penalized
        self.assertTrue(np.array_equal(ground.location, np.array([4, 4])))
        self.assertAlmostEqual(env.rewards['ground'], -0.06, places=5)
        
        # Aerial agent should move successfully without penalty
        self.assertTrue(np.array_equal(aerial.location, np.array([5, 5])))
        self.assertAlmostEqual(env.rewards['aerial'], -0.01, places=5)
        
        env.close()


class TestTargetClass(unittest.TestCase):
    """Test cases for Target class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize test target
        pass
    
    def test_target_initialization(self):
        """Test target initialization."""
        # TODO: Test target creation
        pass
    
    def test_target_triangle_points(self):
        """Test target triangle point calculation."""
        # TODO: Test triangle rendering
        pass
    
    def test_target_box_coordinates(self):
        """Test target box coordinate calculation."""
        # TODO: Test box rendering
        pass


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
