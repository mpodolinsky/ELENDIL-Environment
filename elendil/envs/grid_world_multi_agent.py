from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.env import ParallelEnv

from typing import Optional, List, Dict, Union, Any, Tuple

from agents.agents import Agent, FOVAgent
from agents.observer_agent import ObserverAgent
from agents.target import Target

# TODO review the agent FOV, I want it to represent the agent's perspective in the world
# TODO review the code and streamline. Become familiar with the codebase.
# TODO add the fact that agents can't see behind obstacles.
# Note: Target coordinates are now optional via show_target_coords parameter (default: False)

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    NO_OP = 4

# class GridWorldEnvMultiAgent(AECEnv):
#     ''' A multi-agent grid world environment where agents navigate to find a target while avoiding obstacles.
    
#     This environment follows the PettingZoo AEC (Agent-Environment-Cycle) interface, where agents
#     take turns acting sequentially. Each agent receives individual rewards based on their FOV detection.
    
#     Input parameters:
#      - size: The size of the grid (size x size)
#      - max_steps: Maximum steps per episode
#      - render_mode: Rendering mode ("human", "rgb_array", or None)
#      - show_fov_display: Whether to display agent FOVs in human render mode
#      - intrinsic: Whether to use intrinsic exploration rewards
#      - lambda_fov: Weighting factor for FOV-based rewards (0 to 1)
#      - show_target_coords: Whether agents can see target coordinates in their observations
#      - obstacle_collision_penalty: Penalty applied when agent collides with obstacle (default: -0.05)
#      - no_target: If True, no target is spawned
#      - agents: List of agent configurations (dicts) or Agent instances. Supports any mix of:
#          * FOVAgent instances (pre-instantiated)
#          * ObserverAgent instances (pre-instantiated)
#          * Dict configs for FOVAgent: {"type": "FOVAgent", "name": str, "color": tuple, "fov_size": int, ...}
#          * Dict configs for ObserverAgent: {"type": "ObserverAgent", "name": str, "color": tuple, 
#                                            "fov_base_size": int, "max_altitude": int, 
#                                            "target_detection_probs": tuple, ...}
#          If None, creates 2 default FOVAgents.
#      - target_config: Configuration dictionary for the target
#      - enable_obstacles: Whether to enable obstacles in the environment
#      - num_obstacles: Number of physical obstacles to generate (block movement)
#      - num_visual_obstacles: Number of visual obstacles to generate (block ObserverAgent view only)
     
#      Agent Configuration Examples:
#          # FOVAgent config
#          {"name": "alpha", "type": "FOVAgent", "color": (80, 160, 255), "fov_size": 5}
         
#          # ObserverAgent config
#          {"name": "observer1", "type": "ObserverAgent", "color": (255, 100, 100), 
#           "fov_base_size": 3, "max_altitude": 3, "target_detection_probs": (1.0, 0.66, 0.33)}'''
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

#     def __init__(
#         self,

#         # General settings
#         size=5,
#         max_steps: int = 500,
#         render_mode=None,
#         show_fov_display: bool = True,

#         # Rewards and target settings
#         intrinsic=False,
#         lambda_fov: float = 0.5,
#         show_target_coords: bool = False,
#         obstacle_collision_penalty: float = -0.05,

#         # Agents and target settings
#         no_target=False,
#         agents: Optional[List[Union[Dict, 'Agent']]] = None,
#         target_config: Optional[Dict] = None,

#         # Obstacles
#         enable_obstacles: bool = False,
#         num_obstacles: int = 0,
#         num_visual_obstacles: int = 0,

#     ):
#         self.size = size  # The size of the square grid (default to 5x5)
#         self.window_size = 512  # The size of the PyGame window
#         self.no_target = no_target  # If True, no target is spawned
#         self.enable_obstacles = bool(enable_obstacles)
#         self.num_obstacles = int(max(0, num_obstacles))
#         self.num_visual_obstacles = int(max(0, num_visual_obstacles))
#         self.show_fov_display = bool(show_fov_display)
#         self.lambda_fov = max(-0.5, min(0.5, float(lambda_fov)))  # Clamp between 0 and 1
#         self.show_target_coords = bool(show_target_coords)  # If True, agents can see target coordinates
#         self.obstacle_collision_penalty = float(obstacle_collision_penalty)  # Penalty for hitting obstacles
        
#         # FOV display state
#         self._step_count = 0
#         self.max_steps = max_steps

#         # Detection state tracking for visual indicators
#         self._agent_detects_target = {}  # Track which agents detect the target
#         self._target_detects_agent = {}  # Track which agents the target detects

#         # Build agents list (support custom configs or instances)
#         default_colors = [(80, 160, 255), (255, 180, 60), (100, 200, 100), (255, 100, 100)]
#         self.agent_list: List[Union[FOVAgent, ObserverAgent]] = []

#         if agents is None:
#             # Auto-generate 2 default FOVAgents
#             count = 2
#             for i in range(count):
#                 name = f"agent_{i+1}"
#                 color = default_colors[i % len(default_colors)]
#                 agent = FOVAgent(
#                     name=name,
#                     color=color,
#                     env_size=self.size,
#                     fov_size=3,
#                     outline_width=1,
#                     box_scale=0.8,
#                     show_target_coords=self.show_target_coords
#                 )
#                 self.agent_list.append(agent)
#         else:
#             for idx, agent_or_cfg in enumerate(agents):
#                 # Check if it's already an agent instance or a configuration dict
#                 if isinstance(agent_or_cfg, (FOVAgent, ObserverAgent, Agent)):
#                     # It's already an agent object - use it directly
#                     self.agent_list.append(agent_or_cfg)
#                 elif isinstance(agent_or_cfg, dict):
#                     # It's a configuration dictionary - instantiate the agent
#                     cfg = agent_or_cfg
#                     name = cfg.get("name", f"agent_{idx+1}")
#                     color = tuple(cfg.get("color", default_colors[idx % len(default_colors)]))
#                     outline_width = int(cfg.get("outline_width", 2))
#                     box_scale = float(cfg.get("box_scale", 0.8))
#                     show_coords = cfg.get("show_target_coords", self.show_target_coords)
                    
#                     # Determine agent type from config
#                     agent_type = cfg.get("type", "FOVAgent")
                    
#                     # Check for ObserverAgent-specific parameters
#                     has_observer_params = any(key in cfg for key in 
#                                              ["fov_base_size", "max_altitude", "target_detection_probs"])
                    
#                     if agent_type == "ObserverAgent" or has_observer_params:
#                         # Create ObserverAgent
#                         agent = ObserverAgent(
#                             name=name,
#                             color=color,
#                             env_size=self.size,
#                             fov_base_size=int(cfg.get("fov_base_size", 3)),
#                             outline_width=outline_width,
#                             box_scale=box_scale,
#                             show_target_coords=show_coords,
#                             max_altitude=int(cfg.get("max_altitude", 3)),
#                             target_detection_probs=tuple(cfg.get("target_detection_probs", (1.0, 0.66, 0.33)))
#                         )
#                     else:
#                         # Create FOVAgent
#                         agent = FOVAgent(
#                             name=name,
#                             color=color,
#                             env_size=self.size,
#                             fov_size=int(cfg.get("fov_size", 3)),
#                             outline_width=outline_width,
#                             box_scale=box_scale,
#                             show_target_coords=show_coords
#                         )
                    
#                     self.agent_list.append(agent)
#                 else:
#                     raise ValueError(f"Agent at index {idx} must be either an agent instance or a configuration dict, got {type(agent_or_cfg)}")

#         # Public names and lookup
#         self.agents: List[str] = [a.name for a in self.agent_list]
#         self._agents_by_name: Dict[str, Agent] = {a.name: a for a in self.agent_list}

#         # AEC required properties
#         # Add target to the agent cycle so it acts after all agents
#         self.possible_agents = self.agents.copy()
#         if not self.no_target:
#             self.possible_agents.append("_target")
        
#         self.agent_selection = self.possible_agents[0] if self.possible_agents else None
#         self.rewards = {agent: 0 for agent in self.possible_agents}
#         self.terminations = {agent: False for agent in self.possible_agents}
#         self.truncations = {agent: False for agent in self.possible_agents}
#         self.infos = {agent: {} for agent in self.possible_agents}
#         self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

#         # Per-agent spaces (AEC format)
#         self.action_spaces = {a.name: a.get_action_space() if hasattr(a, 'get_action_space') else a.action_space for a in self.agent_list}
#         self.observation_spaces = {a.name: a.get_observation_space() if hasattr(a, 'get_observation_space') else a.observation_space for a in self.agent_list}
        
#         # Add dummy action/observation space for target (it doesn't use them but needed for AEC)
#         if not self.no_target:
#             self.action_spaces["_target"] = spaces.Discrete(1)  # Dummy action space
#             self.observation_spaces["_target"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Dummy obs space
        
#         # AEC properties for compatibility
#         self.action_space = self.action_spaces
#         self.observation_space = self.observation_spaces

#         # Target location
#         self._target_location = np.array([-1, -1], dtype=int)

#         # Obstacles: list of (x, y, w, h) in cell units
#         self._obstacles: List[tuple] = []
#         # Visual obstacles: block ObserverAgent view but not movement
#         self._visual_obstacles: List[tuple] = []

#         # Mapping actions to movement on the grid
#         self._action_to_direction = {
#             0: np.array([1, 0]),  # Right
#             1: np.array([0, 1]),  # Up
#             2: np.array([-1, 0]),  # Left
#             3: np.array([0, -1]),  # Down
#             4: np.array([0, 0]),  # No operation
#         }

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         self.intrinsic = intrinsic
#         if self.intrinsic:
#             self._visit_counts: Dict[str, np.ndarray] = {a.name: np.zeros((self.size, self.size), dtype=int) for a in self.agent_list}

#         # Initialize target
#         if not self.no_target:
#             if target_config is None: # If no target config is provided, use default values
#                 target_config = {}
#             self.target = Target(
#                 name=target_config.get('name', 'target'),
#                 color=target_config.get('color', (255, 0, 0)),
#                 size=self.size,
#                 movement_speed=target_config.get('movement_speed', 0.3),
#                 movement_range=target_config.get('movement_range', 1),
#                 smooth_movement=target_config.get('smooth_movement', True),
#                 box_cells=target_config.get('box_cells', 3),
#                 outline_width=target_config.get('outline_width', 2),
#                 box_scale=target_config.get('box_scale', 1.0)
#             )
#         else:
#             self.target = None # If no target is needed, set target to None

#         self.window = None
#         self.clock = None
#         self.step_counter = 0


#     def _get_info(self):
#         '''
#         Returns:
#             dict: Distance between each agent and the target
#         '''
#         info = {}
#         for a in self.agent_list:
#             if self.target is not None:
#                 info[f"distance_{a.name}"] = np.linalg.norm(a.location - self.target.location, ord=1)
#             else:
#                 info[f"distance_{a.name}"] = -1 # If no target is needed, set distance to -1
#         return info

#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         '''
#         Resets the environment to a new state.

#         Args:
#             seed: Optional seed for the random number generator
#             options: Optional dictionary of options

#         Returns:
#             None (AEC reset returns None)
#         '''
#         # Initialize random seed if provided
#         if seed is not None:
#             np.random.seed(seed)

#         # Initialize AEC state
#         self.agents = self.possible_agents.copy()
#         self.rewards = {agent: 0 for agent in self.possible_agents}
#         self.terminations = {agent: False for agent in self.possible_agents}
#         self.truncations = {agent: False for agent in self.possible_agents}
#         self.infos = {agent: {} for agent in self.possible_agents}
#         self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        
#         # Initialize agent selector for cycling through agents
#         self._agent_selector = AgentSelector(self.possible_agents)
#         self.agent_selection = self._agent_selector.next()

#         # Optionally generate new obstacles each reset
#         self._obstacles = []
#         if self.enable_obstacles and self.num_obstacles > 0:
#             self._generate_random_obstacles(self.num_obstacles)
        
#         # Generate visual obstacles (block view but not movement)
#         self._visual_obstacles = []
#         if self.enable_obstacles and self.num_visual_obstacles > 0:
#             self._generate_random_visual_obstacles(self.num_visual_obstacles)

#         # Sample positions not inside obstacles (only physical obstacles block spawn)
#         forbidden = self._cells_covered_by_obstacles()
#         choices = [i for i in range(self.size * self.size) if (i // self.size, i % self.size) not in forbidden]
#         num_needed = len(self.agent_list) + (0 if self.no_target else 1)
        
#         if len(choices) < num_needed:
#             # fallback: ignore obstacles if too dense
#             coords = np.random.choice(self.size * self.size, size=num_needed, replace=False)
#         else:
#             coords = np.random.choice(choices, size=num_needed, replace=False)

#         for i, a in enumerate(self.agent_list):
#             self._update_agent_location(a, np.array([coords[i] // self.size, coords[i] % self.size], dtype=int))
        
#         # The target should already be initialized in __init__, just reset it if needed
#         # This code initializes or resets the target object's location at the start of an environment episode.
#         # If the environment is configured to have no target (self.no_target is True), set self.target to None.
#         # Otherwise, compute a new start position for the target (ensuring it is not placed on an obstacle),
#         # and reset (move) the target to that position if the target object exists.

#         if self.no_target:
#             self.target = None
#         else:
#             t_idx = len(self.agent_list)  # The index after all agents (since coords = [agents..., target])
#             target_position = (coords[t_idx] // self.size, coords[t_idx] % self.size)  # Get target cell coordinates
#             if self.target is not None:             # If the target object exists, reset it to the new position (ensuring it is not placed on an obstacle)
#                 self.target.reset(target_position)  # Move target to new position, the target object has its own reset method.

#         if self.intrinsic: # Counting visits to cells if intrinsic exploration is enabled
#             for a in self.agent_list:
#                 self._visit_counts[a.name].fill(0)
#                 self._visit_counts[a.name][tuple(a.location)] += 1

#         self.step_counter = 0

#         if self.render_mode == "human":
#             self._render_frame()

#     def observe(self, agent: str):
#         '''
#         Returns observation for a single agent.
        
#         Args:
#             agent: The name of the agent to observe
            
#         Returns:
#             dict: Observation containing agent location, FOV obstacles, and optionally target location
#         '''
#         # Handle target observation (dummy)
#         if agent == "_target":
#             return np.array([0.0], dtype=np.float32)
        
#         if agent not in self._agents_by_name:
#             raise ValueError(f"Agent {agent} not found")
            
#         agent_obj = self._agents_by_name[agent]
        
#         # Prepare environment state for agent observation
#         env_state = {
#             "agents": self.agent_list,
#             "target": self.target,
#             "obstacles": self._obstacles,
#             "visual_obstacles": self._visual_obstacles,
#             "grid_size": self.size
#         }
        
#         # Use agent's own observation method if available, otherwise fall back to legacy method
#         if hasattr(agent_obj, 'observe') and callable(getattr(agent_obj, 'observe')):
#             return agent_obj.observe(env_state)
#         else:
#             # Legacy observation method
#             fov_obstacles = self._get_agent_fov_obstacles(agent_obj)
#             obs = {
#                 "agent": agent_obj.location,
#                 "obstacles_fov": fov_obstacles
#             }
#             # Only include target coordinates if show_target_coords is True
#             if self.show_target_coords and self.target is not None:
#                 obs["target"] = self.target.location
#             elif self.show_target_coords:
#                 obs["target"] = np.array([-1, -1])
#             return obs

#     def last(self):
#         '''
#         Returns the observation, reward, termination, truncation, and info for the current agent.
        
#         Returns:
#             tuple: (observation, reward, termination, truncation, info) for agent_selection
#         '''
#         agent = self.agent_selection
#         if agent is None:
#             return None, None, None, None, None
        
#         # Handle target turn (returns dummy values)
#         if agent == "_target":
#             observation = self.observe(agent)
#             reward = self.rewards.get(agent, 0.0)
#             termination = self.terminations.get(agent, False)
#             truncation = self.truncations.get(agent, False)
#             info = {}
#             return observation, reward, termination, truncation, info
            
#         observation = self.observe(agent)
#         reward = self.rewards[agent]
#         termination = self.terminations[agent]
#         truncation = self.truncations[agent]
        
#         # Get info for this specific agent
#         info = {}
#         if self.target is not None:
#             agent_obj = self._agents_by_name[agent]
#             info[f"distance_{agent}"] = np.linalg.norm(agent_obj.location - self.target.location, ord=1)
#         else:
#             info[f"distance_{agent}"] = -1
            
#         return observation, reward, termination, truncation, info

#     def step(self, action):
#         """
#         AEC step method - takes action for current agent and advances to next agent.
        
#         Args:
#             action: Integer action (0-4) for the current agent, or None for target
            
#         Returns:
#             None (AEC step returns None)
#         """
#         if action is None:
#             # Handle agent termination/truncation
#             self._accumulate_rewards()
#             self.agent_selection = self._agent_selector.next()
#             return
            
#         agent = self.agent_selection
#         if agent is None:
#             return
        
#         # Check if current "agent" is actually the target
#         if agent == "_target":
#             # Move the target
#             if self.target is not None:
#                 self.target.step(self.step_counter, self._obstacles, self._is_cell_obstacle)
#             # Target doesn't receive rewards
#             self.rewards["_target"] = 0.0
#             self.terminations["_target"] = False
#             self.truncations["_target"] = self.step_counter >= self.max_steps
#             # Advance to next agent
#             self._accumulate_rewards()
#             self.agent_selection = self._agent_selector.next()
#             return
            
#         self.step_counter += 1
#         self._step_count += 1
        
#         # Clear screen and show step info if FOV display is enabled
#         self._clear_screen_and_show_step()

#         # Move the current agent
#         agent_obj = self._agents_by_name[agent]
        
#         # Prepare environment state for agent step
#         env_state = {
#             "agents": self.agent_list,
#             "target": self.target,
#             "obstacles": self._obstacles,
#             "visual_obstacles": self._visual_obstacles,
#             "grid_size": self.size
#         }
        
#         # Track if agent collided with an obstacle
#         obstacle_collision = False
        
#         # Use agent's own step method if available, otherwise fall back to legacy method
#         if hasattr(agent_obj, 'step') and callable(getattr(agent_obj, 'step')):
#             step_result = agent_obj.step(int(action), env_state)
#             proposed_loc = step_result["new_location"]
#             # Check if movement was blocked by obstacle (location didn't change when it should have)
#             if "obstacle_collision" in step_result:
#                 obstacle_collision = step_result["obstacle_collision"]
#             self._update_agent_location(agent_obj, proposed_loc)
#         else:
#             # Legacy movement method
#             direction = self._action_to_direction[int(action)]
#             proposed_loc = np.clip(agent_obj.location + direction, 0, self.size - 1)
            
#             # Block obstacles
#             if self._is_cell_obstacle(proposed_loc):
#                 obstacle_collision = True
#                 proposed_loc = agent_obj.location
                
#             # Block collisions with other agents
#             blocked = any(np.array_equal(proposed_loc, other.location) for other in self.agent_list if other.name != agent)
#             if not blocked:
#                 self._update_agent_location(agent_obj, proposed_loc)
        
#         # Show FOV for this agent if display is enabled and agent has FOV
#         if self.show_fov_display and hasattr(agent_obj, 'fov_size'):
#             # Get the masked FOV from the agent's observation
#             env_state = {
#                 "agents": self.agent_list,
#                 "target": self.target,
#                 "obstacles": self._obstacles,
#                 "visual_obstacles": self._visual_obstacles,
#                 "grid_size": self.size
#             }
#             obs = agent_obj.observe(env_state)
#             fov_map = obs["obstacles_fov"]
#             self._print_fov_display(agent, fov_map, f"(action {action})")

#         # Note: Target now moves as part of the agent cycle (after all agents have acted)
        
#         # Calculate individual reward for this agent
#         reward = 0.0
        
#         # Apply obstacle collision penalty if agent hit an obstacle
#         if obstacle_collision:
#             reward += self.obstacle_collision_penalty
        
#         # Update detection state for visual indicators
#         self._agent_detects_target = {}
#         self._target_detects_agent = {}
        
#         # Check if this agent reached the target (terminates the agent)
#         reached_target = False
#         if not self.no_target and self.target is not None:
#             reached_target = np.array_equal(agent_obj.location, self.target.location)
        
#         if reached_target:
#             reward += -5.0 * self.lambda_fov  # Penalty for reaching target
#         else:
#             # Check if this agent detects target (only for agents with FOV)
#             # For ObserverAgent, check actual observation (accounts for probabilistic detection)
#             agent_detects = False
#             if hasattr(agent_obj, 'fov_size'):
#                 # Get the agent's actual observation to check if target is visible (value 3 in FOV)
#                 env_state = {
#                     "agents": self.agent_list,
#                     "target": self.target,
#                     "obstacles": self._obstacles,
#                     "visual_obstacles": self._visual_obstacles,
#                     "grid_size": self.size
#                 }
#                 obs = agent_obj.observe(env_state)
#                 fov_map = obs["obstacles_fov"]
#                 # Check if target (value 3) is in the observed FOV
#                 agent_detects = np.any(fov_map == 3)
#             self._agent_detects_target[agent] = agent_detects
            
#             # Check if target detects this agent
#             target_detects = False
#             if self.target is not None:
#                 target_detects = self._is_agent_in_target_fov(agent_obj)
#             self._target_detects_agent[agent] = target_detects
                
#             # Penalty: -lambda if agent is in target's FOV - we could call lambda the stealth factor
#             if target_detects:
#                 reward -= self.lambda_fov
#             # Reward: (1-lambda) if target is in agent's FOV
#             elif agent_detects:
#                 reward += (1 - self.lambda_fov)
#             else:
#                 # Small step penalty if no detection
#                 reward += -0.01

#         # Intrinsic exploration rewards (if enabled)
#         if self.intrinsic:
#             if self._visit_counts[agent][tuple(agent_obj.location)] == 0:
#                 reward += 0.025
#             self._visit_counts[agent][tuple(agent_obj.location)] += 1

#         # Store reward for this agent
#         self.rewards[agent] = reward
#         self._cumulative_rewards[agent] += reward
        
#         # Store last agent and reward for rendering
#         self._last_agent = agent
#         self._last_reward = reward
        
#         # Check termination/truncation for this agent
#         # Terminates the agent if it reaches the target
#         self.terminations[agent] = reached_target
#         self.truncations[agent] = self.step_counter >= self.max_steps

#         # Advance to next agent
#         self._accumulate_rewards()
        
#         # Check if all agents are done before updating agents list
#         all_done = all(self.terminations[a] or self.truncations[a] for a in self.possible_agents)
        
#         if all_done:
#             # Environment is done - set agents to empty list
#             self.agents = []
#             self.agent_selection = None
#         else:
#             # Remove done agents from active agents list
#             active_agents = [
#                 a for a in self.possible_agents 
#                 if not (self.terminations[a] or self.truncations[a])
#             ]
            
#             # If the active agents list has changed, update the selector
#             if set(active_agents) != set(self.agents):
#                 self.agents = active_agents
#                 self._agent_selector = AgentSelector(self.agents)
#                 self.agent_selection = self._agent_selector.next()
#             else:
#                 # Normal case: just advance to next agent
#                 self.agent_selection = self._agent_selector.next()

#         if self.render_mode == "human":
#             self._render_frame()

#     def _accumulate_rewards(self):
#         """Accumulate rewards for the current agent."""
#         agent = self.agent_selection
#         if agent is not None:
#             self._cumulative_rewards[agent] += self.rewards[agent]

#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()

#     def render(self):
#         """
#         Render the environment.
        
#         Returns:
#             numpy array if render_mode is "rgb_array", None otherwise
#         """
#         if self.render_mode == "rgb_array":
#             return self._render_frame()
#         elif self.render_mode == "human":
#             self._render_frame()
#             return None
#         else:
#             return None

#     def _render_frame(self):
#         # Initialize pygame for both human and rgb_array modes
#         if not pygame.get_init():
#             pygame.init()
        
#         if self.window is None and self.render_mode == "human":
#             pygame.display.init()
#             # Increase window height to accommodate info panel
#             self.window = pygame.display.set_mode((self.window_size, self.window_size + 120))
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         # Colors (dark theme)
#         BG_COLOR = (30, 30, 30)
#         GRID_COLOR = (60, 60, 60)
#         TARGET_COLOR = (240, 100, 90)  # red-ish
#         INFO_PANEL_COLOR = (50, 50, 50)
#         TEXT_COLOR = (255, 255, 255)

#         # Create canvas with extra height for info panel
#         canvas = pygame.Surface((self.window_size, self.window_size + 120))
#         canvas.fill(BG_COLOR)
#         pix_square = self.window_size / self.size

#         # Grid
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas, GRID_COLOR, (0, pix_square * x), (self.window_size, pix_square * x), 1
#             )
#             pygame.draw.line(
#                 canvas, GRID_COLOR, (pix_square * x, 0), (pix_square * x, self.window_size), 1
#             )

#         # Physical obstacles (filled rectangles - dark gray)
#         if self._obstacles:
#             OBSTACLE_COLOR = (90, 90, 90)
#             for (ox, oy, ow, oh) in self._obstacles:
#                 rect = pygame.Rect(
#                     (ox * pix_square, oy * pix_square),
#                     (ow * pix_square, oh * pix_square),
#                 )
#                 pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)
        
#         # Visual obstacles (semi-transparent light blue - blocks view but not movement)
#         if self._visual_obstacles:
#             VISUAL_OBSTACLE_COLOR = (150, 200, 240, 150)  # Light blue with alpha
#             VISUAL_OBSTACLE_STRIPE_COLOR = (100, 160, 220, 200)  # Darker blue for stripes
            
#             for (ox, oy, ow, oh) in self._visual_obstacles:
#                 # Create temporary surface for this obstacle
#                 temp_surface = pygame.Surface((ow * pix_square, oh * pix_square), pygame.SRCALPHA)
#                 temp_surface.fill(VISUAL_OBSTACLE_COLOR)
                
#                 # Add diagonal stripe pattern
#                 stripe_spacing = 8  # pixels between stripes
#                 stripe_width = 3
                
#                 # Draw diagonal stripes from top-left to bottom-right
#                 for i in range(-int(oh * pix_square), int(ow * pix_square), stripe_spacing):
#                     start_x = i
#                     start_y = 0
#                     end_x = i + int(oh * pix_square)
#                     end_y = int(oh * pix_square)
#                     pygame.draw.line(temp_surface, VISUAL_OBSTACLE_STRIPE_COLOR, 
#                                    (start_x, start_y), (end_x, end_y), stripe_width)
                
#                 canvas.blit(temp_surface, (ox * pix_square, oy * pix_square))

#         # Target (only if spawned and exists)
#         if self.target is not None:
#             # Get triangle points from target (pass pix_square as float)
#             point1, point2, point3 = self.target.get_triangle_points(pix_square)
#             pygame.draw.polygon(canvas, self.target.color, [point1, point2, point3])
            
#             # Draw target box outline
#             top_left, top_right, bottom_right, bottom_left = self.target.get_box_coordinates(pix_square)
#             box_points = [top_left, top_right, bottom_right, bottom_left, top_left]  # Close the polygon
#             pygame.draw.lines(canvas, self.target.color, False, box_points, self.target.outline_width)
            
#             # Draw detection dot if target detects any agent
#             if any(self._target_detects_agent.values()):
#                 target_center = (self.target.location + 0.5) * pix_square
#                 dot_center = target_center + np.array([0, -pix_square * 0.6])  # Position higher above target
#                 pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

#         # Agents (circles or squares)
#         for a in self.agent_list:
#             center = (a.location + 0.5) * pix_square
            
#             # Check if this is an ObserverAgent to render as square
#             if hasattr(a, 'altitude') and hasattr(a, 'fov_base_size'):
#                 # Render as square for ObserverAgent
#                 square_size = int(pix_square / 3)
#                 square_rect = pygame.Rect(
#                     center[0] - square_size // 2,
#                     center[1] - square_size // 2,
#                     square_size,
#                     square_size
#                 )
#                 pygame.draw.rect(canvas, a.color, square_rect)
#             else:
#                 # Render as circle for other agents
#                 pygame.draw.circle(canvas, a.color, center.astype(int), int(pix_square / 3))
            
#             # Draw detection dot if agent detects target
#             if self._agent_detects_target.get(a.name, False):
#                 dot_center = center + np.array([0, -pix_square * 0.6])  # Position higher above agent
#                 pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

#         # Outline boxes using per-agent params
#         def draw_agent_box(a: Agent):
#             if a.fov_size > 1:
#                 # Draw a kxk box centered on the agent, without clamping to grid bounds
#                 k = a.fov_size
#                 offset = k // 2
#                 # Compute top-left in cell coordinates centered on agent
#                 start_cell = (a.location - np.array([offset, offset]))
#                 top_left = start_cell * pix_square
#                 side_px = pix_square * k
#                 rect = pygame.Rect(top_left, (side_px, side_px))
                
#                 # Check if this is an ObserverAgent for special line styles and visible area
#                 if hasattr(a, 'altitude') and hasattr(a, 'fov_base_size'):
#                     # ObserverAgent with flight level-based visible area and line styles
#                     # Flight levels are only 1, 2, 3 (never 0)
#                     if a.altitude >= 1:
#                         # Calculate visible area based on flight level
#                         # Flight Level 1: visible = fov_size - 4 (3x3 for fov_size 7)
#                         # Flight Level 2: visible = fov_size - 2 (5x5 for fov_size 7)
#                         # Flight Level 3: visible = fov_size (7x7 for fov_size 7)
#                         rings_to_mask = 3 - a.altitude
#                         visible_size = a.fov_size - (rings_to_mask * 2)
                        
#                         # Calculate new rectangle for visible area
#                         visible_offset = visible_size // 2
#                         visible_start_cell = (a.location - np.array([visible_offset, visible_offset]))
#                         visible_top_left = visible_start_cell * pix_square
#                         visible_side_px = pix_square * visible_size
#                         visible_rect = pygame.Rect(visible_top_left, (visible_side_px, visible_side_px))
                        
#                         # Draw with appropriate line style based on flight level
#                         if a.altitude == 1:  # Flight level 1 - solid line (3x3)
#                             pygame.draw.rect(canvas, a.color, visible_rect, width=a.outline_width)
#                         elif a.altitude == 2:  # Flight level 2 - dashed line (5x5)
#                             draw_dashed_rect(canvas, a.color, visible_rect, a.outline_width)
#                         elif a.altitude == 3:  # Flight level 3 - dotted line (7x7)
#                             draw_dotted_rect(canvas, a.color, visible_rect, a.outline_width)
#                     else:
#                         # Ground level (altitude 0) - no FOV border
#                         pass
#                 else:
#                     # Regular agent - solid line with full FOV
#                     pygame.draw.rect(canvas, a.color, rect, width=a.outline_width)
#             else:
#                 box_side = pix_square * a.box_scale
#                 top_left = (a.location * pix_square) + ((pix_square - box_side) / 2)
#                 rect = pygame.Rect(top_left, (box_side, box_side))
#                 pygame.draw.rect(canvas, a.color, rect, width=a.outline_width)
        
#         def draw_dashed_rect(surface, color, rect, width):
#             """Draw a dashed rectangle."""
#             x, y, w, h = rect
#             dash_length = 8
#             gap_length = 4
            
#             # Top edge
#             draw_dashed_line(surface, color, (x, y), (x + w, y), dash_length, gap_length, width)
#             # Bottom edge
#             draw_dashed_line(surface, color, (x, y + h), (x + w, y + h), dash_length, gap_length, width)
#             # Left edge
#             draw_dashed_line(surface, color, (x, y), (x, y + h), dash_length, gap_length, width)
#             # Right edge
#             draw_dashed_line(surface, color, (x + w, y), (x + w, y + h), dash_length, gap_length, width)
        
#         def draw_dotted_rect(surface, color, rect, width):
#             """Draw a dotted rectangle."""
#             x, y, w, h = rect
#             dot_spacing = 6
            
#             # Top edge
#             draw_dotted_line(surface, color, (x, y), (x + w, y), dot_spacing, width)
#             # Bottom edge
#             draw_dotted_line(surface, color, (x, y + h), (x + w, y + h), dot_spacing, width)
#             # Left edge
#             draw_dotted_line(surface, color, (x, y), (x, y + h), dot_spacing, width)
#             # Right edge
#             draw_dotted_line(surface, color, (x + w, y), (x + w, y + h), dot_spacing, width)
        
#         def draw_dashed_line(surface, color, start, end, dash_length, gap_length, width):
#             """Draw a dashed line."""
#             x1, y1 = start
#             x2, y2 = end
            
#             # Calculate line length and direction
#             dx = x2 - x1
#             dy = y2 - y1
#             line_length = (dx * dx + dy * dy) ** 0.5
            
#             if line_length == 0:
#                 return
            
#             # Normalize direction
#             dx /= line_length
#             dy /= line_length
            
#             # Draw dashes
#             current_length = 0
#             draw_dash = True
            
#             while current_length < line_length:
#                 if draw_dash:
#                     # Calculate dash end
#                     dash_end_length = min(current_length + dash_length, line_length)
#                     start_x = x1 + dx * current_length
#                     start_y = y1 + dy * current_length
#                     end_x = x1 + dx * dash_end_length
#                     end_y = y1 + dy * dash_end_length
                    
#                     pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)
#                     current_length = dash_end_length
#                 else:
#                     current_length += gap_length
                
#                 draw_dash = not draw_dash
        
#         def draw_dotted_line(surface, color, start, end, dot_spacing, width):
#             """Draw a dotted line."""
#             x1, y1 = start
#             x2, y2 = end
            
#             # Calculate line length and direction
#             dx = x2 - x1
#             dy = y2 - y1
#             line_length = (dx * dx + dy * dy) ** 0.5
            
#             if line_length == 0:
#                 return
            
#             # Normalize direction
#             dx /= line_length
#             dy /= line_length
            
#             # Draw dots
#             current_length = 0
#             while current_length < line_length:
#                 dot_x = x1 + dx * current_length
#                 dot_y = y1 + dy * current_length
#                 pygame.draw.circle(surface, color, (int(dot_x), int(dot_y)), width)
#                 current_length += dot_spacing

#         for a in self.agent_list:
#             draw_agent_box(a)
        
#         # Draw halos around agents that see other agents in their FOV
#         for a in self.agent_list:
#             fov_map = self._get_agent_fov_obstacles(a)
#             if np.any(fov_map == 2):  # Check if agent sees other agents (value 2)
#                 # Draw a halo around the agent
#                 center_x = (a.location[0] + 0.5) * pix_square
#                 center_y = (a.location[1] + 0.5) * pix_square
#                 halo_radius = pix_square * 0.6  # Slightly larger than agent box
#                 pygame.draw.circle(canvas, a.color, (int(center_x), int(center_y)), int(halo_radius), width=3)
            
#         # Draw lines through last 3 visited cells for each agent
#         for a in self.agent_list:
#             if len(a.last_visited_cells) >= 2:
#                 # Convert cell coordinates to pixel coordinates (center of cells)
#                 pixel_points = []
#                 for cell in a.last_visited_cells:
#                     x, y = cell
#                     center_x = (x + 0.5) * pix_square
#                     center_y = (y + 0.5) * pix_square
#                     pixel_points.append((center_x, center_y))
                
#                 # Draw lines between consecutive points
#                 for i in range(len(pixel_points) - 1):
#                     start_pos = pixel_points[i]
#                     end_pos = pixel_points[i + 1]
#                     pygame.draw.line(canvas, a.color, start_pos, end_pos, width=2)

#         # Information panel
#         info_panel_y = self.window_size
#         info_panel_height = 120
        
#         # Draw info panel background
#         info_rect = pygame.Rect(0, info_panel_y, self.window_size, info_panel_height)
#         pygame.draw.rect(canvas, INFO_PANEL_COLOR, info_rect)
#         pygame.draw.rect(canvas, GRID_COLOR, info_rect, 2)  # Border
        
#         # Initialize font
#         try:
#             # Initialize font module if not already done
#             if not pygame.font.get_init():
#                 pygame.font.init()
            
#             font_large = pygame.font.Font(None, 24)
#             font_medium = pygame.font.Font(None, 18)
#             font_small = pygame.font.Font(None, 16)
#         except:
#             try:
#                 font_large = pygame.font.SysFont('Arial', 24)
#                 font_medium = pygame.font.SysFont('Arial', 18)
#                 font_small = pygame.font.SysFont('Arial', 16)
#             except:
#                 try:
#                     # Last resort: use default font
#                     font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
#                     font_medium = pygame.font.Font(pygame.font.get_default_font(), 18)
#                     font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
#                 except:
#                     # If all else fails, create dummy fonts
#                     font_large = None
#                     font_medium = None
#                     font_small = None
        
#         # Only render text if fonts are available
#         if font_large is not None and font_medium is not None and font_small is not None:
#             # Step counter and current agent
#             current_agent_text = f"Step: {self.step_counter}"
#             if hasattr(self, 'agent_selection') and self.agent_selection is not None:
#                 current_agent_text += f" | Current: {self.agent_selection}"
#             step_text = font_large.render(current_agent_text, True, TEXT_COLOR)
#             canvas.blit(step_text, (10, info_panel_y + 10))
            
#             # Agent information
#             y_offset = info_panel_y + 40
#             for i, agent in enumerate(self.agent_list):
#                 # Agent name and position
#                 agent_info = f"{agent.name}: ({agent.location[0]}, {agent.location[1]})"
#                 agent_text = font_medium.render(agent_info, True, agent.color)
#                 canvas.blit(agent_text, (10, y_offset + i * 20))
            
#             # Target information
#             if self.target is not None:
#                 target_info = f"Target: ({self.target.location[0]}, {self.target.location[1]})"
#                 target_text = font_medium.render(target_info, True, self.target.color)
#                 canvas.blit(target_text, (10, y_offset + len(self.agent_list) * 20))
            
#             # Reward information (if available from last step)
#             if hasattr(self, '_last_reward') and hasattr(self, '_last_agent'):
#                 reward_text = font_medium.render(f"Last Reward ({self._last_agent}): {self._last_reward:.3f}", True, TEXT_COLOR)
#                 canvas.blit(reward_text, (self.window_size - 200, info_panel_y + 10))
#             elif hasattr(self, '_last_reward'):
#                 reward_text = font_medium.render(f"Last Reward: {self._last_reward:.3f}", True, TEXT_COLOR)
#                 canvas.blit(reward_text, (self.window_size - 200, info_panel_y + 10))
            
#             # Lambda FOV parameter
#             lambda_text = font_small.render(f"λ FOV: {self.lambda_fov:.2f}", True, TEXT_COLOR)
#             canvas.blit(lambda_text, (self.window_size - 200, info_panel_y + 35))
            
#             # FOV legend
#             legend_text = font_small.render("FOV: 0=Empty, 1=Physical, 2=Agent, 3=Target, 4=Visual", True, TEXT_COLOR)
#             canvas.blit(legend_text, (self.window_size - 200, info_panel_y + 55))

#         # Finalize frame
#         if self.render_mode == "human":
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#             self.clock.tick(self.metadata["render_fps"])
#         else:
#             return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

#     # --- Agent management ---
#     def add_agent(self, name: Optional[str] = None, color: Optional[tuple] = None, action_space: Optional[spaces.Space] = None, observation_space: Optional[spaces.Space] = None, outline_width: Optional[int] = None, box_scale: Optional[float] = None, fov_size: Optional[int] = None) -> str:
#         """Add a new agent at runtime and update spaces and lookups. Returns the agent name."""
#         # Defaults based on env
#         idx = len(self.agent_list)
#         agent_name = name or f"agent_{idx+1}"
#         if agent_name in self._agents_by_name:
#             raise ValueError(f"Agent name already exists: {agent_name}")
#         default_colors = [(80, 160, 255), (255, 180, 60)]
#         agent_color = color or default_colors[idx % len(default_colors)]
#         act_space = action_space or spaces.Discrete(5)
#         ow = int(outline_width) if outline_width is not None else 1
#         bs = float(box_scale) if box_scale is not None else 1.0
#         bc = int(fov_size) if fov_size is not None else 1
#         new_agent = Agent(self.size, agent_name, agent_color, None, ow, bs, bc, self.show_target_coords)

#         # Register
#         self.agent_list.append(new_agent)
#         self.agents.append(agent_name)
#         self._agents_by_name[agent_name] = new_agent

#         # Update combined spaces
#         self.action_space = spaces.Dict({a.name: a.action_space for a in self.agent_list})
#         self.observation_space = spaces.Dict({a.name: a.observation_space for a in self.agent_list})

#         # Init intrinsic map if needed
#         if self.intrinsic:
#             self._visit_counts[agent_name] = np.zeros((self.size, self.size), dtype=int)

#         return agent_name

#     def _update_agent_location(self, agent: 'Agent', new_location: np.ndarray) -> None:
#         """Update agent location and track last 3 visited cells."""
#         agent.location = new_location
#         # Add current location to visited cells (as tuple for immutability)
#         cell_tuple = tuple(new_location)
#         agent.last_visited_cells.append(cell_tuple)
#         # Keep only last 3 cells
#         if len(agent.last_visited_cells) > 3:
#             agent.last_visited_cells.pop(0)

#     def _get_agent_fov_obstacles(self, agent: 'Agent') -> np.ndarray:
#         """Generate field of view obstacle map for an agent - OPTIMIZED VERSION."""
#         fov_size = agent.fov_size
#         if fov_size is None:
#             fov_size = 5
        
#         # Agent's current position
#         agent_x, agent_y = agent.location
#         offset = fov_size // 2
        
#         # Pre-calculate all world coordinates at once (vectorized)
#         world_x_coords = np.arange(agent_x - offset, agent_x - offset + fov_size)
#         world_y_coords = np.arange(agent_y - offset, agent_y - offset + fov_size)
        
#         # Create meshgrid for all coordinates
#         world_x_grid, world_y_grid = np.meshgrid(world_x_coords, world_y_coords, indexing='ij')
        
#         # Check boundaries (vectorized)
#         out_of_bounds = ((world_x_grid < 0) | (world_x_grid >= self.size) | 
#                         (world_y_grid < 0) | (world_y_grid >= self.size))
        
#         # Initialize FOV map with boundary obstacles
#         fov_map = out_of_bounds.astype(int)
        
#         # Only check obstacles and other agents for in-bounds cells
#         valid_mask = ~out_of_bounds
#         if valid_mask.any():
#             # Get valid coordinates
#             valid_x = world_x_grid[valid_mask]
#             valid_y = world_y_grid[valid_mask]
            
#             # Check obstacles, other agents, and target for valid cells only
#             for x, y in zip(valid_x, valid_y):
#                 x_int, y_int = int(x), int(y)
#                 fov_i = int(x - agent_x + offset)
#                 fov_j = int(y - agent_y + offset)
                
#                 # Check if target is at this position
#                 target_present = False
#                 if self.target is not None and (self.target.location[0] == x_int and self.target.location[1] == y_int):
#                     fov_map[fov_i, fov_j] = 3  # Mark target as 3
#                     target_present = True
                
#                 # Check if another agent is at this position (only if no target)
#                 if not target_present:
#                     other_agent_present = False
#                     for other_agent in self.agent_list:
#                         if (other_agent.name != agent.name and 
#                             other_agent.location[0] == x_int and 
#                             other_agent.location[1] == y_int):
#                             fov_map[fov_i, fov_j] = 2  # Mark other agent as 2
#                             other_agent_present = True
#                             break
                    
#                     # If no other agent, check for obstacles
#                     if not other_agent_present and self._is_cell_obstacle_fast(x_int, y_int):
#                         fov_map[fov_i, fov_j] = 1  # Mark obstacle as 1
        
#         return fov_map
    
#     def _is_agent_in_target_fov(self, agent: 'Agent') -> bool:
#         """
#         Check if an agent is within the target's field of view.
        
#         Args:
#             agent: The agent to check
            
#         Returns:
#             True if agent is in target's FOV, False otherwise
#         """
#         if self.target is None:
#             return False
        
#         # Get target's FOV size (use box_cells as FOV size)
#         target_fov_size = self.target.box_cells
#         if target_fov_size is None:
#             target_fov_size = 3  # Default FOV size
        
#         # Target's current position
#         target_x, target_y = self.target.location
#         offset = target_fov_size // 2
        
#         # Agent's position
#         agent_x, agent_y = agent.location
        
#         # Check if agent is within target's FOV bounds
#         fov_left = target_x - offset
#         fov_right = target_x + offset
#         fov_top = target_y - offset
#         fov_bottom = target_y + offset
        
#         return (fov_left <= agent_x <= fov_right and 
#                 fov_top <= agent_y <= fov_bottom)
    
#     def _is_target_in_agent_fov(self, agent: 'Agent') -> bool:
#         """
#         Check if the target is within an agent's field of view.
        
#         Args:
#             agent: The agent whose FOV to check
            
#         Returns:
#             True if target is in agent's FOV, False otherwise
#         """
#         if self.target is None:
#             return False
        
#         # Get agent's FOV size
#         agent_fov_size = agent.fov_size
#         if agent_fov_size is None:
#             agent_fov_size = 5  # Default FOV size
        
#         # Agent's current position
#         agent_x, agent_y = agent.location
#         offset = agent_fov_size // 2
        
#         # Target's position
#         target_x, target_y = self.target.location
        
#         # Check if target is within agent's FOV bounds
#         fov_left = agent_x - offset
#         fov_right = agent_x + offset
#         fov_top = agent_y - offset
#         fov_bottom = agent_y + offset
        
#         return (fov_left <= target_x <= fov_right and 
#                 fov_top <= target_y <= fov_bottom)
    
#     def _print_fov_display(self, agent_name: str, fov_map: np.ndarray, action_info: str = ""):
#         """Print FOV map that can be overwritten, aligned 'facing north' like pygame"""
#         if not self.show_fov_display:
#             return
            
#         print(f"{agent_name.upper()} FOV {action_info}:")
#         # Reverse the rows to match pygame orientation (north at top)
#         for row in reversed(fov_map):
#             print(f"  {' '.join(str(cell) for cell in row)}")
        
#         # Show legend for first agent only to avoid clutter
#         if agent_name == self.agents[0]:
#             print("  Legend: 0=Empty, 1=Physical Obstacle, 2=Agent, 3=Target, 4=Visual Obstacle")
#         print()  # New line after each agent's FOV

#     def _clear_screen_and_show_step(self):
#         """Clear screen and show step information"""
#         if not self.show_fov_display:
#             return
            
#         print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
#         print("="*80)
#         print(f"STEP {self._step_count}".center(80))
#         print("="*80)

#     def _cells_covered_by_obstacles(self) -> set:
#         covered = set()
#         for (ox, oy, ow, oh) in self._obstacles:
#             for dx in range(ow):
#                 for dy in range(oh):
#                     cx, cy = ox + dx, oy + dy
#                     if 0 <= cx < self.size and 0 <= cy < self.size:
#                         covered.add((cx, cy))
#         return covered

#     def _is_cell_obstacle(self, cell: np.ndarray) -> bool:
#         x, y = int(cell[0]), int(cell[1])
#         for (ox, oy, ow, oh) in self._obstacles:
#             if ox <= x < ox + ow and oy <= y < oy + oh:
#                 return True
#         return False
    
#     def _is_cell_obstacle_fast(self, x: int, y: int) -> bool:
#         """Optimized obstacle checking without numpy array creation."""
#         for (ox, oy, ow, oh) in self._obstacles:
#             if ox <= x < ox + ow and oy <= y < oy + oh:
#                 return True
#         return False

#     def _generate_random_obstacles(self, count: int) -> None:
#         """Generate rectangular physical obstacles within grid bounds."""
#         attempts = 0
#         max_attempts = count * 10
#         created = 0
#         while created < count and attempts < max_attempts:
#             attempts += 1
#             # Random top-left
#             ox = int(np.random.randint(0, self.size))
#             oy = int(np.random.randint(0, self.size))
#             # Random width/height (at least 1 cell), capped to remain in bounds
#             max_w = max(1, self.size - ox)
#             max_h = max(1, self.size - oy)
#             # Prefer small obstacles
#             ow = int(np.random.randint(1, min(4, max_w) + 1))
#             oh = int(np.random.randint(1, min(4, max_h) + 1))
#             # Avoid duplicates
#             rect = (ox, oy, ow, oh)
#             if rect in self._obstacles:
#                 continue
#             self._obstacles.append(rect)
#             created += 1
    
#     def _generate_random_visual_obstacles(self, count: int) -> None:
#         """Generate rectangular visual obstacles within grid bounds.
        
#         Visual obstacles block ObserverAgent view but do not block movement.
#         They can overlap with physical obstacles, agents, and target, but not with each other.
#         """
#         attempts = 0
#         max_attempts = count * 20
#         created = 0
#         while created < count and attempts < max_attempts:
#             attempts += 1
#             # Random top-left
#             ox = int(np.random.randint(0, self.size))
#             oy = int(np.random.randint(0, self.size))
#             # Random width/height (at least 1 cell), capped to remain in bounds
#             max_w = max(1, self.size - ox)
#             max_h = max(1, self.size - oy)
#             # Prefer small obstacles (same size distribution as physical obstacles)
#             ow = int(np.random.randint(1, min(4, max_w) + 1))
#             oh = int(np.random.randint(1, min(4, max_h) + 1))
            
#             rect = (ox, oy, ow, oh)
            
#             # Check for exact duplicate
#             if rect in self._visual_obstacles:
#                 continue
            
#             # Check for overlap with other visual obstacles
#             overlaps = False
#             for (vis_ox, vis_oy, vis_ow, vis_oh) in self._visual_obstacles:
#                 # Check if rectangles overlap using standard rectangle intersection test
#                 if not (ox + ow <= vis_ox or vis_ox + vis_ow <= ox or
#                         oy + oh <= vis_oy or vis_oy + vis_oh <= oy):
#                     overlaps = True
#                     break
            
#             if not overlaps:
#                 self._visual_obstacles.append(rect)
#                 created += 1


class GridWorldEnvParallel(ParallelEnv):
    ''' A multi-agent grid world environment where agents navigate to find a target while avoiding obstacles.
    
    This environment follows the PettingZoo Parallel API interface, where all agents
    act simultaneously. Each agent receives individual rewards based on their FOV detection.
    
    Input parameters:
     - size: The size of the grid (size x size)
     - max_steps: Maximum steps per episode
     - render_mode: Rendering mode ("human", "rgb_array", or None)
     - show_fov_display: Whether to display agent FOVs in human render mode
     - intrinsic: Whether to use intrinsic exploration rewards
     - lambda_fov: Weighting factor for FOV-based rewards (0 to 1)
     - show_target_coords: Whether agents can see target coordinates in their observations
     - obstacle_collision_penalty: Penalty applied when agent collides with obstacle (default: -0.05)
     - death_on_sight: If True, agents are terminated when detected by the target (default: False)
     - no_target: If True, no target is spawned
     - agents: List of agent configurations (dicts) or Agent instances. Supports any mix of:
         * FOVAgent instances (pre-instantiated)
         * ObserverAgent instances (pre-instantiated)
         * Dict configs for FOVAgent: {"type": "FOVAgent", "name": str, "color": tuple, "fov_size": int, ...}
         * Dict configs for ObserverAgent: {"type": "ObserverAgent", "name": str, "color": tuple, 
                                           "fov_base_size": int, "max_altitude": int, 
                                           "target_detection_probs": tuple, ...}
         If None, creates 2 default FOVAgents.
     - target_config: Configuration dictionary for a single target, or a list of dictionaries for multiple targets.
                      Each target config can specify: name, color, movement_speed, movement_range, 
                      smooth_movement, box_cells, outline_width, box_scale
     - enable_obstacles: Whether to enable obstacles in the environment
     - num_obstacles: Number of physical obstacles to generate (block movement)
     - num_visual_obstacles: Number of visual obstacles to generate (block ObserverAgent view only)
     
    Agent Configuration Examples:
        # FOVAgent config
        {"name": "alpha", "type": "FOVAgent", "color": (80, 160, 255), "fov_size": 5}
        
        # ObserverAgent config
        {"name": "observer1", "type": "ObserverAgent", "color": (255, 100, 100), 
         "fov_base_size": 3, "max_altitude": 3, "target_detection_probs": (1.0, 0.66, 0.33)}'''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        # General settings
        size=5,
        max_steps: int = 500,
        render_mode=None,
        show_fov_display: bool = True,

        # Rewards and target settings
        intrinsic=False,
        lambda_fov: float = 0.5,
        show_target_coords: bool = False,
        obstacle_collision_penalty: float = -0.05,
        death_on_sight: bool = False,

        # Agents and target settings
        no_target=False,
        agents: Optional[List[Union[Dict, 'Agent']]] = None,
        target_config: Optional[Union[Dict, List[Dict]]] = None,

        # Obstacles
        enable_obstacles: bool = False,
        num_obstacles: int = 0,
        num_visual_obstacles: int = 0,

    ):
        self.size = size  # The size of the square grid (default to 5x5)
        self.window_size = 512  # The size of the PyGame window
        self.no_target = no_target  # If True, no target is spawned
        self.enable_obstacles = bool(enable_obstacles)
        self.num_obstacles = int(max(0, num_obstacles))
        self.num_visual_obstacles = int(max(0, num_visual_obstacles))
        self.show_fov_display = bool(show_fov_display)
        self.lambda_fov = max(-0.5, min(0.5, float(lambda_fov)))  # Clamp between 0 and 1
        self.show_target_coords = bool(show_target_coords)  # If True, agents can see target coordinates
        self.obstacle_collision_penalty = float(obstacle_collision_penalty)  # Penalty for hitting obstacles
        self.death_on_sight = bool(death_on_sight)  # If True, agents are terminated when detected by target
        
        # Parallel environment state
        self._step_count = 0
        self.max_steps = max_steps

        # Detection state tracking for visual indicators
        self._agent_detects_target = {}  # Track which agents detect the target
        self._target_detects_agent = {}  # Track which agents the target detects

        # Build agents list (support custom configs or instances)
        default_colors = [(80, 160, 255), (255, 180, 60), (100, 200, 100), (255, 100, 100)]
        self.agent_list: List[Union[FOVAgent, ObserverAgent]] = []

        if agents is None:
            # Auto-generate 2 default FOVAgents
            count = 2
            for i in range(count):
                name = f"agent_{i+1}"
                color = default_colors[i % len(default_colors)]
                agent = FOVAgent(
                    name=name,
                    color=color,
                    env_size=self.size,
                    fov_size=3,
                    outline_width=1,
                    box_scale=0.8,
                    show_target_coords=self.show_target_coords
                )
                self.agent_list.append(agent)
        else:
            for idx, agent_or_cfg in enumerate(agents):
                # Check if it's already an agent instance or a configuration dict
                if isinstance(agent_or_cfg, (FOVAgent, ObserverAgent, Agent)):
                    # It's already an agent object - use it directly
                    self.agent_list.append(agent_or_cfg)
                elif isinstance(agent_or_cfg, dict):
                    # It's a configuration dictionary - instantiate the agent
                    cfg = agent_or_cfg
                    name = cfg.get("name", f"agent_{idx+1}")
                    color = tuple(cfg.get("color", default_colors[idx % len(default_colors)]))
                    outline_width = int(cfg.get("outline_width", 2))
                    box_scale = float(cfg.get("box_scale", 0.8))
                    show_coords = cfg.get("show_target_coords", self.show_target_coords)
                    
                    # Determine agent type from config
                    agent_type = cfg.get("type", "FOVAgent")
                    
                    # Check for ObserverAgent-specific parameters
                    has_observer_params = any(key in cfg for key in 
                                             ["fov_base_size", "max_altitude", "target_detection_probs"])
                    
                    if agent_type == "ObserverAgent" or has_observer_params:
                        # Create ObserverAgent
                        agent = ObserverAgent(
                            name=name,
                            color=color,
                            env_size=self.size,
                            fov_base_size=int(cfg.get("fov_base_size", 3)),
                            outline_width=outline_width,
                            box_scale=box_scale,
                            show_target_coords=show_coords,
                            max_altitude=int(cfg.get("max_altitude", 3)),
                            target_detection_probs=tuple(cfg.get("target_detection_probs", (1.0, 0.66, 0.33)))
                        )
                    else:
                        # Create FOVAgent
                        agent = FOVAgent(
                            name=name,
                            color=color,
                            env_size=self.size,
                            fov_size=int(cfg.get("fov_size", 3)),
                            outline_width=outline_width,
                            box_scale=box_scale,
                            show_target_coords=show_coords
                        )
                    
                    self.agent_list.append(agent)
                else:
                    raise ValueError(f"Agent at index {idx} must be either an agent instance or a configuration dict, got {type(agent_or_cfg)}")

        # Public names and lookup
        self.agents: List[str] = [a.name for a in self.agent_list]
        self._agents_by_name: Dict[str, Agent] = {a.name: a for a in self.agent_list}

        # Parallel environment properties
        self.possible_agents = self.agents.copy()

        # Per-agent spaces
        self.action_spaces = {a.name: a.get_action_space() if hasattr(a, 'get_action_space') else a.action_space for a in self.agent_list}
        self.observation_spaces = {a.name: a.get_observation_space() if hasattr(a, 'get_observation_space') else a.observation_space for a in self.agent_list}

        # Target location
        self._target_location = np.array([-1, -1], dtype=int)

        # Obstacles: list of (x, y, w, h) in cell units
        self._obstacles: List[tuple] = []
        # Visual obstacles: block ObserverAgent view but not movement
        self._visual_obstacles: List[tuple] = []

        # Mapping actions to movement on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # Right
            1: np.array([0, 1]),  # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Down
            4: np.array([0, 0]),  # No operation
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.intrinsic = intrinsic
        if self.intrinsic:
            self._visit_counts: Dict[str, np.ndarray] = {a.name: np.zeros((self.size, self.size), dtype=int) for a in self.agent_list}

        # Initialize targets (support single or multiple targets)
        if not self.no_target:
            self.targets: List[Target] = []
            
            if target_config is None:
                # Default: create one target with default values
                target_config = {}
            
            # Handle both single dict and list of dicts
            if isinstance(target_config, dict):
                # Single target config
                target_configs = [target_config]
            elif isinstance(target_config, list):
                # Multiple target configs
                target_configs = target_config
            else:
                raise ValueError(f"target_config must be a dict or list of dicts, got {type(target_config)}")
            
            # Create targets from configs
            for idx, cfg in enumerate(target_configs):
                if not isinstance(cfg, dict):
                    raise ValueError(f"Each target config must be a dict, got {type(cfg)} at index {idx}")
                
                target = Target(
                    name=cfg.get('name', f'target_{idx}'),
                    color=cfg.get('color', (255, 0, 0)),
                    size=self.size,
                    movement_speed=cfg.get('movement_speed', 0.3),
                    movement_range=cfg.get('movement_range', 1),
                    smooth_movement=cfg.get('smooth_movement', True),
                    box_cells=cfg.get('box_cells', 3),
                    outline_width=cfg.get('outline_width', 2),
                    box_scale=cfg.get('box_scale', 1.0)
                )
                self.targets.append(target)
            
            # For backward compatibility, keep self.target pointing to first target (or None)
            self.target = self.targets[0] if self.targets else None
        else:
            self.targets = []
            self.target = None # If no target is needed, set target to None

        self.window = None
        self.clock = None
        self.step_counter = 0
        
        # Track terminations for rendering
        self.terminations = {a.name: False for a in self.agent_list}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Resets the environment to a new state.

        Args:
            seed: Optional seed for the random number generator
            options: Optional dictionary of options

        Returns:
            tuple: (observations, infos) dictionaries keyed by agent name
        '''
        # Initialize random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset agents list to all possible agents
        self.agents = self.possible_agents.copy()
        
        # Reset terminations
        self.terminations = {a.name: False for a in self.agent_list}

        # Optionally generate new obstacles each reset
        self._obstacles = []
        if self.enable_obstacles and self.num_obstacles > 0:
            self._generate_random_obstacles(self.num_obstacles)
        
        # Generate visual obstacles (block view but not movement)
        self._visual_obstacles = []
        if self.enable_obstacles and self.num_visual_obstacles > 0:
            self._generate_random_visual_obstacles(self.num_visual_obstacles)

        # Sample positions not inside obstacles (only physical obstacles block spawn)
        forbidden = self._cells_covered_by_obstacles()
        choices = [i for i in range(self.size * self.size) if (i // self.size, i % self.size) not in forbidden]
        num_needed = len(self.agent_list) + (0 if self.no_target else len(self.targets))
        
        if len(choices) < num_needed:
            # fallback: ignore obstacles if too dense
            coords = np.random.choice(self.size * self.size, size=num_needed, replace=False)
        else:
            coords = np.random.choice(choices, size=num_needed, replace=False)

        for i, a in enumerate(self.agent_list):
            self._update_agent_location(a, np.array([coords[i] // self.size, coords[i] % self.size], dtype=int))
        
        # Initialize or reset the targets
        if self.no_target:
            self.targets = []
            self.target = None
        else:
            # Reset all targets to new positions
            for idx, target in enumerate(self.targets):
                t_idx = len(self.agent_list) + idx  # Index after all agents
                target_position = (coords[t_idx] // self.size, coords[t_idx] % self.size)
                target.reset(target_position)
            
            # For backward compatibility, keep self.target pointing to first target
            self.target = self.targets[0] if self.targets else None

        if self.intrinsic: # Counting visits to cells if intrinsic exploration is enabled
            for a in self.agent_list:
                self._visit_counts[a.name].fill(0)
                self._visit_counts[a.name][tuple(a.location)] += 1

        self.step_counter = 0

        # Generate initial observations and infos for all agents
        observations = {}
        infos = {}
        
        for agent_name in self.agents:
            observations[agent_name] = self.observe(agent_name)
            infos[agent_name] = self._get_agent_info(agent_name)

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def action_space(self, agent: str):
        """
        Get the action space for a specific agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Action space for the agent
        """
        return self.action_spaces[agent]
    
    def observation_space(self, agent: str):
        """
        Get the observation space for a specific agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Observation space for the agent
        """
        return self.observation_spaces[agent]

    def step(self, actions: Dict[str, int]):
        """
        Parallel step method - all agents act simultaneously.
        
        Args:
            actions: Dictionary of {agent_name: action} for all active agents
            
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos) dictionaries
        """
        self.step_counter += 1
        self._step_count += 1
        
        # Clear screen and show step info if FOV display is enabled
        self._clear_screen_and_show_step()

        # Prepare environment state for agent steps
        env_state = {
            "agents": self.agent_list,
            "target": self.target,  # For backward compatibility
            "targets": self.targets,  # New: list of all targets
            "obstacles": self._obstacles,
            "visual_obstacles": self._visual_obstacles,
            "grid_size": self.size
        }

        # Process all agent actions simultaneously
        agent_results = {}
        obstacle_collisions = {}
        
        for agent_name, action in actions.items():
            if agent_name not in self._agents_by_name:
                continue
                
            agent_obj = self._agents_by_name[agent_name]
            
            # Track if agent collided with an obstacle
            obstacle_collision = False
            
            # Use agent's own step method if available, otherwise fall back to legacy method
            if hasattr(agent_obj, 'step') and callable(getattr(agent_obj, 'step')):
                step_result = agent_obj.step(int(action), env_state)
                proposed_loc = step_result["new_location"]
                # Check if movement was blocked by obstacle
                if "obstacle_collision" in step_result:
                    obstacle_collision = step_result["obstacle_collision"]
                agent_results[agent_name] = proposed_loc
            else:
                # Legacy movement method
                direction = self._action_to_direction[int(action)]
                proposed_loc = np.clip(agent_obj.location + direction, 0, self.size - 1)
                
                # Block obstacles
                if self._is_cell_obstacle(proposed_loc):
                    obstacle_collision = True
                    proposed_loc = agent_obj.location
                    
                agent_results[agent_name] = proposed_loc
            
            obstacle_collisions[agent_name] = obstacle_collision

        # Resolve agent-to-agent collisions
        self._resolve_agent_collisions(agent_results)

        # Update all agent locations
        for agent_name, new_location in agent_results.items():
            agent_obj = self._agents_by_name[agent_name]
            self._update_agent_location(agent_obj, new_location)

        # Calculate rewards for ALL agents (before target moves)
        rewards = {}
        terminations = {}
        truncations = {}
        
        # Update detection state for visual indicators
        self._agent_detects_target = {}
        self._target_detects_agent = {}
        
        for agent_name in self.agents:
            agent_obj = self._agents_by_name[agent_name]
            
            # Calculate individual reward for this agent
            reward = 0.0
            
            # Apply obstacle collision penalty if agent hit an obstacle
            if obstacle_collisions.get(agent_name, False):
                reward += self.obstacle_collision_penalty
            
            # Check if this agent reached any target (terminates the agent)
            reached_target = False
            if not self.no_target and self.targets:
                for target in self.targets:
                    if np.array_equal(agent_obj.location, target.location):
                        reached_target = True
                        break
            
            if reached_target:
                reward += -1.0 * self.lambda_fov  # Penalty for reaching target
            else:
                # Check if this agent detects any target (only for agents with FOV)
                agent_detects = False
                if hasattr(agent_obj, 'fov_size'):
                    # Get the agent's actual observation to check if any target is visible
                    obs = agent_obj.observe(env_state)
                    fov_map = obs["obstacles_fov"]
                    # Check if target (value 3) is in the observed FOV
                    agent_detects = np.any(fov_map == 3)
                self._agent_detects_target[agent_name] = agent_detects
                
                # Check if any target detects this agent
                target_detects = False
                if self.targets:
                    for target in self.targets:
                        if self._is_agent_in_target_fov(agent_obj, target):
                            target_detects = True
                            break
                self._target_detects_agent[agent_name] = target_detects
                    
                # Penalty: -lambda if agent is in any target's FOV
                if target_detects:
                    reward -= self.lambda_fov
                # Reward: (1-lambda) if any target is in agent's FOV
                elif agent_detects:
                    reward += (1 - self.lambda_fov)
                else:
                    # Small step penalty if no detection
                    reward += -0.01

            # Intrinsic exploration rewards (if enabled)
            if self.intrinsic:
                if self._visit_counts[agent_name][tuple(agent_obj.location)] == 0:
                    reward += 0.025
                self._visit_counts[agent_name][tuple(agent_obj.location)] += 1

            rewards[agent_name] = reward
            
            # Check termination/truncation for this agent
            # Terminates the agent if it reaches the target or if detected by target (death_on_sight)
            detected_by_target = self._target_detects_agent.get(agent_name, False)
            terminated = reached_target or (self.death_on_sight and detected_by_target)
            terminations[agent_name] = terminated
            truncations[agent_name] = self.step_counter >= self.max_steps
            
            # Store termination status for rendering
            self.terminations[agent_name] = terminated

        # NOW move all targets (after all rewards calculated)
        for target in self.targets:
            target.step(self.step_counter, self._obstacles, self._is_cell_obstacle)

        # Generate observations and infos for all agents
        observations = {}
        infos = {}
        
        for agent_name in self.agents:
            observations[agent_name] = self.observe(agent_name)
            infos[agent_name] = self._get_agent_info(agent_name)

        # Remove terminated/truncated agents from active agents list
        self.agents = [
            agent for agent in self.agents 
            if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str):
        '''
        Returns observation for a single agent.
        
        Args:
            agent: The name of the agent to observe
            
        Returns:
            dict: Observation containing agent location, FOV obstacles, and optionally target location
        '''
        if agent not in self._agents_by_name:
            raise ValueError(f"Agent {agent} not found")
            
        agent_obj = self._agents_by_name[agent]
        
        # Prepare environment state for agent observation
        env_state = {
            "agents": self.agent_list,
            "target": self.target,  # For backward compatibility
            "targets": self.targets,  # New: list of all targets
            "obstacles": self._obstacles,
            "visual_obstacles": self._visual_obstacles,
            "grid_size": self.size
        }
        
        # Use agent's own observation method if available, otherwise fall back to legacy method
        if hasattr(agent_obj, 'observe') and callable(getattr(agent_obj, 'observe')):
            return agent_obj.observe(env_state)
        else:
            # Legacy observation method
            fov_obstacles = self._get_agent_fov_obstacles(agent_obj)
            obs = {
                "agent": agent_obj.location,
                "obstacles_fov": fov_obstacles
            }
            # Only include target coordinates if show_target_coords is True
            if self.show_target_coords and self.targets:
                # Include all target locations as a list
                obs["targets"] = np.array([target.location for target in self.targets])
                # For backward compatibility, also include first target as "target"
                obs["target"] = self.targets[0].location
            elif self.show_target_coords:
                obs["targets"] = np.array([])
                obs["target"] = np.array([-1, -1])
            return obs

    def _get_agent_info(self, agent_name: str):
        '''Get info for a specific agent'''
        info = {}
        if agent_name in self._agents_by_name:
            agent_obj = self._agents_by_name[agent_name]
            
            # Include distances to all targets
            if self.targets:
                for idx, target in enumerate(self.targets):
                    distance = np.linalg.norm(agent_obj.location - target.location, ord=1)
                    if idx == 0:
                        # For backward compatibility, first target uses the old key
                        info[f"distance_{agent_name}"] = distance
                    info[f"distance_{agent_name}_to_{target.name}"] = distance
                
                # Also include distance to closest target
                distances = [np.linalg.norm(agent_obj.location - target.location, ord=1) for target in self.targets]
                info[f"distance_{agent_name}_to_closest_target"] = min(distances) if distances else -1
            else:
                info[f"distance_{agent_name}"] = -1
        else:
            info[f"distance_{agent_name}"] = -1
        return info

    def _resolve_agent_collisions(self, agent_results: Dict[str, np.ndarray]):
        """Resolve collisions between agents by preventing overlapping positions."""
        # Create a mapping of proposed positions to agents
        position_to_agents = {}
        for agent_name, proposed_pos in agent_results.items():
            pos_tuple = tuple(proposed_pos)
            if pos_tuple not in position_to_agents:
                position_to_agents[pos_tuple] = []
            position_to_agents[pos_tuple].append(agent_name)
        
        # For each position with multiple agents, keep only the first agent there
        for pos_tuple, agents_at_pos in position_to_agents.items():
            if len(agents_at_pos) > 1:
                # Keep the first agent at this position, move others back to their original positions
                for agent_name in agents_at_pos[1:]:
                    agent_obj = self._agents_by_name[agent_name]
                    agent_results[agent_name] = agent_obj.location  # Revert to original position

    def _update_agent_location(self, agent: 'Agent', new_location: np.ndarray) -> None:
        """Update agent location and track last 3 visited cells."""
        agent.location = new_location
        # Add current location to visited cells (as tuple for immutability)
        cell_tuple = tuple(new_location)
        agent.last_visited_cells.append(cell_tuple)
        # Keep only last 3 cells
        if len(agent.last_visited_cells) > 3:
            agent.last_visited_cells.pop(0)

    def _get_agent_fov_obstacles(self, agent: 'Agent') -> np.ndarray:
        """Generate field of view obstacle map for an agent - OPTIMIZED VERSION."""
        fov_size = agent.fov_size
        if fov_size is None:
            fov_size = 5
        
        # Agent's current position
        agent_x, agent_y = agent.location
        offset = fov_size // 2
        
        # Pre-calculate all world coordinates at once (vectorized)
        world_x_coords = np.arange(agent_x - offset, agent_x - offset + fov_size)
        world_y_coords = np.arange(agent_y - offset, agent_y - offset + fov_size)
        
        # Create meshgrid for all coordinates
        world_x_grid, world_y_grid = np.meshgrid(world_x_coords, world_y_coords, indexing='ij')
        
        # Check boundaries (vectorized)
        out_of_bounds = ((world_x_grid < 0) | (world_x_grid >= self.size) | 
                        (world_y_grid < 0) | (world_y_grid >= self.size))
        
        # Initialize FOV map with boundary obstacles
        fov_map = out_of_bounds.astype(int)
        
        # Only check obstacles and other agents for in-bounds cells
        valid_mask = ~out_of_bounds
        if valid_mask.any():
            # Get valid coordinates
            valid_x = world_x_grid[valid_mask]
            valid_y = world_y_grid[valid_mask]
            
            # Check obstacles, other agents, and target for valid cells only
            for x, y in zip(valid_x, valid_y):
                x_int, y_int = int(x), int(y)
                fov_i = int(x - agent_x + offset)
                fov_j = int(y - agent_y + offset)
                
                # Check if any target is at this position
                target_present = False
                for target in self.targets:
                    if target.location[0] == x_int and target.location[1] == y_int:
                        fov_map[fov_i, fov_j] = 3  # Mark target as 3
                        target_present = True
                        break
                
                # Check if another agent is at this position (only if no target)
                if not target_present:
                    other_agent_present = False
                    for other_agent in self.agent_list:
                        if (other_agent.name != agent.name and 
                            other_agent.location[0] == x_int and 
                            other_agent.location[1] == y_int):
                            fov_map[fov_i, fov_j] = 2  # Mark other agent as 2
                            other_agent_present = True
                            break
                    
                    # If no other agent, check for obstacles
                    if not other_agent_present and self._is_cell_obstacle_fast(x_int, y_int):
                        fov_map[fov_i, fov_j] = 1  # Mark obstacle as 1
        
        return fov_map
    
    def _is_agent_in_target_fov(self, agent: 'Agent', target: Optional[Target] = None) -> bool:
        """
        Check if an agent is within a target's field of view.
        
        Args:
            agent: The agent to check
            target: The target to check against. If None, uses self.target (for backward compatibility)
            
        Returns:
            True if agent is in target's FOV, False otherwise
        """
        # For backward compatibility, use self.target if no target provided
        if target is None:
            target = self.target
        
        if target is None:
            return False
        
        # Get target's FOV size (use box_cells as FOV size)
        target_fov_size = target.box_cells
        if target_fov_size is None:
            target_fov_size = 3  # Default FOV size
        
        # Target's current position
        target_x, target_y = target.location
        offset = target_fov_size // 2
        
        # Agent's position
        agent_x, agent_y = agent.location
        
        # Check if agent is within target's FOV bounds
        fov_left = target_x - offset
        fov_right = target_x + offset
        fov_top = target_y - offset
        fov_bottom = target_y + offset
        
        return (fov_left <= agent_x <= fov_right and 
                fov_top <= agent_y <= fov_bottom)

    def _print_fov_display(self, agent_name: str, fov_map: np.ndarray, action_info: str = ""):
        """Print FOV map that can be overwritten, aligned 'facing north' like pygame"""
        if not self.show_fov_display:
            return
            
        print(f"{agent_name.upper()} FOV {action_info}:")
        # Reverse the rows to match pygame orientation (north at top)
        for row in reversed(fov_map):
            print(f"  {' '.join(str(cell) for cell in row)}")
        
        # Show legend for first agent only to avoid clutter
        if agent_name == self.agents[0]:
            print("  Legend: 0=Empty, 1=Physical Obstacle, 2=Agent, 3=Target, 4=Visual Obstacle")
        print()  # New line after each agent's FOV

    def _clear_screen_and_show_step(self):
        """Clear screen and show step information"""
        if not self.show_fov_display:
            return
            
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print("="*80)
        print(f"PARALLEL STEP {self._step_count}".center(80))
        print("="*80)

    def _cells_covered_by_obstacles(self) -> set:
        covered = set()
        for (ox, oy, ow, oh) in self._obstacles:
            for dx in range(ow):
                for dy in range(oh):
                    cx, cy = ox + dx, oy + dy
                    if 0 <= cx < self.size and 0 <= cy < self.size:
                        covered.add((cx, cy))
        return covered

    def _is_cell_obstacle(self, cell: np.ndarray) -> bool:
        x, y = int(cell[0]), int(cell[1])
        for (ox, oy, ow, oh) in self._obstacles:
            if ox <= x < ox + ow and oy <= y < oy + oh:
                return True
        return False
    
    def _is_cell_obstacle_fast(self, x: int, y: int) -> bool:
        """Optimized obstacle checking without numpy array creation."""
        for (ox, oy, ow, oh) in self._obstacles:
            if ox <= x < ox + ow and oy <= y < oy + oh:
                return True
        return False

    def _generate_random_obstacles(self, count: int) -> None:
        """Generate rectangular physical obstacles within grid bounds."""
        attempts = 0
        max_attempts = count * 10
        created = 0
        while created < count and attempts < max_attempts:
            attempts += 1
            # Random top-left
            ox = int(np.random.randint(0, self.size))
            oy = int(np.random.randint(0, self.size))
            # Random width/height (at least 1 cell), capped to remain in bounds
            max_w = max(1, self.size - ox)
            max_h = max(1, self.size - oy)
            # Prefer small obstacles
            ow = int(np.random.randint(1, min(4, max_w) + 1))
            oh = int(np.random.randint(1, min(4, max_h) + 1))
            # Avoid duplicates
            rect = (ox, oy, ow, oh)
            if rect in self._obstacles:
                continue
            self._obstacles.append(rect)
            created += 1
    
    def _generate_random_visual_obstacles(self, count: int) -> None:
        """Generate rectangular visual obstacles within grid bounds.
        
        Visual obstacles block ObserverAgent view but do not block movement.
        They can overlap with physical obstacles, agents, and target, but not with each other.
        """
        attempts = 0
        max_attempts = count * 20
        created = 0
        while created < count and attempts < max_attempts:
            attempts += 1
            # Random top-left
            ox = int(np.random.randint(0, self.size))
            oy = int(np.random.randint(0, self.size))
            # Random width/height (at least 1 cell), capped to remain in bounds
            max_w = max(1, self.size - ox)
            max_h = max(1, self.size - oy)
            # Prefer small obstacles (same size distribution as physical obstacles)
            ow = int(np.random.randint(1, min(4, max_w) + 1))
            oh = int(np.random.randint(1, min(4, max_h) + 1))
            
            rect = (ox, oy, ow, oh)
            
            # Check for exact duplicate
            if rect in self._visual_obstacles:
                continue
            
            # Check for overlap with other visual obstacles
            overlaps = False
            for (vis_ox, vis_oy, vis_ow, vis_oh) in self._visual_obstacles:
                # Check if rectangles overlap using standard rectangle intersection test
                if not (ox + ow <= vis_ox or vis_ox + vis_ow <= ox or
                        oy + oh <= vis_oy or vis_oy + vis_oh <= oy):
                    overlaps = True
                    break
            
            if not overlaps:
                self._visual_obstacles.append(rect)
                created += 1

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        """
        Render the environment.
        
        Returns:
            numpy array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        else:
            return None

    def _render_frame(self):
        # Initialize pygame for both human and rgb_array modes
        if not pygame.get_init():
            pygame.init()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            # Increase window height to accommodate info panel
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 120))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Colors (dark theme)
        BG_COLOR = (30, 30, 30)
        GRID_COLOR = (60, 60, 60)
        TARGET_COLOR = (240, 100, 90)  # red-ish
        INFO_PANEL_COLOR = (50, 50, 50)
        TEXT_COLOR = (255, 255, 255)

        # Create canvas with extra height for info panel
        canvas = pygame.Surface((self.window_size, self.window_size + 120))
        canvas.fill(BG_COLOR)
        pix_square = self.window_size / self.size

        # Grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, GRID_COLOR, (0, pix_square * x), (self.window_size, pix_square * x), 1
            )
            pygame.draw.line(
                canvas, GRID_COLOR, (pix_square * x, 0), (pix_square * x, self.window_size), 1
            )

        # Physical obstacles (filled rectangles - dark gray)
        if self._obstacles:
            OBSTACLE_COLOR = (90, 90, 90)
            for (ox, oy, ow, oh) in self._obstacles:
                rect = pygame.Rect(
                    (ox * pix_square, oy * pix_square),
                    (ow * pix_square, oh * pix_square),
                )
                pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)
        
        # Visual obstacles (semi-transparent light blue - blocks view but not movement)
        if self._visual_obstacles:
            VISUAL_OBSTACLE_COLOR = (150, 200, 240, 150)  # Light blue with alpha
            VISUAL_OBSTACLE_STRIPE_COLOR = (100, 160, 220, 200)  # Darker blue for stripes
            
            for (ox, oy, ow, oh) in self._visual_obstacles:
                # Create temporary surface for this obstacle
                temp_surface = pygame.Surface((ow * pix_square, oh * pix_square), pygame.SRCALPHA)
                temp_surface.fill(VISUAL_OBSTACLE_COLOR)
                
                # Add diagonal stripe pattern
                stripe_spacing = 8  # pixels between stripes
                stripe_width = 3
                
                # Draw diagonal stripes from top-left to bottom-right
                for i in range(-int(oh * pix_square), int(ow * pix_square), stripe_spacing):
                    start_x = i
                    start_y = 0
                    end_x = i + int(oh * pix_square)
                    end_y = int(oh * pix_square)
                    pygame.draw.line(temp_surface, VISUAL_OBSTACLE_STRIPE_COLOR, 
                                   (start_x, start_y), (end_x, end_y), stripe_width)
                
                canvas.blit(temp_surface, (ox * pix_square, oy * pix_square))

        # Targets (draw all targets)
        for target in self.targets:
            # Get triangle points from target (pass pix_square as float)
            point1, point2, point3 = target.get_triangle_points(pix_square)
            pygame.draw.polygon(canvas, target.color, [point1, point2, point3])
            
            # Draw target box outline
            top_left, top_right, bottom_right, bottom_left = target.get_box_coordinates(pix_square)
            box_points = [top_left, top_right, bottom_right, bottom_left, top_left]  # Close the polygon
            pygame.draw.lines(canvas, target.color, False, box_points, target.outline_width)
            
            # Draw detection dot if this target detects any agent
            # Check if any agent is detected by this specific target
            target_detects = False
            for agent_name in self._agents_by_name:
                agent_obj = self._agents_by_name[agent_name]
                if self._is_agent_in_target_fov(agent_obj, target):
                    target_detects = True
                    break
            
            if target_detects:
                target_center = (target.location + 0.5) * pix_square
                dot_center = target_center + np.array([0, -pix_square * 0.6])  # Position higher above target
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

        # Agents (circles or squares)
        for a in self.agent_list:
            center = (a.location + 0.5) * pix_square
            
            # Use gray color if agent is terminated
            agent_color = (128, 128, 128) if self.terminations.get(a.name, False) else a.color
            
            # Check if this is an ObserverAgent to render as square
            if hasattr(a, 'altitude') and hasattr(a, 'fov_base_size'):
                # Render as square for ObserverAgent
                square_size = int(pix_square / 3)
                square_rect = pygame.Rect(
                    center[0] - square_size // 2,
                    center[1] - square_size // 2,
                    square_size,
                    square_size
                )
                pygame.draw.rect(canvas, agent_color, square_rect)
            else:
                # Render as circle for other agents
                pygame.draw.circle(canvas, agent_color, center.astype(int), int(pix_square / 3))
            
            # Draw detection dot if agent detects target
            if self._agent_detects_target.get(a.name, False):
                dot_center = center + np.array([0, -pix_square * 0.6])  # Position higher above agent
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

        # Outline boxes using per-agent params
        def draw_agent_box(a: Agent):
            # Use gray color if agent is terminated
            box_color = (128, 128, 128) if self.terminations.get(a.name, False) else a.color
            
            if a.fov_size > 1:
                # Draw a kxk box centered on the agent, without clamping to grid bounds
                k = a.fov_size
                offset = k // 2
                # Compute top-left in cell coordinates centered on agent
                start_cell = (a.location - np.array([offset, offset]))
                top_left = start_cell * pix_square
                side_px = pix_square * k
                rect = pygame.Rect(top_left, (side_px, side_px))
                
                # Check if this is an ObserverAgent for special line styles and visible area
                if hasattr(a, 'altitude') and hasattr(a, 'fov_base_size'):
                    # ObserverAgent with flight level-based visible area and line styles
                    # Flight levels are only 1, 2, 3 (never 0)
                    if a.altitude >= 1:
                        # Calculate visible area based on flight level
                        # Flight Level 1: visible = fov_size - 4 (3x3 for fov_size 7)
                        # Flight Level 2: visible = fov_size - 2 (5x5 for fov_size 7)
                        # Flight Level 3: visible = fov_size (7x7 for fov_size 7)
                        rings_to_mask = 3 - a.altitude
                        visible_size = a.fov_size - (rings_to_mask * 2)
                        
                        # Calculate new rectangle for visible area
                        visible_offset = visible_size // 2
                        visible_start_cell = (a.location - np.array([visible_offset, visible_offset]))
                        visible_top_left = visible_start_cell * pix_square
                        visible_side_px = pix_square * visible_size
                        visible_rect = pygame.Rect(visible_top_left, (visible_side_px, visible_side_px))
                        
                        # Draw with appropriate line style based on flight level
                        if a.altitude == 1:  # Flight level 1 - solid line (3x3)
                            pygame.draw.rect(canvas, box_color, visible_rect, width=a.outline_width)
                        elif a.altitude == 2:  # Flight level 2 - dashed line (5x5)
                            draw_dashed_rect(canvas, box_color, visible_rect, a.outline_width)
                        elif a.altitude == 3:  # Flight level 3 - dotted line (7x7)
                            draw_dotted_rect(canvas, box_color, visible_rect, a.outline_width)
                    else:
                        # Ground level (altitude 0) - no FOV border
                        pass
                else:
                    # Regular agent - solid line with full FOV
                    pygame.draw.rect(canvas, box_color, rect, width=a.outline_width)
            else:
                box_side = pix_square * a.box_scale
                top_left = (a.location * pix_square) + ((pix_square - box_side) / 2)
                rect = pygame.Rect(top_left, (box_side, box_side))
                pygame.draw.rect(canvas, box_color, rect, width=a.outline_width)
        
        def draw_dashed_rect(surface, color, rect, width):
            """Draw a dashed rectangle."""
            x, y, w, h = rect
            dash_length = 8
            gap_length = 4
            
            # Top edge
            draw_dashed_line(surface, color, (x, y), (x + w, y), dash_length, gap_length, width)
            # Bottom edge
            draw_dashed_line(surface, color, (x, y + h), (x + w, y + h), dash_length, gap_length, width)
            # Left edge
            draw_dashed_line(surface, color, (x, y), (x, y + h), dash_length, gap_length, width)
            # Right edge
            draw_dashed_line(surface, color, (x + w, y), (x + w, y + h), dash_length, gap_length, width)
        
        def draw_dotted_rect(surface, color, rect, width):
            """Draw a dotted rectangle."""
            x, y, w, h = rect
            dot_spacing = 6
            
            # Top edge
            draw_dotted_line(surface, color, (x, y), (x + w, y), dot_spacing, width)
            # Bottom edge
            draw_dotted_line(surface, color, (x, y + h), (x + w, y + h), dot_spacing, width)
            # Left edge
            draw_dotted_line(surface, color, (x, y), (x, y + h), dot_spacing, width)
            # Right edge
            draw_dotted_line(surface, color, (x + w, y), (x + w, y + h), dot_spacing, width)
        
        def draw_dashed_line(surface, color, start, end, dash_length, gap_length, width):
            """Draw a dashed line."""
            x1, y1 = start
            x2, y2 = end
            
            # Calculate line length and direction
            dx = x2 - x1
            dy = y2 - y1
            line_length = (dx * dx + dy * dy) ** 0.5
            
            if line_length == 0:
                return
            
            # Normalize direction
            dx /= line_length
            dy /= line_length
            
            # Draw dashes
            current_length = 0
            draw_dash = True
            
            while current_length < line_length:
                if draw_dash:
                    # Calculate dash end
                    dash_end_length = min(current_length + dash_length, line_length)
                    start_x = x1 + dx * current_length
                    start_y = y1 + dy * current_length
                    end_x = x1 + dx * dash_end_length
                    end_y = y1 + dy * dash_end_length
                    
                    pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)
                    current_length = dash_end_length
                else:
                    current_length += gap_length
                
                draw_dash = not draw_dash
        
        def draw_dotted_line(surface, color, start, end, dot_spacing, width):
            """Draw a dotted line."""
            x1, y1 = start
            x2, y2 = end
            
            # Calculate line length and direction
            dx = x2 - x1
            dy = y2 - y1
            line_length = (dx * dx + dy * dy) ** 0.5
            
            if line_length == 0:
                return
            
            # Normalize direction
            dx /= line_length
            dy /= line_length
            
            # Draw dots
            current_length = 0
            while current_length < line_length:
                dot_x = x1 + dx * current_length
                dot_y = y1 + dy * current_length
                pygame.draw.circle(surface, color, (int(dot_x), int(dot_y)), width)
                current_length += dot_spacing

        for a in self.agent_list:
            draw_agent_box(a)
        
        # Draw halos around agents that see other agents in their FOV
        for a in self.agent_list:
            fov_map = self._get_agent_fov_obstacles(a)
            if np.any(fov_map == 2):  # Check if agent sees other agents (value 2)
                # Draw a halo around the agent
                center_x = (a.location[0] + 0.5) * pix_square
                center_y = (a.location[1] + 0.5) * pix_square
                halo_radius = pix_square * 0.6  # Slightly larger than agent box
                pygame.draw.circle(canvas, a.color, (int(center_x), int(center_y)), int(halo_radius), width=3)
            
        # Draw lines through last 3 visited cells for each agent
        for a in self.agent_list:
            if len(a.last_visited_cells) >= 2:
                # Convert cell coordinates to pixel coordinates (center of cells)
                pixel_points = []
                for cell in a.last_visited_cells:
                    x, y = cell
                    center_x = (x + 0.5) * pix_square
                    center_y = (y + 0.5) * pix_square
                    pixel_points.append((center_x, center_y))
                
                # Draw lines between consecutive points
                for i in range(len(pixel_points) - 1):
                    start_pos = pixel_points[i]
                    end_pos = pixel_points[i + 1]
                    pygame.draw.line(canvas, a.color, start_pos, end_pos, width=2)

        # Information panel
        info_panel_y = self.window_size
        info_panel_height = 120
        
        # Draw info panel background
        info_rect = pygame.Rect(0, info_panel_y, self.window_size, info_panel_height)
        pygame.draw.rect(canvas, INFO_PANEL_COLOR, info_rect)
        pygame.draw.rect(canvas, GRID_COLOR, info_rect, 2)  # Border
        
        # Initialize font
        try:
            # Initialize font module if not already done
            if not pygame.font.get_init():
                pygame.font.init()
            
            font_large = pygame.font.Font(None, 24)
            font_medium = pygame.font.Font(None, 18)
            font_small = pygame.font.Font(None, 16)
        except:
            try:
                font_large = pygame.font.SysFont('Arial', 24)
                font_medium = pygame.font.SysFont('Arial', 18)
                font_small = pygame.font.SysFont('Arial', 16)
            except:
                try:
                    # Last resort: use default font
                    font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
                    font_medium = pygame.font.Font(pygame.font.get_default_font(), 18)
                    font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
                except:
                    # If all else fails, create dummy fonts
                    font_large = None
                    font_medium = None
                    font_small = None
        
        # Only render text if fonts are available
        if font_large is not None and font_medium is not None and font_small is not None:
            # Step counter and parallel mode indicator
            step_text = f"Parallel Step: {self.step_counter}"
            step_surface = font_large.render(step_text, True, TEXT_COLOR)
            canvas.blit(step_surface, (10, info_panel_y + 10))
            
            # Agent information
            y_offset = info_panel_y + 40
            for i, agent in enumerate(self.agent_list):
                # Agent name and position
                agent_info = f"{agent.name}: ({agent.location[0]}, {agent.location[1]})"
                agent_text = font_medium.render(agent_info, True, agent.color)
                canvas.blit(agent_text, (10, y_offset + i * 20))
            
            # Target information (show all targets)
            target_y_offset = y_offset + len(self.agent_list) * 20
            for idx, target in enumerate(self.targets):
                target_info = f"{target.name}: ({target.location[0]}, {target.location[1]})"
                target_text = font_medium.render(target_info, True, target.color)
                canvas.blit(target_text, (10, target_y_offset + idx * 20))
            
            # Lambda FOV parameter
            lambda_text = font_small.render(f"λ FOV: {self.lambda_fov:.2f}", True, TEXT_COLOR)
            canvas.blit(lambda_text, (self.window_size - 200, info_panel_y + 10))
            
            # FOV legend
            legend_text = font_small.render("FOV: 0=Empty, 1=Physical, 2=Agent, 3=Target, 4=Visual", True, TEXT_COLOR)
            canvas.blit(legend_text, (self.window_size - 200, info_panel_y + 30))

        # Finalize frame
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


class GridWorldEnvParallelExploration(GridWorldEnvParallel):
    ''' A multi-agent grid world environment for exploration where agents navigate to a static goal 
    while avoiding moving targets.
    
    This environment follows the PettingZoo Parallel API interface, where all agents
    act simultaneously. Agents are rewarded +5 for reaching the static goal.
    
    Key differences from GridWorldEnvParallel:
    - Has a static goal that agents must reach (rewarded +5)
    - Moving targets still exist but their positions are NOT visible in observations
    - Agents must navigate to the goal while avoiding the moving targets
    
    Input parameters:
     - All parameters from GridWorldEnvParallel
     - goal_color: Color of the static goal (default: (0, 255, 0) - green)
    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        # General settings
        size=5,
        max_steps: int = 500,
        render_mode=None,
        show_fov_display: bool = True,

        # Rewards and target settings
        intrinsic=False,
        lambda_fov: float = 0.5,
        show_target_coords: bool = False,  # Ignored - goal coords shown instead
        obstacle_collision_penalty: float = -0.05,
        death_on_sight: bool = False,

        # Agents and target settings
        no_target=False,
        agents: Optional[List[Union[Dict, 'Agent']]] = None,
        target_config: Optional[Union[Dict, List[Dict]]] = None,

        # Obstacles
        enable_obstacles: bool = False,
        num_obstacles: int = 0,
        num_visual_obstacles: int = 0,
        
        # Goal settings
        goal_color: Tuple[int, int, int] = (0, 255, 0),  # Green by default
        goal_approach_reward_scale: float = 0.05,

    ):
        # Call parent __init__ but we'll override observation spaces
        super().__init__(
            size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            show_fov_display=show_fov_display,
            intrinsic=intrinsic,
            lambda_fov=lambda_fov,
            show_target_coords=False,  # Force False - we show goal instead
            obstacle_collision_penalty=obstacle_collision_penalty,
            death_on_sight=death_on_sight,
            no_target=no_target,
            agents=agents,
            target_config=target_config,
            enable_obstacles=enable_obstacles,
            num_obstacles=num_obstacles,
            num_visual_obstacles=num_visual_obstacles,
        )
        
        # Static goal location (will be set in reset)
        self.goal_location: np.ndarray = np.array([-1, -1], dtype=int)
        self.goal_color = goal_color
        self.goal_approach_reward_scale = float(goal_approach_reward_scale)
        self._prev_distance_to_goal: Dict[str, Optional[float]] = {}
        
        # Update observation spaces to include goal instead of target
        for agent_name in self.possible_agents:
            agent_obs_space = self.observation_spaces[agent_name].spaces.copy()
            # Remove target-related keys if they exist
            if 'target' in agent_obs_space:
                del agent_obs_space['target']
            if 'targets' in agent_obs_space:
                del agent_obs_space['targets']
            # Add goal position
            agent_obs_space['goal'] = spaces.Box(
                low=0, high=self.size - 1, shape=(2,), dtype=np.int32
            )
            self.observation_spaces[agent_name] = spaces.Dict(agent_obs_space)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Resets the environment to a new state with a static goal.
        
        Args:
            seed: Optional seed for the random number generator
            options: Optional dictionary of options
            
        Returns:
            tuple: (observations, infos) dictionaries keyed by agent name
        '''
        # Call parent reset first
        observations, infos = super().reset(seed=seed, options=options)
        
        # Place static goal at a valid position (not on obstacles, agents, or targets)
        forbidden = self._cells_covered_by_obstacles()
        
        # Add agent positions
        for agent in self.agent_list:
            forbidden.add(tuple(agent.location))
        
        # Add target positions
        for target in self.targets:
            forbidden.add(tuple(target.location))
        
        # Find valid positions for goal
        choices = [
            i for i in range(self.size * self.size) 
            if (i // self.size, i % self.size) not in forbidden
        ]
        
        if len(choices) == 0:
            # Fallback: place goal randomly (might overlap)
            goal_coord = np.random.choice(self.size * self.size)
        else:
            goal_coord = np.random.choice(choices)
        
        self.goal_location = np.array([
            goal_coord // self.size,
            goal_coord % self.size
        ], dtype=int)
        
        # Update observations to include goal position
        for agent_name in self.agents:
            if agent_name in observations:
                # Remove target-related keys if they exist
                if 'target' in observations[agent_name]:
                    del observations[agent_name]['target']
                if 'targets' in observations[agent_name]:
                    del observations[agent_name]['targets']
                # Add goal position
                observations[agent_name]['goal'] = self.goal_location.copy()
        
        # Update infos to include goal distance
        for agent_name in self.agents:
            if agent_name in self._agents_by_name:
                agent_obj = self._agents_by_name[agent_name]
                infos[agent_name]['distance_to_goal'] = np.linalg.norm(
                    agent_obj.location - self.goal_location, ord=1
                )
        
        # Initialize distance tracking for goal-approach shaping rewards
        self._prev_distance_to_goal = {}
        for agent in self.agent_list:
            self._prev_distance_to_goal[agent.name] = self._distance_to_goal(agent)

        return observations, infos

    def step(self, actions: Dict[str, int]):
        """
        Parallel step method - all agents act simultaneously.
        Rewards agents +5 for reaching the static goal.
        
        Args:
            actions: Dictionary of {agent_name: action} for all active agents
            
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos) dictionaries
        """
        # Call parent step but we'll modify rewards and observations
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Dense shaping reward based on change in distance to the goal
        if self.goal_approach_reward_scale != 0.0 and self.goal_location[0] >= 0:
            for agent_name in rewards.keys():
                agent_obj = self._agents_by_name.get(agent_name)
                if agent_obj is None:
                    continue
                current_distance = self._distance_to_goal(agent_obj)
                previous_distance = self._prev_distance_to_goal.get(agent_name)
                if previous_distance is not None and current_distance is not None:
                    delta = previous_distance - current_distance
                    if delta != 0:
                        rewards[agent_name] += self.goal_approach_reward_scale * delta
                self._prev_distance_to_goal[agent_name] = current_distance
        
        # Modify rewards: check if agents reached the goal
        for agent_name in self.agents:
            if agent_name in self._agents_by_name:
                agent_obj = self._agents_by_name[agent_name]
                
                # Check if agent reached the goal
                reached_goal = np.array_equal(agent_obj.location, self.goal_location)
                
                if reached_goal:
                    # Reward +5 for reaching goal
                    rewards[agent_name] += 20.0
                
                # Agent terminates if: reached_goal OR caught by target (death_on_sight)
                # Parent step already handles termination when caught by target
                # We combine both conditions here
                already_terminated = terminations.get(agent_name, False)
                terminated = already_terminated or reached_goal
                terminations[agent_name] = terminated
                self.terminations[agent_name] = terminated
                
                # Update info with goal distance
                if agent_name in infos:
                    infos[agent_name]['distance_to_goal'] = np.linalg.norm(
                        agent_obj.location - self.goal_location, ord=1
                    )
        
        # Modify observations: remove target positions, add goal position
        for agent_name in self.agents:
            if agent_name in observations:
                obs = observations[agent_name]
                # Remove target-related keys if they exist
                if 'target' in obs:
                    del obs['target']
                if 'targets' in obs:
                    del obs['targets']
                # Add goal position
                obs['goal'] = self.goal_location.copy()
        
        return observations, rewards, terminations, truncations, infos

    def _distance_to_goal(self, agent: 'Agent') -> Optional[float]:
        """Compute the Manhattan distance from an agent to the static goal."""
        if agent is None or self.goal_location is None:
            return None
        if np.any(self.goal_location < 0):
            return None
        return float(np.linalg.norm(agent.location - self.goal_location, ord=1))

    def observe(self, agent: str):
        '''
        Returns observation for a single agent.
        Includes goal position but NOT target positions.
        
        Args:
            agent: The name of the agent to observe
            
        Returns:
            dict: Observation containing agent location, FOV obstacles, and goal location
        '''
        # Call parent observe
        obs = super().observe(agent)
        
        # Remove target-related keys if they exist
        if 'target' in obs:
            del obs['target']
        if 'targets' in obs:
            del obs['targets']
        
        # Add goal position
        obs['goal'] = self.goal_location.copy()
        
        return obs

    def _get_agent_fov_obstacles(self, agent: 'Agent') -> np.ndarray:
        """Generate field of view obstacle map for an agent.
        Does NOT include targets (they are hidden from observations).
        """
        fov_size = agent.fov_size
        if fov_size is None:
            fov_size = 5
        
        # Agent's current position
        agent_x, agent_y = agent.location
        offset = fov_size // 2
        
        # Pre-calculate all world coordinates at once (vectorized)
        world_x_coords = np.arange(agent_x - offset, agent_x - offset + fov_size)
        world_y_coords = np.arange(agent_y - offset, agent_y - offset + fov_size)
        
        # Create meshgrid for all coordinates
        world_x_grid, world_y_grid = np.meshgrid(world_x_coords, world_y_coords, indexing='ij')
        
        # Check boundaries (vectorized)
        out_of_bounds = ((world_x_grid < 0) | (world_x_grid >= self.size) | 
                        (world_y_grid < 0) | (world_y_grid >= self.size))
        
        # Initialize FOV map with boundary obstacles
        fov_map = out_of_bounds.astype(int)
        
        # Only check obstacles and other agents for in-bounds cells
        valid_mask = ~out_of_bounds
        if valid_mask.any():
            # Get valid coordinates
            valid_x = world_x_grid[valid_mask]
            valid_y = world_y_grid[valid_mask]
            
            # Check obstacles, other agents, and goal for valid cells only
            # NOTE: Targets are NOT included in FOV (they are hidden)
            for x, y in zip(valid_x, valid_y):
                x_int, y_int = int(x), int(y)
                fov_i = int(x - agent_x + offset)
                fov_j = int(y - agent_y + offset)
                
                # Check if goal is at this position
                if (self.goal_location[0] == x_int and self.goal_location[1] == y_int):
                    fov_map[fov_i, fov_j] = 5  # Mark goal as 5
                    continue
                
                # Check if another agent is at this position
                other_agent_present = False
                for other_agent in self.agent_list:
                    if (other_agent.name != agent.name and 
                        other_agent.location[0] == x_int and 
                        other_agent.location[1] == y_int):
                        fov_map[fov_i, fov_j] = 2  # Mark other agent as 2
                        other_agent_present = True
                        break
                
                # If no other agent, check for obstacles
                if not other_agent_present and self._is_cell_obstacle_fast(x_int, y_int):
                    fov_map[fov_i, fov_j] = 1  # Mark obstacle as 1
        
        return fov_map

    def _render_frame(self):
        """Render the environment frame including the static goal."""
        # Initialize pygame for both human and rgb_array modes
        if not pygame.get_init():
            pygame.init()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 120))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Colors (dark theme)
        BG_COLOR = (30, 30, 30)
        GRID_COLOR = (60, 60, 60)
        INFO_PANEL_COLOR = (50, 50, 50)
        TEXT_COLOR = (255, 255, 255)

        # Create canvas with extra height for info panel
        canvas = pygame.Surface((self.window_size, self.window_size + 120))
        canvas.fill(BG_COLOR)
        pix_square = self.window_size / self.size

        # Grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, GRID_COLOR, (0, pix_square * x), (self.window_size, pix_square * x), 1
            )
            pygame.draw.line(
                canvas, GRID_COLOR, (pix_square * x, 0), (pix_square * x, self.window_size), 1
            )

        # Physical obstacles
        if self._obstacles:
            OBSTACLE_COLOR = (90, 90, 90)
            for (ox, oy, ow, oh) in self._obstacles:
                rect = pygame.Rect(
                    (ox * pix_square, oy * pix_square),
                    (ow * pix_square, oh * pix_square),
                )
                pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)
        
        # Visual obstacles
        if self._visual_obstacles:
            VISUAL_OBSTACLE_COLOR = (150, 200, 240, 150)
            VISUAL_OBSTACLE_STRIPE_COLOR = (100, 160, 220, 200)
            
            for (ox, oy, ow, oh) in self._visual_obstacles:
                temp_surface = pygame.Surface((ow * pix_square, oh * pix_square), pygame.SRCALPHA)
                temp_surface.fill(VISUAL_OBSTACLE_COLOR)
                
                stripe_spacing = 8
                stripe_width = 3
                
                for i in range(-int(oh * pix_square), int(ow * pix_square), stripe_spacing):
                    start_x = i
                    start_y = 0
                    end_x = i + int(oh * pix_square)
                    end_y = int(oh * pix_square)
                    pygame.draw.line(temp_surface, VISUAL_OBSTACLE_STRIPE_COLOR, 
                                   (start_x, start_y), (end_x, end_y), stripe_width)
                
                canvas.blit(temp_surface, (ox * pix_square, oy * pix_square))

        # Draw static goal (green circle) - BEFORE targets so it's behind them
        goal_center = (self.goal_location + 0.5) * pix_square
        goal_radius = int(pix_square * 0.3)
        pygame.draw.circle(canvas, self.goal_color, goal_center.astype(int), goal_radius)
        pygame.draw.circle(canvas, (255, 255, 255), goal_center.astype(int), goal_radius, 2)

        # Targets (draw all targets) - on top of goal
        for target in self.targets:
            point1, point2, point3 = target.get_triangle_points(pix_square)
            pygame.draw.polygon(canvas, target.color, [point1, point2, point3])
            
            top_left, top_right, bottom_right, bottom_left = target.get_box_coordinates(pix_square)
            box_points = [top_left, top_right, bottom_right, bottom_left, top_left]
            pygame.draw.lines(canvas, target.color, False, box_points, target.outline_width)
            
            target_detects = False
            for agent_name in self._agents_by_name:
                agent_obj = self._agents_by_name[agent_name]
                if self._is_agent_in_target_fov(agent_obj, target):
                    target_detects = True
                    break
            
            if target_detects:
                target_center = (target.location + 0.5) * pix_square
                dot_center = target_center + np.array([0, -pix_square * 0.6])
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))

        # Agents
        for a in self.agent_list:
            center = (a.location + 0.5) * pix_square
            agent_color = (128, 128, 128) if self.terminations.get(a.name, False) else a.color
            
            if hasattr(a, 'altitude') and hasattr(a, 'fov_base_size'):
                square_size = int(pix_square / 3)
                square_rect = pygame.Rect(
                    center[0] - square_size // 2,
                    center[1] - square_size // 2,
                    square_size,
                    square_size
                )
                pygame.draw.rect(canvas, agent_color, square_rect)
            else:
                pygame.draw.circle(canvas, agent_color, center.astype(int), int(pix_square / 3))
            
            if self._agent_detects_target.get(a.name, False):
                dot_center = center + np.array([0, -pix_square * 0.6])
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))

        # Draw agent FOV boxes (simplified - using parent's logic would be complex)
        for a in self.agent_list:
            if a.fov_size > 1:
                k = a.fov_size
                offset = k // 2
                start_cell = (a.location - np.array([offset, offset]))
                top_left = start_cell * pix_square
                side_px = pix_square * k
                rect = pygame.Rect(top_left, (side_px, side_px))
                box_color = (128, 128, 128) if self.terminations.get(a.name, False) else a.color
                pygame.draw.rect(canvas, box_color, rect, width=a.outline_width)

        # Information panel
        info_panel_y = self.window_size
        info_panel_height = 120
        
        info_rect = pygame.Rect(0, info_panel_y, self.window_size, info_panel_height)
        pygame.draw.rect(canvas, INFO_PANEL_COLOR, info_rect)
        pygame.draw.rect(canvas, GRID_COLOR, info_rect, 2)
        
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font_large = pygame.font.Font(None, 24)
            font_medium = pygame.font.Font(None, 18)
            font_small = pygame.font.Font(None, 16)
        except:
            try:
                font_large = pygame.font.SysFont('Arial', 24)
                font_medium = pygame.font.SysFont('Arial', 18)
                font_small = pygame.font.SysFont('Arial', 16)
            except:
                font_large = None
                font_medium = None
                font_small = None
        
        if font_large is not None and font_medium is not None and font_small is not None:
            step_text = f"Parallel Step: {self.step_counter}"
            step_surface = font_large.render(step_text, True, TEXT_COLOR)
            canvas.blit(step_surface, (10, info_panel_y + 10))
            
            y_offset = info_panel_y + 40
            for i, agent in enumerate(self.agent_list):
                agent_info = f"{agent.name}: ({agent.location[0]}, {agent.location[1]})"
                agent_text = font_medium.render(agent_info, True, agent.color)
                canvas.blit(agent_text, (10, y_offset + i * 20))
            
            # Goal information
            goal_info = f"Goal: ({self.goal_location[0]}, {self.goal_location[1]})"
            goal_text = font_medium.render(goal_info, True, self.goal_color)
            canvas.blit(goal_text, (10, y_offset + len(self.agent_list) * 20))
            
            # Target information (show all targets)
            target_y_offset = y_offset + len(self.agent_list) * 20 + 20
            for idx, target in enumerate(self.targets):
                target_info = f"{target.name}: ({target.location[0]}, {target.location[1]})"
                target_text = font_medium.render(target_info, True, target.color)
                canvas.blit(target_text, (10, target_y_offset + idx * 20))
            
            lambda_text = font_small.render(f"λ FOV: {self.lambda_fov:.2f}", True, TEXT_COLOR)
            canvas.blit(lambda_text, (self.window_size - 200, info_panel_y + 10))
            
            legend_text = font_small.render("FOV: 0=Empty, 1=Physical, 2=Agent, 5=Goal, 4=Visual", True, TEXT_COLOR)
            canvas.blit(legend_text, (self.window_size - 200, info_panel_y + 30))

        # Finalize frame
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


if __name__ == "__main__":
    env = GridWorldEnvMultiAgent(render_mode="human", size=7, intrinsic=True)
    env.reset()
    
    print("Action spaces:", env.action_spaces)
    print("Observation spaces:", env.observation_spaces)
    print("Possible agents:", env.possible_agents)
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_spaces[agent].sample()
        env.step(action)
    
    env.close()
    print("Episode completed!")


