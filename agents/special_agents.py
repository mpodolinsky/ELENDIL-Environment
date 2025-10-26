import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple
from .agents import BaseAgent

class GlobalViewAgent(BaseAgent):
    """
    Agent with global view of the environment instead of FOV.
    
    This agent can see the entire grid state, making it useful for
    testing and comparison with FOV-based agents.
    """
    
    def _setup_agent(
        self,
        outline_width: int = 1,
        box_scale: float = 1.0,
        **kwargs
    ) -> None:
        """Setup global view agent specific attributes."""
        self.outline_width = outline_width
        self.box_scale = max(0.05, min(1.0, box_scale))
        
        # Define observation space with global grid view
        obs_dict = {
            "agent": spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int),
            "global_grid": spaces.Box(0, 3, shape=(self.env_size, self.env_size), dtype=int),
            "target": spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int),
        }
        
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Discrete(5)
    
    def get_observation_space(self) -> spaces.Space:
        """Return the agent's observation space."""
        return self.observation_space
    
    def get_action_space(self) -> spaces.Space:
        """Return the agent's action space."""
        return self.action_space
    
    def observe(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate global view observation from environment state.
        
        Args:
            env_state: Dictionary containing environment state
            
        Returns:
            Dictionary containing agent's global observation
        """
        # Create global grid map
        global_grid = np.zeros((self.env_size, self.env_size), dtype=int)
        
        # Mark obstacles
        for (ox, oy, ow, oh) in env_state["obstacles"]:
            global_grid[ox:ox+ow, oy:oy+oh] = 1
        
        # Mark agents
        for agent in env_state["agents"]:
            if agent.name != self.name:
                global_grid[agent.location[0], agent.location[1]] = 2
        
        # Mark target
        if env_state.get("target") is not None:
            target_loc = env_state["target"].location
            global_grid[target_loc[0], target_loc[1]] = 3
        
        obs = {
            "agent": self.location,
            "global_grid": global_grid,
            "target": env_state["target"].location if env_state.get("target") is not None else np.array([-1, -1])
        }
        
        return obs
    
    def step(self, action: int, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent step based on action.
        
        Args:
            action: Action to execute (0-4)
            env_state: Current environment state
            
        Returns:
            Dictionary containing step results
        """
        # Action to direction mapping
        action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Down
            4: np.array([0, 0]),   # No operation
        }
        
        direction = action_to_direction.get(action, np.array([0, 0]))
        proposed_location = np.clip(self.location + direction, 0, self.env_size - 1)
        
        # Check for obstacles
        if self._is_cell_obstacle(proposed_location, env_state["obstacles"]):
            proposed_location = self.location
        
        # Check for collisions with other agents
        if self._is_cell_occupied(proposed_location, env_state["agents"]):
            proposed_location = self.location
        
        return {
            "new_location": proposed_location,
            "action_taken": action,
            "movement_blocked": np.array_equal(proposed_location, self.location) and action != 4
        }
    
    def _is_cell_obstacle(self, location: np.ndarray, obstacles: list) -> bool:
        """Check if a cell contains an obstacle."""
        for (ox, oy, ow, oh) in obstacles:
            if (ox <= location[0] < ox + ow and oy <= location[1] < oy + oh):
                return True
        return False
    
    def _is_cell_occupied(self, location: np.ndarray, agents: list) -> bool:
        """Check if a cell is occupied by another agent."""
        for agent in agents:
            if agent.name != self.name and np.array_equal(agent.location, location):
                return True
        return False

class TelepathicAgent(BaseAgent):
    """
    Agent that can see other agents' locations and intentions.
    
    This agent has access to information about other agents' locations
    and can optionally see their last actions or planned movements.
    """
    
    def _setup_agent(
        self,
        fov_size: int = 3,
        outline_width: int = 1,
        box_scale: float = 1.0,
        see_other_actions: bool = False,
        **kwargs
    ) -> None:
        """Setup telepathic agent specific attributes."""
        # Validate and set FOV size
        if fov_size % 2 == 0:
            fov_size += 1
        
        if fov_size < 3 or fov_size > self.env_size or fov_size <= 1:
            fov_size = 3
        
        self.fov_size = fov_size
        self.outline_width = outline_width
        self.box_scale = max(0.05, min(1.0, box_scale))
        self.see_other_actions = see_other_actions
        
        # Define observation space with telepathic information
        # Use a reasonable maximum number of other agents (can be up to env_size^2 - 1)
        max_other_agents = min(10, self.env_size * self.env_size - 1)  # Reasonable upper bound
        
        obs_dict = {
            "agent": spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int),
            "obstacles_fov": spaces.Box(0, 1, shape=(self.fov_size, self.fov_size), dtype=int),
            "other_agents_locations": spaces.Box(-1, self.env_size - 1, shape=(max_other_agents, 2), dtype=int),
        }
        
        if see_other_actions:
            obs_dict["other_agents_actions"] = spaces.Box(0, 4, shape=(max_other_agents,), dtype=int)
        
        obs_dict["target"] = spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int)
        
        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Discrete(5)
    
    def get_observation_space(self) -> spaces.Space:
        """Return the agent's observation space."""
        return self.observation_space
    
    def get_action_space(self) -> spaces.Space:
        """Return the agent's action space."""
        return self.action_space
    
    def observe(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate telepathic observation from environment state.
        
        Args:
            env_state: Dictionary containing environment state
            
        Returns:
            Dictionary containing agent's telepathic observation
        """
        # Get standard FOV obstacles
        fov_obstacles = self._get_fov_obstacles(env_state)
        
        # Get other agents' locations
        other_agents_locations = []
        for agent in env_state["agents"]:
            if agent.name != self.name:
                other_agents_locations.append(agent.location)
        
        # Pad with [-1, -1] to fill the fixed-size observation space
        max_other_agents = self.observation_space.spaces["other_agents_locations"].shape[0]
        while len(other_agents_locations) < max_other_agents:
            other_agents_locations.append(np.array([-1, -1]))
        
        obs = {
            "agent": self.location,
            "obstacles_fov": fov_obstacles,
            "other_agents_locations": np.array(other_agents_locations),
            "target": env_state["target"].location if env_state.get("target") is not None else np.array([-1, -1])
        }
        
        # Add other agents' actions if enabled (would need to be passed in env_state)
        if self.see_other_actions and "other_actions" in env_state:
            obs["other_agents_actions"] = np.array(env_state["other_actions"])
        
        return obs
    
    def step(self, action: int, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent step based on action.
        
        Args:
            action: Action to execute (0-4)
            env_state: Current environment state
            
        Returns:
            Dictionary containing step results
        """
        # Same movement logic as FOVAgent
        action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Down
            4: np.array([0, 0]),   # No operation
        }
        
        direction = action_to_direction.get(action, np.array([0, 0]))
        proposed_location = np.clip(self.location + direction, 0, self.env_size - 1)
        
        # Check for obstacles
        if self._is_cell_obstacle(proposed_location, env_state["obstacles"]):
            proposed_location = self.location
        
        # Check for collisions with other agents
        if self._is_cell_occupied(proposed_location, env_state["agents"]):
            proposed_location = self.location
        
        return {
            "new_location": proposed_location,
            "action_taken": action,
            "movement_blocked": np.array_equal(proposed_location, self.location) and action != 4
        }
    
    def _get_fov_obstacles(self, env_state: Dict[str, Any]) -> np.ndarray:
        """Generate FOV obstacles map for this agent."""
        fov_size = self.fov_size
        agent_x, agent_y = self.location
        offset = fov_size // 2
        
        # Initialize FOV map
        fov_map = np.zeros((fov_size, fov_size), dtype=int)
        
        # Generate world coordinates for FOV
        for i in range(fov_size):
            for j in range(fov_size):
                world_x = agent_x - offset + i
                world_y = agent_y - offset + j
                
                # Check boundaries
                if (world_x < 0 or world_x >= self.env_size or 
                    world_y < 0 or world_y >= self.env_size):
                    fov_map[i, j] = 1  # Boundary obstacle
                    continue
                
                # Check for target
                if (env_state.get("target") is not None and
                    env_state["target"].location[0] == world_x and
                    env_state["target"].location[1] == world_y):
                    fov_map[i, j] = 3  # Target
                    continue
                
                # Check for other agents
                for other_agent in env_state["agents"]:
                    if (other_agent.name != self.name and
                        other_agent.location[0] == world_x and
                        other_agent.location[1] == world_y):
                        fov_map[i, j] = 2  # Other agent
                        break
                
                # Check for obstacles (only if no target or other agent)
                if fov_map[i, j] == 0 and self._is_cell_obstacle(np.array([world_x, world_y]), env_state["obstacles"]):
                    fov_map[i, j] = 1  # Obstacle
        
        return fov_map
    
    def _is_cell_obstacle(self, location: np.ndarray, obstacles: list) -> bool:
        """Check if a cell contains an obstacle."""
        for (ox, oy, ow, oh) in obstacles:
            if (ox <= location[0] < ox + ow and oy <= location[1] < oy + oh):
                return True
        return False
    
    def _is_cell_occupied(self, location: np.ndarray, agents: list) -> bool:
        """Check if a cell is occupied by another agent."""
        for agent in agents:
            if agent.name != self.name and np.array_equal(agent.location, location):
                return True
        return False
