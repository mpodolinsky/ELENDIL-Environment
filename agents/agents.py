import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent environment.
    
    This class defines the interface that all agent types must implement,
    ensuring modularity and extensibility for different agent behaviors.
    """
    
    def __init__(self, name: str, color: Tuple[int, int, int], env_size: int, **kwargs):
        self.name = name
        self.color = color
        self.env_size = env_size
        self.location = np.array([-1, -1], dtype=int)
        self.last_visited_cells = []
        
        # Initialize agent-specific attributes
        self._setup_agent(**kwargs)
        
    @abstractmethod
    def _setup_agent(self, **kwargs) -> None:
        """Setup agent-specific attributes and spaces."""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """Return the agent's observation space."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """Return the agent's action space."""
        pass
    
    @abstractmethod
    def observe(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate observation from environment state.
        
        Args:
            env_state: Dictionary containing environment state information
            
        Returns:
            Dictionary containing agent's observation
        """
        pass
    
    @abstractmethod
    def step(self, action: int, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent step and return step results.
        
        Args:
            action: Action to execute
            env_state: Current environment state
            
        Returns:
            Dictionary containing step results (new_location, etc.)
        """
        pass
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.location = np.array([-1, -1], dtype=int)
        self.last_visited_cells = []
    
    def update_location(self, new_location: np.ndarray) -> None:
        """Update agent location and track visited cells."""
        self.location = new_location.copy()
        cell_tuple = tuple(new_location)
        self.last_visited_cells.append(cell_tuple)
        # Keep only last 3 cells
        if len(self.last_visited_cells) > 3:
            self.last_visited_cells.pop(0)

class FOVAgent(BaseAgent):
    """
    Standard FOV-based agent with configurable field of view.
    
    This agent observes its environment through a local field of view
    and can optionally see target coordinates.
    """
    
    def _setup_agent(
        self,
        fov_size: int = 3,
        outline_width: int = 1,
        box_scale: float = 1.0,
        show_target_coords: bool = False,
        **kwargs
    ) -> None:
        """Setup FOV agent specific attributes."""
        # Validate and set FOV size
        if fov_size % 2 == 0:
            fov_size += 1
            print(f"FOV size is even, incrementing by 1 to {fov_size}")
        
        if fov_size < 3 or fov_size > self.env_size or fov_size <= 1:
            fov_size = 3
            print(f"FOV size invalid, setting to 3")
        
        self.fov_size = fov_size
        self.outline_width = outline_width
        self.box_scale = max(0.05, min(1.0, box_scale))
        self.show_target_coords = show_target_coords
        
        # Define observation space
        obs_dict = {
            "agent": spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int),
            "obstacles_fov": spaces.Box(0, 1, shape=(self.fov_size, self.fov_size), dtype=int),
        }
        
        if show_target_coords:
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
        Generate FOV-based observation from environment state.
        
        Args:
            env_state: Dictionary containing:
                - agents: List of all agents
                - target: Target object (if exists)
                - obstacles: List of obstacles
                - grid_size: Size of the grid
                
        Returns:
            Dictionary containing agent's observation
        """
        # Get FOV obstacles map
        fov_obstacles = self._get_fov_obstacles(env_state)
        
        obs = {
            "agent": self.location,
            "obstacles_fov": fov_obstacles
        }
        
        # Add target coordinates if enabled
        if self.show_target_coords and env_state.get("target") is not None:
            obs["target"] = env_state["target"].location
        elif self.show_target_coords:
            obs["target"] = np.array([-1, -1])
        
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
        
        obstacle_collision = False
        
        # Check for obstacles
        if self._is_cell_obstacle(proposed_location, env_state["obstacles"]):
            obstacle_collision = True
            proposed_location = self.location
        
        # Check for collisions with other agents
        if self._is_cell_occupied(proposed_location, env_state["agents"]):
            proposed_location = self.location
        
        return {
            "new_location": proposed_location,
            "action_taken": action,
            "movement_blocked": np.array_equal(proposed_location, self.location) and action != 4,
            "obstacle_collision": obstacle_collision
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

# Legacy Agent class for backward compatibility
class Agent(FOVAgent):
    """
    Legacy Agent class for backward compatibility.
    
    This class maintains the same interface as the original Agent class
    while using the new modular architecture internally.
    """
    
    def __init__(
        self,
        env_size: int,
        name: str,
        color: tuple,
        observation_space: Optional[spaces.Space] = None,  # Unused, kept for compatibility
        outline_width: int = 1,
        box_scale: float = 1.0,
        fov_size: int = 3,
        show_target_coords: bool = False,
    ) -> None:
        """Initialize agent with legacy interface."""
        super().__init__(
            name=name,
            color=color,
            env_size=env_size,
            fov_size=fov_size,
            outline_width=outline_width,
            box_scale=box_scale,
            show_target_coords=show_target_coords
        )
        
