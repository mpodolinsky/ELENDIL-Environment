import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple
from .agents import BaseAgent

class ObserverAgent(BaseAgent):
    """
    Observer agent with FOV and altitude capabilities.
    
    This agent has:
    - Configurable FOV size (user input + 2)
    - 5 discrete movement actions (same as other agents)
    - 3 discrete altitude actions (increase, remain, reduce)
    - Combined action space of 8 discrete actions total
    """
    
    def _setup_agent(
        self,
        fov_base_size: int = 3,  # Base FOV size, actual will be fov_base_size + 4
        outline_width: int = 1,
        box_scale: float = 1.0,
        show_target_coords: bool = False,
        max_altitude: int = 3,  # Maximum altitude level (1, 2, 3)
        target_detection_probs: Tuple[float, float, float] = (1.0, 0.66, 0.33),  # Detection probs for FL1, FL2, FL3
        **kwargs
    ) -> None:
        """
        Setup observer agent specific attributes.
        
        Args:
            target_detection_probs: Tuple of 3 floats for target detection probabilities
                                   at flight levels 1, 2, 3 respectively.
                                   Default: (1.0, 0.66, 0.33)
        """
        # Calculate actual FOV size (base + 4)
        self.fov_base_size = fov_base_size
        self.fov_size = fov_base_size + 4
        
        # Store target detection probabilities for each flight level
        if isinstance(target_detection_probs, (list, tuple)) and len(target_detection_probs) == 3:
            self.target_detection_probs = tuple(target_detection_probs)
        else:
            raise ValueError("target_detection_probs must be a list or tuple of 3 floats")
        
        # Validate FOV size
        if self.fov_size % 2 == 0:
            self.fov_size += 1
            print(f"FOV size is even, incrementing by 1 to {self.fov_size}")
        
        if self.fov_size < 3 or self.fov_size > self.env_size or self.fov_size <= 1:
            self.fov_size = 5  # Default to 5 if invalid
            print(f"FOV size invalid, setting to 5")
        
        self.outline_width = outline_width
        self.box_scale = max(0.05, min(1.0, box_scale))
        self.show_target_coords = show_target_coords
        self.max_altitude = max_altitude
        
        # Initialize altitude (starts at flight level 1, flight levels are 1, 2, 3)
        self.altitude = 1
        
        # Define observation space with flight level information
        obs_dict = {
            "agent": spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int),
            "obstacles_fov": spaces.Box(-10, 4, shape=(self.fov_size, self.fov_size), dtype=int),  # -10 for masked, 0-4 for visible
            "flight_level": spaces.Discrete(4),  # Flight levels: 1, 2, 3 (never 0)
        }
        
        if show_target_coords:
            obs_dict["target"] = spaces.Box(0, self.env_size - 1, shape=(2,), dtype=int)
        
        self.observation_space = spaces.Dict(obs_dict)
        
        # Combined action space: 5 movement + 3 altitude = 8 total actions
        # Actions 0-4: Movement (same as other agents)
        # Actions 5-7: Altitude control
        self.action_space = spaces.Discrete(8)
    
    def get_observation_space(self) -> spaces.Space:
        """Return the agent's observation space."""
        return self.observation_space
    
    def get_action_space(self) -> spaces.Space:
        """Return the agent's action space."""
        return self.action_space
    
    def _apply_flight_level_masking(self, fov_map: np.ndarray) -> np.ndarray:
        """
        Apply flight level-based masking to FOV.
        
        Args:
            fov_map: The FOV map to mask
            
        Returns:
            Masked FOV map with restricted areas set to -10
        """
        # Create a copy to avoid modifying the original
        masked_fov = fov_map.copy()
        
        # Calculate visible area based on flight level
        # Flight Level 1: loses 2 rings (visible = fov_size - 4)
        # Flight Level 2: loses 1 ring (visible = fov_size - 2) 
        # Flight Level 3: loses 0 rings (visible = fov_size)
        
        if self.altitude >= 1:
            # Calculate how many rings to mask from each edge
            rings_to_mask = 3 - self.altitude
            
            if rings_to_mask > 0:
                # Mask outer rings by setting them to -10
                for ring in range(rings_to_mask):
                    # Top and bottom rows
                    masked_fov[ring, :] = -10
                    masked_fov[-(ring + 1), :] = -10
                    # Left and right columns
                    masked_fov[:, ring] = -10
                    masked_fov[:, -(ring + 1)] = -10
        
        return masked_fov

    def observe(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate FOV-based observation with altitude information and flight level masking.
        
        Args:
            env_state: Dictionary containing environment state
            
        Returns:
            Dictionary containing agent's observation with altitude and masked FOV
        """
        # Get FOV obstacles map
        fov_obstacles = self._get_fov_obstacles(env_state)
        
        # Apply flight level-based masking
        masked_fov = self._apply_flight_level_masking(fov_obstacles)
        
        obs = {
            "agent": self.location,
            "obstacles_fov": masked_fov,
            "flight_level": self.altitude  # Current flight level (1, 2, or 3)
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
            action: Action to execute (0-7)
                0-4: Movement actions (same as other agents)
                5: Increase altitude
                6: Remain at current altitude
                7: Reduce altitude
            
        Returns:
            Dictionary containing step results
        """
        movement_action = None
        altitude_action = None
        
        # Parse action
        if action < 5:
            # Movement action
            movement_action = action
            altitude_action = 6  # Remain at altitude (default)
        else:
            # Altitude action
            altitude_action = action - 5  # Convert to 0, 1, 2
            movement_action = 4  # No movement (default)
        
        # Handle altitude change
        altitude_changed = self._handle_altitude_change(altitude_action)
        
        # Handle movement (only if on ground level or if movement action is specified)
        new_location = self.location
        movement_blocked = False
        
        if movement_action < 4:  # If movement action (not "no operation")
            new_location, movement_blocked = self._handle_movement(movement_action, env_state)
        
        return {
            "new_location": new_location,
            "action_taken": action,
            "movement_action": movement_action,
            "altitude_action": altitude_action,
            "altitude_changed": altitude_changed,
            "movement_blocked": movement_blocked,
            "new_altitude": self.altitude,
            "obstacle_collision": False  # ObserverAgent flies over obstacles
        }
    
    def _handle_altitude_change(self, altitude_action: int) -> bool:
        """
        Handle altitude changes based on action.
        
        Args:
            altitude_action: 0=increase, 1=remain, 2=reduce
            
        Returns:
            bool: True if altitude changed
        """
        old_altitude = self.altitude
        
        if altitude_action == 0:  # Increase altitude
            self.altitude = min(self.altitude + 1, self.max_altitude)
        elif altitude_action == 2:  # Reduce altitude
            self.altitude = max(self.altitude - 1, 1)
        # altitude_action == 1 means remain at current altitude (no change)
        
        return self.altitude != old_altitude
    
    def _handle_movement(self, movement_action: int, env_state: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
        """
        Handle movement based on action.
        
        Args:
            movement_action: Movement action (0-4)
            env_state: Environment state
            
        Returns:
            Tuple of (new_location, movement_blocked)
        """
        # Action to direction mapping
        action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Down
            4: np.array([0, 0]),   # No operation
        }
        
        direction = action_to_direction.get(movement_action, np.array([0, 0]))
        proposed_location = np.clip(self.location + direction, 0, self.env_size - 1)
        
        # ObserverAgent flies at altitude >= 1, so obstacles don't block movement
        # No obstacle collision for ObserverAgent (it flies over them)
        movement_blocked = np.array_equal(proposed_location, self.location) and movement_action != 4
        
        return proposed_location, movement_blocked
    
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
                
                # Check for target with flight level-based detection probability
                if (env_state.get("target") is not None and
                    env_state["target"].location[0] == world_x and
                    env_state["target"].location[1] == world_y):
                    
                    # Flight level-based target detection probability
                    # Use altitude - 1 to index into target_detection_probs (0-indexed)
                    if self.altitude >= 1 and self.altitude <= 3:
                        detection_prob = self.target_detection_probs[self.altitude - 1]
                        if np.random.random() < detection_prob:
                            fov_map[i, j] = 3  # Target detected
                        else:
                            fov_map[i, j] = 0  # Target missed (shows as empty)
                    continue
                
                # Check for visual obstacles (they hide what's beneath them)
                visual_obstacles = env_state.get("visual_obstacles", [])
                if self._is_cell_in_visual_obstacle(world_x, world_y, visual_obstacles):
                    fov_map[i, j] = 4  # Visual obstacle
                    continue
                
                # Check for other agents (only if at same or lower altitude)
                for other_agent in env_state["agents"]:
                    if (other_agent.name != self.name and
                        other_agent.location[0] == world_x and
                        other_agent.location[1] == world_y):
                        # Only see other agents if they are at same or lower altitude
                        other_altitude = getattr(other_agent, 'altitude', 0)
                        if other_altitude <= self.altitude:
                            fov_map[i, j] = 2  # Other agent
                            break
                
                # ObserverAgent flies at altitude >= 1, so it doesn't see ground obstacles
                # (physical obstacles are not rendered in FOV for aerial agents)
        
        return fov_map
    
    def _is_cell_obstacle(self, location: np.ndarray, obstacles: list) -> bool:
        """Check if a cell contains a physical obstacle."""
        for (ox, oy, ow, oh) in obstacles:
            if (ox <= location[0] < ox + ow and oy <= location[1] < oy + oh):
                return True
        return False
    
    def _is_cell_in_visual_obstacle(self, x: int, y: int, visual_obstacles: list) -> bool:
        """Check if a cell is covered by a visual obstacle."""
        for (ox, oy, ow, oh) in visual_obstacles:
            if ox <= x < ox + ow and oy <= y < oy + oh:
                return True
        return False
    
    def _is_cell_occupied(self, location: np.ndarray, agents: list) -> bool:
        """Check if a cell is occupied by another agent."""
        for agent in agents:
            if agent.name != self.name and np.array_equal(agent.location, location):
                return True
        return False
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        super().reset()
        self.altitude = 0
    
    def update_location(self, new_location: np.ndarray) -> None:
        """Update agent location and track visited cells."""
        super().update_location(new_location)
        # Note: Altitude is handled separately and doesn't affect location tracking
