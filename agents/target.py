import numpy as np
from typing import Tuple, Optional
import random


class Target:
    """
    A triangular target that moves around the map with smooth random motion.
    """
    
    def __init__(
        self, 
        name: str = "target",
        color: Tuple[int, int, int] = (255, 0, 0),  # Red by default
        size: int = 5,
        initial_position: Optional[Tuple[int, int]] = None,
        movement_speed: float = 0.3,  # Probability of moving each step
        movement_range: int = 1,  # Maximum cells to move in one step
        smooth_movement: bool = True,
        box_cells: int = 3,  # Size of the box around target (must be odd)
        outline_width: int = 2,  # Width of the box outline
        box_scale: float = 1.0  # Scale factor for the box
    ):
        self.name = name
        self.color = color
        self.size = size  # Grid size
        self.movement_speed = movement_speed
        self.movement_range = movement_range
        self.smooth_movement = smooth_movement
        
        # Box parameters
        self.box_cells = max(1, box_cells if box_cells % 2 == 1 else box_cells + 1)  # Ensure odd
        self.outline_width = max(1, outline_width)
        self.box_scale = max(0.1, box_scale)
        
        # Position tracking
        if initial_position is not None:
            self.location = np.array(initial_position, dtype=int)
        else:
            # Random initial position
            self.location = np.array([
                random.randint(0, size - 1),
                random.randint(0, size - 1)
            ], dtype=int)
        
        # Movement state for smooth motion
        self._target_direction = np.array([0, 0], dtype=float)
        self._movement_momentum = 0.0
        self._last_move_step = 0
        
    def step(self, step_count: int, obstacles: list = None, is_cell_obstacle_func=None) -> None:
        """
        Update target position with smooth random movement.
        
        Args:
            step_count: Current step number for movement timing
            obstacles: List of obstacles in the environment
            is_cell_obstacle_func: Function to check if a cell is an obstacle
        """
        # Decide whether to change direction (smooth movement)
        if self.smooth_movement:
            # Change direction occasionally for smooth motion
            if random.random() < self.movement_speed * 0.3:  # 30% of movement probability
                # Choose new random direction (cardinal only)
                directions = [
                    np.array([1, 0]),   # Right
                    np.array([-1, 0]),  # Left
                    np.array([0, 1]),   # Up
                    np.array([0, -1]),  # Down
                ]
                self._target_direction = random.choice(directions).astype(float)
        
        # Move with some probability
        if random.random() < self.movement_speed:
            if self.smooth_movement:
                # Use target direction (already cardinal)
                move_direction = self._target_direction.copy()
            else:
                # Pure random movement (cardinal only)
                cardinal_directions = [
                    np.array([1, 0]),   # Right
                    np.array([-1, 0]),  # Left
                    np.array([0, 1]),   # Up
                    np.array([0, -1]),  # Down
                ]
                move_direction = random.choice(cardinal_directions).astype(float)
            
            # Calculate new position (cardinal movement only)
            new_position = self.location + move_direction.astype(int)
            
            # Clip to grid bounds
            new_position = np.clip(new_position, 0, self.size - 1)
            
            # Check for obstacles before moving
            can_move = True
            if is_cell_obstacle_func is not None:
                can_move = not is_cell_obstacle_func(new_position)
            
            # Only update position if no obstacle
            if can_move:
                self.location = new_position
                self._last_move_step = step_count
            else:
                # If blocked by obstacle, try to find alternative direction
                # Try cardinal directions first
                cardinal_directions = [
                    np.array([1, 0]),   # Right
                    np.array([-1, 0]),  # Left
                    np.array([0, 1]),   # Up
                    np.array([0, -1]),  # Down
                ]
                
                # Shuffle directions for randomness
                random.shuffle(cardinal_directions)
                
                for direction in cardinal_directions:
                    alt_position = self.location + direction
                    alt_position = np.clip(alt_position, 0, self.size - 1)
                    
                    if is_cell_obstacle_func is None or not is_cell_obstacle_func(alt_position):
                        self.location = alt_position
                        self._last_move_step = step_count
                        break
    
    def get_triangle_points(self, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the three points of the triangle for rendering.
        
        Args:
            cell_size: Size of each grid cell in pixels
            offset: Offset for rendering position
            
        Returns:
            Tuple of three (x, y) coordinate tuples for the triangle vertices
        """
        # Convert grid position to pixel position
        center_x = self.location[0] * cell_size + cell_size // 2 + offset[0]
        center_y = self.location[1] * cell_size + cell_size // 2 + offset[1]
        
        # Triangle size (smaller than cell)
        triangle_size = cell_size * 0.6
        
        # Calculate triangle vertices (pointing up)
        point1 = (center_x, center_y - triangle_size // 2)  # Top point
        point2 = (center_x - triangle_size // 2, center_y + triangle_size // 2)  # Bottom left
        point3 = (center_x + triangle_size // 2, center_y + triangle_size // 2)  # Bottom right
        
        return point1, point2, point3
    
    def get_box_coordinates(self, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the four corner coordinates of the box around the target.
        
        Args:
            cell_size: Size of each grid cell in pixels
            offset: Offset for rendering position
            
        Returns:
            Tuple of four (x, y) coordinate tuples for the box corners (top-left, top-right, bottom-right, bottom-left)
        """
        # Convert grid position to pixel position
        center_x = self.location[0] * cell_size + cell_size // 2 + offset[0]
        center_y = self.location[1] * cell_size + cell_size // 2 + offset[1]
        
        # Calculate box size based on box_cells and scale
        box_pixel_size = int(cell_size * self.box_cells * self.box_scale)
        half_size = box_pixel_size // 2
        
        # Calculate box corners (centered on target)
        top_left = (center_x - half_size, center_y - half_size)
        top_right = (center_x + half_size, center_y - half_size)
        bottom_right = (center_x + half_size, center_y + half_size)
        bottom_left = (center_x - half_size, center_y + half_size)
        
        return top_left, top_right, bottom_right, bottom_left
    
    def reset(self, new_position: Optional[Tuple[int, int]] = None) -> None:
        """
        Reset target to a new position.
        
        Args:
            new_position: New position to place target, or None for random
        """
        if new_position is not None:
            self.location = np.array(new_position, dtype=int)
        else:
            self.location = np.array([
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ], dtype=int)
        
        # Reset movement state
        self._target_direction = np.array([0, 0], dtype=float)
        self._movement_momentum = 0.0
        self._last_move_step = 0
    
    def __repr__(self) -> str:
        return f"Target(name='{self.name}', location={self.location.tolist()}, color={self.color})"
