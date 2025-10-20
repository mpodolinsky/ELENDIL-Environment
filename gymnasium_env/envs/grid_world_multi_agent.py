from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from typing import Optional, List, Dict, Union

from agents.agents import Agent
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

class GridWorldEnvMultiAgent(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        render_mode=None,
        size=5,
        intrinsic=False,
        no_target=False,
        agent_outline_width=1,
        agent_box_scale=1.0,
        agent_fov_size=1,
        agents: Optional[List[Union[Dict, 'Agent']]] = None,
        enable_obstacles: bool = False,
        num_obstacles: int = 0,
        num_default_agents: int = 2,
        show_fov_display: bool = True,
        target_config: Optional[Dict] = None,
        lambda_fov: float = 0.5,
        max_steps: int = 500,
        show_target_coords: bool = False,
    ):
        self.size = size  # The size of the square grid (default to 5x5)
        self.window_size = 512  # The size of the PyGame window
        self.no_target = no_target  # If True, no target is spawned
        self.enable_obstacles = bool(enable_obstacles)
        self.num_obstacles = int(max(0, num_obstacles))
        self.show_fov_display = bool(show_fov_display)
        self.lambda_fov = max(-0.5, min(0.5, float(lambda_fov)))  # Clamp between 0 and 1
        self.show_target_coords = bool(show_target_coords)  # If True, agents can see target coordinates
        
        # FOV display state
        self._step_count = 0
        self.max_steps = max_steps
        # Detection state tracking for visual indicators
        self._agent_detects_target = {}  # Track which agents detect the target
        self._target_detects_agent = {}  # Track which agents the target detects
        # Back-compat defaults for per-agent visual params
        default_outline_width = int(agent_outline_width)
        default_box_scale = float(agent_box_scale)
        default_fov_size = int(agent_fov_size if agent_fov_size % 2 == 1 else agent_fov_size + 1)

        # Build agents list (support custom configs)
        default_colors = [(80, 160, 255), (255, 180, 60)]
        self.agent_list: List[Agent] = []
        if agents is None:
            # Auto-generate N default agents
            count = max(1, int(num_default_agents))
            self.agent_list = []
            for i in range(count):
                name = f"agent_{i+1}"
                color = default_colors[i % len(default_colors)]
                agent = Agent(self.size, name, color, None, default_outline_width, default_box_scale, default_fov_size, self.show_target_coords)
                self.agent_list.append(agent)
        else:
            self.agent_list = []
            for idx, cfg in enumerate(agents):
                if isinstance(cfg, Agent):
                    # Use provided Agent instance as-is
                    self.agent_list.append(cfg)
                    continue
                # Otherwise treat as config dict
                name = cfg.get("name", f"agent_{idx+1}")
                color = cfg.get("color", default_colors[idx % len(default_colors)])
                act_space = cfg.get("action_space", spaces.Discrete(5))
                outline_width = int(cfg.get("outline_width", default_outline_width))
                box_scale = float(cfg.get("box_scale", default_box_scale))
                fov_size = int(cfg.get("fov_size", default_fov_size))
                
                agent = Agent(self.size, name, color, None, outline_width, box_scale, fov_size, self.show_target_coords)
                self.agent_list.append(agent)

        # Public names and lookup
        self.agents: List[str] = [a.name for a in self.agent_list]
        self._agents_by_name: Dict[str, Agent] = {a.name: a for a in self.agent_list}

        # Combined spaces from agents
        self.action_space = spaces.Dict({a.name: a.action_space for a in self.agent_list})
        self.observation_space = spaces.Dict({a.name: a.observation_space for a in self.agent_list})

        # Target location
        self._target_location = np.array([-1, -1], dtype=int)
        # Obstacles: list of (x, y, w, h) in cell units
        self._obstacles: List[tuple] = []

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

        # Initialize target
        if not self.no_target:
            if target_config is None:
                target_config = {}
            self.target = Target(
                name=target_config.get('name', 'target'),
                color=target_config.get('color', (255, 0, 0)),
                size=self.size,
                movement_speed=target_config.get('movement_speed', 0.3),
                movement_range=target_config.get('movement_range', 1),
                smooth_movement=target_config.get('smooth_movement', True),
                box_cells=target_config.get('box_cells', 3),
                outline_width=target_config.get('outline_width', 2),
                box_scale=target_config.get('box_scale', 1.0)
            )
        else:
            self.target = None

        self.window = None
        self.clock = None
        self.step_counter = 0

    def _get_obs(self):
        obs = {}
        for a in self.agent_list:
            fov_obstacles = self._get_agent_fov_obstacles(a)
            agent_obs = {
                "agent": a.location, 
                "obstacles_fov": fov_obstacles
            }
            
            # Only include target coordinates if show_target_coords is True
            if self.show_target_coords:
                agent_obs["target"] = self.target.location if self.target is not None else np.array([-1, -1])
            
            obs[a.name] = agent_obs
        return obs

    def _get_info(self):
        info = {}
        for a in self.agent_list:
            if self.target is not None:
                info[f"distance_{a.name}"] = np.linalg.norm(a.location - self.target.location, ord=1)
            else:
                info[f"distance_{a.name}"] = -1
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Optionally generate new obstacles each reset
        self._obstacles = []
        if self.enable_obstacles and self.num_obstacles > 0:
            self._generate_random_obstacles(self.num_obstacles)

        # Sample positions not inside obstacles
        forbidden = self._cells_covered_by_obstacles()
        choices = [i for i in range(self.size * self.size) if (i // self.size, i % self.size) not in forbidden]
        num_needed = len(self.agent_list) + (0 if self.no_target else 1)
        if len(choices) < num_needed:
            # fallback: ignore obstacles if too dense
            coords = self.np_random.choice(self.size * self.size, size=num_needed, replace=False)
        else:
            coords = self.np_random.choice(choices, size=num_needed, replace=False)

        for i, a in enumerate(self.agent_list):
            self._update_agent_location(a, np.array([coords[i] // self.size, coords[i] % self.size], dtype=int))
        
        if self.no_target:
            if self.target is not None:
                self.target = None
        else:
            if self.target is None:
                # Create target if it doesn't exist
                self.target = Target(
                    name='target',
                    color=(255, 0, 0),
                    size=self.size,
                    movement_speed=0.3,
                    movement_range=1,
                    smooth_movement=True,
                    box_cells=3,
                    outline_width=2,
                    box_scale=1.0
                )
            t_idx = len(self.agent_list)
            target_position = (coords[t_idx] // self.size, coords[t_idx] % self.size)
            self.target.reset(target_position)

        if self.intrinsic:
            for a in self.agent_list:
                self._visit_counts[a.name].fill(0)
                self._visit_counts[a.name][tuple(a.location)] += 1

        self.step_counter = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Supports two modes:
        - Batch step for both agents in one call: action = {"agent_1": a1, "agent_2": a2}
        - Single-agent step: action = {"agent": "agent_1"|"agent_2"|0|1, "action": a}
          or a tuple/list (agent_id, a) where agent_id is "agent_1"/"agent_2" or 0/1.
        """
        self.step_counter += 1
        self._step_count += 1
        
        # Clear screen and show step info if FOV display is enabled
        self._clear_screen_and_show_step()

        # Helper to normalize agent identifier to key string
        def _normalize_agent_key(agent_identifier: Union[str, int, np.integer]) -> str:
            if isinstance(agent_identifier, str):
                if agent_identifier in self._agents_by_name:
                    return agent_identifier
            elif isinstance(agent_identifier, (int, np.integer)):
                idx = int(agent_identifier)
                if 0 <= idx < len(self.agent_list):
                    return self.agent_list[idx].name
            raise ValueError("Invalid agent identifier. Use valid agent name or index.")

        # Case 1: batch two-agent action dict
        if isinstance(action, dict) and all(name in action for name in self.agents):
            # batch actions for all agents
            proposed: Dict[str, np.ndarray] = {}
            for name in self.agents:
                a_val = int(action[name])  # type: ignore[index]
                direction = self._action_to_direction[a_val]
                current = self._agents_by_name[name].location
                candidate = np.clip(current + direction, 0, self.size - 1)
                # block if obstacle cell
                if self._is_cell_obstacle(candidate):
                    candidate = current
                proposed[name] = candidate

            # resolve collisions simply: later agents can't move into occupied cells
            occupied = set()
            for name in self.agents:
                target_loc = proposed[name]
                key = (int(target_loc[0]), int(target_loc[1]))
                if key in occupied:
                    continue
                # also avoid moving into someone who isn't moving
                can_move = True
                for other_name in self.agents:
                    if other_name == name:
                        continue
                    other = self._agents_by_name[other_name]
                    other_prop = proposed.get(other_name, other.location)
                    if np.array_equal(target_loc, other.location) and np.array_equal(other_prop, other.location):
                        can_move = False
                        break
                if can_move:
                    self._update_agent_location(self._agents_by_name[name], target_loc)
                    occupied.add(key)
            
            # Show FOV for ALL agents after batch processing
            if self.show_fov_display:
                for name in self.agents:
                    fov_map = self._get_agent_fov_obstacles(self._agents_by_name[name])
                    self._print_fov_display(name, fov_map, f"(action {action[name]})")

        else:
            # Case 2: single-agent action
            if isinstance(action, dict) and "agent" in action and "action" in action:
                agent_key = _normalize_agent_key(action["agent"])  # type: ignore[index]
                a = int(action["action"])  # type: ignore[index]
            elif isinstance(action, (list, tuple)) and len(action) == 2:
                agent_key = _normalize_agent_key(action[0])
                a = int(action[1])
            else:
                raise ValueError(
                    "Unsupported action format. Use {agent_1:a1, agent_2:a2} or {agent: id, action: a} or (id, a)."
                )

            direction = self._action_to_direction[a]
            agent = self._agents_by_name[agent_key]
            proposed_loc = np.clip(agent.location + direction, 0, self.size - 1)
            # Block obstacles
            if self._is_cell_obstacle(proposed_loc):
                proposed_loc = agent.location
            blocked = any(np.array_equal(proposed_loc, other.location) for other in self.agent_list if other.name != agent_key)
            if not blocked:
                self._update_agent_location(agent, proposed_loc)
            
            # Show FOV for this agent if display is enabled (regardless of movement)
            if self.show_fov_display:
                fov_map = self._get_agent_fov_obstacles(agent)
                self._print_fov_display(agent_key, fov_map, f"(action {a})")

        # Move target if it exists
        if self.target is not None:
            self.target.step(self.step_counter, self._obstacles, self._is_cell_obstacle)
        
        # FOV-based reward system
        reward = 0.0
        
        # Update detection state for visual indicators
        self._agent_detects_target = {}
        self._target_detects_agent = {}
        
        # Check if any agent reached the target (termination condition)
        reached_any = False
        if not self.no_target and self.target is not None:
            reached_any = any(np.array_equal(a.location, self.target.location) for a in self.agent_list)
        
        if reached_any:
            reward = 0.0  # Success reward
        else:
            # Apply FOV-based rewards for each agent and track detection
            for agent in self.agent_list:
                # Check if agent detects target
                agent_detects = self._is_target_in_agent_fov(agent)
                self._agent_detects_target[agent.name] = agent_detects
                
                # Check if target detects agent
                target_detects = self._is_agent_in_target_fov(agent)
                self._target_detects_agent[agent.name] = target_detects
                
                # Penalty: -lambda if agent is in target's FOV
                # If lambda is 1, the agent only cares about avoiding the target
                if target_detects:
                    reward -= (self.lambda_fov)

                # Reward: (1-lambda) if target is in agent's FOV
                # If lambda is 0, the agent only cares about finding the target
                elif agent_detects:
                    reward += (1-self.lambda_fov)
                
            
            # Add small step penalty if no FOV interactions
            if reward == 0.0:
                reward = -0.01

        # Intrinsic exploration rewards (if enabled)
        if self.intrinsic:
            for a in self.agent_list:
                if self._visit_counts[a.name][tuple(a.location)] == 0:
                    reward += 0.025
                self._visit_counts[a.name][tuple(a.location)] += 1

        # terminated = reached_any
        terminated = False # no termination condition
        truncated = self.step_counter >= self.max_steps

        # Store last reward for display
        self._last_reward = reward

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

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

        # Obstacles (filled rectangles)
        if self._obstacles:
            OBSTACLE_COLOR = (90, 90, 90)
            for (ox, oy, ow, oh) in self._obstacles:
                rect = pygame.Rect(
                    (ox * pix_square, oy * pix_square),
                    (ow * pix_square, oh * pix_square),
                )
                pygame.draw.rect(canvas, OBSTACLE_COLOR, rect)

        # Target (only if spawned and exists)
        if self.target is not None:
            # Get triangle points from target
            point1, point2, point3 = self.target.get_triangle_points(int(pix_square))
            pygame.draw.polygon(canvas, self.target.color, [point1, point2, point3])
            
            # Draw target box outline
            top_left, top_right, bottom_right, bottom_left = self.target.get_box_coordinates(int(pix_square))
            box_points = [top_left, top_right, bottom_right, bottom_left, top_left]  # Close the polygon
            pygame.draw.lines(canvas, self.target.color, False, box_points, self.target.outline_width)
            
            # Draw detection dot if target detects any agent
            if any(self._target_detects_agent.values()):
                target_center = (self.target.location + 0.5) * pix_square
                dot_center = target_center + np.array([0, -pix_square * 0.6])  # Position higher above target
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

        # Agents (circles)
        for a in self.agent_list:
            center = (a.location + 0.5) * pix_square
            pygame.draw.circle(canvas, a.color, center.astype(int), int(pix_square / 3))
            
            # Draw detection dot if agent detects target
            if self._agent_detects_target.get(a.name, False):
                dot_center = center + np.array([0, -pix_square * 0.6])  # Position higher above agent
                pygame.draw.circle(canvas, (255, 255, 0), dot_center.astype(int), int(pix_square / 8))  # Yellow dot

        # Outline boxes using per-agent params
        def draw_agent_box(a: Agent):
            if a.fov_size > 1:
                # Draw a kxk box centered on the agent, without clamping to grid bounds
                k = a.fov_size
                offset = k // 2
                # Compute top-left in cell coordinates centered on agent
                start_cell = (a.location - np.array([offset, offset]))
                top_left = start_cell * pix_square
                side_px = pix_square * k
                rect = pygame.Rect(top_left, (side_px, side_px))
                pygame.draw.rect(canvas, a.color, rect, width=a.outline_width)
            else:
                box_side = pix_square * a.box_scale
                top_left = (a.location * pix_square) + ((pix_square - box_side) / 2)
                rect = pygame.Rect(top_left, (box_side, box_side))
                pygame.draw.rect(canvas, a.color, rect, width=a.outline_width)

        for a in self.agent_list:
            draw_agent_box(a)
            
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
            # Step counter
            step_text = font_large.render(f"Step: {self.step_counter}", True, TEXT_COLOR)
            canvas.blit(step_text, (10, info_panel_y + 10))
            
            # Agent information
            y_offset = info_panel_y + 40
            for i, agent in enumerate(self.agent_list):
                # Agent name and position
                agent_info = f"{agent.name}: ({agent.location[0]}, {agent.location[1]})"
                agent_text = font_medium.render(agent_info, True, agent.color)
                canvas.blit(agent_text, (10, y_offset + i * 20))
            
            # Target information
            if self.target is not None:
                target_info = f"Target: ({self.target.location[0]}, {self.target.location[1]})"
                target_text = font_medium.render(target_info, True, self.target.color)
                canvas.blit(target_text, (10, y_offset + len(self.agent_list) * 20))
            
            # Reward information (if available from last step)
            if hasattr(self, '_last_reward'):
                reward_text = font_medium.render(f"Last Reward: {self._last_reward:.3f}", True, TEXT_COLOR)
                canvas.blit(reward_text, (self.window_size - 200, info_panel_y + 10))
            
            # Lambda FOV parameter
            lambda_text = font_small.render(f"λ FOV: {self.lambda_fov:.2f}", True, TEXT_COLOR)
            canvas.blit(lambda_text, (self.window_size - 200, info_panel_y + 35))
            
            # FOV legend
            legend_text = font_small.render("FOV: 0=Empty, 1=Obstacle, 2=Agent, 3=Target", True, TEXT_COLOR)
            canvas.blit(legend_text, (self.window_size - 200, info_panel_y + 55))

        # Finalize frame
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    # --- Agent management ---
    def add_agent(self, name: Optional[str] = None, color: Optional[tuple] = None, action_space: Optional[spaces.Space] = None, observation_space: Optional[spaces.Space] = None, outline_width: Optional[int] = None, box_scale: Optional[float] = None, fov_size: Optional[int] = None) -> str:
        """Add a new agent at runtime and update spaces and lookups. Returns the agent name."""
        # Defaults based on env
        idx = len(self.agent_list)
        agent_name = name or f"agent_{idx+1}"
        if agent_name in self._agents_by_name:
            raise ValueError(f"Agent name already exists: {agent_name}")
        default_colors = [(80, 160, 255), (255, 180, 60)]
        agent_color = color or default_colors[idx % len(default_colors)]
        act_space = action_space or spaces.Discrete(5)
        ow = int(outline_width) if outline_width is not None else 1
        bs = float(box_scale) if box_scale is not None else 1.0
        bc = int(fov_size) if fov_size is not None else 1
        new_agent = Agent(self.size, agent_name, agent_color, None, ow, bs, bc, self.show_target_coords)

        # Register
        self.agent_list.append(new_agent)
        self.agents.append(agent_name)
        self._agents_by_name[agent_name] = new_agent

        # Update combined spaces
        self.action_space = spaces.Dict({a.name: a.action_space for a in self.agent_list})
        self.observation_space = spaces.Dict({a.name: a.observation_space for a in self.agent_list})

        # Init intrinsic map if needed
        if self.intrinsic:
            self._visit_counts[agent_name] = np.zeros((self.size, self.size), dtype=int)

        return agent_name

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
                
                # Check if target is at this position
                target_present = False
                if self.target is not None and (self.target.location[0] == x_int and self.target.location[1] == y_int):
                    fov_map[fov_i, fov_j] = 3  # Mark target as 3
                    target_present = True
                
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
    
    def _is_agent_in_target_fov(self, agent: 'Agent') -> bool:
        """
        Check if an agent is within the target's field of view.
        
        Args:
            agent: The agent to check
            
        Returns:
            True if agent is in target's FOV, False otherwise
        """
        if self.target is None:
            return False
        
        # Get target's FOV size (use box_cells as FOV size)
        target_fov_size = self.target.box_cells
        if target_fov_size is None:
            target_fov_size = 3  # Default FOV size
        
        # Target's current position
        target_x, target_y = self.target.location
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
    
    def _is_target_in_agent_fov(self, agent: 'Agent') -> bool:
        """
        Check if the target is within an agent's field of view.
        
        Args:
            agent: The agent whose FOV to check
            
        Returns:
            True if target is in agent's FOV, False otherwise
        """
        if self.target is None:
            return False
        
        # Get agent's FOV size
        agent_fov_size = agent.fov_size
        if agent_fov_size is None:
            agent_fov_size = 5  # Default FOV size
        
        # Agent's current position
        agent_x, agent_y = agent.location
        offset = agent_fov_size // 2
        
        # Target's position
        target_x, target_y = self.target.location
        
        # Check if target is within agent's FOV bounds
        fov_left = agent_x - offset
        fov_right = agent_x + offset
        fov_top = agent_y - offset
        fov_bottom = agent_y + offset
        
        return (fov_left <= target_x <= fov_right and 
                fov_top <= target_y <= fov_bottom)
    
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
            print("  Legend: 0=Empty, 1=Obstacle, 2=Agent, 3=Target")
        print()  # New line after each agent's FOV

    def _clear_screen_and_show_step(self):
        """Clear screen and show step information"""
        if not self.show_fov_display:
            return
            
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        print("="*80)
        print(f"STEP {self._step_count}".center(80))
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
        # Generate rectangular obstacles within grid bounds
        # Sizes are chosen randomly with small extents
        attempts = 0
        max_attempts = count * 10
        created = 0
        while created < count and attempts < max_attempts:
            attempts += 1
            # Random top-left
            ox = int(self.np_random.integers(0, self.size))
            oy = int(self.np_random.integers(0, self.size))
            # Random width/height (at least 1 cell), capped to remain in bounds
            max_w = max(1, self.size - ox)
            max_h = max(1, self.size - oy)
            # Prefer small obstacles
            ow = int(self.np_random.integers(1, min(4, max_w) + 1))
            oh = int(self.np_random.integers(1, min(4, max_h) + 1))
            # Avoid duplicates
            rect = (ox, oy, ow, oh)
            if rect in self._obstacles:
                continue
            self._obstacles.append(rect)
            created += 1

        # Note: rendering return handled inside _render_frame; nothing to return here


if __name__ == "__main__":
    env = GridWorldEnvMultiAgent(render_mode="human", size=7, intrinsic=True)
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    print("Initial obs:", obs)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    while not done:
        # Random actions for both agents
        action = {
            "agent_1": env.action_space["agent_1"].sample(),
            "agent_2": env.action_space["agent_2"].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    print("Total reward:", total_reward)


