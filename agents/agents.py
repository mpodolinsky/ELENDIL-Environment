import numpy as np
from gymnasium import spaces

class Agent:
    '''
    An agent in the grid world environment.
    
    Inputs:
        env_size: int,
        name: str,
        color: tuple,
        outline_width: int = 1,
        box_scale: float = 1.0,
        fov_size: int = 3,
        show_target_coords: bool = False,

    Attributes:
        name: str,
        color: tuple,
        action_space: spaces.Discrete(5),
        outline_width: int,
        box_scale: float,
        fov_size: int,
        show_target_coords: bool,
        location: np.ndarray,

    All agents of this class have an action space of Discrete(5) and an observation space of Dict(agent: Box(0, env_size - 1, shape=(2,), dtype=int), obstacles_fov: Box(0, 1, shape=(fov_size, fov_size), dtype=int)).
    If show_target_coords is True, the observation space includes a target box.
    '''
    def __init__(
        self,
        env_size: int,          # The size of the grid world
        name: str,              # The name of the agent
        color: tuple,           # The color of the agent
        outline_width: int,     # The width of the agent's outline
        box_scale: float,       # The scale of the agent's box
        fov_size: int,          # The size of the agent's field of view (must be odd)
        show_target_coords: bool, # Whether to show the target coordinates in the observation space
    ) -> None:
        '''Initializes the agent's observation space, action space, and other attributes.'''

        if fov_size % 2 == 0:
            fov_size += 1
            self.fov_size = fov_size
            print(f"FOV size is even, incrementing by 1 to {fov_size}")
        if fov_size < 3 or fov_size > env_size or fov_size <= 1:
            self.fov_size = 3
            print(f"FOV size is less than 3, setting to 3")

        else:
            self.fov_size = fov_size

        obs_dict = {
            "agent": spaces.Box(0, env_size - 1, shape=(2,), dtype=int),
            "obstacles_fov": spaces.Box(0, 1, shape=(self.fov_size, self.fov_size), dtype=int),
        }

        if show_target_coords:
            obs_dict["target"] = spaces.Box(0, env_size - 1, shape=(2,), dtype=int)
        self.observation_space = spaces.Dict(obs_dict)

        self.name = name
        self.color = color
        self.action_space = spaces.Discrete(5)
        self.outline_width = outline_width
        self.box_scale = box_scale

        if not (0.05 <= self.box_scale <= 1.0):
            self.box_scale = max(0.05, min(1.0, self.box_scale))

        self.location = np.array([-1, -1], dtype=int)

        # Track last 3 cells visited used for rendering
        self.last_visited_cells = []
        
