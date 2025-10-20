import numpy as np
from gymnasium import spaces

class Agent:
    def __init__(
        self,
        env_size: int,
        name: str,
        color: tuple,
        observation_space: spaces.Space,
        outline_width: int = 1,
        box_scale: float = 1.0,
        fov_size: int = 3,
        show_target_coords: bool = False,

    ) -> None:

        if fov_size % 2 == 0:
            fov_size += 1
            self.fov_size = fov_size
            print(f"FOV size is even, incrementing by 1 to {fov_size}")
        if fov_size < 3:
            self.fov_size = 3
            print(f"FOV size is less than 3, setting to 3")
        else:
            self.fov_size = fov_size

        base_obs = {
            "agent": spaces.Box(0, env_size - 1, shape=(2,), dtype=int),
        }
        
        # Only include target coordinates if show_target_coords is True
        if show_target_coords:
            base_obs["target"] = spaces.Box(0, env_size - 1, shape=(2,), dtype=int)

        self.observation_space = spaces.Dict({
            **base_obs,
            "obstacles_fov": spaces.Box(0, 1, shape=(fov_size, fov_size), dtype=int),  # matches fov_size=6
        })

        self.name = name
        self.color = color
        self.action_space = spaces.Discrete(5)
        self.outline_width = int(outline_width)
        self.box_scale = float(box_scale)
        if self.fov_size < 1:
            self.fov_size = 1
        if not (0.05 <= self.box_scale <= 1.0):
            self.box_scale = max(0.05, min(1.0, self.box_scale))
        self.location = np.array([-1, -1], dtype=int)
        # Track last 3 cells visited
        self.last_visited_cells = []
        
