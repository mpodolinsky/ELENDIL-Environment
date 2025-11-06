from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from typing import Optional

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, intrinsic = False):
        self.size = size  # The size of the square grid (default to 5x5)
        self.window_size = 512  # The size of the PyGame window

        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # Action space - available actions 
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space.n = self.observation_space.shape

        # Mapping actions to movement on the grid
        self._action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Up 
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1]), # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # We have these following initalizations if we use the "human" render mode
        self.intrinsic = intrinsic

        if self.intrinsic:
            self.visit_count = np.zeros((self.size, self.size), dtype=int)

        self.window = None
        self.clock = None

    def _get_obs(self):
        '''
        Convert internal state of observation format

        Returns:
            dict: Observation of agent and target position
        '''
        return {"agent": self._agent_location, "target": self._target_location}
    
    # We can also define additional information to be returned by .reset() and .step() 
    # for example we can return the Manhattan distance for progress and debugging

    def _get_info(self):
        '''
        Returns:
            Manhattan distance between target and agent [used for debugging]
        '''

        return {
            "distance" : np.linalg.norm(self._agent_location - self._target_location, ord=1)
            }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''Start a new episode.

        Args: 
            seed: Seed for random number generator
            options: Additional configuration options

        Returns:
            tuple: (observation, info) for the initial state
        '''

        super().reset(seed=seed) # calls .reset() in the parent class gym.Env

        # Randomly place agent
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # select size=2 numbers withing (0,self.size)

        # Randomly place target [ensure that it is different from the agent]
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while np.all(self._agent_location == self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        
        if self.intrinsic:
            self.visit_count.fill(0)
            self.visit_count[tuple(self._agent_location)] += 1

        if self.render_mode == "human":
            self._render_frame()

        self.step_counter = 0

        return observation, info

     
    # --- STEP ---
    # The step function is the core environment logic, this is where physics, rules, and reward live

    # --> Agent takes action
    # 1. Convert action from discrete to movement
    # 2. Update the state [agent position]
    # 3. Compute reward 
    # 4. Determine if episode should end [i.e. if goal was reached]
    # 5. Return all required information

    def step(self, action):
        '''Execute one time step in the environment.

        Args:
            action: action within [0,3]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        '''

        direction = self._action_to_direction[action]

        # Clips the coordinates if we go out of bounds
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use this here
        truncated = True if self.step_counter >= 50 else False

        # Simple reward: +1 for reaching target, 0 otherwise

        reward = 1 if terminated else -0.01 # small negative reward to encourage efficiency
        
        if self.intrinsic:
            reward += 0.05 if self.visit_count[tuple(self._agent_location)] == 0 else 0
            self.visit_count[tuple(self._agent_location)] += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        self.step_counter += 1

        return observation, reward, terminated, truncated, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human", size=15)
    env = FlattenObservation(env)
    obs, info = env.reset()
    print(type(obs))
    done = False

    print("Initial Observation:", obs)

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        done = terminated or truncated

    env.close()