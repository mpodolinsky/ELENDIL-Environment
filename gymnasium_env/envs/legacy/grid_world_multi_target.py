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

class GridWorldEnvMulti_Target(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, intrinsic = False):
        self.size = size  # The size of the square grid (default to 5x5)
        self.window_size = 512  # The size of the PyGame window

        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_red": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_green": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location_red = np.array([-1, -1], dtype=int)
        self._target_location_green = np.array([-1, -1], dtype=int)

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
        self.step_counter = 0

    def _get_obs(self):
        '''
        Convert internal state of observation format

        Returns:
            dict: Observation of agent and target position
        '''
        return {"agent": self._agent_location, "target_red": self._target_location_red, "target_green": self._target_location_green}
    
    # We can also define additional information to be returned by .reset() and .step() 
    # for example we can return the Manhattan distance for progress and debugging

    def _get_info(self):
        '''
        Returns:
            Manhattan distance between target and agent [used for debugging]
        '''

        return {
            "distance_red" : np.linalg.norm(self._agent_location - self._target_location_red, ord=1),
            "distance_green" : np.linalg.norm(self._agent_location - self._target_location_green, ord=1)
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
        coords = self.np_random.choice(self.size * self.size, size=3, replace=False)
        self._agent_location = np.array([coords[0] // self.size, coords[0] % self.size], dtype=int)
        self._target_location_red = np.array([coords[1] // self.size, coords[1] % self.size], dtype=int)
        self._target_location_green = np.array([coords[2] // self.size, coords[2] % self.size], dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        
        if self.intrinsic:
            self.visit_count.fill(0)
            self.visit_count[tuple(self._agent_location)] += 1

        if self.render_mode == "human":
            self._render_frame()

        self.visited_targets = {"red": False, "green": False}
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
        self.step_counter += 1

        direction = self._action_to_direction[action]

        # Clips the coordinates if we go out of bounds
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        reward = 0

        # Check if agent is on red target and hasn't visited it yet
        if not self.visited_targets["red"] and np.array_equal(self._agent_location, self._target_location_red):
            self.visited_targets["red"] = True
            reward += 1
        elif self.visited_targets["red"] and np.array_equal(self._agent_location, self._target_location_red):
            reward -= 0.2  # Penalty for revisiting red target

        # Check if agent is on green target and hasn't visited it yet
        if not self.visited_targets["green"] and np.array_equal(self._agent_location, self._target_location_green):
            self.visited_targets["green"] = True
            reward += 1
        elif self.visited_targets["green"] and np.array_equal(self._agent_location, self._target_location_green):
            reward -= 0.2  # Penalty for revisiting green target

        # Episode terminates when both targets have been visited
        terminated = all(self.visited_targets.values())
        truncated = self.step_counter > 64
        # truncated = False
    
        if self.intrinsic:
            reward += 0.05 if self.visit_count[tuple(self._agent_location)] == 0 else 0
            self.visit_count[tuple(self._agent_location)] += 1

        # Punishement for each step to encourage efficiency
        reward -= 0.01

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # --- Init pygame if needed ---
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # --- Dark mode colors ---
        BG_COLOR     = (30, 30, 30)     # dark background
        GRID_COLOR   = (60, 60, 60)     # subtle grid
        RED_TARGET   = (220, 50, 50)    # red
        GREEN_TARGET = (50, 200, 100)   # green
        AGENT_COLOR  = (80, 160, 255)   # light blue

        # --- Setup canvas ---
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(BG_COLOR)
        pix_square = self.window_size / self.size

        # --- Grid ---
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, GRID_COLOR,
                (0, pix_square * x), (self.window_size, pix_square * x), 1
            )
            pygame.draw.line(
                canvas, GRID_COLOR,
                (pix_square * x, 0), (pix_square * x, self.window_size), 1
            )

        # --- Targets (flat squares) ---
        for color, loc in [(RED_TARGET, self._target_location_red),
                        (GREEN_TARGET, self._target_location_green)]:
            rect = pygame.Rect(
                pix_square * loc,
                (pix_square, pix_square),
            )
            pygame.draw.rect(canvas, color, rect)

        # --- Agent (flat circle) ---
        center = (self._agent_location + 0.5) * pix_square
        pygame.draw.circle(
            canvas, AGENT_COLOR,
            center.astype(int),
            int(pix_square / 3)
        )

        # --- Render ---
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


        
if __name__ == "__main__":
    env = GridWorldEnvMulti_Target(render_mode="human", size=3, intrinsic=True)
    env = FlattenObservation(env)
    obs, info = env.reset()
    print(type(obs))
    done = False

    print("Initial Observation:", obs)

    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        total_reward += reward
        done = terminated or truncated

    env.close()
    print("\nTest run finished.")
    print(f"Final Observation: {obs}")
    print(f"Final Reward: {total_reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Final Info: {info}")



    