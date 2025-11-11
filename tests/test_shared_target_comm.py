#!/usr/bin/env python3

import numpy as np
from gymnasium.spaces.utils import unflatten

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel


def main():
    agent_configs = [
        {"name": "ground", "type": "FOVAgent"},
        {"name": "air", "type": "ObserverAgent"},
    ]

    env = GridWorldEnvParallel(
        size=7,
        agents=agent_configs,
        no_target=False,
        goal_enabled=False,
        target_velocity=False,
        show_fov_display=False,
        shared_target_coords=True,
        target_config={"movement_speed": 0},
    )

    obs, _ = env.reset(seed=123)

    def shared(agent_name, obs_flat):
        raw = env._raw_observation_spaces[agent_name]
        obs_dict = unflatten(raw, obs_flat)
        return obs_dict["shared_target"]

    agent_names = list(env._agents_by_name.keys())

    print("Initial shared target values:")
    for agent_name in agent_names:
        print(f"  {agent_name}: {shared(agent_name, obs[agent_name])}")

    # Force both agents to see the target this turn
    target_loc = env.target.location.copy()
    viewing_offset = np.array([0, 1])
    view_loc = np.clip(target_loc + viewing_offset, 0, env.size - 1)
    env._agents_by_name["ground"].location = view_loc.copy()
    env._agents_by_name["air"].location = view_loc.copy()

    # Step once (observations returned correspond to previous shared state)
    actions = {agent: 4 for agent in env.agents}
    obs_after_step, *_ = env.step(actions)

    print("\nAfter detection step (immediate observations reflect previous turn):")
    for agent_name in agent_names:
        print(f"  {agent_name}: {shared(agent_name, obs_after_step[agent_name])}")

    # Obtain fresh observation without progressing time to see shared coords updated
    print("\nObservations for next turn (should include target coordinates):")
    for agent_name in agent_names:
        obs_flat = env.observe(agent_name)
        print(f"  {agent_name}: {shared(agent_name, obs_flat)}")

    # Move agents away so no one sees the target on the next step
    env._agents_by_name["ground"].location = np.array([0, 0])
    env._agents_by_name["air"].location = np.array([6, 6])
    obs_after_miss, _, terminations, _, _ = env.step({agent: 4 for agent in env.agents})

    print("\nAfter a turn with no detections (still showing previous turn's coords):")
    for agent_name in agent_names:
        obs_flat = obs_after_miss.get(agent_name)
        if obs_flat is None:
            print(f"  {agent_name}: <terminated>")
        else:
            print(f"  {agent_name}: {shared(agent_name, obs_flat)}")

    # Now check the next observation to ensure it reset to -1
    print("\nNext turn observations (expect [-1, -1]):")
    for agent_name in agent_names:
        if terminations.get(agent_name):
            print(f"  {agent_name}: <terminated>")
            continue
        obs_flat = env.observe(agent_name)
        print(f"  {agent_name}: {shared(agent_name, obs_flat)}")


if __name__ == "__main__":
    main()

