<!-- 3c397ee3-d642-40d0-b8c6-160b052d060f 50f74387-03ac-4b6f-aba2-cb0862f62ade -->
# Migrate GridWorldEnvMultiAgent to PettingZoo AECEnv

## 1. Update Dependencies

**File: `pyproject.toml`**

- Add `pettingzoo` to dependencies list

## 2. Update Imports and Class Inheritance

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

- Change `from gymnasium import gym` to import from PettingZoo
- Replace `class GridWorldEnvMultiAgent(gym.Env)` with `class GridWorldEnvMultiAgent(AECEnv)`
- Import required PettingZoo classes: `from pettingzoo import AECEnv`

## 3. Implement AEC Required Properties

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Add AECEnv-required properties:

- `possible_agents`: List of all agent names (e.g., ["agent_1", "agent_2"])
- `agent_selection`: Current agent whose turn it is
- `agents`: List of active agents (starts as copy of possible_agents)
- `rewards`: Dict mapping agent names to their individual rewards
- `terminations`: Dict mapping agent names to boolean termination status
- `truncations`: Dict mapping agent names to boolean truncation status
- `infos`: Dict mapping agent names to info dicts

## 4. Refactor reset() Method

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Current signature: `reset(seed, options) -> (obs_dict, info_dict)`

Change to AEC signature: `reset(seed, options) -> None`

- Initialize `self.agents = self.possible_agents.copy()`
- Set `self.agent_selection` to first agent (e.g., "agent_1")
- Initialize `self.rewards = {agent: 0 for agent in self.agents}`
- Initialize `self.terminations = {agent: False for agent in self.agents}`
- Initialize `self.truncations = {agent: False for agent in self.agents}`
- Initialize `self.infos = {agent: {} for agent in self.agents}`
- Initialize `self._cumulative_rewards = {agent: 0 for agent in self.agents}`
- Remove return statement (AEC reset returns None)

## 5. Implement observe() Method

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Create new method: `observe(agent: str) -> observation`

- Use existing `_get_agent_fov_obstacles()` logic
- Return observation for single agent: `{"agent": location, "obstacles_fov": fov_map, "target": target_loc (if enabled)}`
- Preserve existing FOV observation logic

## 6. Implement last() Method

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Create new method: `last() -> tuple`

- Return `(observation, reward, termination, truncation, info)` for `agent_selection`
- Use `observe(self.agent_selection)` for observation
- Return accumulated reward for current agent from `self.rewards[self.agent_selection]`

## 7. Refactor step() Method

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Current: Batch mode with dict of all agent actions → returns (obs_dict, shared_reward, term, trunc, info)

Change to: `step(action: int) -> None`

- Remove all batch stepping logic (lines 287-327)
- Remove single-agent stepping format parsing (lines 328-339)
- Take single integer action for `self.agent_selection`
- Move only the selected agent using existing movement logic
- Calculate **individual reward** for this agent only based on:
  - `+self.lambda_fov` if agent detects target (`_is_target_in_agent_fov`)
  - `-self.lambda_fov` if target detects agent (`_is_agent_in_target_fov`)
  - `-0.01` small step penalty if no detection
  - `+0.025` intrinsic exploration bonus (if enabled and unvisited cell)
- Store reward in `self.rewards[self.agent_selection]`
- Update `self._cumulative_rewards[self.agent_selection]`
- Move target using existing `self.target.step()` logic
- Check termination/truncation conditions and update dicts
- Advance `self.agent_selection` to next agent (cycle through agents)
- No return value (AEC step returns None)

## 8. Update Action/Observation Spaces

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

Current: `self.action_space = spaces.Dict({agent.name: agent.action_space})`

Change to per-agent spaces:

- Keep `self.action_spaces = {agent.name: spaces.Discrete(5) for agent in self.agent_list}`
- Keep `self.observation_spaces = {agent.name: agent.observation_space for agent in self.agent_list}`
- Remove the Dict wrapping

## 9. Update Rendering

**File: `gymnasium_env/envs/grid_world_multi_agent.py`**

- Keep existing `_render_frame()` logic
- Call rendering after each agent step if `render_mode == "human"`
- Ensure FOV display shows only current agent's FOV

## 10. Update **main** Test Block

**File: `gymnasium_env/envs/grid_world_multi_agent.py`** (lines 864-883)

Replace batch action loop with AEC loop:

```python
env = GridWorldEnvMultiAgent(render_mode="human", size=7, intrinsic=True)
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
env.close()
```

## 11. Remove Unused Code

- Delete batch stepping logic (lines ~287-327)
- Delete single-agent format parsing (lines ~328-339)
- Remove `_get_obs()` method (replaced by `observe()`)
- Clean up docstrings mentioning batch mode

## Key Preservation Points

- **FOV Observation Logic**: Keep `_get_agent_fov_obstacles()` intact
- **Reward Calculation**: Preserve FOV-based reward logic (lines 381-411) but apply per-agent
- **Collision Detection**: Keep existing collision and obstacle logic
- **Target Movement**: Keep `self.target.step()` behavior
- **Detection Methods**: Keep `_is_target_in_agent_fov()` and `_is_agent_in_target_fov()` unchanged

### To-dos

- [ ] Add pettingzoo to pyproject.toml dependencies
- [ ] Update imports and change class inheritance from gym.Env to AECEnv
- [ ] Add AEC required properties in __init__ (possible_agents, agent_selection, rewards, terminations, truncations, infos)
- [ ] Refactor reset() to AEC signature (returns None, initializes agent states)
- [ ] Implement observe(agent) method for per-agent observations
- [ ] Implement last() method to return current agent's observation/reward/termination
- [ ] Refactor step() to AEC signature (single action, no return, individual rewards)
- [ ] Update action_space and observation_space to per-agent dicts
- [ ] Update __main__ test block to use AEC agent_iter() pattern
- [ ] Remove batch stepping code and update docstrings