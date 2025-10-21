# Multi-Agent Grid World Environment

A sophisticated multi-agent reinforcement learning environment built with PettingZoo's AEC (Agent-Environment-Cycle) interface, featuring field-of-view (FOV) based observations and dynamic target tracking.

## Overview

This environment simulates a grid-based world where multiple agents navigate to find a moving target while avoiding obstacles and other agents. Each agent operates with limited field-of-view observations, creating realistic partial observability scenarios for multi-agent reinforcement learning research.

## Features

### Environment Features
- **Multi-Agent Sequential Stepping**: Agents take turns acting using PettingZoo's AEC interface
- **Dynamic Target Movement**: Moving target that changes position each step
- **Obstacle Avoidance**: Configurable obstacles that agents must navigate around
- **FOV-Based Observations**: Agents see only their local field-of-view, not the entire grid
- **Individual Rewards**: Each agent receives rewards based on their own FOV detection

### Agent Architecture

The environment supports a modular agent architecture with different agent types:

#### Base Agent System
All agents inherit from the `BaseAgent` abstract class, which defines the core interface:
- `get_observation_space()`: Returns the agent's observation space
- `get_action_space()`: Returns the agent's action space
- `observe(env_state)`: Generates observations from the environment state
- `step(action, env_state)`: Processes actions and returns results
- `reset()`: Resets the agent to initial state

This modular design allows for easy creation of specialized agent types with different observation spaces, action spaces, and behaviors.

#### FOVAgent (Field-of-View Agent)
Standard agent with local field-of-view based observations:
- **Action Space**: Discrete(5) - Movement in 4 directions + no-op
- **Observation Space**:
  - Agent position (x, y coordinates)
  - FOV obstacles map with encoding:
    - `0` = Empty squares
    - `1` = Obstacles
    - `2` = Other agents
    - `3` = Target
  - Optional target coordinates
- **Visual Representation**: Circular shape with solid FOV border

#### ObserverAgent (Altitude-Aware Agent)
Advanced agent with flight level capabilities and expanded FOV:
- **Action Space**: Discrete(8) - 5 movement actions + 3 altitude actions
  - Movement: Right (0), Up (1), Left (2), Down (3), No-op (4)
  - Altitude: Increase (5), Maintain (6), Decrease (7)
- **Flight Levels**: Operates at 3 distinct altitudes (1, 2, 3)
  - Always starts at flight level 1
  - Cannot descend below level 1
- **Observation Space**:
  - Agent position (x, y coordinates)
  - Flight level (single integer: 1, 2, or 3)
  - Altitude settings (current altitude value)
  - FOV obstacles map (size = base_fov_size + 4)
    - `-10` = Masked area (outside visible range for current flight level)
    - `0` = Empty squares
    - `1` = Obstacles
    - `2` = Other agents (at same or lower altitude)
    - `3` = Target (probabilistic detection)
- **Visual Representation**: Square shape with dynamic FOV border
  - Flight Level 1: Solid line border (3x3 visible area)
  - Flight Level 2: Dashed line border (5x5 visible area)
  - Flight Level 3: Dotted line border (7x7 visible area)

### Flight Level Mechanics (ObserverAgent)

#### FOV Masking System
The ObserverAgent's field-of-view is dynamically masked based on altitude:
- **Flight Level 3**: Full FOV visible (e.g., 7x7 for base_fov_size=3)
- **Flight Level 2**: Outer ring masked, middle area visible (5x5 visible)
- **Flight Level 1**: Two outer rings masked, core area visible (3x3 visible)

Masked cells are set to `-10` in the observation, indicating they are outside the agent's current visual range.

#### Altitude-Based Interactions
- **Obstacle Avoidance**: Agents at altitude > 1 can fly over ground obstacles (value 1)
- **Agent Detection**: Agents only see other agents at the same or lower altitude
- **Target Movement**: Target remains on ground level

#### Probabilistic Target Detection
Target detection varies by flight level, simulating reduced accuracy at higher altitudes:
- **Flight Level 1**: 100% detection probability (always sees target if in FOV)
- **Flight Level 2**: 66% detection probability (2 in 3 chance)
- **Flight Level 3**: 33% detection probability (1 in 3 chance)

Detection probabilities are configurable per agent instance via the `target_detection_probs` parameter (tuple of 3 floats).

When detection fails, the target cell appears as empty (0) instead of showing the target (3), even though the target is physically present in that location.

### Visual Features
- **Real-time Rendering**: PyGame-based visualization
- **Agent Halos**: Visual indicators when agents can see each other
- **FOV Display**: Optional field-of-view visualization
- **Movement Trails**: Track agent movement history
- **Information Panel**: Real-time step, agent, and reward information

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd HA-SPO2V-Env

# Install dependencies
pip install pettingzoo gymnasium pygame pyyaml
```

### Basic Usage

```python
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent

# Create agents
agents = [
    FOVAgent(name="ground_agent", color=(255, 100, 100), env_size=15, fov_size=5),
    ObserverAgent(name="aerial_agent", color=(100, 100, 255), env_size=15, fov_base_size=3)
]

# Create environment
env = GridWorldEnvMultiAgent(
    agents=agents,
    size=15,
    render_mode="human",  # or "rgb_array" for headless
    show_fov_display=True,
    enable_obstacles=True,
    num_obstacles=3
)

# Reset environment
env.reset()

# Run episode
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_spaces[agent].sample()
    
    env.step(action)

env.close()
```

## Configuration

### Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | int | 5 | Grid size (size x size) |
| `max_steps` | int | 500 | Maximum steps per episode |
| `render_mode` | str | None | "human", "rgb_array", or None |
| `show_fov_display` | bool | True | Display agent FOVs in human mode |
| `intrinsic` | bool | False | Enable intrinsic exploration rewards |
| `lambda_fov` | float | 0.5 | FOV reward weighting factor (0-1) |
| `show_target_coords` | bool | False | Include target coordinates in observations |
| `no_target` | bool | False | Disable target spawning |
| `enable_obstacles` | bool | False | Enable obstacles in environment |
| `num_obstacles` | int | 0 | Number of obstacles to generate |

### Agent Configuration

#### Using FOVAgent
```python
from agents.agents import FOVAgent

agents = [
    FOVAgent(
        name="agent_1",
        color=(80, 160, 255),
        env_size=15,
        fov_size=5,
        outline_width=2,
        box_scale=0.8
    ),
    FOVAgent(
        name="agent_2", 
        color=(255, 180, 60),
        env_size=15,
        fov_size=3,
        outline_width=1,
        box_scale=1.0
    )
]

env = GridWorldEnvMultiAgent(agents=agents, size=15, ...)
```

#### Using ObserverAgent
```python
from agents.observer_agent import ObserverAgent

agents = [
    ObserverAgent(
        name="observer_1",
        color=(100, 100, 255),
        env_size=15,
        fov_base_size=3,  # Actual FOV will be 7x7 (base + 4)
        max_altitude=3,
        target_detection_probs=(1.0, 0.66, 0.33),  # FL1, FL2, FL3
        show_target_coords=False
    )
]

env = GridWorldEnvMultiAgent(agents=agents, size=15, ...)
```

#### Mixed Agent Types
```python
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent

agents = [
    FOVAgent(name="ground_agent", color=(255, 100, 100), env_size=15, fov_size=5),
    ObserverAgent(name="aerial_agent", color=(100, 100, 255), env_size=15, fov_base_size=3)
]

env = GridWorldEnvMultiAgent(agents=agents, size=15, ...)
```

## Reward System

Each agent receives individual rewards based on their FOV detection:

- **`-1.0`**: Reaching the target (penalty - agents should observe, not intercept)
- **`+(1-λ)`**: Detecting target in FOV (λ = lambda_fov)
- **`-λ`**: Being detected by target's FOV
- **`-0.01`**: Small step penalty when no detection occurs
- **`+0.025`**: Intrinsic exploration bonus (if enabled) for visiting new cells

Note: The negative reward for reaching the target encourages agents to maintain observation distance rather than intercepting the target.

## Action Space

### FOVAgent Action Space
Discrete(5) - Standard movement actions:
- `0` = Move Right
- `1` = Move Up  
- `2` = Move Left
- `3` = Move Down
- `4` = No Operation

### ObserverAgent Action Space
Discrete(8) - Movement and altitude control:
- `0` = Move Right
- `1` = Move Up
- `2` = Move Left
- `3` = Move Down
- `4` = No Operation
- `5` = Increase Altitude (up to level 3)
- `6` = Maintain Altitude (no change)
- `7` = Decrease Altitude (minimum level 1)

## Research Applications

This environment is designed for:

- **Multi-Agent Reinforcement Learning**: Sequential agent interactions
- **Partial Observability**: FOV-based limited observations
- **Cooperative/Competitive Scenarios**: Configurable reward structures
- **Exploration Research**: Intrinsic motivation and FOV-based rewards
- **AEC Framework Studies**: PettingZoo compatibility research

## Project Structure

```
HA-SPO2V-Env/
├── gymnasium_env/
│   ├── envs/
│   │   ├── grid_world_multi_agent.py  # Main AEC environment
│   │   └── test.py                    # Test script
│   └── wrappers/                      # Environment wrappers
├── agents/
│   ├── agents.py                      # BaseAgent, FOVAgent classes
│   ├── observer_agent.py              # ObserverAgent with flight levels
│   ├── special_agents.py              # GlobalViewAgent, TelepathicAgent
│   └── target.py                      # Target class definition
├── configs/                           # Configuration files
├── tests/                             # Unit test suite
└── videos/                            # Recorded episodes
```

## Testing

Run the test script to see the environment in action:

```bash
python gymnasium_env/envs/test.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PettingZoo](https://pettingzoo.farama.org/) for multi-agent RL
- Uses [Gymnasium](https://gymnasium.farama.org/) as the base RL framework
- Rendered with [PyGame](https://www.pygame.org/) for visualization

## Documentation

For detailed API documentation and advanced usage examples, see the inline code documentation in the source files.

---

**Note**: This environment follows PettingZoo's AEC (Agent-Environment-Cycle) standard, ensuring compatibility with the broader multi-agent RL ecosystem.