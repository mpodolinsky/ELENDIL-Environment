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

### Observation System
- **Agent Position**: Current location in the grid
- **FOV Obstacles**: Local view with encoding:
  - `0` = Empty squares
  - `1` = Obstacles
  - `2` = Other agents
  - `3` = Target
- **Optional Target Coordinates**: Can be enabled for specific research needs

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

# Create environment
env = GridWorldEnvMultiAgent(
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

Agents can be customized with:

```python
agents = [
    {
        "name": "agent_1",
        "color": (80, 160, 255),
        "fov_size": 5,
        "outline_width": 2,
        "box_scale": 0.8
    },
    {
        "name": "agent_2", 
        "color": (255, 180, 60),
        "fov_size": 3,
        "outline_width": 1,
        "box_scale": 1.0
    }
]

env = GridWorldEnvMultiAgent(agents=agents, ...)
```

## Reward System

Each agent receives individual rewards based on their FOV detection:

- **`+1.0`**: Successfully reaching the target
- **`+(1-λ)`**: Detecting target in FOV (λ = lambda_fov)
- **`-λ`**: Being detected by target's FOV
- **`-0.01`**: Small step penalty when no detection occurs
- **`+0.025`**: Intrinsic exploration bonus (if enabled) for visiting new cells

## Action Space

All agents use a discrete action space with 5 actions:
- `0` = Move Right
- `1` = Move Up  
- `2` = Move Left
- `3` = Move Down
- `4` = No Operation

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
│   │   ├── grid_world_multi_agent.py  # Main environment
│   │   └── test.py                    # Test script
│   └── wrappers/                      # Environment wrappers
├── agents/
│   ├── agents.py                      # Agent class definition
│   └── target.py                      # Target class definition
├── configs/                           # Configuration files
├── tests/                             # Test suite
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