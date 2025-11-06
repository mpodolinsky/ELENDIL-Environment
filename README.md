# ELENDIL - Environment for Limited-Exposure Navigation for Diverse Intercommunicative Learners

A multi-agent reinforcement learning environment built with PettingZoo's AEC (Agent-Environment-Cycle) and Parallel APIs, featuring field-of-view (FOV) based observations and dynamic target tracking.

## Overview

This environment simulates a grid-based world where multiple agents navigate to find a moving target while avoiding obstacles and other agents. Each agent operates with limited field-of-view observations, creating realistic partial observability scenarios for multi-agent reinforcement learning research.

## Features

### Environment Features
- **Dual API Support**: Both AEC (sequential) and Parallel (simultaneous) agent interactions
- **Dynamic Target Movement**: Moving target that changes position each step
- **Dual Obstacle System**: 
  - Physical obstacles (dark gray) block agent movement
  - Visual obstacles (light blue with stripes) block ObserverAgent view but not movement (simulating buildings/cover)
- **FOV-Based Observations**: Agents see only their local field-of-view, not the entire grid
- **Individual Rewards**: Each agent receives rewards based on their own FOV detection
- **Fair Reward Attribution**: In Parallel API, agents are rewarded before target moves
- **Flexible Agent Configuration**: 
  - Configure agents via YAML files or inline dictionaries
  - Automatic agent type detection
  - Support for any number and mix of agent types
  - No need to pre-instantiate agents in external repositories

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
  - FOV obstacles map (size = base_fov_size + 4)
    - `-10` = Masked area (outside visible range for current flight level)
    - `0` = Empty squares
    - `1` = Physical obstacles (not shown for ObserverAgent at altitude >= 1)
    - `2` = Other agents (at same or lower altitude)
    - `3` = Target (probabilistic detection)
    - `4` = Visual obstacles (block view, hide what's beneath them)
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
cd ELENDIL

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (this will install elendil package in editable mode)
pip install -e .
```

**Note**: The `pip install -e .` command installs the `elendil` package in development mode, making it available for import as `from elendil.envs.grid_world_multi_agent import ...`

### Basic Usage

The environment supports both **AEC (Sequential)** and **Parallel** APIs. Choose based on your needs:

#### AEC API (Sequential Agent Interactions)

```python
import yaml
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

# Load configurations
with open("configs/agent_configs/ground_agent.yaml") as f:
    ground_agent_config = yaml.safe_load(f)

with open("configs/agent_configs/air_observer_agent.yaml") as f:
    air_observer_config = yaml.safe_load(f)

with open("configs/target_configs/target_config.yaml") as f:
    target_config = yaml.safe_load(f)

# Create AEC environment - agents take turns sequentially
env = GridWorldEnvMultiAgent(
    agents=[ground_agent_config, air_observer_config],
    size=15,
    render_mode="human",
    enable_obstacles=True,
    num_obstacles=3,
    target_config=target_config
)

env.reset()

# AEC API: Sequential agent interactions
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_spaces[agent].sample()
    
    env.step(action)

env.close()
```

#### Parallel API (Simultaneous Agent Interactions) - **NEW!**

```python
import yaml
from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel

# Load configurations
with open("configs/agent_configs/ground_agent.yaml") as f:
    ground_agent_config = yaml.safe_load(f)

with open("configs/agent_configs/air_observer_agent.yaml") as f:
    air_observer_config = yaml.safe_load(f)

with open("configs/target_configs/target_config.yaml") as f:
    target_config = yaml.safe_load(f)

# Create Parallel environment - all agents act simultaneously
env = GridWorldEnvParallel(
    agents=[ground_agent_config, air_observer_config],
    size=15,
    render_mode="human",
    enable_obstacles=True,
    num_obstacles=3,
    target_config=target_config
)

observations, infos = env.reset()

# Parallel API: Simultaneous agent interactions
while env.agents:
    # Generate actions for all agents
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    
    # All agents act simultaneously
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

#### API Comparison

| Feature | AEC API | Parallel API |
|---------|---------|--------------|
| **Agent Interaction** | Sequential (turn-based) | Simultaneous |
| **Performance** | Sequential overhead | Better performance |
| **Integration** | PettingZoo standard | Compatible with many RL frameworks |
| **Reward Timing** | Target moves after each agent | Agents rewarded before target moves |
| **Use Cases** | Turn-based games, strict ordering | Real-time scenarios, natural interactions |

**Key Benefits of Parallel API:**
- More natural agent interactions
- Better performance (no sequential overhead)  
- Easier integration with many RL frameworks
- Fairer reward attribution (agents rewarded before target moves)

#### Option 2: Inline Configuration

```python
from elendil.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

# Define agents as configuration dictionaries
agent_configs = [
    {
        "name": "ground_agent",
        "type": "FOVAgent",
        "color": (80, 160, 255),
        "fov_size": 5
    },
    {
        "name": "aerial_agent",
        "type": "ObserverAgent",
        "color": (255, 100, 100),
        "fov_base_size": 3,
        "max_altitude": 3,
        "target_detection_probs": (1.0, 0.66, 0.33)
    }
]

# Create environment
env = GridWorldEnvMultiAgent(
    agents=agent_configs,
    size=15,
    render_mode="human",
    enable_obstacles=True,
    num_obstacles=3
)

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
| `num_obstacles` | int | 0 | Number of physical obstacles to generate (block movement) |
| `num_visual_obstacles` | int | 0 | Number of visual obstacles to generate (block ObserverAgent view) |

### Agent Configuration

The environment supports flexible agent configuration through **dictionaries** (recommended) or pre-instantiated objects. Agent configurations can be loaded from YAML files or defined inline.

#### Configuration Files

The `configs/` directory contains agent configuration files:

**Ground Agent (FOVAgent)** - `configs/ground_agent.yaml`:
```yaml
name: "alpha"
type: "FOVAgent"
color: [80, 160, 255]  # Blue
fov_size: 5
outline_width: 2
box_scale: 0.9
show_target_coords: False
```

**Air Observer Agent (ObserverAgent)** - `configs/air_observer_agent.yaml`:
```yaml
name: "epsilon"
type: "ObserverAgent"
color: [255, 100, 100]  # Red
fov_base_size: 3
max_altitude: 3
target_detection_probs: [1.0, 0.66, 0.33]
outline_width: 2
box_scale: 0.7
show_target_coords: False
```

#### Configuration Parameters

**FOVAgent Configuration:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique agent identifier |
| `type` | str | "FOVAgent" (optional, auto-detected) |
| `color` | tuple/list | RGB color values |
| `fov_size` | int | Field of view size |
| `outline_width` | int | Visual outline thickness (default: 2) |
| `box_scale` | float | Agent box scale (default: 0.8) |
| `show_target_coords` | bool | Override env setting (default: False) |

**ObserverAgent Configuration:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique agent identifier |
| `type` | str | "ObserverAgent" (optional, auto-detected) |
| `color` | tuple/list | RGB color values |
| `fov_base_size` | int | Base FOV size (actual: base_size + 4) |
| `max_altitude` | int | Maximum altitude level (default: 3) |
| `target_detection_probs` | tuple/list | Detection probability per FL [FL1, FL2, FL3] |
| `outline_width` | int | Visual outline thickness (default: 2) |
| `box_scale` | float | Agent box scale (default: 0.8) |
| `show_target_coords` | bool | Override env setting (default: False) |

#### Using Configuration Dictionaries

```python
# Define agents directly as dictionaries
agent_configs = [
    {
        "name": "agent_1",
        "type": "FOVAgent",
        "color": (80, 160, 255),
        "fov_size": 5
    },
    {
        "name": "observer_1",
        "type": "ObserverAgent",
        "color": (255, 100, 100),
        "fov_base_size": 3,
        "max_altitude": 3,
        "target_detection_probs": (1.0, 0.66, 0.33)
    }
]

env = GridWorldEnvMultiAgent(agents=agent_configs, size=15)
```

#### Automatic Type Detection

The environment can automatically detect agent type based on parameters:
```python
agent_configs = [
    {"name": "ground", "fov_size": 5},           # Auto-detected as FOVAgent
    {"name": "aerial", "fov_base_size": 3}       # Auto-detected as ObserverAgent
]
```

#### Pre-Instantiated Agents (Still Supported)

```python
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent

agents = [
    FOVAgent(name="ground_agent", color=(255, 100, 100), env_size=15, fov_size=5),
    ObserverAgent(name="aerial_agent", color=(100, 100, 255), env_size=15, fov_base_size=3)
]

env = GridWorldEnvMultiAgent(agents=agents, size=15)
```

#### Mixed Configuration and Instances

```python
from agents.agents import FOVAgent

agents = [
    FOVAgent(name="agent_1", color=(80, 160, 255), env_size=15, fov_size=5),  # Instance
    {"name": "observer_1", "type": "ObserverAgent", "fov_base_size": 3}       # Config
]

env = GridWorldEnvMultiAgent(agents=agents, size=15)
```

For more details on configuration-based setup, see `AGENT_CONFIG_GUIDE.md` and `example_config_based.py`.

## Reward System

Each agent receives individual rewards based on their FOV detection:

- **`-3λ`**: Reaching the target location (penalty - agents should observe, not intercept, where λ = lambda_fov)
- **`-0.05`**: Colliding with an obstacle (default, configurable via `obstacle_collision_penalty`)
- **`+(1-λ)`**: Detecting target in FOV (λ = lambda_fov)
- **`-λ`**: Being detected by target's FOV
- **`-0.01`**: Small step penalty when no detection occurs
- **`+0.025`**: Intrinsic exploration bonus (if enabled) for visiting new cells

**Notes:**
- The negative reward for reaching the target encourages agents to maintain observation distance rather than intercepting the target. The episode continues even when the target is reached.
- ObserverAgents flying at altitude ≥ 1 do not receive obstacle collision penalties as they fly over ground obstacles.
- Rewards are cumulative - an agent can receive multiple penalties/rewards in a single step.

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
ELENDIL/
├── elendil/                              # Main package
│   ├── envs/
│   │   ├── grid_world_multi_agent.py     # Both AEC and Parallel environments
│   │   ├── legacy/                       # Legacy environment implementations
│   │   └── __init__.py
│   ├── wrappers/                         # Environment wrappers
│   └── __init__.py
├── agents/                               # Agent implementations
│   ├── agents.py                         # BaseAgent, FOVAgent classes
│   ├── observer_agent.py                 # ObserverAgent with flight levels
│   ├── special_agents.py                 # GlobalViewAgent, TelepathicAgent
│   ├── target.py                         # Target class definition
│   └── __init__.py
├── configs/                              # Configuration files
│   ├── agent_configs/                    # Agent configuration YAML files
│   ├── env_configs/                      # Environment configuration files
│   └── target_configs/                   # Target configuration files
├── tests/                                # Test files and scripts
│   ├── test_aec.py                       # AEC API test script
│   ├── test_parallel.py                  # Parallel API test script
│   ├── test_observer_agent.py            # ObserverAgent tests
│   ├── test_modular_agents.py            # Modular agent tests
│   ├── demo_capabilities.py              # Capability demonstration
│   ├── example_config_based.py           # Configuration examples
│   ├── visualize_obstacles.py            # Obstacle visualization
│   └── AGENT_CONFIG_GUIDE.md             # Configuration guide
├── examples/                             # Example scripts and demos
│   └── api_comparison_demo.py            # AEC vs Parallel API comparison
├── docs/                                 # Documentation and assets
│   ├── q_learning_training_rewards.png   # Training results
│   └── TODO.md                           # Project TODO list
├── images/                               # Image assets
│   └── obstacle_examples/                # Obstacle visualization examples
├── pyproject.toml                        # Package configuration
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Testing

### Test Both APIs

Run the comparison script to see both AEC and Parallel APIs in action:

**From project root:**
```bash
python examples/api_comparison_demo.py
```

**From examples directory:**
```bash
cd examples
python api_comparison_demo.py
```

### Test Individual APIs

**AEC API (Sequential):**

From project root:
```bash
python tests/test_aec.py
```

From tests directory:
```bash
cd tests
python test_aec.py
```

**Parallel API (Simultaneous):**

From project root:
```bash
python tests/test_parallel.py
```

From tests directory:
```bash
cd tests
python test_parallel.py
```

### Virtual Environment Setup

Make sure you have the `elendil` package installed in your virtual environment:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# Verify installation
python -c "import elendil; print('elendil package installed successfully!')"
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'elendil'**

This error occurs when the `elendil` package is not installed in your virtual environment. Solution:

```bash
# Make sure you're in the project root directory
cd /path/to/ELENDIL

# Activate your virtual environment
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# Verify installation
python -c "import elendil; print('Success!')"
```

**2. FileNotFoundError when running from elendil/envs directory**

When running scripts from the `elendil/envs/` directory, the config file paths are automatically adjusted. If you get file not found errors, make sure you're running from the correct directory or the config files exist in the expected locations.

**3. Virtual Environment Issues**

If you have multiple virtual environments (`.venv`, `.venv_new`, etc.), make sure you're using the correct one:

```bash
# Check which Python you're using
which python

# Check which packages are installed
pip list | grep elendil

# If elendil is not found, install it
pip install -e .
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

### Configuration Guides
- **`AGENT_CONFIG_GUIDE.md`**: Comprehensive guide to agent configuration system
  - Configuration dictionary format
  - YAML file integration
  - Automatic type detection
  - Migration guide from old approach
- **`example_config_based.py`**: Five detailed examples showing configuration usage
- **`demo_capabilities.py`**: Scripted demonstration of agent capabilities

### API Documentation
For detailed API documentation, see the inline code documentation in the source files.

---

**Note**: This environment follows PettingZoo's AEC (Agent-Environment-Cycle) standard, ensuring compatibility with the broader multi-agent RL ecosystem.
