# Agent Configuration Guide

## Overview

The `GridWorldEnvMultiAgent` environment now supports **flexible agent initialization** through configuration dictionaries, making it easy to use in other repositories without pre-instantiating agents.

## Quick Start

### Option 1: Configuration Dictionaries (Recommended)

```python
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

# Define agents as configs
agent_configs = [
    {
        "name": "alpha",
        "type": "FOVAgent",
        "color": (80, 160, 255),
        "fov_size": 5
    },
    {
        "name": "observer1",
        "type": "ObserverAgent",
        "color": (255, 100, 100),
        "fov_base_size": 3,
        "max_altitude": 3,
        "target_detection_probs": (1.0, 0.66, 0.33)
    }
]

# Create environment - agents are instantiated automatically!
env = GridWorldEnvMultiAgent(
    size=15,
    agents=agent_configs,
    enable_obstacles=True,
    num_obstacles=4
)
```

### Option 2: Pre-Instantiated Agents (Still Supported)

```python
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent

# Pre-instantiate agents
agents = [
    FOVAgent(name="alpha", color=(80, 160, 255), env_size=15, fov_size=5),
    ObserverAgent(name="obs", color=(255, 100, 100), env_size=15, fov_base_size=3)
]

env = GridWorldEnvMultiAgent(size=15, agents=agents)
```

### Option 3: Mix and Match

```python
# You can mix configs and instances!
from agents.agents import FOVAgent

agents = [
    FOVAgent(name="alpha", color=(80, 160, 255), env_size=15, fov_size=5),  # Instance
    {"name": "observer1", "type": "ObserverAgent", "fov_base_size": 3}      # Config
]

env = GridWorldEnvMultiAgent(size=15, agents=agents)
```

## Agent Types

### FOVAgent Configuration

```python
{
    "name": "agent_name",           # Required: unique identifier
    "type": "FOVAgent",             # Optional: auto-detected if fov_size present
    "color": (80, 160, 255),        # Optional: RGB tuple (default: auto-assigned)
    "fov_size": 5,                  # Required for FOVAgent: field of view size
    "outline_width": 2,             # Optional: visual outline thickness (default: 2)
    "box_scale": 0.9,               # Optional: agent box scale (default: 0.8)
    "show_target_coords": False     # Optional: override env setting (default: False)
}
```

### ObserverAgent Configuration

```python
{
    "name": "observer_name",        # Required: unique identifier
    "type": "ObserverAgent",        # Optional: auto-detected if fov_base_size present
    "color": (255, 100, 100),       # Optional: RGB tuple (default: auto-assigned)
    "fov_base_size": 3,             # Required for ObserverAgent: base FOV size
    "max_altitude": 3,              # Optional: maximum altitude level (default: 3)
    "target_detection_probs": (1.0, 0.66, 0.33),  # Optional: detection probability per altitude
    "outline_width": 2,             # Optional: visual outline thickness (default: 2)
    "box_scale": 0.8,               # Optional: agent box scale (default: 0.8)
    "show_target_coords": False     # Optional: override env setting (default: False)
}
```

## Automatic Type Detection

The environment can **automatically detect** the agent type based on parameters:

```python
# No "type" field needed!
agent_configs = [
    {"name": "alpha", "fov_size": 5},            # Auto-detected as FOVAgent
    {"name": "obs", "fov_base_size": 3},         # Auto-detected as ObserverAgent
]
```

**Detection Rules:**
- If config has `fov_base_size`, `max_altitude`, or `target_detection_probs` → **ObserverAgent**
- Otherwise, or if config has `fov_size` → **FOVAgent**

## Loading from YAML

```yaml
# agents_config.yaml
agents:
  - name: scout
    type: FOVAgent
    color: [80, 160, 255]
    fov_size: 7
    
  - name: observer
    type: ObserverAgent
    color: [255, 100, 100]
    fov_base_size: 3
    max_altitude: 3
    target_detection_probs: [1.0, 0.66, 0.33]
```

```python
import yaml

with open("agents_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Convert lists to tuples
for cfg in config["agents"]:
    if "color" in cfg:
        cfg["color"] = tuple(cfg["color"])
    if "target_detection_probs" in cfg:
        cfg["target_detection_probs"] = tuple(cfg["target_detection_probs"])

env = GridWorldEnvMultiAgent(size=15, agents=config["agents"])
```

## Default Behavior

If `agents=None`, the environment creates **2 default FOVAgents**:

```python
env = GridWorldEnvMultiAgent(size=10)  # Creates agent_1 and agent_2
```

## Support for Any Number of Agents

You can create any number of agents, mixing both types:

```python
agent_configs = [
    {"name": f"fov_{i}", "fov_size": 5} for i in range(5)
] + [
    {"name": f"obs_{i}", "fov_base_size": 3, "max_altitude": 3} for i in range(3)
]

env = GridWorldEnvMultiAgent(size=20, agents=agent_configs)
# Creates 5 FOVAgents + 3 ObserverAgents = 8 total agents
```

## Benefits

✅ **Easier Integration**: No need to import agent classes in other repos  
✅ **YAML/JSON Friendly**: Store configurations in files  
✅ **Flexible**: Mix configs and instances  
✅ **Auto-Detection**: Smart type inference  
✅ **Scalable**: Support any number of agents  
✅ **Backward Compatible**: Existing code still works  

## Examples

See `example_config_based.py` for comprehensive examples including:
- Pure configuration-based initialization
- Automatic type detection
- Minimal configs with defaults
- Loading from YAML
- Default agent creation

## Complete Environment Example

```python
import yaml
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

# Load configs
with open("configs/target_config.yaml") as f:
    target_config = yaml.safe_load(f)

# Define agents
agent_configs = [
    {"name": "alpha", "color": (80, 160, 255), "fov_size": 5},
    {"name": "beta", "color": (255, 180, 60), "fov_size": 3},
    {
        "name": "observer1",
        "color": (255, 100, 100),
        "fov_base_size": 3,
        "max_altitude": 3,
        "target_detection_probs": (1.0, 0.66, 0.33)
    }
]

# Create environment
env = GridWorldEnvMultiAgent(
    render_mode="human",
    size=15,
    agents=agent_configs,
    target_config=target_config,
    enable_obstacles=True,
    num_obstacles=4,
    num_visual_obstacles=3,
    lambda_fov=0.5
)

# Use environment
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_spaces[agent].sample()
    env.step(action)

env.close()
```

## Migration Guide

### Before (old approach)

```python
from agents.agents import FOVAgent
from agents.observer_agent import ObserverAgent

agents = [
    FOVAgent(name="a", color=(80,160,255), env_size=15, fov_size=5, ...),
    ObserverAgent(name="o", color=(255,100,100), env_size=15, fov_base_size=3, ...)
]

for agent in agents:
    agent._setup_agent(...)  # Manual setup

env = GridWorldEnvMultiAgent(size=15, agents=agents)
```

### After (new approach)

```python
agent_configs = [
    {"name": "a", "color": (80,160,255), "fov_size": 5},
    {"name": "o", "color": (255,100,100), "fov_base_size": 3}
]

env = GridWorldEnvMultiAgent(size=15, agents=agent_configs)
# That's it! Everything is handled automatically.
```

---

For more details, see the full API documentation in the `GridWorldEnvMultiAgent` class docstring.

