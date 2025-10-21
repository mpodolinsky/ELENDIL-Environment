# TODO List

## Target Movement Issue
- [ ] **Think about how to deal with the fact that the target moves twice as much as the other agents**
  - Currently the target moves every step while agents take turns acting sequentially
  - This means the target moves twice as frequently as individual agents
  - Need to consider if this creates unfair advantage/disadvantage
  - Possible solutions:
    - Make target move only every other step
    - Make target move at half the speed
    - Synchronize target movement with agent turns
    - Keep current behavior if it's intentional design choice

## Visual Enhancements
- [x] **Add halo around agent when they see other agents in FOV**
  - When FOV contains value 2 (other agents), draw a small halo around the agent
  - Visual indicator to show mutual agent awareness (both agents get halos when they can see each other)
  - Implemented while maintaining AEC compliance

## New Agent Type Development
- [x] **Create modular agent system with different agent types**
  - ✅ **COMPLETED** - Implemented modular architecture with `BaseAgent` abstract class
  - ✅ Created `FOVAgent` (standard FOV-based agent)
  - ✅ Created `GlobalViewAgent` (sees entire grid instead of FOV)
  - ✅ Created `TelepathicAgent` (sees other agents' locations)
  - ✅ Maintained backward compatibility with legacy `Agent` class
  - ✅ Environment now supports mixed agent types seamlessly
  - ✅ Each agent handles its own observation generation and step logic
  - **Files Created**: `agents/special_agents.py`, `test_modular_agents.py`
  - **Next Step**: Create new branch for advanced agent types as originally planned

## ObserverAgent Boundary Logic
- [ ] **Discuss with Christian: ObserverAgent sees 0 outside of playing area**
  - Current behavior: ObserverAgent FOV shows 0 (empty) for cells outside the grid boundaries
  - Potential issue: This might give misleading information about "safe" empty spaces
  - Alternative: Should out-of-bounds areas be treated as obstacles (1) instead?
  - Decision needed: How should the ObserverAgent perceive grid boundaries in its large FOV?

## Future Items
- [ ] Add more items here as requested...

