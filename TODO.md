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

## Future Items
- [ ] Add more items here as requested...

