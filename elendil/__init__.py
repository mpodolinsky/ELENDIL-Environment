from gymnasium.envs.registration import register

register(
    id="elendil/GridWorld-v0",
    entry_point="elendil.envs.legacy:GridWorldEnv",
)
register(
    id="elendil/GridWorld-intrinsic",
    entry_point="elendil.envs.legacy:GridWorldEnvIntrinsic",
)
register(
    id="elendil/GridWorld-multi-target",
    entry_point="elendil.envs.legacy:GridWorldEnvMulti_Target",
)
register(
    id="elendil/GridWorld-multi-agent",
    entry_point="elendil.envs:GridWorldEnvMultiAgent",
)