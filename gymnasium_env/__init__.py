from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs.legacy:GridWorldEnv",
)
register(
    id="gymnasium_env/GridWorld-intrinsic",
    entry_point="gymnasium_env.envs.legacy:GridWorldEnvIntrinsic",
)
register(
    id="gymnasium_env/GridWorld-multi-target",
    entry_point="gymnasium_env.envs.legacy:GridWorldEnvMulti_Target",
)
register(
    id="gymnasium_env/GridWorld-multi-agent",
    entry_point="gymnasium_env.envs:GridWorldEnvMultiAgent",
)