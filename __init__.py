from gymnasium.envs.registration import register
print("register")

register(
     id="AlphaBaseEnv-v0",
     entry_point="tasks.env_n:AlphaBaseEnv",
     max_episode_steps=2700,
)