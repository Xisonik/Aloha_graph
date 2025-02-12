# from tasks.env import CLGRENV, CLGRCENV
from tasks.env import CLGRCENV
from gymnasium.envs.registration import register
print("register")

# register(
#      id="rlmodel-v0",
#      entry_point="tasks.env:CLGRENV",
#      max_episode_steps=512,
# )

register(
     id="rlmodel-v1",
     entry_point="tasks.env:CLGRCENV",
     max_episode_steps=512,
)
