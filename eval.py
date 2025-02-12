import argparse

import carb
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
import gymnasium

config = MainConfig()

# load_policy = asdict(config).get('load_policy', None)
from pathlib import Path
d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
load_policys = [general_path + "/models/SAC/nav_bc_actual_380000_steps.zip",
                general_path + "/models/SAC/nav_without_bc_380000_steps.zip",
                general_path + "/models/SAC/test01_380000_steps.zip"]
log_dir = asdict(config).get('train_log_dir', None)
env = gymnasium.make("tasks:rlmodel-v1", config=config)
for load_policy in load_policys:
    model = SAC.load(load_policy,verbose=1,tensorboard_log=log_dir,)
    ss = 0
    false = 0
    for t in range(70):
        obs, info = env.reset()
        done = False
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info, message_to_collback = env.step(action)
            if terminated or truncated:
                if terminated:
                    ss += 1
                if truncated:
                    false += 1
                obs, info = env.reset()

    f = open(general_path + "/logs/eval_log_many.txt", "a+")
    sr = ss/(ss+false)
    message = load_policy + " SR: " + str(sr)
    f.write(message + "\n")
    f.close()
env.close()