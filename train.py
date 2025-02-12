# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse

import carb
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
import gymnasium


config = MainConfig()
tuning = asdict(config).get('tuning', None)
log_dir = asdict(config).get('train_log_dir', None)
env = gymnasium.make("tasks:rlmodel-v1", config=config)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="nav_with_bc_test4")

total_timesteps = 2500000

if tuning:
    load_policy = asdict(config).get('load_policy', None)
    model = SAC.load(load_policy, env=env, verbose=1,tensorboard_log=log_dir,)
else:
    model = SAC("MlpPolicy", env, verbose=1,tensorboard_log=log_dir,)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.save(log_dir + "/SAC_policy")

env.close()


