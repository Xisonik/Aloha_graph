import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"

import random
from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim
from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
import torch
import os
import os
import gzip
import pickle
import argparse
import json

class Graph_manager():
    def __init__(self, config = MainConfig()):
        self.config = config
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval = asdict(config).get('eval', None)
        self.evalp = asdict(config).get('eval_print', None)
        self.learn_emb = False
        self.init_embedding_nn()

    def init_embedding_nn(self):
        device  = self.device
        self.embedding_net = SceneEmbeddingNetwork(object_feature_dim=518).to(device)
        self.embedding_net.to(self.device)
        self.embedding_net.load_state_dict(torch.load(asdict(self.config).get('load_emb_nn', None), map_location=device))
        if (not self.eval and not self.evalp) or self.learn_emb:
            self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)

    def get_graph_embedding(self):
        with open(general_path + '/img/goal_bowl.png', "rb") as file:
            objects = pickle.load(file)

        # Read bbox info
        bbox_path = general_path + "/scene/scene_test/0"
        with open(bbox_path, "r") as flie:
            bbox_pose = json.load(flie)

        features = []

        current_bowl_pose = self.poses_bowl[self.num_of_envs]
        
        id_order = [7, 0, 1, 2, 8]
        for target_id in id_order:
            for item in bbox_pose:
                if item["id"] == target_id:
                    clip_descriptor = objects[str(target_id)]
                    bbox_extent = np.array([item['size']["x"], item['size']["y"], item['size']["z"]])
                    if target_id == 8:
                        bbox_center = self.poses_bowl[self.num_of_envs]
                    else:
                        bbox_center = np.array([item['center']['position']["x"], item['center']['position']["y"], item['center']['position']["z"]])
                    object_feature = np.concatenate([clip_descriptor, bbox_extent, bbox_center])  # (512 + 3 + 3 = 518)
                    features.append(object_feature)

        # to tensor (num_object, 518)
        object_features = torch.tensor(features, dtype=torch.float32).to(self.device)  # (7, 518)

        return object_features
