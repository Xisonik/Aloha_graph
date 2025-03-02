import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
d = Path().resolve()  # .parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"

import random
from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim
from dataclasses import asdict, dataclass
from configs.main_config import MainConfig
import torch
import os
import gzip
import pickle
import argparse
import json
import open_clip

class Graph_manager():
    def __init__(self, config=MainConfig()):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval = asdict(config).get('eval', None)
        self.evalp = asdict(config).get('eval_print', None)
        self.learn_emb = False
        self.init_embedding_nn()

    def init_embedding_nn(self):
        device = self.device
        self.embedding_net = SceneEmbeddingNetwork(object_feature_dim=518).to(device)
        self.embedding_net.to(self.device)
        self.embedding_net.load_state_dict(torch.load(asdict(self.config).get('load_emb_nn', None), map_location=device))
        if (not self.eval and not self.evalp) or self.learn_emb:
            self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)

    # Load open_clip model and processor
    def load_clip_model(self):
        # Choose model and weights
        model_name = "EVA02-B-16"
        pretrained = "merged2b_s8b_b131k"

        # Load model and processor
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
        clip_model = clip_model.to("cuda")  # Move model to GPU
        clip_tokenizer = open_clip.get_tokenizer(model_name)

        return clip_model, clip_tokenizer

    # Generate CLIP embedding for text description
    def get_clip_embedding(self, text, model, tokenizer):
        # Tokenize and encode text
        text_tokens = tokenizer(text).to("cuda")  # Move input to GPU
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)  # Generate text embedding
            text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
        return text_features.squeeze(0).cpu()  # Return 512-dimensional vector and move to CPU

    # Read JSON file and generate feature tensor
    def json_to_tensor(self, json_file, model, tokenizer):
        # Read JSON file
        with open(json_file, "r") as f:
            data = json.load(f)

        # Initialize list to store features of all objects
        object_features = []

        # Iterate over each object
        for obj_key, obj_data in data.items():
            # Extract bbox_extent, bbox_center, bbox_volume
            bbox_extent = torch.tensor(obj_data["bbox_extent"], dtype=torch.float32)  # 3 dimensions
            bbox_center = torch.tensor(obj_data["bbox_center"], dtype=torch.float32)  # 3 dimensions
            bbox_volume = torch.tensor([obj_data["bbox_volume"]], dtype=torch.float32)  # 1 dimension

            # Generate CLIP embedding for text description
            object_tag = obj_data["object_tag"]
            text_embedding = self.get_clip_embedding(object_tag, model, tokenizer)  # 512 dimensions

            # Concatenate all features into a 518-dimensional vector
            features = torch.cat([bbox_extent, bbox_center, text_embedding], dim=0)  # 518 dimensions
            object_features.append(features)

        # Stack features of all objects into a tensor
        object_tensor = torch.stack(object_features)  # Shape: (num_objects, 518)
        return object_tensor

    # Calculate similarity between each object and target (e.g., "bowl")
    def calculate_similarity_to_target(self, object_tensor, target_text, clip_model, clip_tokenizer):
        # Get CLIP embedding for target text (e.g., "bowl")
        target_embedding = self.get_clip_embedding(target_text, clip_model, clip_tokenizer)  # 512 dimensions

        # Extract CLIP embeddings from object_tensor (last 512 dimensions)
        object_clip_embeddings = object_tensor[:, -512:]  # Shape: (num_objects, 512)

        # Calculate cosine similarity between each object and target
        similarities = torch.nn.functional.cosine_similarity(
            object_clip_embeddings, target_embedding.unsqueeze(0), dim=1
        )  # Shape: (num_objects,)

        return similarities

    # Weighted pooling of object features
    def weighted_pooling(self, object_tensor, weights):
        # Normalize weights to sum to 1
        weights = weights / torch.sum(weights)  # Shape: (num_objects,)

        # Expand weights to match the feature dimension
        weights = weights.unsqueeze(1)  # Shape: (num_objects, 1)

        # Weighted sum of object features
        pooled_vector = torch.sum(object_tensor * weights, dim=0)  # Shape: (518,)
        return pooled_vector

    def get_graph_embedding(self, num_of_envs=0):
        # Load open_clip model
        clip_model, clip_tokenizer = self.load_clip_model()

        # JSON file path
        json_file = general_path + "/scene_graph_2_26/" + str(num_of_envs + 1) + ".json"  # Replace with your JSON file path

        # Generate feature tensor
        object_tensor = self.json_to_tensor(json_file, clip_model, clip_tokenizer)

        # Calculate similarity to target (e.g., "bowl")
        target_text = "bowl"  # Replace with your target object
        similarities = self.calculate_similarity_to_target(object_tensor, target_text, clip_model, clip_tokenizer)

        # Weighted pooling of object features
        pooled_vector = self.weighted_pooling(object_tensor, similarities)

        # Print results
        # print("Object tensor shape:", object_tensor.shape)  # Output: (num_objects, 518)
        # print("Similarities to target ('{}'):".format(target_text), similarities)
        # print("Pooled vector shape:", pooled_vector.shape)  # Output: (518,)
        # print("Pooled vector:", pooled_vector)

        return pooled_vector
