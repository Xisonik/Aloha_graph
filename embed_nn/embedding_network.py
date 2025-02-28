import torch
import torch.nn as nn

class SceneEmbeddingNetwork(nn.Module):
    def __init__(self, object_feature_dim=518, object_embedding_dim=128, scene_embedding_dim=32):
        """
        SceneEmbeddingNetwork processes features of multiple objects in a scene 
        and generates a single embedding for the entire scene.

        Args:
        object_feature_dim (int): Dimensionality of input features for each object.
        object_embedding_dim (int): Dimensionality of the intermediate object embeddings.
        scene_embedding_dim (int): Dimensionality of the final scene embedding.
        """
        super(SceneEmbeddingNetwork, self).__init__()
        # input_dim = 512(Clip-embedding) + 3 bbox extent(width,high, length) + 3 bbox center(x,y,z) = 518
        # output_dim = 32
        # Object-level feature encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(object_feature_dim, 256),  # First linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(256, object_embedding_dim)  # Second linear layer
        )
        
        # Scene-level feature encoder
        self.scene_encoder = nn.Sequential(
            nn.Linear(object_embedding_dim, 64),  # First linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(64, scene_embedding_dim)  # Second linear layer
        )

    def forward(self, object_features):

        # 1: Encode individual object features
        object_embeddings = self.object_encoder(object_features)  # Shape: (num_objects, object_embedding_dim) --> 4*512
        
        # 2: Aggregate embeddings of all objects in the scene
        # Here, we use mean pooling. Alternative methods like max pooling or attention can also be used.
        aggregated_embedding = torch.mean(object_embeddings, dim=0)  # Shape: (object_embedding_dim)
        
        # 3: Encode the aggregated feature into the final scene embedding
        scene_embedding = self.scene_encoder(aggregated_embedding)  # Shape: (scene_embedding_dim)
        
        return scene_embedding
