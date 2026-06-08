import os 
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetEncoder(nn.Module):

    def __init__(self, embedding_dim=224):
        super().__init__()
        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1280
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.embedding(x)

        x = nn.functional.normalize(x, p=2, dim=1)

        return x
    

def load_model(architecture = EfficientNetEncoder(embedding_dim= 224)):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(base_dir, "model.pth")
     
    architecture.load_state_dict(torch.load(model_weights_path,weights_only= True))

    return architecture