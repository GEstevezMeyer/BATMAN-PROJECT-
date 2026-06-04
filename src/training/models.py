import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class SOCOFIngEncoder(nn.Module): 
    def __init__(self, dimension, filters_out=[16,32,64], kernel_size=[3,3,3], dense_out=[64,128]):
        super().__init__()
        self.convLayers = self._create_conv_layers(filters_out, kernel_size)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            flat_dim = self.flatten(self.convLayers(dummy)).shape[1]
        
        self.denseLayers = self._create_dense_layers(flat_dim, dense_out)
        self.encoder = nn.Linear(dense_out[-1], dimension)

    def _create_conv_layers(self, filters_out, kernel_size):
        layers = []
        filters_in = 3
        for f_out, k in zip(filters_out, kernel_size):
            padding = k // 2
            layers += [
                nn.Conv2d(filters_in, f_out, kernel_size=k, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(3)
            ]
            filters_in = f_out
        return nn.Sequential(*layers)
    
    def _create_dense_layers(self, flat_dim, dense_out): 
        layers = []
        in_dim = flat_dim
        for out_dim in dense_out:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        return nn.Sequential(*layers)
    
    def forward(self, x): 
        x = self.convLayers(x)
        x = self.flatten(x)
        x = self.denseLayers(x)
        x = self.encoder(x)
        return nn.functional.normalize(x, p=2, dim=1) 
    

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

    
class SOCOFingEmbedding(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor, positive, negative):

        u = self.encoder(anchor)
        v = self.encoder(positive)
        w = self.encoder(negative)

        return u, v, w
    