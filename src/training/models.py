import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class SOCOFIngEncoder(nn.Module): 
    def __init__(self,dimention,filters_out = [16,32,64],kernel_size = [3,3,3],dense_out = [64,128]):
        super().__init__()

        self.filters_in = 3
        self.dense_in = None

        self.dense_out = dense_out
        self.filters_out = filters_out
        self.kernel_size = kernel_size

        self.convLayers = self._create_conv_layers()
        self.flatten = nn.Flatten()
        self.denseLayers = None
        self.encoder = nn.Linear(self.dense_out[-1],dimention)

    def _create_conv_layers(self):
        layers = []
        for i in range(len(self.filters_out)):
            padding = self.kernel_size[i]//2
            layers.append(nn.Conv2d(self.filters_in,self.filters_out[i],
                                    kernel_size=self.kernel_size[i],padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(3))
            self.filters_in = self.filters_out[i]


        return nn.Sequential(*layers)
    
    def _create_dense_layers(self): 
        layers = []
        for i in range(len(self.dense_out)):
            layers.append(nn.Linear(self.dense_in,self.dense_out[i]))
            layers.append(nn.ReLU())
            self.dense_in = self.dense_out[i]

        return nn.Sequential(*layers)
    
    def forward(self,input): 
        x = self.convLayers(input)
        x = self.flatten(x)

        if self.denseLayers is None: 
            self.dense_in = x.shape[1]
            self.denseLayers = self._create_dense_layers()
            self.denseLayers = self.denseLayers.to("cuda")

        x = self.denseLayers(x)
        x = self.encoder(x)

        x = nn.functional.normalize(x, p=2, dim=1)
        
        return x   
    

class EfficientNetEncoder(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1280
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
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
    