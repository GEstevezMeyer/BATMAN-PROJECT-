import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.training.pipeline import main_pipeline
from src.utils.config import named_action , import_config

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
    




@named_action
def create_embedding_model(config_path = "config.toml"):
    config_model = import_config(config_path)["model"]
    encoder = SOCOFIngEncoder(**config_model)
    
    embedding_model = SOCOFingEmbedding(encoder)

    return embedding_model


def training_epoch_embedding(model, dataloader, optimizer, loss_function, device="cuda"):

    model.train()
    total_loss = 0.0
    total_embeddings = []
    total_labels = []

    for anchor, positive, negative, labels in tqdm(dataloader):

        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        anchor_label, positive_label, negative_label = labels 

        loss = loss_function(anchor_emb,positive_emb,negative_emb)
    
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_embeddings.extend(anchor_emb.detach().cpu().numpy())
        total_embeddings.extend(positive_emb.detach().cpu().numpy())
        total_embeddings.extend(negative_emb.detach().cpu().numpy())

        total_labels.extend(anchor_label)
        total_labels.extend(positive_label)
        total_labels.extend(negative_label)

    total_embeddings = np.array(total_embeddings)
    total_labels = np.array(total_labels)

    silhouette = silhouette_score(total_embeddings,total_labels)
    tolerance = compute_tolerance_metric(total_embeddings,total_labels)

    return total_loss,silhouette,tolerance

def compute_tolerance_metric(total_embeddings,total_labels,treshold_errors = 7): 
    nn = NearestNeighbors(n_neighbors= 11)
    nn.fit(total_embeddings)
    
    distances, indices = nn.kneighbors(total_embeddings)
    correct = 0
    n = len(total_embeddings)

    for i in range(n): 
        
        nearest_neighbor = []
        success = 1
        current_errors = 0
        j = 0

        while (j <= 10) and success: 

            nearest_neighbor = indices[i][j]

            if total_labels[i] != total_labels[nearest_neighbor]:
                current_errors+= 1
            
            if current_errors >= treshold_errors:
                success = 0

            j+= 1
        
        correct+= success 
               
    tolerance = correct/ n

    return tolerance
    

def training_model(model, dataloader, optimizer, loss_function, device="cuda", epochs=10):

    model = model.to(device)
    

    for i in range(epochs):

        print("-----------------")
        print(f"Epochs: {i}")

        total_loss, silhouette, tolerance = training_epoch_embedding(
            model,
            dataloader,
            optimizer,
            loss_function,
            device
        )

        print(f"Total_loss: {total_loss} | silhouette score: {silhouette}| tolerance: {tolerance}")
        print("-----------------")

    return model


def main_training(data_path = "DATA/SOCOFing/Real", config_path = "config.toml",epochs = 10):
    dataloader = main_pipeline(data_path=data_path)
    model = create_embedding_model(config_path=config_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_function = nn.TripletMarginLoss(margin=1.0,p=2)
    trained_model = training_model(model,dataloader,optimizer,loss_function,epochs=epochs)
    

    return 0



if __name__ == "__main__": 
    print(main_training()) 
