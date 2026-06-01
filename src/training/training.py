import torch 
import torch.nn as nn

from tqdm import tqdm
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os 

from src.training.pipeline import main_pipeline
from src.utils.config import named_action , import_config, return_device
from src.utils.math_utils import normalize_vectors
from src.training.models import *


@named_action
def create_embedding_model(config_path = "config.toml"):

    config = import_config(config_path)
    config_architecture = config["architecture"]
    config_model = config["model"]

    if config_architecture["encoder"] == "soco":
        encoder = SOCOFIngEncoder(**config_model)
    elif config_architecture["encoder"] == "efn":
        encoder = EfficientNetEncoder(config_model["dimention"])
    
    embedding_model = SOCOFingEmbedding(encoder)

    return embedding_model




def training_epoch_embedding(model, dataloader, optimizer, loss_function, device="cuda"):

    model.train()
    
    total_loss = 0.0
    normL2 = []
    total_embeddings = []
    total_labels = []

    for anchor, positive, negative, labels in tqdm(dataloader):

        optimizer.zero_grad()

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        
        loss = loss_function(anchor_emb,positive_emb,negative_emb)
    
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        normL2.append(loss.item()**2)

        

    normL2 = np.array(normL2)
    normL2 = np.sqrt(np.sum(normL2)).item()

    mean_loss = total_loss/len(dataloader)
    total_loss = round(total_loss,3)


    return total_loss,normL2,mean_loss


def validation(model,dataloader,dataloader_validation = None,device = "cuda"):
    model.eval()
    encoder = model.encoder 

    total_embeddings = []
    total_labels = []

    with torch.no_grad():
        for anchor, _, _, labels in tqdm(dataloader):
            anchor = anchor.to(device)

            outputs = encoder(anchor)
            total_embeddings.extend(outputs.detach().cpu().numpy())

            anchor_label,_, _ = labels 
            total_labels.extend(anchor_label)

    total_embeddings = np.array(total_embeddings)
    total_labels = np.array(total_labels)

    silhouette = silhouette_score(total_embeddings,total_labels)
    tolerance = compute_tolerance_metric(total_embeddings,total_labels)
    recall = compute_recall(encoder,total_embeddings,dataloader_validation,total_labels,device=device)

    return silhouette,tolerance,recall


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

 

def compute_recall(encoder,total_embeddings,dataloader_validation,total_labels,device = "cuda"):
    
    nn = NearestNeighbors(n_neighbors= 1)
    nn.fit(total_embeddings)
    
    correct = 0
    total_n = 0

    with torch.no_grad():
        for query, _, _, labels in dataloader_validation:
            query = query.to(device)

            query_embeddings = encoder(query)
            query_embeddings = (query_embeddings.detach().cpu().numpy())
            
            query_labels,_,_ = labels 
            query_labels = np.array(query_labels)

            neighbors,indices = nn.kneighbors(query_embeddings)
            
            n = len(indices)
            
            for i in range(n): 
                if total_labels[indices[i][0]] == query_labels[i]:
                    correct+= 1

            total_n+= n

    
    return round(correct/total_n,3)

    

def training_model(model, dataloader,dataloader_validation, optimizer, loss_function, schedulers = None, device="cuda", epochs=10):

    model = model.to(device)
    
    training_total_loss = []
    training_normL2 = []
    training_mean_loss = []
    training_silhouette = []
    training_tolerance = []

    validation_recall = []
    
    for i in range(epochs):

        print("----------------------------")
        print(f"Epochs: {i}")


        total_loss,normL2,mean_loss = training_epoch_embedding(
            model,
            dataloader,
            optimizer,
            loss_function,
            device
        )

        if not(schedulers is None): 
            schedulers.step()
        print("------------Training---------------")
        silhouette,tolerance,recall_validation= validation(model,dataloader,dataloader_validation,device=device)

        print(f"Total_loss: {total_loss} | normL2: {normL2}| mean_loss: {mean_loss}")
        print(f"silhouette score: {silhouette}| tolerance: {tolerance}")

        print("------------Validation---------------")
        print(f" recall: {recall_validation}")
        print("---------------------------")

        training_total_loss.append(total_loss)
        training_normL2.append(normL2)
        training_mean_loss.append(mean_loss)
        training_silhouette.append(silhouette)
        training_tolerance.append(tolerance)
        
        validation_recall.append(recall_validation)

        history_validation = {
            "recall": validation_recall
        }

        history_training = {
            "total_loss": training_total_loss,
            "normL2": training_normL2,
            "mean_loss": training_mean_loss,
            "silhouette": training_silhouette,
            "tolerance": training_tolerance,
        }

        history = {
            "training": history_training,
            "validation": history_validation
        }

    return model,history

   

def main_training(data_path = "DATA/SOCOFing/Real", config_path = "config.toml",epochs = 10,scheduler = True):
    device = return_device()
    dataloader, dataloader_validation = main_pipeline(data_path=data_path)
    model = create_embedding_model(config_path=config_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None

    if scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma= 0.5)
    

    loss_function = nn.TripletMarginLoss(margin=1.0,p=2)
    trained_model,history = training_model(model,dataloader,dataloader_validation,optimizer,loss_function,
                                   schedulers=scheduler,epochs=epochs,device=device)

    
    return history


#TODO try more metrics and organize this code 


if __name__ == "__main__": 
    print(main_training(scheduler=False,epochs=10)) 
