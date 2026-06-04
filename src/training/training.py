import torch 
from pytorch_metric_learning import miners, losses, distances

from tqdm import tqdm
import numpy as np 
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


from src.training.pipeline import main_pipeline
from src.utils.config import named_action , import_config, return_device, dump_history,save_model_weights
from src.training.models import *



@named_action
def create_embedding_model(config_path = "config.toml"):

    config = import_config(config_path)
    config_architecture = config["architecture"]
    config_model = config["model"]

    if config_architecture["encoder"] == "soco":
        encoder = SOCOFIngEncoder(**config_model)
    elif config_architecture["encoder"] == "efn":
        encoder = EfficientNetEncoder(config_model["dimension"])
    
    

    return encoder




def training_epoch_embedding(model, dataloader, optimizer, loss_function,miner, device="cuda"):

    model.train()
    
    total_loss = 0.0
    normL2 = []
    total_embeddings = []
    total_labels = []

    for images,labels in tqdm(dataloader):

        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        hard_pairs = miner(embeddings, labels)


        loss = loss_function(embeddings, labels, hard_pairs)
    
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        normL2.append(loss.item()**2)

        

    normL2 = np.array(normL2)
    normL2 = np.sqrt(np.sum(normL2)).item()

    mean_loss = total_loss/len(dataloader)
    total_loss = round(total_loss,3)


    return total_loss,normL2,mean_loss


def validation(encoder,dataloader,dataloader_validation = None,device = "cuda"):
    encoder.eval()

    total_embeddings = []
    total_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc= "Transforming training dataset into embeddings"):
            images = images.to(device)
            
            embeddings = encoder(images)
            total_embeddings.extend(embeddings.detach().cpu().numpy())
            total_labels.extend(labels)

    total_embeddings = np.array(total_embeddings)
    total_labels = np.array(total_labels)

    silhouette = silhouette_score(total_embeddings,total_labels)
    tolerance = compute_tolerance_metric(total_embeddings,total_labels)
    recall = compute_recall(encoder,total_embeddings,dataloader_validation,total_labels,device=device)

    return silhouette,tolerance,recall


def compute_tolerance_metric(total_embeddings,total_labels,treshold_errors = 4): 

    nn = NearestNeighbors(n_neighbors= 11)
    nn.fit(total_embeddings)
    
    distances, indices = nn.kneighbors(total_embeddings)
    correct = 0
    n = len(total_embeddings)

    for i in tqdm(range(n),desc="Computing tolerance metric"): 
        
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
        for query,query_labels in tqdm(dataloader_validation,desc= "Computing recall metric"):
            query = query.to(device)

            query_embeddings = encoder(query)
            query_embeddings = (query_embeddings.detach().cpu().numpy())
            
            neighbors,indices = nn.kneighbors(query_embeddings)
            n = len(indices)
            
            for i in range(n): 
                if total_labels[indices[i][0]] == query_labels[i]:
                    correct+= 1

            total_n+= n

    
    return round(correct/total_n,3)

    

def training_model(model, dataloader,dataloader_evaluation,dataloader_validation, optimizer, 
                   loss_function,miner, device="cuda", epochs=10):

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
            loss_function,miner,
            device
        )

        
        print("------------Training---------------")
        silhouette,tolerance,recall_validation= validation(model,dataloader_evaluation,
                                                           dataloader_validation,device=device)

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

   

def main_training(data_path = "DATA/SOCOFing/Real", config_path = "config.toml",
                  res_path = "res/history.json",epochs = 10):
    device = return_device()
    dataloader,dataloader_evaluation,dataloader_validation = main_pipeline(data_path=data_path)
    model = create_embedding_model(config_path=config_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   
    distance = distances.CosineSimilarity()  
    loss_function = losses.TripletMarginLoss(margin=0.2, distance=distance)
    miner = miners.TripletMarginMiner(
        margin=0.2,
        distance=distance,
        type_of_triplets="semihard"  
    ) 
    
    trained_model,history = training_model(model,dataloader,dataloader_evaluation,dataloader_validation,
                                           optimizer,loss_function,miner,
                                           epochs=epochs,device=device)

    
    dump_history(history,res_path=res_path)
    save_model_weights(trained_model)
    

    return 0





if __name__ == "__main__": 
    print(main_training(epochs=15)) 
