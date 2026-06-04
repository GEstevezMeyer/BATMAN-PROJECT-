from tqdm import tqdm 
import numpy as np 
import torch
from src.utils.config import return_device


def transform_data(encoder,dataloader):
    device = return_device()

    total_embeddings = []
    total_labels = []

    encoder.to(device)
    encoder.eval()

    with torch.no_grad():

        for images , labels in tqdm(dataloader,desc= "transforming images to embeddings"): 
            
            images = images.to(device)
            embeddings = encoder(images)

            total_embeddings.extend(embeddings.detach().cpu().numpy())
            total_labels.extend(labels)

        
        total_embeddings = np.array(total_embeddings)
        total_labels = np.array(total_labels)

        return total_embeddings, total_labels
    
def transform_loading_data(encoder,dataloader):
    device = return_device()

    total_ids = []
    total_embeddings = []
    total_labels = []

    encoder.to(device)
    encoder.eval()

    with torch.no_grad():

        for ids,images , labels in tqdm(dataloader,desc= "transforming images to embeddings"): 
            
            images = images.to(device)
            embeddings = encoder(images)

            total_embeddings.extend(embeddings.detach().cpu().tolist())
            total_labels.extend(labels)
            total_ids.extend(ids)


        return total_ids,total_embeddings, total_labels
    

def transform_image(encoder,image):
    device = return_device()

    encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        image = image.to(device)
        embedding = encoder(image)

    return embedding


if __name__ == "__main__": 

    from src.utils.config import load_model
    from src.training.models import EfficientNetEncoder
    from src.etl.load import load_data

    encoder = load_model(EfficientNetEncoder())
    dataloader = load_data()

    print(transform_loading_data(encoder,dataloader))

