import pandas as pd 
import numpy as np
from PIL import Image 
import os
from src.utils.config import import_config, create_metadata,named_action


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_metric_learning import samplers



class SOCOFingDataset(Dataset): 
    def __init__(self,data_path,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], 
                features_function = lambda x: x, augmentation_transform = None): 
        self.metadata = create_metadata(data_path)

        if augmentation_transform is None: 
            augmentation_transform = []

        self.transform = transforms.Compose([transforms.Resize((224,224)),*augmentation_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)
        ])
        self.features_function = features_function

        self.paths = self.metadata["path"].tolist()
        self.labels = self.metadata["labels"].astype("int").tolist()

    def return_labels(self): 
        return self.labels
    
    def _read_image(self,path:str): 

        image = Image.open(path).convert("RGB")
        image = self.features_function(image)
        image = self.transform(image)

        return image 
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):

        path = self.paths[index]
        label = self.labels[index]

        image = self._read_image(path)
        
        return image,label
    
    
def create_TrainingDataLoader(dataset:Dataset,config_path:str = "config.toml"): 
    config = import_config(config_path)["training"]
    sampler = samplers.MPerClassSampler(dataset.labels, m=4, batch_size=32)
    dataLoader = DataLoader(dataset,config["batch_size"],sampler=sampler)

    return dataLoader

def create_ValidationDataLoader(dataset:Dataset):
    return DataLoader(dataset,64,shuffle=True)


@named_action 
def main_pipeline(data_path:str = "DATA/SOCOFing/Real",validation_data_path = "DATA/SOCOFing/Altered/Altered-Medium"):
    dataset = SOCOFingDataset(data_path)
    dataset_validation = SOCOFingDataset(validation_data_path)
    
    labels = dataset.labels
    dataloader = create_TrainingDataLoader(dataset)
    dataloader_evaluation = create_ValidationDataLoader(dataset)
    dataloader_validation = create_ValidationDataLoader(dataset_validation)
    
    

    return dataloader,dataloader_evaluation,dataloader_validation,labels 



if __name__ == "__main__": 

    main_pipeline()