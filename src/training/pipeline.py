import pandas as pd 
import numpy as np
from PIL import Image 
import os
from src.utils.config import import_config, create_metadata,named_action


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



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

    
    
    def _read_image(self,path:str): 

        image = Image.open(path).convert("RGB")
        image = self.features_function(image)
        image = self.transform(image)

        return image 
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):

        anchor = self.metadata.iloc[index]

        sub_positive_df = self.metadata[
            (self.metadata["label"] == anchor["label"]) &
            (self.metadata["path"] != anchor["path"])
        ]
        sub_negative_df = self.metadata[self.metadata["label"] != anchor["label"]]

        n_positive = len(sub_positive_df)
        n_negative = len(sub_negative_df)

        positive = sub_positive_df.iloc[np.random.randint(0,n_positive)]
        negative = sub_negative_df.iloc[np.random.randint(0,n_negative)]

        anchor_label = anchor["label"]
        positive_label = anchor["label"]
        negative_label = negative["label"]

        anchor = self._read_image(anchor["path"])
        positive = self._read_image(positive["path"])
        negative = self._read_image(negative["path"])

        labels = (anchor_label,positive_label,negative_label)

    

        return anchor,positive,negative,labels
    

def create_DataLoader(dataset:Dataset,config_path:str = "config.toml"): 
    config = import_config(config_path)["training"]
    dataLoader = DataLoader(dataset,config["batch_size"],shuffle = True)

    return dataLoader


@named_action 
def main_pipeline(data_path:str = "DATA/SOCOFing/Real"):

    dataset = SOCOFingDataset(data_path)
    dataloader = create_DataLoader(dataset)

    return dataloader



if __name__ == "__main__": 

    main_pipeline()