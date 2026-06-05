from src.utils.config import create_loading_metadata,import_config
from src.training.pipeline import SOCOFingDataset,create_ValidationDataLoader
from torchvision import transforms
from qdrant_client import QdrantClient

class LoadingDataset(SOCOFingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self._metadata = create_loading_metadata(args[0])  

       
        self.paths = self._metadata["path"].tolist()
        self.labels = self._metadata["labels"].astype("int").tolist()
        self.ids = self._metadata["ids"].tolist()

    def __getitem__(self, index):

        path = self.paths[index]
        label = self.labels[index]
        ids = self.ids[index]

        image = self._read_image(path)
        
        return ids,image,label


def load_data(data_path:str = "DATA/SOCOFing/Real",config_path:str = "config.toml"):
    config = import_config(config_path)
    config_dataset = config["dataset"]

    dataset = LoadingDataset(data_path,mean=config_dataset["mean"],std= config_dataset["std"])
    dataloader = create_ValidationDataLoader(dataset=dataset)

    return dataloader

def read_image(image,config_path = "config.toml"):

    config = import_config(config_path)
    config_dataset = config["dataset"]

    transform = transforms.Compose([transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config_dataset["mean"],std= config_dataset["std"])
        ])

    return transform(image)



if __name__ == "__main__": 
    print(load_data())

    












    

    

