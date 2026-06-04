from src.utils.config import load_model,import_config
from src.training.models import EfficientNetEncoder

from src.etl.transform import transform_loading_data
from src.etl.load import load_data
from src.etl.ingest import ingest

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

import shutil





def main(collection_name,data_path = "DATA/SOCOFing/Real",config_path = "config.toml",
         model_weigth_path = "res/model.pth",embeddings_database = "./embeddings_database"):
    
    config = import_config(config_path)

    config_model = config["model"]
    encoder = load_model(EfficientNetEncoder(embedding_dim=config_model["dimension"]),
                         model_weights_path=model_weigth_path)
    dataloader = load_data(data_path=data_path,config_path=config_path)

    total_ids, total_embeddings, total_labels = transform_loading_data(encoder,dataloader)

    client = QdrantClient(path=embeddings_database)
    
    client.create_collection(collection_name=collection_name,
                             vectors_config=VectorParams(size = config_model["dimension"], distance= Distance.COSINE))
    
    ingest(client,collection_name,total_ids,total_embeddings,total_labels)

    return 0

    


# TODO create the relation of the labels with the actual name of the "subjects" and make test of the database

if __name__ == "__main__": 
    shutil.rmtree("./embeddings_database")
    print(main("test1"))