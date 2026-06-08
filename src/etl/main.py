from src.utils.config import load_model,import_config
from src.training.models import EfficientNetEncoder

from src.etl.transform import transform_loading_data
from src.etl.load import load_data
from src.etl.ingest import ingest,ingest_db

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from dotenv import load_dotenv
import mysql.connector
import os 




def main(collection_name,data_path = "DATA/SOCOFing/Real",config_path = "config.toml",
         model_weigth_path = "res/model.pth",embeddings_database = "./embeddings_database"):
    
    load_dotenv()
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")

    config = import_config(config_path)
    config_model = config["model"]
    encoder = load_model(EfficientNetEncoder(embedding_dim=config_model["dimension"]),
                         model_weights_path=model_weigth_path)
    dataloader = load_data(data_path=data_path,config_path=config_path)

    total_ids, total_embeddings, total_labels = transform_loading_data(encoder,dataloader)

    client = QdrantClient(host=HOST,port =PORT)
    
    client.create_collection(collection_name=collection_name,
                             vectors_config=VectorParams(size = config_model["dimension"], distance= Distance.COSINE))
    
    ingest(client,collection_name,total_ids,total_embeddings,total_labels)

    return 0

def main_sql(data_path = "DATA/SOCOFing/Real"):
    load_dotenv()
    HOST_DB = os.getenv("HOST_DB")
    PORT_DB = os.getenv("PORT_DB")
    USER = os.getenv("USER")
    PASSWORD = os.getenv("PASSWORD")
    DATABASE = os.getenv("DATABASE")

    config = {
        "user": USER,
        "password": PASSWORD,
        "port": int(PORT_DB),
        "host": HOST_DB,
        "database": DATABASE
    }

    link = mysql.connector.connect(**config)
    cursor = link.cursor()
    ingest_db(cursor)
    link.commit()
    
    return 0

    



if __name__ == "__main__": 
    print(main_sql())