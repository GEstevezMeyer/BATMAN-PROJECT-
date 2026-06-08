import os
import httpx
import docker 
import mysql.connector

from dotenv import load_dotenv 
from docker.errors import NotFound,APIError
from qdrant_client import QdrantClient

def check_container(container_id:str) -> bool:
    
    client = docker.from_env()
    
    try: 
        container = client.containers.get(container_id= container_id)
        if container.status != "running":
            container.start()

        return True

    except NotFound:
        return False
    except APIError as e:
        return False
    
def connect_sql():
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

    return link


def connect_qdrant() -> QdrantClient:
    load_dotenv()
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")
    QDRANT_CONTAINER = os.getenv("QDRANT_CONTAINER")

    containerFlag = check_container(QDRANT_CONTAINER)

    if not containerFlag: return None

    try:
        client = QdrantClient(host=HOST,port= PORT)
        return client
    except httpx.ConnectError:
        return None
    

def knn_search(client, query_vector, k=1, collection_name="test1"):
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,         
        with_payload=True
    )

    return results.points[0]

def import_vectors(client):
    points, offset = client.scroll(
        collection_name="test1",
        with_vectors=True, 
        limit=10000
    )
    
    embeddings = []
    labels = []
    
    for point in points:
        embeddings.append(point.vector)       
        labels.append(point.payload["role"])   
    
    return embeddings, labels



def fetch_criminal(link,criminal_id:int) -> list: 
    with link.cursor() as cursor:
        cursor.execute("SELECT * FROM criminals WHERE criminals.id = %s",(int(criminal_id),))
        row = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]

    criminal_metadata = dict(zip(columns, row))
    
    return criminal_metadata
