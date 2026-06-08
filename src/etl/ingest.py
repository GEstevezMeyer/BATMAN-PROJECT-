from qdrant_client.models import PointStruct
from tqdm import tqdm
from src.utils.config import create_loading_metadata
import numpy as np
import mysql.connector 






def ingest(client,collection_name,total_ids,total_embeddings,total_labels):
    n = len(total_labels)

    points = [PointStruct(id=i, vector=total_embeddings[i], payload={"role": int(total_labels[i]),
                                                                     "finger_name": str(total_ids[i])}) 
     for i in tqdm(range(n), desc="Loading Points")]
    
    client.upsert(collection_name=collection_name,wait = True,points=points)




def ingest_db(cursor):
    metadata = create_loading_metadata("DATA/SOCOFing/Real")
    ids = metadata["labels"].unique().tolist()

    n = len(ids)

    name = ids
    danger_level = ["mid" if level < 10 else "high"  for level in np.random.uniform(5,20,size=n).tolist()]
    last_known_location = ["unknown" if i < n//2 else "Gotham" for i in range(n)]
    status = ["Arkham" if level < 10 else "Free"  for level in np.random.uniform(5,20,size=n).tolist()]

    for i in range(n):
        criminals = (ids[i],name[i],last_known_location[i],danger_level[i],status[i])
        cursor.execute(
            "INSERT INTO criminals (id,name, danger_level,last_known_location,status) VALUES (%s, %s, %s, %s,%s)",
            criminals)   
        

    

    











