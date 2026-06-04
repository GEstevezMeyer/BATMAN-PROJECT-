from qdrant_client.models import PointStruct
from tqdm import tqdm






def ingest(client,collection_name,total_ids,total_embeddings,total_labels):
    n = len(total_labels)

    points = [PointStruct(id=i, vector=total_embeddings[i], payload={"role": int(total_labels[i]),
                                                                     "finger_name": str(total_ids[i])}) 
     for i in tqdm(range(n), desc="Loading Points")]
    
    client.upsert(collection_name=collection_name,wait = True,points=points)








