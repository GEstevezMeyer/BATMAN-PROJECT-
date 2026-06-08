from django.shortcuts import render

from models_ai.architecture import EfficientNetEncoder,load_model
from models_ai.inference import read_image,transform_image
from function.plots import plot_qdrant_embeddings
from function.utils import connect_qdrant, connect_sql,import_vectors,knn_search,fetch_criminal

import os 




# Create your views here

def home(request): 

    encoder = load_model()
    client = connect_qdrant()
    link = connect_sql()

    graph_html = None
    inference = None

    if client is None:
        return render(
        request,
        "home.html",
        {"graph": graph_html}
    )

    embeddings, labels = import_vectors(client)

    if request.method == "POST":
        images = request.FILES.getlist('images')
        if images:
            images = read_image(images[0])
            query_vector = transform_image(encoder,images)
            query_label = knn_search(client,query_vector).payload["role"]
            graph_html = plot_qdrant_embeddings(embeddings,labels,highlight_label=query_label)
            criminal_metadata = fetch_criminal(link,query_label)

            return render(
                request,
                "home_analyze.html",
                    {"graph": graph_html,
                    "criminal_metadata":criminal_metadata}
                )
        else:
            graph_html = plot_qdrant_embeddings(embeddings,labels)
    else: 
        graph_html = plot_qdrant_embeddings(embeddings,labels)
    
    return render(
        request,
        "home.html",
        {"graph": graph_html}
    )
    




