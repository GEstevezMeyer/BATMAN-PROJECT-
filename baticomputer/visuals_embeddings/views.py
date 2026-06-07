from django.shortcuts import render
from qdrant_client import QdrantClient
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from dotenv import load_dotenv
import docker 
from docker.errors import NotFound,APIError
from .models_ai.architecture import EfficientNetEncoder,load_model
from .models_ai.inference import read_image,transform_image

import httpx
import os 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models_ai', 'model.pth')

architecture = EfficientNetEncoder(embedding_dim=224)
encoder = load_model(architecture, MODEL_PATH) 

# Create your views here.

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

def plot_qdrant_embeddings(embeddings, labels, method=PCA, highlight_label=None):
    import numpy as np
    
    visualizer = method(n_components=3)
    embeddings = visualizer.fit_transform(embeddings)

    labels = np.array(labels)

   
    mask_highlight = labels == highlight_label
    mask_normal    = labels != highlight_label

    traces = []

    
    traces.append(go.Scatter3d(
        x=embeddings[mask_normal, 0],
        y=embeddings[mask_normal, 1],
        z=embeddings[mask_normal, 2],
        mode='markers',
        name='embeddings',
        marker=dict(
            size=4,
            color=labels[mask_normal],
            colorscale=[
                [0.0,  '#000d1a'],
                [0.25, '#003366'],
                [0.5,  '#0055aa'],
                [0.75, '#00aaff'],
                [1.0,  '#00eeff'],
            ],
            showscale=True,
            opacity=0.9,
            line=dict(width=0),
            colorbar=dict(
                title=dict(
                    text="Label",
                    font=dict(color='#00aaff', family='monospace')
                ),
                tickfont=dict(color='#00aaff', family='monospace'),
            )
        ),
        text=labels[mask_normal],
        hovertemplate="[ %{text} ]<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
    ))

    
    if highlight_label and mask_highlight.any():
        traces.append(go.Scatter3d(
            x=embeddings[mask_highlight, 0],
            y=embeddings[mask_highlight, 1],
            z=embeddings[mask_highlight, 2],
            mode='markers',
            name=str(highlight_label),
            marker=dict(
                size=6,           
                color='#ff3c3c',  
                opacity=1.0,
                line=dict(width=0),
            ),
            text=labels[mask_highlight],
            hovertemplate="[ %{text} ]<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
        ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(
            text="[ BATICOMPUTER — EMBEDDING SCAN ]",
            font=dict(color='#00eeff', size=18, family='monospace'),
            x=0.5
        ),
        paper_bgcolor='#000d1a',
        scene=dict(
            bgcolor='#000d1a',
            xaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                showticklabels=True,
                tickfont=dict(color='#003d7a', family='monospace'),
                title=dict(text='', font=dict(color='#003d7a')),
                showbackground=False,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                showticklabels=True,
                tickfont=dict(color='#003d7a', family='monospace'),
                title=dict(text='', font=dict(color='#003d7a')),
                showbackground=False,
            ),
            zaxis=dict(
                showgrid=False, zeroline=False, showline=False,
                showticklabels=True,
                tickfont=dict(color='#003d7a', family='monospace'),
                title=dict(text='', font=dict(color='#003d7a')),
                showbackground=False,
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def home(request): 

    client = connect_qdrant()
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
            inference = transform_image(encoder,images)

            query_vector = inference.cpu().numpy().squeeze().tolist()
            query_label = knn_search(client,query_vector).payload["role"]

            graph_html = plot_qdrant_embeddings(embeddings,labels,highlight_label=query_label)

    else: 
        graph_html = plot_qdrant_embeddings(embeddings,labels)
    
    return render(
        request,
        "home.html",
        {"graph": graph_html}
    )
    




