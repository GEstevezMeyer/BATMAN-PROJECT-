from django.shortcuts import render, HttpResponse
from qdrant_client import QdrantClient
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.graph_objects as go


# Create your views here.

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

def plot_qdrant_embeddings(embeddings,labels,method = PCA): 
    visualizer = method(n_components = 3)

    embeddings = visualizer.fit_transform(embeddings)

    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            showscale=True
        ),
        text=labels  
    )])

    fig.update_layout(title=f"Embeddings")
    
    graph_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    return graph_html

def home(request): 

    client = QdrantClient(host="localhost",port ="6333")
    embeddings, labels = import_vectors(client)

    graph_html = plot_qdrant_embeddings(embeddings,labels)

    return render(
        request,
        "home.html",
        {"graph": graph_html}
    )

