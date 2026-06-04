import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import math 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
from src.etl.transform import transform_data


def plot_metrics(history: dict):
    
    keys = list(history.keys())
    epochs = [i + 1 for i in range(len(history[keys[0]]))]

    n = len(keys)
    number_of_rows = min(n, 3)
    number_of_columns = math.ceil(n / number_of_rows)

    fig, ax = plt.subplots(number_of_rows, number_of_columns)
    ax = np.array(ax).reshape(number_of_rows, number_of_columns)

    for i, key in enumerate(keys):
        ax_x, ax_y = i % number_of_rows, i // number_of_rows
        ax[ax_x, ax_y].plot(epochs, history[key])
        ax[ax_x, ax_y].set_title(key)

    for i in range(n, number_of_rows * number_of_columns):
        ax[i % number_of_rows, i // number_of_rows].set_visible(False)

    plt.tight_layout()
    plt.show()
        
def plot_pca(encoder,dataloader,device = "cuda"):
    pca = PCA(n_components=3)
    total_embeddings,total_labels = transform_data(encoder=encoder,dataloader=dataloader,device=device)
    
    total_embeddings = pca.fit_transform(total_embeddings)

    fig = go.Figure(data=[go.Scatter3d(
        x=total_embeddings[:, 0],
        y=total_embeddings[:, 1],
        z=total_embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=total_labels,
            colorscale='Viridis',
            showscale=True
        ),
        text=total_labels  
    )])

    fig.update_layout(title="PCA of Embeddings")
    fig.show()



def plot_tsne(encoder,dataloader,device = "cuda"):
    tsnee = TSNE(n_components=3)

    total_embeddings,total_labels = transform_data(encoder=encoder,dataloader=dataloader,device=device)

    total_embeddings = tsnee.fit_transform(total_embeddings)

    fig = go.Figure(data=[go.Scatter3d(
        x=total_embeddings[:, 0],
        y=total_embeddings[:, 1],
        z=total_embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=total_labels,
            colorscale='Viridis',
            showscale=True
        ),
        text=total_labels  
    )])

    fig.update_layout(title="TSNE of Embeddings")
    fig.show()


