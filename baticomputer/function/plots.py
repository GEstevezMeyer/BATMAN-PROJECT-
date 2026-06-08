import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.graph_objects as go


def plot_qdrant_embeddings(embeddings, labels, method=PCA, highlight_label=None):
   
    
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
