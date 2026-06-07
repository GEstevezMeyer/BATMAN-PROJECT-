# BATMAN-PROJECT

## Problematic

When investigating crimes in Gotham City, Batman has the problem that, to analyze fingerprints, he needs to return to the Batcave to identify the criminal. Returning wastes precious time that could instead be used to capture criminals. Your job is to create a solution to Batman’s problem and design a UI that allows him to traverse the criminals database.


## Demo of the idea:

[![Baticomputer Demo](https://img.youtube.com/vi/TQWIGXrmxMo/maxresdefault.jpg)](https://youtu.be/TQWIGXrmxMo)


## Methodologie 


For building this encoder we are using a **Triplet Margin Loss** function. The idea behind the Triplet Margin Loss is that we have an **anchor** — in this case an image of a criminal — a **positive** that is the fingerprint of the same criminal, and a **negative** that is the fingerprint of a different criminal. When the embeddings of the anchor and the negative are vastly different and the embeddings of the anchor and the positive are close, the loss will be at its minimum.

$$\mathcal{L} = \max\left( \| f(a) - f(p) \|_2^2 - \| f(a) - f(n) \|_2^2 + \alpha,\ 0 \right)$$

Where:
- $f(\cdot)$ is the encoder function
- $a$ is the anchor
- $p$ is the positive sample
- $n$ is the negative sample
- $\alpha$ is the margin

For training we use a sampler, because for the model to learn it needs to compare every image against the others. As a learning metric we use the **Silhouette Score**, which gives us an idea of the clusters being created. Our ultimate goal is for each criminal to form a unique and well-separated cluster. I made a custom metric that I call *Tolerance*. The core idea of this metric is to observe the grouping of the embeddings in the space. To evaluate the quality of the clusters we define a **Tolerance Metric**:

$$\text{Tolerance} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\left[ \sum_{j=1}^{k} \mathbb{1}\left[ y_i \neq y_{N_j(i)} \right] < \tau \right]$$

Where $k = 10$ neighbors are evaluated per sample and $\tau$ is the maximum number of
allowed mismatches before a sample is considered incorrectly clustered.
A tolerance of **1.0** means every sample has at most $\tau - 1$ neighbors from a different class.

For validation we use a simple **Recall@1**:

$$\text{Recall@1} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\left[ y_i = y_{N_1(i)} \right]$$

Where $N_1(i)$ is the nearest neighbor of sample $i$, excluding itself.