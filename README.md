# SpEx

This is the official repository for the paper SpEx: A Spectral Approach to Explainable Clustering, accepted to NeurIPS 2025.

This repository provides implementations for our main algorithm - SpEx, for general reference graphs, with an efficient implementation for the special case of a clique graph (the edge set is the set of cliques of equally-labeled data points, as set by a reference clustering), alongside IMM, EMN and an adapted version of CART discussed in the paper. 

## Installation

The code requires external repositories to run fully, namely CLIP to process some of the datasets and ExKMC for the relevant experiment. It is advised to use the `Makefile` file to install all required dependencies. For the main experiment for the smaller datasets, downloading the requirements listed in the `requirements.txt` is enough.

## Basic Usage

All of the mentioned algorithms share a common interface - `Tree`, provided with a `train` function to build an explainable tree based on a refernce and a `predict` function to cluster input data. For all algorithms, the prediction method requires a dataset in the form of a numpy array of shape (number of samples, dimension of samples). The train function however, is algorithm specific. 

Our main algorithm requires a graph's adjacency matrix as an input. For the specific case of the clique graph a specific implementation that does not require an external adjacency matrix is available. For kNN - we have a dedicated function. In general, any `scipy.sparse` adjacency matrix should work.

Usage is presented in the following example:

```python
# Example usage

# Get some synthetic dataset
from sklearn.datasets import make_blobs

k = 10
seed = None
box = (-2, 2)
std = 1
n_samples = 6000
dimension = 50
data, clustering_true, centers = make_blobs(n_samples=, centers=k, cluster_std=std,
                                                    random_state=seed, n_features=dimension, center_box=box, return_centers=True)

# Init trees
from init_tree import init_tree_by_name

spex_knn = init_tree_by_name('graph')
spex_clique = init_tree_by_name('clique')
emn = init_tree_by_name('emn')
imm = init_tree_by_name('imm')
cart = init_tree_by_name('cart')

# Get kNN graph for SpEx kNN
num_neighbors = 50

from graph_utils import get_nearest_neighbors
nn_graph = get_nearest_neighbors(data, num_neighbors)

# Get a reference clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, random_state=1234)
clustering_kmeans = kmeans.fit_predict(data)
centers_kmeans = np.zeros((k, data.shape[1]))
for i in range(k):
  centers_kmeans[i,:] = data[clustering_kmeans == i, :].mean(axis=0)

# Generate the trees
# All algorithms except EMN and IMM support expansion of the number of clusters beyond the ground truth (k)
k_prime = k 

# For SpEx kNN, k must be provided
spex_knn.train(data, nn_graph, k_prime) 
# For SpEx Clique and CART, it can be inferred from the reference clustering (default is None).
spex_clique.train(data, clustering_kmeans, k_prime)
cart.train(data, clustering_kmeans, k_prime)

imm.train(data, clustering_kmeans, centers_kmeans)
emn.train(data, clustering_kmeans, centers_kmeans)

# Predict
knn_pred = spex_knn.predict(data)
clique_pred = spex_clique.predict(data)
...

# Evaluate
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

print(adjusted_rand_score(clustering_true, knn_pred)) # ARI with respect to ground truth
print(adjusted_rand_score(clustering_kmeans, knn_pred)) # ARI with respect to reference clustering
...

```

## Reproducing the paper

To reproduce our results, use the provided Jupyter notebooks - `main_experiments.ipynb` (for our main experiment), `price_of_explainability_experiment.ipynb` and `comparison_with_ExKMC.ipynb`.

### Dataset availablility 

Note that some of the datasets used in the paper require a local copy of the dataset, namely "pathbased", "R15" and "ecoli".

### Kernel IMM

This repository contains code from the [Kernel IMM paper](https://openreview.net/forum?id=FAGtjl7HOw&noteId=ojbtGqOTg8), in the "kimm" folder and in the main notebook.