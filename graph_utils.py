from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sprs

def get_nearest_neighbors(data: np.ndarray, n_neighbors: int) -> sprs.csr_matrix:
    """
    Generates symmetrized nearest neighbor graph.

    Args:
        data: np.ndarray
            The data to get nearest neighbors for.
            Dimensions should be (num samples, sample dimension).
        n_neighbors: int
            The number of nearest neighbors to get.

    Returns:
        scipy.sparse.csr_matrix
            The adjacency matrix of the symmetrized nearest neighbor graph.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    mat = nbrs.kneighbors_graph()
    return mat + mat.T

def get_ref_clique_graph(clustering: np.ndarray) -> np.ndarray:
    """
    Generates a graph, where edges are cliques of equally-labeled samples (self loops are removed).

    Args:
        clustering: np.ndarray
            A reference clustering of the data.

    Returns:
        np.ndarray
            The adjacency matrix of the clique graph.
    """
    graph = (clustering[:, None] == clustering).astype(float, copy=False)
    graph -= np.eye(graph.shape[0])
    return graph