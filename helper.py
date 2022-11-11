from IPython import embed
import numpy as np
from sklearn.neighbors import NearestNeighbors


def typicality(X, K):
    """
    Args:
        K: hyperparameter deciding on the number of nearest neighbours to use
        X: array (n_samples, n_features)
    Returns:
        t: typicality array of shape (n_samples, 1)
    """
    knn = NearestNeighbors(n_neighbors=K)
    nbrs = knn.fit(X)
    distances, _ = nbrs.kneighbors(X)
    t = 1 / (np.mean(distances, axis=1)+0.000001)
    return t.reshape(-1,1)