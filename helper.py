import pandas as pd
from IPython import embed
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
import scipy
from sklearn.metrics import accuracy_score
from scipy.sparse.csgraph import shortest_path

def typicality(X, K):
    """
    Args:
        K: hyperparameter deciding on the number of nearest neighbours to use
        X: array (n_samples, n_features)
    Returns:
        t: typicality array of shape (n_samples, 1)
    """
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    distances, _ = knn.kneighbors(X)
    t = 1 / (np.mean(distances, axis=1) + 0.000001)
    return t.reshape(-1, 1)


def adjacency_graph(x: np.array, delta: float):
    n_vertices = len(x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        graph[u,u]=1
        for v in range(u):
            if np.linalg.norm(x[u,:]-x[v,:]) < delta:
                graph[u, v] = 1
                graph[v, u] = 1

    return graph

def weighted_graph(dataset: pd.DataFrame, kernel_fn, kernel_hyperparam, sampled_labels=None):
    n_vertices = len(x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        for v in range(u):
            # if sampled_labels is not None:
            if len(dataset.queries)>=1:
                if (u in dataset.queries)&(v in dataset.queries):
                    if (dataset.y[u] == dataset.y[v]) & (u != v):
                        graph[u,v]= 1000
                        graph[v,u]= 1000
                        # otherwise the weight is kept as zero
                else:
                    w= kernel_fn(dataset.x[u,:], dataset.x[v,:], kernel_hyperparam)
                    graph[u, v] = w
                    graph[v, u] = w
            else:
                w = kernel_fn(dataset.x[u,:], dataset.x[v,:], kernel_hyperparam)
                graph[u, v] = w
                graph[v, u] = w
    return graph

def get_nearest_neighbour(x: np.array):
    knn = NearestNeighbors(n_neighbors=2).fit(X=x)
    _, idx = knn.kneighbors(x)
    idx = idx[:, 1]  # contains the nearest neighbour for each point
    return idx

def get_purity(x, pseudo_labels, nn_idx, delta, plot_unpure_balls=False):
    """
    Args:
        x: np.array
        pseudo_labels: labels predicted by some clustering algorithm
        nn_idx: id of the nearest neighbour of each point
        delta: radius

    Returns:
        purity(delta)
    """
    purity = np.zeros(len(x), dtype=int)
    graph = adjacency_graph(x, delta)

    for u in range(len(x)):
        covered_vertices = np.where(graph[u, :] == 1)[0]
        purity[u] = int(np.all(pseudo_labels[nn_idx[covered_vertices]] == pseudo_labels[nn_idx[u]]))

    if plot_unpure_balls:
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=x[:,0], y=x[:,1], hue=pseudo_labels, palette="Set2")
        for u in range(len(x)):
            if purity[u] == 0:
                plt.plot(x[u, 0], x[u, 1], color="red", marker="P")
                ax.add_patch(plt.Circle((x[u, 0], x[u, 1]), delta, color='red', fill=False, alpha=0.5))
        plt.title(f"Points that are not pure and their covering balls of radius {delta}")
        plt.show()

    return np.mean(purity)


def get_radius(alpha: float, x: np.array, pseudo_labels,
                         search_range=[0.5, 3.5], search_step=0.05, plot_unpure_balls=False):
    """
    Args:
        x: np.array
        k: number of classes in the data
        graph: adjacency matrix with 0 diagonals, an edge for points in the B-delta ball
        alpha: purity threshold
    Returns:
        purity_radius: min{r: purity(r)>=alpha}
    """

    # Get the indices of the 1 nearest neighbours for each point
    nn_idx = get_nearest_neighbour(x)
    # Purity of each point

    radiuses = np.arange(search_range[0], search_range[1], search_step)

    while len(radiuses)>1:
        id = int(len(radiuses) / 2)
        purity= get_purity(x, pseudo_labels, nn_idx, radiuses[id])

        if purity>=alpha:
            radiuses=radiuses[id:]
        else:
            radiuses=radiuses[:id]
    purity_radius = radiuses[0]

    if plot_unpure_balls:
        get_purity(x, pseudo_labels, nn_idx, purity_radius, True)

    return purity_radius

def check_P_cover(x, labeled_idx, radius: float, P=None):
    assert(isinstance(P, int))
    n_vertices= len(x)
    unlabeled_idx = np.array(list(set(np.arange(n_vertices)) - set(labeled_idx)))
    adjacency= adjacency_graph(x, radius)
    dist_matrix= shortest_path(csgraph= adjacency, directed=False, indices= unlabeled_idx, return_predecessors=False)
    paths_length= np.partition(dist_matrix, 1, axis=1)[:, 1]
    max_length= np.max(shortest_path)
    return max_length<=P

def check_cover(x, labeled_idx, radius: float, p_cover:float):
    n_vertices = len(x)
    unlabeled_idx = np.array(list(set(np.arange(n_vertices)) - set(labeled_idx)))
    adjacency = adjacency_graph(x, radius)
    covered= adjacency[unlabeled_idx.reshape(-1,1), labeled_idx.reshape(1,-1)]
    if ((covered.shape[0]==0) or (covered.shape[1]==0)):
        covered_points=0
    else:
        covered_points= np.sum(np.max(covered, axis=1))/(len(unlabeled_idx))
    return covered_points>=p_cover



def accuracy_clustering(true_labels, pseudo_labels):
    assert(true_labels.shape==pseudo_labels.shape)
    true_labels= true_labels.astype(int)
    pseudo_labels= pseudo_labels.astype(int)
    true = np.unique(true_labels)
    pseudo= np.unique(pseudo_labels)
    permutations = [p for p in multiset_permutations(true)]
    accuracy=[]
    for i, p in enumerate(permutations):
        true_labels_copy= true_labels.copy()
        for label in true:
            true_labels_copy[true_labels==label]= p[label]
        accuracy.append(accuracy_score(true_labels_copy, pseudo_labels))
    return np.max(accuracy)



