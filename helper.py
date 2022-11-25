import pandas as pd
from IPython import embed
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


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


def adjacency_graph(dataset: pd.DataFrame, delta: float):
    n_vertices = len(dataset)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        graph[u,u]=1
        for v in range(u):
            if np.linalg.norm(dataset.iloc[u, :-1] - dataset.iloc[v, :-1]) < delta:
                graph[u, v] = 1
                graph[v, u] = 1

    return graph

def get_knn_labels(dataset, n_clusters, plot_kmeans=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=22).fit(dataset.iloc[:, :-1])
    pseudo_labels = kmeans.predict(dataset.iloc[:, :-1])
    if plot_kmeans:
        sns.scatterplot(data=dataset, x="x1", y="x2", hue=pseudo_labels, palette="Set2")
        plt.title("Pseudo-labels from k-means")
        plt.show()
    return pseudo_labels

def get_nearest_neighbour(dataset):
    knn = NearestNeighbors(n_neighbors=2).fit(X=dataset.iloc[:, :-1])
    _, idx = knn.kneighbors(dataset.iloc[:, :-1])
    idx = idx[:, 1]  # contains the nearest neighbour for each point
    return idx

def get_purity_labeled_data(dataset, pseudo_labels, nn_idx, delta):
    purity = np.zeros(len(dataset), dtype=int)
    graph = adjacency_graph(dataset, delta)

    for u in range(len(dataset)):
        covered_vertices = np.where(graph[u, :] == 1)[0]
        purity[u] = int(np.all(pseudo_labels[nn_idx[covered_vertices]] == pseudo_labels[nn_idx[u]]))
    return purity


def estimate_emperical_purity(delta, dataset: pd.DataFrame, pseudo_labels, plot_unpure_balls=False):
    """
    Args:
        dataset: pd.DataFrame
        delta: radius of the purity, float or list
        k: number of classes in the data
        graph: adjacency matrix with 0 diagonals, an edge for points in the B-delta ball
    Returns:
        pi(delta): 1-NN purity, using pseudo-labels assigned by k-means
    """
    # Get the indices of the 1 nearest neighbours for each point
    idx= get_nearest_neighbour(dataset)
    # Purity of each point
    purity = get_purity_labeled_data(dataset, pseudo_labels, idx, delta)

    if plot_unpure_balls:
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(data=dataset, x="x1", y="x2", hue=pseudo_labels, palette="Set2")
        for u in range(len(dataset)):
            if purity[u] == 0:
                plt.plot(dataset.iloc[u, 0], dataset.iloc[u, 1], color="red", marker="P")
                ax.add_patch(
                    plt.Circle((dataset.iloc[u, 0], dataset.iloc[u, 1]), delta, color='red', fill=False, alpha=0.5))
        plt.title(f"Points that are not pure and their covering balls of radius {delta}")
        plt.show()

    return np.mean(purity)


def get_emperical_radius(alpha: float, dataset: pd.DataFrame, pseudo_labels,
                         search_range=[0.5, 3.5], search_step=0.05, plot_unpure_balls=False):
    """
    Args:
        dataset: pd.DataFrame
        k: number of classes in the data
        graph: adjacency matrix with 0 diagonals, an edge for points in the B-delta ball
        alpha: purity threshold
    Returns:
        purity_radius: min{r: purity(r)>=alpha}
    """

    # Get the indices of the 1 nearest neighbours for each point
    idx = get_nearest_neighbour(dataset)
    # Purity of each point

    radiuses = np.arange(search_range[0], search_range[1], search_step)

    while len(radiuses)>1:
        id = int(len(radiuses) / 2)
        purity= np.mean(get_purity_labeled_data(dataset, pseudo_labels, idx, radiuses[id]))
        print(len(radiuses), radiuses[id], purity)

        if purity>=alpha:
            radiuses=radiuses[id:]
        else:
            radiuses=radiuses[:id]
    print(radiuses)
    purity_radius = radiuses[0]

    if plot_unpure_balls:
        purity = get_purity_labeled_data(dataset, pseudo_labels, idx, purity_radius)
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(data=dataset, x="x1", y="x2", hue=pseudo_labels, palette="Set2")
        for u in range(len(dataset)):
            if purity[u] == 0:
                plt.plot(dataset.iloc[u, 0], dataset.iloc[u, 1], color="red", marker="P")
                ax.add_patch(
                    plt.Circle((dataset.iloc[u, 0], dataset.iloc[u, 1]), purity_radius, color='red', fill=False, alpha=0.5))
        plt.title(f"Points that are not pure and their covering balls of radius {purity_radius}")
        plt.show()
    return purity_radius


