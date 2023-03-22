import pandas as pd
from IPython import embed
import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
import faiss
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment

def adjacency_graph(x: np.array, radius: float):
    n_vertices = len(x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        graph[u,u]=1
        for v in range(u):
            if np.linalg.norm(x[u,:]-x[v,:]) < radius:
                graph[u, v] = 1
                graph[v, u] = 1
    return graph

def adjacency_graph_faiss(x: np.array, radius:float):
    n_features= x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    lims, D, I = index.range_search(x.astype('float32'), radius**2) # because faiss uses squared L2 error
    return lims, D, I

def weighted_graph(dataset: pd.DataFrame, kernel_fn, kernel_hyperparam, sampled_labels=None):
    n_vertices = len(dataset.x)
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

def get_nn_faiss(x: np.array):
    n_features = x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    D, I = index.search(x.astype('float32'), 2)
    idx= I[:,1]
    return idx
def coclust(true_labels, pseudo_labels):
    cm = confusion_matrix(true_labels, pseudo_labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js= indexes[1]
    cm2 = cm[:, js]
    accuracy= np.trace(cm2) / np.sum(cm2)

    return accuracy

def get_purity_faiss(x, pseudo_labels, nn_idx, radius, plot_unpure_balls=False):
    """
    Args:
        x: np.array
        pseudo_labels: labels predicted by some clustering algorithm
        nn_idx: id of the nearest neighbour of each point
        radius: purity radius

    Returns:
        purity(radius)
    """
    purity = np.zeros(len(x), dtype=int)
    lims, D, I = adjacency_graph_faiss(x, radius)

    for u in range(len(x)):
        #covered_vertices = np.where(graph[u, :] == 1)[0]
        covered_vertices= I[lims[u]:lims[u+1]]
        purity[u] = int(np.all(pseudo_labels[nn_idx[covered_vertices]] == pseudo_labels[nn_idx[u]]))

    if plot_unpure_balls:
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=x[:,0], y=x[:,1], hue=pseudo_labels, palette="Set2")
        for u in range(len(x)):
            if purity[u] == 0:
                plt.plot(x[u, 0], x[u, 1], color="red", marker="P")
                ax.add_patch(plt.Circle((x[u, 0], x[u, 1]), radius, color='red', fill=False, alpha=0.5))
        plt.title(f"Points that are not pure and their covering balls of radius {radius}")
        plt.show()

    return np.mean(purity)

def get_radius_faiss(alpha: float, x: np.array, pseudo_labels,
                         search_range=[0.5, 3.5], search_step=0.05, plot_unpure_balls=False):
    """
    Args:
        alpha: purity threshold
        x: np.array
        k: number of classes in the data
        graph: adjacency matrix with 0 diagonals, an edge for points in the B-delta ball
    Returns:
        purity_radius: min{r: purity(r)>=alpha}
    """

    # Get the indices of the 1 nearest neighbours for each point
    nn_idx = get_nn_faiss(x)
    # Radiuses for the search
    radiuses = np.arange(search_range[0], search_range[1], search_step)

    while len(radiuses)>1:
        id = int(len(radiuses) / 2)
        purity= get_purity_faiss(x, pseudo_labels, nn_idx, radiuses[id])

        if purity>=alpha:
            radiuses=radiuses[id:]
        else:
            radiuses=radiuses[:id]
    purity_radius = radiuses[0]

    if plot_unpure_balls:
        get_purity_faiss(x, pseudo_labels, nn_idx, purity_radius, True)

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

def cover(x, labeled_idx, radius:float):
    n_vertices = len(x)
    unlabeled_idx = np.array(list(set(np.arange(n_vertices)) - set(labeled_idx)))
    adjacency = adjacency_graph(x, radius)
    covered= adjacency[unlabeled_idx.reshape(-1,1), labeled_idx.reshape(1,-1)]
    if ((covered.shape[0]==0) or (covered.shape[1]==0)):
        covered_points=0
    else:
        covered_points= np.sum(np.max(covered, axis=1))/(len(unlabeled_idx))
    return covered_points

def check_cover(x, labeled_idx, radius: float, p_cover:float):
    covered_points= cover(x, labeled_idx, radius)
    return covered_points>=p_cover


def get_covered_points(dataset, radius, id):
    n_vertices= len(dataset.x)
    distances= np.linalg.norm(dataset.x[id,:]- dataset.x[np.arange(n_vertices), :], axis=1)
    covered= (distances<= radius)
    return covered







