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

def adjacency_graph(x: np.array, radiuses):
    n_vertices = len(x)
    if isinstance(radiuses, float) or radiuses.ndim == 0:
        radiuses= np.repeat(radiuses, repeats= len(x))
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        covered_u= (np.linalg.norm(x[u,:]- x[np.arange(n_vertices),:], axis=1)< radiuses[u])
        graph[u, covered_u]=1
    return graph

def update_adjacency_graph_labeled(dataset):
    n_vertices = len(dataset.x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    # Update all points covered by u with radius ru
    for u in range(n_vertices):
        covered_u= (np.linalg.norm(dataset.x[u,:]- dataset.x[np.arange(n_vertices),:], axis=1)< dataset.radiuses[u])
        graph[u, covered_u]=1
    # Remove all incoming edges of all points covered by labeled points
    if len(dataset.queries)>0:
        all_covered= np.where(np.max(graph[dataset.queries,:], axis=0)==1)[0]
        graph[:,all_covered]=0
    return graph


def adjacency_graph_faiss(x: np.array, initial_radius:float):
    n_features= x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    lims, D, I = index.range_search(x.astype('float32'), initial_radius**2) # because faiss uses squared L2 error
    return lims, D, I


def update_adjacency_radius_faiss(dataset, new_radiuses, lims, D, I):
    n_vertices= len(dataset.x)
    #replace lims, D, I using the new radiuses
    for u in np.where(dataset.radiuses!=new_radiuses):
        new_distances= np.linalg.norm(dataset.x[u, :] - dataset.x[np.arange(n_vertices), :], axis=1)
        old_covered_points= I[lims[u]:lims[u+1]]
        new_covered_points= np.where(new_distances < new_radiuses[u])[0]  # indices are already sorted
        new_distances= new_distances[new_covered_points]
        length_diff= lims[u+1]- lims[u]- len(new_covered_points)

        #update the neighbours and distances
        I= np.concatenate((I[:lims[u]], new_covered_points, I[lims[u+1]:]), axis=0)
        D= np.concatenate((D[:lims[u]], new_distances, D[lims[u+1]:]), axis=0)
        # update the indices
        lims[u + 1:] = lims[u + 1:] - length_diff

        # if the points whose radius we're changing was labeled, we also need to remove/recover incoming edges
        if (new_radiuses[u]>dataset.radiuses[u])&(dataset.labeled[u]==1):
            # remove all incoming edges of the newly covered points
            for v in new_covered_points[np.invert(np.isin(new_covered_points, old_covered_points))]:
                # find points covered by v
                covered_by_v= I[lims[v]:lims[v+1]]
                # remove them from the list of neighbours I and update lims, D accordingly
                covered_by_v_I_bool= np.isin(I,covered_by_v)
                covered_by_v_I_id= np.where(covered_by_v_I_bool)[0]
                # n_covered = np.zeros(len(lims))
                # for l in range(1, len(lims)):
                #     n_covered[l] = np.sum(lims[l - 1] <= covered_by_v_I_id < lims[l])
                n_covered= np.array([np.sum(lims[l - 1] <= covered_by_v_I_id < lims[l]) for l in range(1, len(lims))])
                lims = lims - np.cumsum(n_covered)
                lims = lims.astype(int)
                I= I[np.invert(covered_by_v_I_bool)]
                D= D[np.invert(covered_by_v_I_bool)]

        elif (new_radiuses[u]< dataset.radiuses[u])&(dataset.labeled[u]==1):
            # need to recover incoming edges of points that are not covered anymore (if they are not covered by other labeled points!!)
            for v in old_covered_points[np.invert(np.isin(old_covered_points, new_covered_points))]:
                # get all points that have an edge going to v and add v back to their list of neighbours
                incoming_edge_vertices_v_distances= np.linalg.norm(dataset.x[v, :] - dataset.x[np.arange(n_vertices), :], axis=1)
                incoming_edge_vertices_v_bool= (incoming_edge_vertices_v_distances<new_radiuses[v])
                incoming_edge_vertices_v_id= np.where(incoming_edge_vertices_v_bool)[0]
                for l in incoming_edge_vertices_v_id:
                    # recover only if incoming_edge_vertices_v_id is not covered by a labeled point
                    covered_by_other_labeled= np.max(np.linalg.norm(dataset.x[l, :] - dataset.x[dataset.queries[dataset.queries!=u], :], axis=1)<dataset.radiuses[dataset.queries])
                    if not covered_by_other_labeled:
                        I= np.insert(I, lims[l+1], v)
                        D= np.insert(D, lims[l+1], incoming_edge_vertices_v_distances[l])
                        lims[l+1:]= lims[l+1:]+1
    dataset.radiuses= new_radiuses
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

def get_all_covered_points(dataset):
    n_vertices= len(dataset.x)
    all_covered= np.array([], dtype=int)
    for i, id in enumerate(dataset.queries):
        distances = np.linalg.norm(dataset.x[id, :] - dataset.x[np.arange(n_vertices), :], axis=1)
        covered= np.where(distances<= dataset.radiuses[i])[0]
        all_covered = np.concatenate((all_covered, covered), axis=0).astype(int)
    return np.unique(all_covered)








