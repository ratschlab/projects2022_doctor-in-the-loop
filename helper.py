import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython import embed

def get_cover(algorithm, dataset, lims_ref, D_ref, I_ref):
    #get the percentage of datapoints within the radius of a labeled point
    # Get all covered points
    if algorithm=="pc" or algorithm=="coverpc":
        covered = np.array([], dtype=int)
        for u in dataset.queries:
            # TODO: improve this
            covered = np.concatenate((covered, I_ref[lims_ref[u]:lims_ref[u + 1]]), axis=0)
        covered= np.concatenate((covered, dataset.queries), axis=0)
        covered= np.unique(covered)
    elif algorithm=="adpc" or algorithm=="partialadpc":
        R = np.repeat(dataset.radiuses, repeats=(lims_ref[1:] - lims_ref[:-1]).astype(int))
        mask = D_ref < R ** 2
        I_temp, D_temp = I_ref[mask], D_ref[mask]
        mask_split = np.split(mask, lims_ref)[1:-1]
        not_covered = np.array([np.invert(mask_split[u]).sum() for u in range(len(dataset.x))], dtype=int)
        lims_temp= lims_ref.copy()
        lims_temp[1:] = lims_ref[1:] - np.cumsum(not_covered)
        covered = np.array([], dtype=int)
        for u in dataset.queries:
            # TODO: improve this
            covered = np.concatenate((covered, I_temp[lims_temp[u]:lims_temp[u + 1]]), axis=0)
        covered = np.concatenate((covered, dataset.queries), axis=0)
        covered = np.unique(covered)

    return len(covered)/dataset.n_points


def get_optimistic_radiuses(dataset):
    optimistic_radiuses = dataset.radiuses.copy()
    for c in np.unique(dataset.y):
        mask_labeled= (dataset.labeled==1)*(dataset.y==c)
        if mask_labeled.sum()>0:
            # fit index to labeled data of class c 
            index = faiss.IndexFlatL2(dataset.d)  # build the index
            index.add(dataset.x[mask_labeled].astype("float32")) # fit to labeled data
            D, I = index.search(dataset.x[dataset.y==c].astype("float32"), 1) # find the nearest labeled point (0 for already queried points)
            D = np.sqrt(D) 
            optimistic_radiuses[dataset.y==c]= np.minimum(optimistic_radiuses[dataset.y==c], D.squeeze())
    return optimistic_radiuses

def get_pessimistic_radiuses(dataset, new_query_id, gamma):
    rc = dataset.radiuses[new_query_id]
    pessimistic_radiuses = dataset.radiuses.copy()

    dist_to_labeled = np.linalg.norm(dataset.x[new_query_id, :] - dataset.x[dataset.queries, :], axis=1)
    diff_radiuses = dataset.radiuses[dataset.queries] + rc - dist_to_labeled

    #Get points with balls that intersect but have different labels and reduce their radius
    mask_pessimistic = (diff_radiuses > 0) * (dataset.y[new_query_id] != dataset.y[dataset.queries])
    pessimistic_radiuses[dataset.queries[mask_pessimistic]]= np.maximum(0, dataset.radiuses[dataset.queries]-gamma*diff_radiuses)[mask_pessimistic]

    if mask_pessimistic.any():
        pessimistic_radiuses[new_query_id] = rc - gamma * np.max(diff_radiuses[mask_pessimistic])
    return pessimistic_radiuses

def get_knn_weighted_radiuses(dataset, K, deg, radius):
    index_knn = faiss.IndexFlatL2(dataset.d)  # build the index
    index_knn.add(dataset.x[dataset.queries].astype("float32"))  # fit it to the labeled data
    n_neighbours = K if len(dataset.queries) >= K else len(dataset.queries)
    D_neighbours, I_neighbours = index_knn.search(dataset.x[dataset.labeled == 0].astype("float32"), n_neighbours)  # find K-nn for all
    D_neighbours= np.sqrt(D_neighbours)
    new_radiuses = dataset.radiuses.copy()

    gauss_distances = np.exp(-D_neighbours ** deg / new_radiuses[dataset.labeled == 0].reshape(-1, 1) ** deg)
    use_self = True

    if use_self:
        norm = (gauss_distances.sum(axis=1, keepdims=True) + 1)
        alpha = (1 / norm)[:, 0]
        weights = gauss_distances / norm

        new_radiuses[dataset.labeled == 0] = (weights * dataset.radiuses[dataset.queries[I_neighbours]]).sum(axis=1) + alpha * radius
    else:
        alpha = 1 / 2
        weights = gauss_distances / (gauss_distances.sum(axis=1).reshape(-1, 1))

        new_radiuses[dataset.labeled == 0] = alpha * (weights * dataset.radiuses[dataset.queries[I_neighbours]]).sum(axis=1) + (1 - alpha) * dataset.radiuses[dataset.labeled == 0]

    return new_radiuses

def adjacency_graph_faiss(x: np.array, initial_radius: float):
    n_features = x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    lims, D, I = index.range_search(x.astype('float32'), initial_radius ** 2)  # because faiss uses squared L2 error
    return lims, D, I


def remove_incoming_edges_faiss(dataset, lims, D, I, query_idx=None):
    if len(dataset.queries) > 0:
        if query_idx is None:
            query_idx = dataset.queries
        if isinstance(query_idx, np.int64):
            query_idx = np.array([query_idx])
        # Get all covered
        covered = np.array([])
        for u in query_idx:
            # TODO: improve this
            covered = np.concatenate((covered, I[lims[u]:lims[u + 1]]), axis=0)
        covered = np.unique(covered)
        I_covered_bool = np.isin(I, covered)
        I_covered_idx = np.where(I_covered_bool)[0]
        # Remove all incoming edges to the covered vertices
        I_split = np.split(I_covered_bool, lims)[1:-1]
        n_covered = np.array([I_split[u].sum() for u in range(len(dataset.x))], dtype=int)

        lims[1:] = lims[1:] - np.cumsum(n_covered)
        I = np.delete(I, I_covered_idx)
        D = np.delete(D, I_covered_idx)
    return lims, D, I


def update_adjacency_radius_faiss(dataset, new_radiuses, lims_ref, D_ref, I_ref, lims, D, I):
    assert (len(dataset.radiuses) == len(new_radiuses))
    assert (lims_ref[-1] == D_ref.shape[0] == I_ref.shape[0])
    assert (lims[-1] == D.shape[0] == I.shape[0])

    if (dataset.radiuses != new_radiuses).any():
        R = np.repeat(new_radiuses, repeats=(lims_ref[1:] - lims_ref[:-1]).astype(int))
        mask = D_ref < R ** 2
        I, D = I_ref[mask], D_ref[mask]
        mask_split = np.split(mask, lims_ref)[1:-1]
        not_covered = np.array([np.invert(mask_split[u]).sum() for u in range(len(dataset.x))], dtype=int)
        lims[1:] = lims_ref[1:] - np.cumsum(not_covered)
        lims, D, I = remove_incoming_edges_faiss(dataset, lims, D, I)

    dataset.radiuses = new_radiuses
    return lims, D, I

# Condition to reduce radiuses is if ri+rj>d or ri+rj-d>0 or diff>0
# The way we reduce the radiuses right now is ri= ri-diff/2 and rj=rj-diff/2 such that ri+rj= ri+rj-diff=d --> the two balls don't intersect at all anymore
# We could change the update to rij= rij-gamma*diff such that ri+rj= ri+rj-gamma*distance with gamma in (0,0.5)
def reduce_intersected_balls_faiss(dataset, new_query_id, lims_ref, D_ref, I_ref, lims, D, I, gamma= 0.5, reduction_method= "pessimistic"):
    rc = dataset.radiuses[new_query_id]

    if len(dataset.queries) > 0:
        if reduction_method=="pessimistic":
            new_radiuses= get_pessimistic_radiuses(dataset, new_query_id, gamma)

        elif reduction_method=="mix":
            optimistic_radiuses= get_optimistic_radiuses(dataset)
            pessimistic_radiuses= get_pessimistic_radiuses(dataset, new_query_id, gamma)
            new_radiuses= np.maximum(optimistic_radiuses, pessimistic_radiuses)  
    else:
        new_radiuses = dataset.radiuses

    # Update for changed radiuses and new observed point
    lims, D, I = update_adjacency_radius_faiss(dataset, new_radiuses, lims_ref, D_ref, I_ref, lims, D, I)

    return lims, D, I

def shift_covering_centroids(dataset, radius, class_centroids, point_centroids, new_query_id, lims_ref, D_ref, I_ref, lims, D, I, shift=0.1):
    if len(dataset.queries)>0:
        rc = dataset.radiuses[new_query_id]
        old_point_centroids= point_centroids.copy()

        dist_to_labeled = np.linalg.norm(point_centroids[new_query_id, :] - point_centroids[dataset.queries, :], axis=1)
        diff_radiuses = dataset.radiuses[dataset.queries] + rc - dist_to_labeled

        # Get points with balls that intersect but have different labels and shift their centroids towards the center 
        mask = (diff_radiuses > 0) * (dataset.y[new_query_id] != dataset.y[dataset.queries])

        # Shift all centroids of points in mask_pessimistic and c_id towards their respective class centroids
        for c in np.unique(dataset.y[mask]):
            mask_label= dataset.y[dataset.queries]==c
            if mask_label.sum()>0:
                # if class centroid exists, shift the point centroid towards it
                point_centroids[mask*mask_label]= (1-shift) * point_centroids[mask*mask_label] + (shift)* class_centroids[dataset.y[mask*mask_label]]
        
        if mask.any():
            point_centroids[new_query_id]= (1-shift) * point_centroids[new_query_id] + (shift)* class_centroids[dataset.y[new_query_id]]
        
        # Update the points covered by the new point centroids
        for i in np.where(old_point_centroids!=point_centroids):
            i=0
            # get all the queries within radius of i that are not labeled
            covered= np.where(np.linalg.norm(dataset.x - point_centroids[i, :], axis=1) < radius)[0]
            # replace them in the list of neightbours for query i
            # adjust lims, D, I
            # remove all incoming edges 
            


              
        







def reduce_all_interesected_balls_faiss(dataset, lims_ref, D_ref, I_ref, lims, D, I, gamma= 0.5, reduction_method= "pessimistic"):
    pessimistic_radiuses = dataset.radiuses.copy()
    a = np.repeat(dataset.x[dataset.queries, :][:, :, None], len(dataset.queries), axis=-1)
    b = np.repeat(dataset.x[dataset.queries, :].T[None, :, :], len(dataset.queries), axis=0)

    # Pairwise distances and radiuses
    dist_labeled= np.sqrt(((a-b)**2).sum(axis=1))
    sum_radiuses= (dataset.radiuses[dataset.queries].reshape(-1,1)+dataset.radiuses[dataset.queries].reshape(1,-1))
    diff_dist = sum_radiuses-dist_labeled
    diff_label= (dataset.y[dataset.queries].reshape(-1,1)!=dataset.y[dataset.queries].reshape(1,-1))

    # Get points with balls that intersect but have different labels and reduce their radius
    mask_pessimistic= np.where((diff_label)*(diff_dist>0)*(np.triu(np.ones(shape=(len(dataset.queries),len(dataset.queries))),1)))
    mask_pessimistic= np.unique(np.concatenate((mask_pessimistic[0], mask_pessimistic[1])))
    diff_radiuses = diff_dist.max(axis=1)

    pessimistic_radiuses[dataset.queries[mask_pessimistic]]= np.maximum(0, dataset.radiuses[dataset.queries]-gamma*diff_radiuses)[mask_pessimistic]
    
    new_radiuses= pessimistic_radiuses
    
    lims, D, I = update_adjacency_radius_faiss(dataset, new_radiuses, lims_ref, D_ref, I_ref, lims, D, I)

    return lims, D, I


    

def get_nn_faiss(x: np.array):
    n_features = x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    D, I = index.search(x.astype('float32'), 2)
    idx = I[:, 1]
    return idx


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
        # covered_vertices = np.where(graph[u, :] == 1)[0]
        covered_vertices = I[lims[u]:lims[u + 1]]
        purity[u] = int(np.all(pseudo_labels[nn_idx[covered_vertices]] == pseudo_labels[nn_idx[u]]))

    if plot_unpure_balls:
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=pseudo_labels, palette="Set2")
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

    while len(radiuses) > 1:
        id = int(len(radiuses) / 2)
        purity = get_purity_faiss(x, pseudo_labels, nn_idx, radiuses[id])

        if purity >= alpha:
            radiuses = radiuses[id:]
        else:
            radiuses = radiuses[:id]
    purity_radius = radiuses[0]

    if plot_unpure_balls:
        get_purity_faiss(x, pseudo_labels, nn_idx, purity_radius, True)

    return purity_radius
