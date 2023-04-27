import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def adjacency_graph_faiss(x: np.array, initial_radius: float):
    n_features = x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    lims, D, I = index.range_search(x.astype('float32'), initial_radius ** 2)  # because faiss uses squared L2 error
    return lims, D, I


def remove_incoming_edges_faiss(dataset, lims, D, I, query_idx=None):
    if len(dataset.queries) > 0:
        if query_idx == None:
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


def reduce_intersected_balls_faiss(dataset, new_query_id, lims_ref, D_ref, I_ref, lims, D, I, random_regime):
    rc = dataset.radiuses[new_query_id]
    if len(dataset.queries) > 0:
        dist_to_labeled = np.linalg.norm(dataset.x[new_query_id, :] - dataset.x[dataset.queries, :], axis=1)
        diff_radiuses = dataset.radiuses[dataset.queries] + rc - dist_to_labeled
        new_radiuses = dataset.radiuses.copy()
        mask = (diff_radiuses > 0) * (dataset.y[new_query_id] != dataset.y[dataset.queries])
        new_radiuses[dataset.queries[mask]] = \
        np.maximum(0, 0.5 * (dataset.radiuses[dataset.queries] - rc + dist_to_labeled))[mask]

        if mask.any():
            new_radiuses[new_query_id] = rc - 0.5 * np.max(diff_radiuses[mask])
        # Change the points that are considered as covered in the graph
    else:
        new_radiuses = dataset.radiuses

    dataset.observe(new_query_id, random_regime, rc)
    # Update for changed radiuses and new observed point
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