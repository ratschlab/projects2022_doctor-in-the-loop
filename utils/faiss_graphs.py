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


def get_nn_faiss(x: np.array):
    n_features = x.shape[1]
    index = faiss.IndexFlatL2(n_features)  # build the index
    index.add(x.astype('float32'))
    D, I = index.search(x.astype('float32'), 2)
    idx = I[:, 1]
    return idx


