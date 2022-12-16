import numpy as np
import faiss
from IPython import embed
from datasets import PointClouds, CenteredCircles
from helper import adjacency_graph, get_radius, get_purity, get_nearest_neighbour
from helper import adjacency_graph_faiss, get_radius_faiss, get_purity_faiss, get_nn_faiss


M = 150
B = 5
threshold = 0.9
p_cover = 1
gamma = 3

center = [4, 5]
radiuses = [0.5, 3, 5]
samples = [1000, 4000, 5000]
std = [0.5, 0.5, 0.55]

circles_data = CenteredCircles(center, radiuses, samples, std)
radius = 0.5
threshold = 0.9
x = circles_data.x.astype("float32")
pseudo_labels = circles_data.y


def check_nn_faiss():
    nn_idx = get_nearest_neighbour(x)
    nn_idx_faiss = get_nn_faiss(x)
    print(np.all(nn_idx == nn_idx_faiss))


def check_get_purity_faiss():
    nn_idx = get_nearest_neighbour(x)
    nn_idx_faiss = get_nn_faiss(x)
    purity = get_purity(x, pseudo_labels, nn_idx, radius)
    purity_faiss = get_purity_faiss(x, pseudo_labels, nn_idx_faiss, radius)
    print(purity_faiss, purity)

def check_get_radius_faiss():
    radius_faiss = get_radius_faiss(threshold, x, pseudo_labels, [0, 10], 0.01)
    radius = get_radius(threshold, x, pseudo_labels, [0, 10], 0.01)
    print(radius_faiss, radius)

def check_adjacency_faiss():
    G = adjacency_graph(x, radius)
    lims, D, I = adjacency_graph_faiss(x, radius)
    out_degrees = np.sum(G, axis=1)
    out_degrees_faiss = lims[1:] - lims[:-1]
    print(np.all(out_degrees == out_degrees_faiss))
    same = True
    for u in range(len(G)):
        if not np.all(np.where(G[u, :] == 1)[0] == I[lims[u]:lims[u + 1]]):
            same = False
    print(same)

def check_prob_cover_faiss():
    lims, D, I= adjacency_graph_faiss(x, radius)
    out_degrees_faiss = (lims[1:]-lims[:-1])
    max_out_degree_id= np.argmax(out_degrees_faiss[circles_data.labeled==0])
    c_id_faiss = np.arange(circles_data.n_points)[circles_data.labeled == 0][max_out_degree_id]

    covered_vertices_faiss= I[lims[c_id_faiss]:lims[c_id_faiss+1]]
    covered_idx_faiss= np.sort(np.where(np.isin(I, covered_vertices_faiss))[0])
    old_lims, old_I, old_covered_idx_faiss, old_out_degrees_faiss= lims.copy(), I.copy(), covered_idx_faiss.copy(), out_degrees_faiss.copy()

    n_covered= np.zeros(len(lims))
    for l in range(1, len(lims)):
        n_covered[l]= np.sum(((lims[l-1]<=old_covered_idx_faiss)&(old_covered_idx_faiss<lims[l])))
    lims= old_lims-np.cumsum(n_covered)
    I = np.delete(I, old_covered_idx_faiss)
    out_degrees_faiss= (lims[1:]-lims[:-1])
    print("Sampling for faiss done")


    # Use the usual method to compare
    G= adjacency_graph(x, radius)
    out_degrees= np.sum(G, axis=1)
    max_out_degree = np.argmax(out_degrees[circles_data.labeled == 0])
    c_id = np.arange(circles_data.n_points)[circles_data.labeled == 0][max_out_degree]
    covered_vertices = np.where(G[c_id, :] == 1)[0]
    G[:, covered_vertices] = 0
    old_out_degrees= out_degrees
    out_degrees= np.sum(G, axis=1)

    #Check that the usual method and faiss implementation correspond
    print(np.all(old_out_degrees_faiss == old_out_degrees))
    print(np.all(out_degrees_faiss== out_degrees))
    # print(np.sum(old_out_degrees_faiss - out_degrees_faiss) == len(covered_idx_faiss))
    # print(np.sum(old_out_degrees- out_degrees) == len(covered_idx_faiss))


check_nn_faiss()
check_prob_cover_faiss()
embed()

