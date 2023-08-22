import numpy as np
import faiss
from IPython import embed
from datasets import PointClouds, CenteredCircles
from utils.faiss_graphs import adjacency_graph, get_nearest_neighbour
from utils.faiss_graphs import adjacency_graph_faiss, get_radius_faiss, get_purity_faiss, get_nn_faiss
from activelearners import ProbCoverSampler_Faiss
from clustering import MyKMeans, MySpectralClustering, OracleClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from activelearners import ActiveLearner
from clustering import ClusteringAlgo
from sklearn.metrics import confusion_matrix
from sympy.utilities.iterables import multiset_permutations



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
    circles_data.restart()
    x= circles_data.x
    radius= 0.5
    lims, D, I= adjacency_graph_faiss(x, radius)

    for i in range(5):
        print(i, lims.shape, I.shape)
        out_degrees_faiss = (lims[1:]-lims[:-1])
        max_out_degree_id= np.argmax(out_degrees_faiss[circles_data.labeled==0])
        c_id_faiss = np.arange(circles_data.n_points)[circles_data.labeled == 0][max_out_degree_id]
        print(c_id_faiss, np.max(out_degrees_faiss))

        covered_vertices_faiss= I[lims[c_id_faiss]:lims[c_id_faiss+1]]
        covered_idx_faiss= np.sort(np.where(np.isin(I, covered_vertices_faiss))[0])
        print("Covered idx:", covered_idx_faiss.shape)

        n_covered= np.zeros(len(lims))
        for l in range(1, len(lims)):
            n_covered[l]= np.sum(((lims[l-1]<=covered_idx_faiss)&(covered_idx_faiss<lims[l])))
        lims= lims-np.cumsum(n_covered)
        lims= lims.astype(int)
        I = np.delete(I, covered_idx_faiss)
        print(f"Iteration {i} done")
        print("*")


    # Use the usual method to compare
    G= adjacency_graph(x, radius)
    out_degrees= np.sum(G, axis=1)
    max_out_degree = np.argmax(out_degrees[circles_data.labeled == 0])
    c_id = np.arange(circles_data.n_points)[circles_data.labeled == 0][max_out_degree]
    covered_vertices = np.where(G[c_id, :] == 1)[0]
    G[:, covered_vertices] = 0


def check_probcover_kmeans():
    clustering_seed= np.random.randint(1000)
    circles_data.restart()
    clustering= MyKMeans(circles_data, 3, random_clustering=clustering_seed)
    learner= ProbCoverSampler(circles_data, 0.9,
                                         clustering, [True, False],
                                         search_range=[0, 10], search_step=0.01)
    learner.query(5)
    learner.query(5)
    print(circles_data.queries)

    circles_data.restart()
    clustering= MyKMeans(circles_data, 3, random_clustering=clustering_seed)
    learner_faiss= ProbCoverSampler_Faiss(circles_data, 0.9,
                                         clustering, [True, False],
                                         search_range=[0, 10], search_step=0.01)
    learner_faiss.query(5)
    learner_faiss.query(5)
    print(circles_data.queries)

    print(accuracy_clustering(learner.pseudo_labels, learner_faiss.pseudo_labels))

def check_probcover_spectral():
    clustering_seed= np.random.randint(1000)

    circles_data.restart()
    clustering= MySpectralClustering(circles_data, 3, gamma= 3, random_clustering=clustering_seed)
    learner_faiss= ProbCoverSampler_Faiss(circles_data, 0.9,
                                         clustering, [True, False],
                                         search_range=[0, 10], search_step=0.01)
    learner_faiss.query(5)
    learner_faiss.query(5)
    print(circles_data.queries)


    circles_data.restart()
    clustering= MySpectralClustering(circles_data, 3, gamma= 3, random_clustering=clustering_seed)
    learner= ProbCoverSampler(circles_data, 0.9,
                                         clustering, [True, False],
                                         search_range=[0, 10], search_step=0.01)
    learner.query(5)
    learner.query(5)
    print(circles_data.queries)

    print(accuracy_clustering(learner.pseudo_labels, learner_faiss.pseudo_labels))


