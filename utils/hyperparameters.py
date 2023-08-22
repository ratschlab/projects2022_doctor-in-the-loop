import numpy as np
import pandas as pd
import faiss
from utils.data import get_data
import matplotlib.pyplot as plt
import seaborn as sns
from utils.faiss_graphs import adjacency_graph_faiss
from clustering import MyKMeans
import os
import math

def floor_twodecimals(value):
    rounded = math.floor(value * 100) / 100
    return rounded

def get_init_radiuses(alpha, dataset, pseudo_labels= None):
    if pseudo_labels is None:
        pseudo_labels= dataset.y
    radiuses= np.zeros(shape= (dataset.n_points,))
    for c in np.unique(pseudo_labels):
        idx_not_c= np.where(pseudo_labels!=c)[0]
        idx_c= np.where(pseudo_labels==c)[0]

        index = faiss.IndexFlatL2(dataset.x.shape[1])  # build the index
        index.add(dataset.x[idx_not_c].astype('float32'))
        D, I = index.search(dataset.x[idx_c].astype('float32'), 1)
        D= np.sqrt(D)
        radiuses[idx_c]= D.squeeze()

    quant= np.quantile(radiuses, 1-alpha)
    return floor_twodecimals(quant), radiuses


def get_purity_faiss(x, pseudo_labels, radius, plot_unpure_balls=False):
    """
    Args:
        x: np.array
        pseudo_labels: labels predicted by some clustering algorithm
        radius: purity radius

    Returns:
        purity(radius)
    """
    purity = np.zeros(len(x), dtype=int)
    lims, D, I = adjacency_graph_faiss(x, radius)

    for u in range(len(x)):
        covered_vertices = I[lims[u]:lims[u + 1]]
        purity[u] = int(np.all(pseudo_labels[covered_vertices] == pseudo_labels[u]))

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

    # Radiuses for the search
    radiuses = np.arange(search_range[0], search_range[1], search_step)

    while len(radiuses) > 1:
        id = int(len(radiuses) / 2)
        purity = get_purity_faiss(x, pseudo_labels, radiuses[id])

        if purity >= alpha:
            radiuses = radiuses[id:]
        else:
            radiuses = radiuses[:id]
    purity_radius = radiuses[0]

    if plot_unpure_balls:
        get_purity_faiss(x, pseudo_labels, purity_radius, True)

    return purity_radius



# args= {"dataset": "cifar100", "n_epochs": 100, "algorithm": "adpc", "separable": "not", "radius": 0.9,
#        "run": "a", "gamma": 0.5, "reduction_method": "pessimistic", "gauss": 4, "hard_threshold": 0.0, "sd": 1,
#        "tsh": 0.95}
# args= pd.Series(args)

# for epochs in [100,200,400,1000]:
#     for sd in [1,2,3,4,5,6,7,8,9,10]:
#         args.n_epochs= epochs
#         args.sd= sd
#         np.random.seed(args.sd)
#         dataset, _, _, _ = get_data(args)
#         # _, _, true_radiuses= get_fullpurity_radiuses(args, "true")
#         pseudo_labels= MyKMeans(dataset, dataset.C).pseudo_labels
#         quantile_75, hard_thresh, kmeans_radiuses= get_fullpurity_radiuses(args, "kmeans_radiuses", pseudo_labels, compute=True)
#         print(epochs, sd, hard_thresh, quantile_75)
#
#
#
