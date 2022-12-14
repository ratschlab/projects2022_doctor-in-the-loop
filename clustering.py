import numpy as np
import pandas as pd
import scipy.linalg
# from datasets import PointClouds
from IPython import embed
from sklearn.cluster import KMeans, SpectralClustering
import seaborn as sns
import matplotlib.pyplot as plt
from helper import weighted_graph


class ClusteringAlgo:
    def __init__(self, dataset, n_clusters):
        self.n_clusters=n_clusters
        self.dataset= dataset

    def plot(self):
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=self.dataset.x[:, 0], y=self.dataset.x[:, 1], hue=self.pseudo_labels, palette="Set2")

        sns.scatterplot(x=self.dataset.x[self.sampled_idx, 0], y=self.dataset.x[self.sampled_idx, 1], color="red", marker="P", s=150)

        if self.name=="Spectral clustering":
            plt.title(f"Spectral clustering with gamma {self.gamma} and {len(self.sampled_idx)} queries")
        else:
            plt.title(f"{self.name}")
        plt.show()


class MyKMeans(ClusteringAlgo):
    def __init__(self, dataset, n_clusters):
        super(MyKMeans, self).__init__(dataset, n_clusters)
        self.name="Kmeans"

    def fit_labeled(self, sampled_idx=np.array([], dtype=int)):
        self.sampled_idx=sampled_idx
        #Does not take sampled_idx into account
        kmeans = KMeans(n_clusters=self.n_clusters).fit(self.dataset.x)
        self.pseudo_labels= kmeans.labels_
        self.centroids= kmeans.cluster_centers_


class MySpectralClustering(ClusteringAlgo):
    def __init__(self, dataset, n_clusters, gamma):
        super(MySpectralClustering, self).__init__(dataset, n_clusters)
        self.gamma=gamma
        self.name= "Spectral clustering"

    def fit_labeled(self, sampled_idx=np.array([], dtype=int)):
        self.sampled_idx= sampled_idx
        spectralclustering = SpectralClustering(n_clusters=self.n_clusters, n_init=10, gamma=self.gamma, affinity='rbf',
                                                assign_labels='kmeans').fit(self.dataset.x)
        if len(self.sampled_idx)==0:
            self.pseudo_labels= spectralclustering.labels_
        elif len(self.sampled_idx)>0:
            affinity_matrix = spectralclustering.affinity_matrix_
            labels = np.array(self.dataset.y[sampled_idx]).reshape(-1, 1)
            for u in sampled_idx:
                for v in sampled_idx:
                    if (self.dataset.y[u] == self.dataset.y[v]) & (u != v):
                        affinity_matrix[u, v] = 1
                        affinity_matrix[v, u] = 1
                    elif (self.dataset.y[u] != self.dataset.y[v]):
                        affinity_matrix[u, v] = 0
                        affinity_matrix[v, u] = 0

            spectralclustering = SpectralClustering(n_clusters=self.n_clusters, n_init=10, affinity='precomputed',
                                                    assign_labels='kmeans').fit(affinity_matrix)
            self.pseudo_labels = spectralclustering.labels_



