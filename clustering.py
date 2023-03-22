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
    def __init__(self, dataset, n_clusters, random_clustering=None):
        self.n_clusters=n_clusters
        self.dataset= dataset
        self.random_clustering= random_clustering


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

class OracleClassifier(ClusteringAlgo):
    def __init__(self, dataset, random_clustering=None):
        n_clusters= np.unique(dataset.y).shape[0]
        super(OracleClassifier, self).__init__(dataset, n_clusters, random_clustering)
        self.name="Ground truth classifier"
        self.pseudo_labels= self.dataset.y


class MyKMeans(ClusteringAlgo):
    def __init__(self, dataset, n_clusters, random_clustering=None):
        super(MyKMeans, self).__init__(dataset, n_clusters, random_clustering)
        self.name="Kmeans"
        kmeans = KMeans(n_clusters=self.n_clusters, random_state= self.random_clustering).fit(self.dataset.x)
        self.pseudo_labels= kmeans.labels_

class MySpectralClustering(ClusteringAlgo):
    def __init__(self, dataset, n_clusters, gamma, alpha=1, random_clustering=None, enforce_closeness= True, enforce_separability=True):
        super(MySpectralClustering, self).__init__(dataset, n_clusters, random_clustering)
        self.gamma=gamma
        self.name= "Spectral clustering"
        self.enforce_closeness= enforce_closeness
        self.enforce_separability= enforce_separability
        self.alpha= alpha
        spectralclustering = SpectralClustering(n_clusters=self.n_clusters, n_init=10, gamma=self.gamma, affinity='rbf',
                                                assign_labels='kmeans', random_state=self.random_clustering).fit(
            self.dataset.x)
        self.affinity_matrix= spectralclustering.affinity_matrix_
        self.pseudo_labels = spectralclustering.labels_

    def fit_labeled(self, sampled_idx):
        for u in sampled_idx:
            for v in sampled_idx:
                if (self.dataset.y[u] == self.dataset.y[v]) & (u != v) & self.enforce_closeness:
                    self.affinity_matrix[u, v] = self.alpha
                    self.affinity_matrix[v, u] = self.alpha
                elif (self.dataset.y[u] != self.dataset.y[v]) & self.enforce_separability:
                    self.affinity_matrix[u, v] = 0
                    self.affinity_matrix[v, u] = 0

        spectralclustering = SpectralClustering(n_clusters=self.n_clusters, n_init=10, affinity='precomputed',
                                                assign_labels='kmeans', random_state=self.random_clustering).fit(self.affinity_matrix)
        self.pseudo_labels = spectralclustering.labels_

