from datasets import MixedClusters, PointClouds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed
from clustering import OracleClassifier, MyKMeans, MySpectralClustering
from activelearners import ProbCoverSampler_Faiss
from copy import deepcopy
from activelearners import ProbCoverSampler


cluster_centers= np.array([[0,0]])
cluster_std= [0.3]
cluster_samples= np.array([400])
dataset= MixedClusters(3, cluster_centers, cluster_std, cluster_samples, random_state=1)

# dataset= PointClouds([[1,2], [2,3], [0.75,3.25]], [0.4, 0.4, 0.4], np.array([200,200, 200]), random_state=1)
dataset.plot_dataset()
clustering= MyKMeans(dataset, 2)
learner= ProbCoverSampler(dataset, 0.95, clustering)
learner.update_radius(0.3)
learner.adaptive_query(15, K=5)
dataset.plot_al(plot_circles=True)


if __name__ == '__main__':
    cluster_centers = np.array([[0, 0]])
    cluster_std = [0.3]
    cluster_samples = np.array([400])
    dataset = MixedClusters(3, cluster_centers, cluster_std, cluster_samples, random_state=1)

    # dataset= PointClouds([[1,2], [2,3], [0.75,3.25]], [0.4, 0.4, 0.4], np.array([200,200, 200]), random_state=1)
    dataset.plot_dataset()
    clustering = MyKMeans(dataset, 2)
    learner = ProbCoverSampler(dataset, 0.95, clustering)
    learner.update_radius(0.3)
    learner.adaptive_query(15, K=5)
    dataset.plot_al(plot_circles=True)
