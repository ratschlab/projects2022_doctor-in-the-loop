import numpy as np
from datasets import CenteredCircles, PointClouds, MixedClusters, CIFAR10_simclr, TwoMoons
from activelearners import ProbCoverSampler, active_learning_algo
from clustering import MySpectralClustering, MyKMeans, ClusteringAlgo
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
from models import Classifier1NN
from helper import check_cover, get_radius, cover


## Initialize the circles dataset
center=[4,5]
radiuses=[0.5, 3, 5]
samples= [30, 100 ,150]
std=[0.5, 0.5, 0.55]

circles_data=CenteredCircles(center, radiuses, samples, std)
#circles_data.plot_dataset()


## Initialize the point clouds dataset
m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)

clouds_data= PointClouds(cluster_centers, cluster_std, cluster_samples)

## Initialize the mixed clusters dataset
m = 400
cluster_centers = [(0, -5), (-6, 0), (5, 4)]
cluster_std = [0.5, 0.8, 1.2]
p=np.array([0.3, 0.4, 0.3])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)

mixed_clusters= MixedClusters(cluster_centers, cluster_std, cluster_samples)


## Initialize CIFAR10 extracted features dataset
cifar10_features= CIFAR10_simclr(n_epochs=100)


# Experiments on the circles dataset
def experiment1():
    clustering_spectral = MySpectralClustering(circles_data, 3, 1)
    clustering_kmeans = MyKMeans(circles_data, 3)

    active_learning_algo(circles_data, clustering_spectral, 150, 5, 0.9, 0.5)
    active_learning_algo(circles_data, clustering_kmeans, 150, 5, 0.9, 0.5)


def experiment2(dataset, M, B, k, clustering_method: str, initial_purity_threshold, p_cover,
                gamma,
                plot_clustering= False):
    dataset.restart()
    accuracy= []
    radiuses= []
    covers= []
    if clustering_method=="spectral":
        clustering= MySpectralClustering(dataset, k, gamma)
    elif clustering_method=="kmeans":
        clustering = MyKMeans(dataset, k)

    activelearner = ProbCoverSampler(dataset, initial_purity_threshold,
                                     clustering, [False, False],
                                     search_range=[0, 10], search_step=0.01)
    model = Classifier1NN(dataset)
    covered = check_cover(dataset.x, dataset.queries, activelearner.radius, p_cover)

    while (len(dataset.queries)<=M-B)&(not covered):
        activelearner.query(B)
        activelearner.update_labeled(plot_clustering)
        model.update()
        covers.append(cover(dataset.x, dataset.queries, activelearner.radius))
        radiuses.append(activelearner.radius)
        accuracy.append(model.accuracy)
        covered= check_cover(dataset.x, dataset.queries, activelearner.radius, p_cover)

    return np.array(covers), np.array(radiuses), np.array(accuracy)

def experiment3(dataset, M, B, k, threshold, p_cover, gamma, plot_lowerbound= False):
    dataset.restart()
    cover_spectral, r_spectral, accuracy_spectral= experiment2(dataset, M, B, k, "spectral", threshold,  p_cover, gamma)
    cover_kmeans, r_kmeans, accuracy_kmeans= experiment2(dataset, M, B, k, "kmeans", threshold, p_cover, gamma)

    sns.lineplot(x= np.arange(len(accuracy_spectral)), y= accuracy_spectral, label="Spectral")
    sns.lineplot(x= np.arange(len(accuracy_kmeans)), y= accuracy_kmeans, label="Kmeans")
    if plot_lowerbound:
        sns.lineplot(x= np.arange(len(r_spectral)), y=cover_spectral+r_spectral-1, label= "Spectral lower bound on accuracy")
        sns.lineplot(x= np.arange(len(r_kmeans)), y=cover_kmeans+r_kmeans-1, label= "Kmeans lower bound on accuracy")
    plt.title(f"Accuracy of a nearest neighbour classifier on dataset {dataset.name}")
    plt.show()


######## Experiments

M = 150
B = 5
threshold = 0.9
p_cover = 1
gamma=3


embed()
experiment3(circles_data, M, B, 3, threshold, p_cover, gamma)
experiment3(circles_data, M, B, 3, threshold, p_cover, gamma, True)

experiment2(cifar10_features, 1000, 50, 10, "kmeans", 0.9, 0.3, 3)
# experiment3(mixed_clusters, M, B, 2, threshold, p_cover, gamma)
# experiment3(clouds_data, M, B, 4, threshold, p_cover, gamma)
