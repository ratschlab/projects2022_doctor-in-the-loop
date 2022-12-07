import numpy as np
from datasets import CenteredCircles, PointClouds
from activelearners import ProbCoverSampler, SpectralProbCover
from clustering import SemiSpectralClustering, SemiKMeans, ClusteringAlgo
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
from helper import accuracy_clustering

## Initialize the circles dataset
center=[4,5]
radiuses=[0.5, 3, 5]
samples= [30, 50 , 100]
std=[0.1, 0.15, 0.3]

circles_data=CenteredCircles(center,radiuses, samples, std)


## Initialize the point clouds dataset
m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
# cluster_std=[0.2, 0.5, 0.5, 0.5]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)

clouds_data= PointClouds(cluster_centers, cluster_std, cluster_samples)



def figure1():
    # Experiments on the circles dataset
    query_idx= np.random.randint(180, size= 10)
    true_labels= np.array(circles_data.dataset["y"])


    # Spectral Clustering with wrong hyperparameter
    spectral= SemiSpectralClustering(dataset=circles_data.dataset, n_clusters=3, gamma=2)
    spectral.fit()
    spectral.plot()
    accuracy_wrong_spectral= accuracy_clustering(true_labels, spectral.pseudo_labels)

    spectral.fit(query_idx)
    spectral.plot()
    accuracy_wrong_spectral_labeled= accuracy_clustering(true_labels, spectral.pseudo_labels)

    # Spectral Clustering with right hyperparameter
    spectral= SemiSpectralClustering(dataset=circles_data.dataset, n_clusters=3, gamma=5)
    spectral.fit()
    spectral.plot()
    accuracy_right_spectral= accuracy_clustering(true_labels, spectral.pseudo_labels)

    spectral.fit(query_idx)
    spectral.plot()
    accuracy_right_spectral_labeled= accuracy_clustering(true_labels, spectral.pseudo_labels)


    # KMeans
    kmeans= SemiKMeans(dataset=circles_data.dataset, n_clusters=3)
    kmeans.fit()
    kmeans.plot()


    accuracy_kmeans= accuracy_clustering(true_labels, kmeans.pseudo_labels)

    print(f"Accuracy of kmeans: {accuracy_kmeans} \n"
          f"Accuracy of spectral clustering with bad hyperparameter and no labels: {accuracy_wrong_spectral}\n"
          f"Accuracy of spectral clustering with bad hyperparameter and {len(query_idx)} labels: {accuracy_wrong_spectral_labeled}\n"
          f"Accuracy of spectral clustering with good hyperparameter and no labels: {accuracy_right_spectral}\n"
          f"Accuracy of spectral clustering with good hyperparameter and {len(query_idx)} labels: {accuracy_right_spectral_labeled}")



def figure2():
    # Experiments on the point clouds dataset
    query_idx= np.random.randint(180, size= 10)

    # Spectral Clustering with wrong hyperparameter
    spectral= SemiSpectralClustering(dataset=clouds_data.dataset, n_clusters=4, gamma=2)
    spectral.fit()
    spectral.plot()
    spectral.fit(query_idx)
    spectral.plot()

    # Spectral Clustering with right hyperparameter
    spectral= SemiSpectralClustering(dataset=clouds_data.dataset, n_clusters=4, gamma=5)
    spectral.fit()
    spectral.plot()
    spectral.fit(query_idx)
    spectral.plot()

    # KMeans
    kmeans= SemiKMeans(dataset=clouds_data.dataset, n_clusters=4)
    kmeans.fit()
    kmeans.plot()

    true_labels= np.array(circles_data.dataset["y"])
    pseudo_labels= spectral.pseudo_labels


probcover_learner= ProbCoverSampler(dataset=circles_data.dataset, purity_radius=None, purity_threshold=0.95, k=3, plot=[True, False],
                                         search_range=[0,5], search_step=0.01)



probcover_learner_spectral= SpectralProbCover(dataset=circles_data.dataset, purity_radius=None, purity_threshold=0.95, k=3, gamma=5, plot=[True, False],
                                         search_range=[0,5], search_step=0.01)


probcover_learner.demo_2dplot(15, 5, final_plot=True)
probcover_learner_spectral.demo_2dplot(15, 5, final_plot=True)



embed()

###################
# print("weighted kmeans")
#
# c_weightedkmeans, l_weightedkmeans= weighted_kmeans(circles_data.dataset, k=3,
#                                                     sampled_labels=query_idx,
#                                                     labeled_importance=0.6,
#                                                     plot_clustering=True)
#
#
# c_kmeans, l_kmeans= kmeans(circles_data.dataset, 3, plot_kmeans=True)
# query_idx= probcover_learner.query(10,5)


