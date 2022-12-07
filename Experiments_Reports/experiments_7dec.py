import numpy as np
from datasets import CenteredCircles, PointClouds, ActiveDataset
from activelearners import ProbCoverSampler, SpectralProbCover
from clustering import MySpectralClustering, MyKMeans, ClusteringAlgo
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
from helper import accuracy_clustering, check_cover, get_purity, get_radius, get_nearest_neighbour, accuracy_score


## Initialize the circles dataset
center=[4,5]
radiuses=[0.5, 3, 5]
samples= [30, 50 ,100]
std=[0.1, 0.15, 0.3]

circles_data=CenteredCircles(center, radiuses, samples, std)
circles_data.plot_dataset()
circles_data.plot_al()

## Initialize the point clouds dataset
m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
# cluster_std=[0.2, 0.5, 0.5, 0.5]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)

clouds_data= PointClouds(cluster_centers, cluster_std, cluster_samples)

# Experiments on the circles dataset

def active_learning_algo(dataset, M, B, k, gamma, initial_purity_threshold, p_cover=1):
    dataset.restart()
    activelearner= SpectralProbCover(dataset=circles_data, purity_radius=None, purity_threshold=initial_purity_threshold, k=3, gamma=gamma,
                                     plot=[True, True],
                                     search_range=[0,10], search_step=0.01)

    purity_threshold= initial_purity_threshold
    nn_idx= get_nearest_neighbour(dataset.x)

    #Clustering pseudo-labels
    clustering= MySpectralClustering(dataset, k, gamma)
    clustering.fit()
    clustering.plot()
    pseudo_labels= clustering.pseudo_labels

    #Purity radius
    purity_radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
    covered= check_cover(dataset.x, dataset.queries, purity_radius, p_cover)

    while len(dataset.queries)<M:
        if ((not covered) & (len(dataset.queries)+B<=M)):
            print("Dataset not covered yet")
            activelearner.query(B)
            covered= check_cover(dataset.x, dataset.queries, purity_radius, p_cover)
        elif covered:
            print(f"Dataset covered with {len(dataset.queries)} queries")
            clustering= MySpectralClustering(dataset, k, gamma)
            clustering.fit(dataset.queries)
            clustering.plot()
            old_labels= pseudo_labels
            old_radius= purity_radius
            pseudo_labels= clustering.pseudo_labels
            while covered:
                old_radius = purity_radius

                changed=(accuracy_clustering(old_labels, pseudo_labels)<1)
                if changed:
                    purity_radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    print(f"The labels were changed with {100*accuracy_clustering(old_labels, pseudo_labels)}% overlap and the new radius is {purity_radius}")
                else:
                    purity_threshold+=(1-purity_threshold)/2
                    purity_radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    print(f"The labels were not changed so we set a higher purity threshold {purity_threshold} with radius {purity_radius}")

                if purity_radius> old_radius:
                    print(f"Radius is bigger, so we raise purity threshold and start the process again")
                    purity_threshold+=(1-purity_threshold)/2
                    purity_radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    activelearner.purity_radius= purity_radius
                    print(f'Changed active learners purity threshold to {purity_threshold} and radius to {purity_radius}')
                    covered= check_cover(dataset.x, dataset.queries, purity_radius, p_cover)

                elif purity_radius< old_radius:
                    covered= check_cover(dataset.x, dataset.queries, purity_radius, p_cover)
                    print(f"Radius is smaller")
                    if not covered:
                        activelearner.purity_radius= purity_radius
                        print(f'Changed active learners purity threshold to {purity_threshold} '
                              f'and radius to {purity_radius}')

                elif purity_radius==old_radius:
                    print("Radius is the same")
                    return purity_radius



active_learning_algo(circles_data, 40, 5, 3, 1, 0.9, 1)