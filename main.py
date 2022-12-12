import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import seaborn as sns
from activelearners import ActiveLearner, RandomSampler, TypiclustSampler, ProbCoverSampler
from datasets import PointClouds
from helper import get_emperical_radius, estimate_emperical_purity, accuracy_clustering


m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
# cluster_std=[0.2, 0.5, 0.5, 0.5]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)
M =15
B=5
K=5
show_all_clusters=True
clustered_2d_data = PointClouds(cluster_centers, cluster_std, cluster_samples)
df= clustered_2d_data.dataset
embed()
def figure1():
    ## Initializing the dataset
    clustered_2d_data= PointClouds(cluster_centers, cluster_std, cluster_samples)
    clustered_2d_data.plot2d()

    ## Initializing the active learners
    random_learner= RandomSampler(dataset=clustered_2d_data.dataset)
    typiclust_learner= TypiclustSampler(dataset=clustered_2d_data.dataset, n_neighbours=K)
    probcover_learner_1= ProbCoverSampler(dataset=clustered_2d_data.dataset, purity_radius=2,purity_threshold=None, k=4,  plot=[True, True])
    probcover_learner_2= ProbCoverSampler(dataset=clustered_2d_data.dataset, purity_radius=None, purity_threshold=0.95, k=4,plot=[True, True],
                                         search_range=[2.3,3], search_step=0.1)
    ## ProbCover demo

    probcover_learner_1.demo_2dplot(10,5, final_plot=True)
    probcover_learner_2.demo_2dplot(10, 5, final_plot=True)

def analysing_typiclust_transition():
    cluster_std=[0.3, 1]
    cluster_centers=[(0, 0), (4, 1)]
    incr=20
    m=200
    B=1
    K=5
    assert (m % incr == 0)
    for k in range(1, int(m / incr)):
        cluster_samples = np.array([k * incr, m - k * incr])
        clustered_points = PointClouds(cluster_centers, cluster_std, cluster_samples)
        dataset = clustered_points.dataset

        typiclust_learner = TypiclustSampler(dataset=dataset, n_neighbours=K)
        queries_idx = typiclust_learner.query(m, B)
        n_iter = int(200 / B)
        queries_per_cluster = np.zeros(shape=(n_iter, 2))

        for i in range(n_iter):
            temp = dataset.iloc[queries_idx[i * B: (i + 1) * B]]
            for y in range(2):
                queries_per_cluster[i, y] = sum(temp["y"] == y)

        queries_per_cluster = np.cumsum(queries_per_cluster, axis=0)
        embed()

figure1()


