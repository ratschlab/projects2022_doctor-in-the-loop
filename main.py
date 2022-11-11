import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import seaborn as sns
from activelearners import ActiveLearner, RandomSampler, TypiclustSampler
from datasets import PointClouds


m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)
M =15
B=5
K=5
show_all_clusters=True

# Initialize the dataset and the active learners
clustered_2d_data= PointClouds(cluster_centers, cluster_std, cluster_samples)
random_learner= RandomSampler(dataset=clustered_2d_data.dataset)
typiclust_learner=TypiclustSampler(dataset=clustered_2d_data.dataset, n_neighbours=K)

#Demo plots, showing all clusters in Typiclust or only the B largest uncovered clusters
def figure1():
    clustered_2d_data.plot2d()
    typiclust_learner.query(M, B, plot=[True,False])
    typiclust_learner.query(M, B, plot=[False, True])

## Plots of random and typiclust sampling process
def figure2():
    random_learner.demo_2dplot(30,5, all_plots=True)

def figure3():
    typiclust_learner.demo_2dplot(30,5, all_plots=True)

def figure4(cluster_std=[0.3, 1], cluster_centers= [(0,0), (4,1)], incr=20, m=200, B=5, plot_points=True):
    assert(m%incr==0)
    for k in range(1, int(m / incr)):
        cluster_samples= np.array([k * incr, m - k * incr])
        clustered_points= PointClouds(cluster_centers, cluster_std, cluster_samples)
        dataset=clustered_points.dataset

        if plot_points==True:
            sns.scatterplot(data=dataset[dataset["y"] == 0], x="x1", y="x2",
                            label=f"0: {k * incr} samples and {cluster_std[0]} std")
            sns.scatterplot(data=dataset[dataset["y"] == 1], x="x1", y="x2",
                            label=f"1: {m - k * incr} samples and {cluster_std[1]} std")
            plt.show()

        typiclust_learner = TypiclustSampler(dataset=dataset, n_neighbours=K)
        queries_idx = typiclust_learner.query(m, B)
        n_iter= int(200 / B)
        queries_per_cluster = np.zeros(shape=(n_iter, 2))

        for i in range(n_iter):
            temp = dataset.iloc[queries_idx[i*B: (i+1)*B]]
            for y in range(2):
                queries_per_cluster[i,y] = sum(temp["y"] == y)

        queries_per_cluster = np.cumsum(queries_per_cluster, axis=0)
        sns.lineplot(x=np.arange(n_iter), y=queries_per_cluster[:n_iter, 0],
                     label="std:{}, samples:{}".format(cluster_std[0], cluster_samples[0]))
        sns.lineplot(x=np.arange(n_iter), y=queries_per_cluster[:n_iter, 1],
                     label="std:{}, samples:{}".format(cluster_std[1], cluster_samples[1]))
        plt.show()

def figure5():
    figure4(cluster_std=[0.3, 1], cluster_centers=[(0, 0), (4, 1)], incr=20, m=200, B=5)
    figure4(cluster_std=[0.9, 3], cluster_centers=[(0, 0), (4, 1)], incr=20, m=200, B=5)
    figure4(cluster_std=[0.3, 1], cluster_centers=[(0, 0), (0, 0)], incr=20, m=200, B=5)

def figure6():
    figure4(cluster_std=[0.3, 0.7], cluster_centers=[(0, 0), (4, 1)], incr=20, m=200, B=5, plot_points=False)
    figure4(cluster_std=[0.3, 0.5], cluster_centers=[(0, 0), (4, 1)], incr=20, m=200, B=5, plot_points=False)
    figure4(cluster_std=[0.3, 0.3], cluster_centers=[(0, 0), (4, 1)], incr=20, m=200, B=5, plot_points=False)

figure6()













