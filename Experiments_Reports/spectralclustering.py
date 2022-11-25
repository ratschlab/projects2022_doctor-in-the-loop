import numpy as np
import pandas as pd
import scipy.linalg
from datasets import PointClouds
from IPython import embed
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.5, 1.2, 1.2, 1.2]
# cluster_std=[0.2, 0.5, 0.5, 0.5]
p=np.array([0.3, 0.2, 0.2, 0.3])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)
M =15
B=5
K=5
show_all_clusters=True
clustered_2d_data = PointClouds(cluster_centers, cluster_std, cluster_samples)
df= clustered_2d_data.dataset

def exp_kernel(x1, x2, bw):
    """
    Args:
        x1,x2: two np.array
        bw: bandwidth parameter in the exponential kernel
    """
    K=np.linalg.norm(x1 - x2)**2
    return np.exp(-(K**2)/(2*bw))


def weighted_graph(dataset: pd.DataFrame, kernel_fn, kernel_hyperparam, sampled_labels=None):
    n_vertices = len(dataset)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        for v in range(u):
            if sampled_labels is not None:
                if (u in sampled_labels)&(v in sampled_labels):
                    if (dataset.iloc[u]["y"] == dataset.iloc[v]["y"]) & (u != v):
                        graph[u,v]= 1000
                        graph[v,u]= 1000
                        # otherwise the weight is kept as zero
                else:
                    w= kernel_fn(dataset.iloc[u, :-1], dataset.iloc[v,:-1], kernel_hyperparam)
                    graph[u, v] = w
                    graph[v, u] = w
            else:
                w = kernel_fn(dataset.iloc[u, :-1], dataset.iloc[v, :-1], kernel_hyperparam)
                graph[u, v] = w
                graph[v, u] = w
    return graph


def degree_matrix(graph_adjacency):
    D= np.zeros(shape=graph_adjacency.shape)
    n_vertices= D.shape[0]
    for i in range(n_vertices):
        D[i,i]= np.sum(graph_adjacency[i, :])
    return D


def spectral_clustering(dataset, k, kernel_fn, kernel_hyperparam, sampled_labels=None):
    W= weighted_graph(dataset, kernel_fn, kernel_hyperparam, sampled_labels)
    D= degree_matrix(W)
    L= D-W
    D_inv= scipy.linalg.inv(D)
    D_inv_sqrt= scipy.linalg.sqrtm(D_inv, disp=True, blocksize=64)
    LG= np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt)
    eig, v= scipy.linalg.eig(LG)

    id= np.argsort(eig)[1:k]
    v2= v[:,id]
    phi2= np.matmul(D_inv_sqrt, v2)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(phi2)
    pseudo_labels=kmeans.labels_
    return pseudo_labels

def weighted_kmeans(dataset, k, sampled_labels, labeled_importance, tol=0.00001):

    #TODO: Implement the case np.unique(sampled_labels).shape<k, then how do we initialize?

    # Initialise the centroids
    centers= np.array(dataset.iloc[sampled_labels].groupby("y").mean())
    update=tol+10
    while update>tol:
        # Cluster assignments
        distance=np.zeros(shape=(len(dataset), k))
        for cluster in range(k):
            center= centers[cluster, :]
            distance[:,cluster]=np.linalg.norm(dataset.iloc[:,0:2]-center, axis=1)
        cluster_assignments= np.argmin(distance, axis=1)
        dataset["cluster"]= cluster_assignments

        # Weights proportional to cluster size
        n_cluster_total= np.array(dataset.groupby("cluster").count().iloc[:,1])
        n_cluster_labeled = np.array(dataset.iloc[sampled_labels, :].groupby("cluster").count().iloc[:, 1])
        weights= np.zeros(len(dataset))
        labeled = np.zeros(len(dataset))
        labeled[sampled_labels] = 1

        for cluster in range(k):
            weights[np.where((cluster_assignments==cluster)&(labeled==1))[0]]= labeled_importance/n_cluster_labeled[cluster]
            weights[np.where((cluster_assignments==cluster)&(labeled==0))[0]]= labeled_importance/(n_cluster_total[cluster]-n_cluster_labeled[cluster])

        old_centers= centers
        centers= dataset.iloc[:,0:2].multiply(weights, axis="index")
        centers["cluster"]=dataset["cluster"]
        centers= np.array(centers.groupby("cluster").sum())

        update= np.max(np.linalg.norm(old_centers-centers, axis=1))
        print(update)


    pseudo_labels=dataset["cluster"]
    dataset=dataset.drop(columns="cluster")

    return pseudo_labels










idx=np.random.choice(400,40, replace=False)
sampled_labels=np.concatenate([idx.reshape(-1,1), df.iloc[idx]["y"].values.reshape(-1,1)],axis=1)

embed()
pseudo_labels, weights, all, labeled, centers, = weighted_kmeans(df, 4, idx, 0.6)
pseudo_semisupervised= spectral_clustering(df, 4, exp_kernel, 2, idx)
pseudo_unsupervised=spectral_clustering(df, 4, exp_kernel, 2)
true_labels=df["y"].values


sns.scatterplot(data=df, x="x1", y="x2", hue=pseudo_semisupervised, palette="Set2")
plt.show()
sns.scatterplot(data=df, x="x1", y="x2", hue=pseudo_unsupervised, palette="Set2")
plt.show()

pseudo_semisupervised==pseudo_unsupervised