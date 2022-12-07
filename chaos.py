from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

def logistic_regression_boundary(data, idx, title:str, show_plot=False, C=1, multi_class="multinomal", max_iter=1000):
    X= np.array(data.drop(['y', 'cluster_id'], axis=1))
    y= np.array(data["y"], dtype=int)

    logistic_reg = LogisticRegression(C=1, multi_class='multinomial', max_iter=1000)
    logistic_reg.fit(X[idx,:], y[idx])
    if show_plot:
        plot_decision_regions(X,y, clf=logistic_reg, scatter_kwargs={'s':0})
        sns.scatterplot(data=X[idx,:], x=X[idx,0], y=X[idx,1], color="black", marker="+", s=150)
        plt.title(title)
        plt.show()

    return logistic_reg.score(X[idx,:], y[idx]), logistic_reg.score(X,y)



def weighted_kmeans(dataset, k, sampled_labels, labeled_importance, tol=0.01, max_iter=1000000, plot_clustering=False):
    # Initialise the centroids
    ## Initialise the centroids as the mean of all labeled points in a class (closest point so that a cluster is never empty)
    df= dataset.copy()
    id_centers= []
    centers= np.array(df.iloc[sampled_labels].groupby("y").mean())
    for i, center in enumerate(centers):
        distance = np.linalg.norm(df.iloc[:, 0:2] - center, axis=1)
        id = np.argmin(distance)
        l = 2
        while id in id_centers:
            id = np.argpartition(distance, range(l))[l - 1]
            l += 1
        id_centers.append(id)
        centers[i] = np.array(df.iloc[id, 0:2])

    ## if some classes had no labels, we initialize the remaining centroids as random points
    if centers.shape[0]<k:
        centers_idx= np.random.choice(np.array(list(set(np.arange(180))-set(id_centers))),
                                       size=k-centers.shape[0], replace=False)
        centers_random= np.array(df.iloc[centers_idx, 0:2])
        centers= np.concatenate([centers, centers_random], axis=0)

    update=tol+10
    iter=0
    while (update>tol)&(iter<max_iter):
        print(iter)
        # Cluster assignments
        distance=np.zeros(shape=(len(df), k))
        for cluster in range(k):
            center= centers[cluster, :]
            distance[:,cluster]=np.linalg.norm(df.iloc[:,0:2]-center, axis=1)
        cluster_assignments= np.argmin(distance, axis=1)
        df["cluster"]= cluster_assignments

        # Weights proportional to cluster size
        n_cluster_labeled = np.zeros(np.unique(df["y"]).shape)
        n_cluster_total = np.zeros(np.unique(df["y"]).shape)
        for i, label in enumerate(np.unique(df["y"])):
            n_cluster_total[i]= len(df.loc[df["cluster"] == label])
            n_cluster_labeled[i]= len(df.iloc[sampled_labels, :].loc[df['y'] == label])

        weights= np.zeros(len(df))
        labeled = np.zeros(len(df))
        labeled[sampled_labels] = 1

        for cluster in range(k):
            weights[np.where((cluster_assignments==cluster)&(labeled==1))[0]]= labeled_importance/n_cluster_labeled[cluster]
            weights[np.where((cluster_assignments==cluster)&(labeled==0))[0]]= labeled_importance/(n_cluster_total[cluster]-n_cluster_labeled[cluster])

        old_centers= centers
        centers= df.iloc[:,0:2].multiply(weights, axis="index").groupby(df["cluster"]).sum()
        centers= np.array(centers)
        id_centers=[]
        for i, center in enumerate(centers):
            distance = np.linalg.norm(df.iloc[:, 0:2] - center, axis=1)
            id = np.argmin(distance)
            l = 2
            while id in id_centers:
                id = np.argpartition(distance, range(l))[l - 1]
                l += 1
            id_centers.append(id)
            centers[i] = np.array(df.iloc[id, 0:2])

        ## if some classes had no labels, we initialize the remaining centroids as random points
        if centers.shape[0] < k:
            centers_idx = np.random.choice(np.array(list(set(np.arange(180)) - set(id_centers))),
                                           size=k - centers.shape[0], replace=False)
            centers_random = np.array(df.iloc[centers_idx, 0:2])
            centers = np.concatenate([centers, centers_random], axis=0)

        update= np.max(np.linalg.norm(old_centers-centers, axis=1))
        iter+=1

    pseudo_labels=df["cluster"]

    if plot_clustering:
        sns.scatterplot(data=dataset, x="x1", y="x2", hue=pseudo_labels, palette="Set2")
        sns.scatterplot(data=dataset.iloc[sampled_labels,:],  x="x1", y="x2", color="red", marker="P", s=150)
        sns.scatterplot(x=centers[:,0], y=centers[:,1], color="black", label="Centers")
        plt.title(f"Pseudo-labels from k-means clustering with label importance {labeled_importance}")
        plt.show()

    return centers, pseudo_labels



def my_spectral_clustering(dataset, k, kernel_fn, kernel_hyperparam, sampled_labels=None):
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



def degree_matrix(graph_adjacency):
    D= np.zeros(shape=graph_adjacency.shape)
    n_vertices= D.shape[0]
    for i in range(n_vertices):
        D[i,i]= np.sum(graph_adjacency[i, :])
    return D
