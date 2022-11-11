import pandas as pd
from sklearn.datasets import make_blobs
from IPython import embed
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PointClouds:
    def __init__(self, cluster_centers, cluster_std, cluster_samples):
        self.n_features= len(cluster_centers[0])
        self.n_cluster= len(cluster_std)
        self.cluster_centers= cluster_centers
        self.cluster_std= cluster_std
        self.cluster_samples= cluster_samples.astype(int)
        X, y = make_blobs(n_samples=self.cluster_samples, cluster_std=self.cluster_std,
                          centers=self.cluster_centers, n_features=self.n_features) #random_state=1 as parameter also?
        self.dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        columns= ['x%s' % i for i in range(1,self.n_features+1)]
        columns.append("y")
        self.dataset = pd.DataFrame(data=self.dataset, columns=columns)

    def plot2d(self):
        #Check that the data is 2-dimensional
        assert(self.dataset.iloc[:,0:-1].shape[1]==2)
        sns.scatterplot(data=self.dataset, x="x1", y="x2", hue="y", palette="Set2")
        plt.show()


