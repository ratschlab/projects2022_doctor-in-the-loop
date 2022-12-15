import pandas as pd
from sklearn.datasets import make_blobs
from IPython import embed
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ActiveDataset:
    def __init__(self, n_points, random_state= None):
        self.n_points= n_points

        if random_state is not None:
            state = np.random.get_state()
            np.random.seed(random_state)
        self.x, self.y= self.generate_data()
        if random_state is not None:
            np.random.set_state(state)

        self.labeled = np.zeros(self.n_points, dtype=int)
        self.queries = list()


    def restart(self):
        self.labeled = np.zeros(self.n_points, dtype=int)
        self.queries = np.array([], dtype=int)

    def observe(self, idx):
        self.labeled[idx] = 1
        if isinstance(idx, int):
            idx = np.array([idx])
        elif idx.ndim == 0:
            idx = np.array([idx])

        self.queries = np.concatenate((self.queries, idx), axis=0).astype(int)

    def plot_dataset(self):
        #Check that the data is 2-dimensional
        assert(self.x.shape[1]==2)

        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x= self.x[:,0], y=self.x[:,1], hue=self.y, palette="Set2")
        plt.show()

    def plot_al(self):
        # Check that the data is 2-dimensional
        assert (self.x.shape[1] == 2)

        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=self.y, palette="Set2")
        sns.scatterplot(x=self.x[self.queries, 0], y=self.x[self.queries, 1], color= "red", marker="P", s=150)
        plt.show()



class PointClouds(ActiveDataset):
    def __init__(self, cluster_centers, cluster_std, cluster_samples, random_state=None):
        self.n_features= len(cluster_centers[0])
        self.n_cluster= len(cluster_std)
        self.cluster_centers= cluster_centers
        self.cluster_std= cluster_std
        self.cluster_samples= cluster_samples.astype(int)
        super(PointClouds, self).__init__(np.sum(cluster_samples), random_state)
        self.name= "Clouds"

    def generate_data(self):
        x, y = make_blobs(n_samples=self.cluster_samples, cluster_std=self.cluster_std,
                          centers=self.cluster_centers, n_features=self.n_features)
        return x, y


class CenteredCircles(ActiveDataset):
    def __init__(self, center, radiuses, samples, std, random_state=None):
        assert (len(radiuses) == len(samples))
        self.n_features = 2
        self.n_cluster = len(radiuses)
        self.center = center
        self.radiuses = radiuses
        self.samples = samples
        self.std = std
        super(CenteredCircles, self).__init__(np.sum(samples), random_state)
        self.name= "Centered circles"

    def generate_data(self):
        x1, x2, y= np.array([]), np.array([]), np.array([])
        for k in range(len(self.radiuses)):
            theta = np.linspace(0,2*np.pi, self.samples[k])
            x1= np.append(x1, np.cos(theta)*self.radiuses[k]+self.center[0]+self.std[k]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[k])).reshape(-1,1)
            x2= np.append(x2, np.sin(theta)*self.radiuses[k]+self.center[1]+self.std[k]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[k])).reshape(-1,1)
            y= np.append(y, np.ones(self.samples[k])*k)
        x= np.concatenate((x1,x2), axis=1)
        return x,y

class MixedClusters(ActiveDataset):
    def __init__(self, cluster_centers, cluster_std, cluster_samples, random_state=None):
        self.n_features= len(cluster_centers[0])
        self.n_cluster= len(cluster_std)
        self.cluster_centers= cluster_centers
        self.cluster_std= cluster_std
        self.cluster_samples= cluster_samples.astype(int)
        super(MixedClusters, self).__init__(np.sum(cluster_samples), random_state)
        self.name= "Mixed clouds"

    def generate_data(self):
        x, y = make_blobs(n_samples=self.cluster_samples, cluster_std=self.cluster_std,
                          centers=self.cluster_centers, n_features=self.n_features)
        labels= np.zeros(len(y))

        classes= np.unique(y)
        for c in classes:
            idx= np.where(y==c)[0]
            clustering= KMeans(n_clusters=2).fit(x[idx,:])
            labels[idx]= clustering.labels_
        return x, labels


class CIFAR10_simclr(ActiveDataset):
    def __init__(self, n_epochs, random_state=None):
        self.path= "../cifar10features_simclr/"
        self.n_epochs= n_epochs
        super(CIFAR10_simclr, self).__init__(10000, random_state)


    def generate_data(self):
        y= np.load(self.path + "cifar10_labels.npy")
        x= np.load(self.path + f"features_{self.n_epochs}epochs.npy")
        return x, y.squeeze()



class TwoMoons(ActiveDataset):
    def __init__(self, ellipse_centers, ellipse_radius, std, samples, random_state=None):
        assert(len(samples)==2)
        self.n_features= len(ellipse_centers[0])
        self.centers= ellipse_centers
        self.radiuses= ellipse_radius
        self.std= std
        self.samples= np.array(samples).astype(int)
        super(TwoMoons, self).__init__(np.sum(samples), random_state)
        self.name= "Moons"

    def generate_data(self):
        a1, b1= self.radiuses[0]
        a2, b2= self.radiuses[1]
        #x1, x2, y = np.array([]), np.array([]), np.array([])
        top_theta= np.linspace(0,np.pi, self.samples[0])
        bottom_theta= np.linspace(np.pi, 2*np.pi, self.samples[1])
        x1= np.append(a1*np.cos(top_theta)+ self.centers[0][0]+ self.std[0]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[0]),
                      a2*np.cos(bottom_theta)+ self.centers[1][0]+ self.std[1]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[1])).reshape(-1,1)
        x2= np.append(b1*np.sin(top_theta)+ self.centers[0][1]+ self.std[0]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[0]),
                      b2*np.sin(bottom_theta)+ self.centers[1][1]+ self.std[1]*np.random.normal(loc=0.0, scale=1.0, size=self.samples[1])).reshape(-1,1)
        y= np.append(np.zeros(self.samples[0]), np.ones(self.samples[1]))
        x = np.concatenate((x1, x2), axis=1)
        return x, y


