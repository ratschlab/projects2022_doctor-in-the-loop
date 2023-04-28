import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

class ActiveDataset:
    def __init__(self, n_points, random_state=None):
        self.n_points = n_points
        self.random_state = random_state

        if self.random_state is not None:
            state = np.random.get_state()
            np.random.seed(self.random_state)
        self.x, self.y = self.generate_data()
        if self.random_state is not None:
            np.random.set_state(state)

        self.labeled = np.zeros(self.n_points, dtype=int)
        self.queries = np.array([], dtype=int)
        self.radiuses = np.zeros(self.n_points)
        self.regime = np.array([], dtype=int)
    def restart(self):
        self.labeled = np.zeros(self.n_points, dtype=int)
        self.queries = np.array([], dtype=int)
        self.radiuses = np.zeros(self.n_points)
        self.regime = np.array([], dtype=int)

    def observe(self, idx, regime, radiuses=None):
        self.labeled[idx] = 1
        if radiuses is not None:
            self.radiuses[idx] = radiuses
        if isinstance(idx, int):
            idx = np.array([idx])
        elif idx.ndim == 0:
            idx = np.array([idx])

        if isinstance(regime, int):
            regime = np.array([regime])
        elif regime.ndim == 0:
            regime = np.array([regime])

        self.queries = np.concatenate((self.queries, idx), axis=0).astype(int)
        self.regime = np.concatenate((self.regime, regime), axis=0).astype(int)

    def plot_dataset(self, save=False, path=""):
        assert (self.x.shape[1] == 2)
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=self.y, palette="Set2")
        if save:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def get_labeled_data(self):
        return self.x[self.queries], self.y[self.queries]

    def get_all_data(self):
        return self.x, self.y

    def plot_al(self, plot_circles=False, color='red', save= False, path=""):
        # Check that the data is 2-dimensional
        assert (self.x.shape[1] == 2)

        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=self.y, palette="Set2")
        sns.scatterplot(x=self.x[self.queries, 0], y=self.x[self.queries, 1], color=color, marker="P", s=150)
        if plot_circles:
            for i, u in enumerate(self.queries):
                ax.add_patch(
                    plt.Circle((self.x[u, 0], self.x[u, 1]), self.radiuses[u], color=color, fill=False, alpha=0.5))
        if save:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()


    def make_separable(self, linear= True, degree= 3):
        if linear:
            clf = SVC(kernel="linear")
        else:
            clf = SVC(kernel="rbf", degree=degree)
        x_train, y_train = self.get_all_data()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_train)
        self.y= y_pred

class PointClouds(ActiveDataset):
    def __init__(self, cluster_centers, cluster_std, cluster_samples, random_state=None):
        self.n_features = len(cluster_centers[0])
        self.n_cluster = len(cluster_std)
        self.cluster_centers = cluster_centers
        self.cluster_std = cluster_std
        self.cluster_samples = cluster_samples.astype(int)
        super(PointClouds, self).__init__(np.sum(cluster_samples), random_state)
        self.name = "Clouds"

    def generate_data(self):
        x, y = make_blobs(n_samples=self.cluster_samples, cluster_std=self.cluster_std,
                          centers=self.cluster_centers, n_features=self.n_features)
        return x, y


class FourWays(ActiveDataset):
    def __init__(self, cluster_std, cluster_samples, random_state=None, separable=False):
        self.n_cluster = 4
        self.cluster_std = cluster_std
        self.cluster_samples = cluster_samples.astype(int)
        self.separable = separable

        super(FourWays, self).__init__(np.sum(cluster_samples), random_state)
        self.name = "Fourways"

    def generate_data(self):
        x, y = make_blobs(n_samples=self.cluster_samples, centers=np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]),
                       cluster_std=self.cluster_std, n_features=2)

        if self.separable:
            sep_labels = []
            for k in x:
                if k[0] < 0 and k[1] < 0:
                    sep_labels.append(0)
                elif k[0] < 0 and k[1] > 0:
                    sep_labels.append(1)
                elif k[0] > 0 and k[1] < 0:
                    sep_labels.append(2)
                elif k[0] > 0 and k[1] > 0:
                    sep_labels.append(3)
            y = np.array(sep_labels)

        return x, y


class CenteredCircles(ActiveDataset):
    def __init__(self, center, circle_radiuses, samples, std, random_state=None):
        assert (len(circle_radiuses) == len(samples))
        self.n_features = 2
        self.n_cluster = len(circle_radiuses)
        self.center = center
        self.r = circle_radiuses
        self.samples = samples
        self.std = std
        super(CenteredCircles, self).__init__(np.sum(samples), random_state)
        self.name = "Centered circles"

    def generate_data(self):
        x1, x2, y = np.array([]), np.array([]), np.array([])
        for k in range(len(self.r)):
            theta = np.linspace(0, 2 * np.pi, self.samples[k])
            x1 = np.append(x1, np.cos(theta) * self.r[k] + self.center[0] + self.std[k] * np.random.normal(loc=0.0,
                                                                                                           scale=1.0,
                                                                                                           size=
                                                                                                           self.samples[
                                                                                                               k])).reshape(
                -1, 1)
            x2 = np.append(x2, np.sin(theta) * self.r[k] + self.center[1] + self.std[k] * np.random.normal(loc=0.0,
                                                                                                           scale=1.0,
                                                                                                           size=
                                                                                                           self.samples[
                                                                                                               k])).reshape(
                -1, 1)
            y = np.append(y, np.ones(self.samples[k]) * k)
        x = np.concatenate((x1, x2), axis=1)
        return x, y


class MixedClusters(ActiveDataset):
    def __init__(self, n_classes, cluster_centers, cluster_std, cluster_samples, random_state=None):
        self.n_features = len(cluster_centers[0])
        self.n_cluster = len(cluster_std)
        self.n_classes = n_classes
        self.cluster_centers = cluster_centers
        self.cluster_std = cluster_std
        self.cluster_samples = cluster_samples.astype(int)
        super(MixedClusters, self).__init__(np.sum(cluster_samples), random_state)
        self.name = "Mixed clouds"

    def generate_data(self):
        x, y = make_blobs(n_samples=self.cluster_samples, cluster_std=self.cluster_std,
                          centers=self.cluster_centers, n_features=self.n_features)
        labels = np.zeros(len(y))

        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            clustering = KMeans(n_clusters=self.n_classes).fit(x[idx, :])
            labels[idx] = clustering.labels_
        return x, labels


class TwoMoons(ActiveDataset):
    def __init__(self, ellipse_centers, ellipse_radius, cluster_std, cluster_samples, random_state=None):
        assert (len(cluster_samples) == 2)
        self.n_features = len(ellipse_centers[0])
        self.centers = ellipse_centers
        self.radiuses = ellipse_radius
        self.std = cluster_std
        self.samples = np.array(cluster_samples).astype(int)
        super(TwoMoons, self).__init__(np.sum(cluster_samples), random_state)
        self.name = "Moons"

    def generate_data(self):
        a1, b1 = self.radiuses[0]
        a2, b2 = self.radiuses[1]
        # x1, x2, y = np.array([]), np.array([]), np.array([])
        top_theta = np.linspace(0, np.pi, self.samples[0])
        bottom_theta = np.linspace(np.pi, 2 * np.pi, self.samples[1])
        x1 = np.append(a1 * np.cos(top_theta) + self.centers[0][0] + self.std[0] * np.random.normal(loc=0.0, scale=1.0,
                                                                                                    size=self.samples[
                                                                                                        0]),
                       a2 * np.cos(bottom_theta) + self.centers[1][0] + self.std[1] * np.random.normal(loc=0.0,
                                                                                                       scale=1.0, size=
                                                                                                       self.samples[
                                                                                                           1])).reshape(
            -1, 1)
        x2 = np.append(b1 * np.sin(top_theta) + self.centers[0][1] + self.std[0] * np.random.normal(loc=0.0, scale=1.0,
                                                                                                    size=self.samples[
                                                                                                        0]),
                       b2 * np.sin(bottom_theta) + self.centers[1][1] + self.std[1] * np.random.normal(loc=0.0,
                                                                                                       scale=1.0, size=
                                                                                                       self.samples[
                                                                                                           1])).reshape(
            -1, 1)
        y = np.append(np.zeros(self.samples[0]), np.ones(self.samples[1]))
        x = np.concatenate((x1, x2), axis=1)
        return x, y


class CIFAR_simclr(ActiveDataset):
    def __init__(self, n_classes, n_epochs, random_state=None):
        # TODO Change path so that it works in general
        self.path = f"/Users/victoriabarenne/projects2022_doctor-in-the-loop/cifar{n_classes}features_simclr/"
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.name = f"CIFAR{n_classes}_{self.n_epochs}epochs"
        super(CIFAR_simclr, self).__init__(10000, random_state)

    def generate_data(self):
        x = np.load(self.path + f"features_{self.n_epochs}epochs.npy")
        y = np.load(self.path + f"cifar{self.n_classes}_labels.npy")

        return x, y.squeeze()

    def split(self, train_idx, test_idx):
        # create a copy
        train, test = type(self)(self.n_classes, self.n_epochs, self.random_state), type(self)(self.n_classes,
                                                                                               self.n_epochs,
                                                                                               self.random_state)
        train.x, train.y, train.labeled = self.x[train_idx], self.y[train_idx], self.labeled[train_idx]
        test.x, test.y, test.labeled = self.x[test_idx], self.y[test_idx], self.labeled[test_idx]
        train.n_points, test.n_points = len(train_idx), len(test_idx)
        query_in_train = np.isin(self.queries, train_idx)
        query_in_test = np.isin(self.queries, test_idx)
        assert (np.all(query_in_train.astype(int) + query_in_test.astype(int) == 1))
        train.queries = list(np.array(self.queries)[query_in_train])
        test.queries = list(np.array(self.queries)[query_in_test])
        return train, test
