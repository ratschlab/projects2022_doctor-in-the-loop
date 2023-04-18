import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from activelearners import ProbCoverSampler_Faiss, BALDSampler
from clustering import MyKMeans
from datasets import PointClouds
from IPython import embed
if __name__ == "__main__":

    cluster_centers = np.array([[0, 0]])

    cluster_std = [0.3]
    cluster_samples = np.array([400])
    # dataset = MixedClusters(3, cluster_centers, cluster_std, cluster_samples, random_state=1)

    dataset = PointClouds([[1, 2], [2, 3], [0.75, 3.25]], [0.38, 0.4, 0.4], np.array([200, 200, 200]), random_state=1)
    test_dataset = PointClouds([[1, 2], [2, 3], [0.75, 3.25]], [0.38, 0.4, 0.4], np.array([50, 50, 50]), random_state=2)

    dataset.plot_dataset()

    sampler= BALDSampler(dataset, 3)
    sampler.query(3)
    for _ in range(10):
        sampler.query(1)
        dataset.plot_al()
    embed()

    clustering = MyKMeans(dataset, 3)
    learner = ProbCoverSampler_Faiss(dataset, 0.95, clustering)
    learner.update_radius(0.3)
    scores = []
    for _ in range(200):
        learner.adaptive_query(1, K=5)
        if _ % 20 == 0:
            dataset.plot_al(plot_circles=True)
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()

            clf = KNeighborsClassifier(1)
            clf.fit(x_train, y_train)
            perf = clf.score(x_test, y_test)
            scores.append(perf)


    dataset.restart()
    learner = ProbCoverSampler_Faiss(dataset, 0.95, clustering)
    learner.update_radius(0.3)
    pc_scores = []
    for _ in range(200):
        learner.query(1)
        if _ % 20 == 0:
            dataset.plot_al(plot_circles=True, color='blue')
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()

            clf = KNeighborsClassifier(1)
            clf.fit(x_train, y_train)
            perf = clf.score(x_test, y_test)
            pc_scores.append(perf)
    print("PC SCORES {}".format(pc_scores))
    print("Adaptative SCORES {}".format(scores))
    x_train, y_train = dataset.get_all_data()
    x_test, y_test = test_dataset.get_all_data()

    clf = KNeighborsClassifier(1)
    clf.fit(x_train, y_train)
    perf = clf.score(x_test, y_test)
    print("Perf all data {}".format(perf))
