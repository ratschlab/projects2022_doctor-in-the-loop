import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

from activelearners import ProbCoverSampler_Faiss, RandomSampler
from clustering import MyKMeans
from datasets import FourWays
import argparse
import sys

def plot_results(perf_prob, perf_pure, perf_random, perf_all, idxs=None):
    if idxs is None:
        idxs = np.arange(len(perf_prob))
        assert len(perf_pure) == len(perf_prob)
    fig, ax = plt.subplots(dpi=150)
    # ax.axis('equal')
    sns.lineplot(x=idxs, y=perf_prob, label='ProbCover', c=sns.color_palette()[0])
    sns.lineplot(x=idxs, y=perf_pure, label='PureCover', c=sns.color_palette()[2])
    sns.lineplot(x=idxs, y=perf_random, label='Random', c=sns.color_palette()[3])

    ax.hlines(idxs[0], idxs[1], perf_all, linestyle='--', color='black')
    # sns.scatterplot(x=self.x[self.queries, 0], y=self.x[self.queries, 1], color=color, marker="P", s=150)
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--separable', default=False, action=argparse.BooleanOptionalAction,
                        help='Whether to make problem separable')

    parser.add_argument('--std',
                        required=True, type=float,
                        help="std of clusters")
    parser.add_argument('--n-points',
                        required=True, type=int,
                        help="number of points per class")
    parser.add_argument('--tsh',
                        required=False, type=float, default=0.95,
                        help="purity threshold")
    parser.add_argument('--radius',
                        required=False, type=float, default=None,
                        help="purity threshold")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    dataset = FourWays(np.ones(4) * args.std, np.array([args.n_points, args.n_points,
                                                        args.n_points, args.n_points]),
                       random_state=1, separable=args.separable)
    test_dataset = FourWays(np.ones(4) * args.std, np.array([args.n_points//2, args.n_points//2,
                                                             args.n_points//2, args.n_points//2]),
                            random_state=2, separable=args.separable)

    dataset.plot_dataset()
    clustering = MyKMeans(dataset, 4)
    learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
    if args.radius is not None:
        learner.update_radius(args.radius)
    print("RADIUS = {}".format(learner.radius))
    scores = []
    for _ in range(100):
        learner.adaptive_query(1, K=5)
        if _ % 1 == 0:
            if _ % 20 == 0:
                dataset.plot_al(plot_circles=True)
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()

            clf = KNeighborsClassifier(1)
            clf.fit(x_train, y_train)
            perf = clf.score(x_test, y_test)
            scores.append(perf)

        # print(scores)
        # print(dataset.radiuses[dataset.queries])

    dataset.restart()
    learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
    print("RADIUS = {}".format(learner.radius))

    if args.radius is not None:
        learner.update_radius(args.radius)
    pc_scores = []
    for _ in range(100):
        learner.query(1)
        if _ % 1 == 0:
            if _ % 20 == 0:
                dataset.plot_al(plot_circles=True, color='blue')
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()

            clf = KNeighborsClassifier(1)
            clf.fit(x_train, y_train)
            perf = clf.score(x_test, y_test)
            pc_scores.append(perf)

    dataset.restart()
    learner = RandomSampler(dataset)

    random_scores = []
    for _ in range(100):
        learner.query(1)
        if _ % 1 == 0:
            if _ % 20 == 0:
                dataset.plot_al(plot_circles=True, color='blue')
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()

            clf = KNeighborsClassifier(1)
            clf.fit(x_train, y_train)
            perf = clf.score(x_test, y_test)
            random_scores.append(perf)


    print("PC SCORES {}".format(pc_scores))
    print("Adaptative SCORES {}".format(scores))
    print("Random SCORES {}".format(scores))

    x_train, y_train = dataset.get_all_data()
    x_test, y_test = test_dataset.get_all_data()

    clf = KNeighborsClassifier(1)
    clf.fit(x_train, y_train)
    perf = clf.score(x_test, y_test)
    plot_results(pc_scores, scores, random_scores, perf, np.arange(0, 100, 1))
    print("Perf all data {}".format(perf))

