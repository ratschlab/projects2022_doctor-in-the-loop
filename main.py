import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import sklearn

from activelearners import ProbCoverSampler_Faiss, RandomSampler
from clustering import MyKMeans
from datasets import FourWays, PointClouds, CenteredCircles, TwoMoons
import argparse
import sys
from IPython import embed
import os
import pandas as pd



def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--separable', type= str, default="not",
                        help='Whether to make problem separable') # others are "linear" and "nonlinear"

    parser.add_argument('--n_points', type= int, required= True,
                        help="number of points per class")


    parser.add_argument('--gauss', type= int, required=True,
                        help= 'norm for the gaussian weighting')

    parser.add_argument('--std', type= float, required=True,
                        help="std of clusters")

    parser.add_argument('--tsh', type=float, default=0.95,
                        help="purity threshold")

    parser.add_argument('--radius', type= float, required= True,
                        help="initial radius for the adaptive probcover")

    parser.add_argument('--seed', type= int, default=None,
                        help= "run seed")

    parser.add_argument('--run', type=str, default="run2",
                        help='folder name')  # others are "linear" and "nonlinear"

    parser.add_argument('--neighbours_rad', type=int, default= 5,
                        help="number of neighbours for radius update using Knn average")

    parser.add_argument('--dilat', type= float, default= 1,
                        help= 'multiplicative term to drive the cluster centers further apart')
    return parser

def simulate_run(algorithm: "str", dataset, test_dataset, args, seed=None, plot=False):
    dataset.restart()
    n_classes = len(np.unique(dataset.y))
    random_regime, random_transition= False, dataset.n_points
    if algorithm == "random":
        random_transition=0
        learner = RandomSampler(dataset)
    elif algorithm == "adaptive_pc" or algorithm == "pc":
        clustering = MyKMeans(dataset, n_classes, random_clustering=seed)
        if algorithm=="adaptive_pc":
            learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering, radius= args.radius)
        elif algorithm=="pc":
            learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
    scores = []
    for _ in range(dataset.n_points):
        # Query new points
        if algorithm == "random":
            learner.query(1)
        elif algorithm == "pc":
            if random_regime==False:
                random_regime= learner.query(1)
                random_transition= len(dataset.queries)-1
            else:
                _ = learner.query(1)
        elif algorithm=="adaptive_pc":
            if random_regime == False:
                random_regime = learner.adaptive_query(1, K=args.neighbours_rad, deg= args.gauss)
                random_transition = len(dataset.queries) - 1
            else:
                _ = learner.adaptive_query(1, K=args.neighbours_rad, deg= args.gauss)

        if plot and len(dataset.queries)%5==0 and len(dataset.queries)<=100:
            dataset.plot_al(True)
        # Train a 1-NN classifier and get the test accuracy
        x_train, y_train = dataset.get_labeled_data()
        x_test, y_test = test_dataset.get_all_data()
        model = KNeighborsClassifier(1)
        model.fit(x_train, y_train)
        perf = model.score(x_test, y_test)
        scores.append(perf)
    return scores, dataset.queries, dataset.radiuses, dataset.regime, random_transition

def get_fullsup_accuracy(dataset, test_dataset):
    x_train, y_train = dataset.get_all_data()
    x_test, y_test = test_dataset.get_all_data()

    model = KNeighborsClassifier(1)
    model.fit(x_train, y_train)
    perf = model.score(x_test, y_test)
    return perf

def get_data(dataset_name: str, args):
    if dataset_name=="clouds":
        cluster_centers = [[0.5, 2], [1.5, 2], [3, 2.5], [3, 5], [5, 3], [5, 2], [3, 4], [4.5, 4.5], [1.5, 3.5], [0, 4]]
        cluster_centers = np.array(cluster_centers) * args.dilat
        cluster_std = np.repeat(args.std, len(cluster_centers))
        cluster_samples = np.repeat(args.n_points, len(cluster_centers))

        dataset = PointClouds(cluster_centers, cluster_std, cluster_samples, random_state=1)
        dataset_test = PointClouds(cluster_centers, cluster_std, cluster_samples // 2, random_state=2)

    if args.separable != "not":
        dataset.make_separable(linear=(args.separable == True))
        dataset_test.make_separable(linear=(args.separable == True))
    return dataset, dataset_test

if __name__ == "__main__":
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    if args.seed is not None:
        np.random.seed(args.seed)

    # get the training and testing data
    dataset, dataset_test = get_data("clouds", args)
    run_path= f"./{args.run}/{args.separable}_{args.n_points}_{args.std}_{args.gauss}_{args.tsh}_{args.radius}_{args.seed}"

    # simulate the runs
    random_scores, random_queries, _, random_regime, _ = simulate_run("random", dataset, dataset_test, args)
    pc_scores, pc_queries, pc_radiuses, pc_regime, pc_transition = simulate_run("pc", dataset, dataset_test, args)
    adpc_scores,  adpc_queries, adpc_radiuses, adpc_regime, adpc_transition = simulate_run("adaptive_pc", dataset, dataset_test, args)
    real_score= get_fullsup_accuracy(dataset, dataset_test)

    # saving the run into csv files
    scores = pd.DataFrame({'pc_scores': pc_scores, 'adaptive_scores': adpc_scores, 'random_scores': random_scores,
                           'full_scores': np.repeat(real_score, len(pc_scores))})
    queries = pd.DataFrame({'pc_queries': pc_queries, 'pc_radiuses': pc_radiuses,
                            'adaptive_queries': adpc_queries, 'adaptive_radiuses': adpc_radiuses,
                            'random_queries': random_queries})
    transitions = pd.DataFrame({'pc': pc_regime, 'adaptive': adpc_regime})

    if not os.path.exists(run_path):
        os.makedirs(run_path)
    scores.to_csv(f'{run_path}/scores.csv')
    queries.to_csv(f'{run_path}/queries.csv')
    transitions.to_csv(f'{run_path}/transitions.csv')
    dataset.plot_dataset(save=True, path= f'{run_path}/train.png')
    dataset_test.plot_dataset(save=True, path= f'{run_path}/test.png')
    print("done")
