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
import time


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--separable', type= str, default="not",
                        help='Whether to make problem separable') # others are "linear" and "nonlinear"

    parser.add_argument('--n_points', type= int, required= True,
                        help="number of points per class")

    parser.add_argument('--eval_freq', type=int, required= True,
                        help= "how often to evaluate using 1-NN neighbours")

    parser.add_argument('--gauss', type= int, required=True,
                        help= 'norm for the gaussian weighting')

    parser.add_argument('--std', type= float, required=True,
                        help="std of clusters")

    parser.add_argument('--tsh', type=float, default=0.95,
                        help="purity threshold")

    parser.add_argument('--radius', type= float, required= True,
                        help="initial radius for the adaptive probcover")

    parser.add_argument('--sd', type= int, default=None,
                        help= "run seed")

    parser.add_argument('--run', type=str, default="runs-28-04",
                        help='folder name')  # others are "linear" and "nonlinear"

    parser.add_argument('--neighbours_rad', type=int, default= 5,
                        help="number of neighbours for radius update using Knn average")

    parser.add_argument('--dilat', type= float, default= 1,
                        help= 'multiplicative term to drive the cluster centers further apart')
    return parser

def simulate_random(dataset, test_dataset, args):
    dataset.restart()
    scores=[]
    learner = RandomSampler(dataset)
    for _ in range(dataset.n_points):
        learner.query(1)
        # Train a 1-NN classifier and get the test accuracy
        if len(dataset.queries)%args.eval_freq==0:
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()
            model = KNeighborsClassifier(1)
            model.fit(x_train, y_train)
            perf = model.score(x_test, y_test)
            scores.append(perf)
    return scores, dataset.queries


def simulate_PC(algorithm: "str", dataset, test_dataset, args, plot=False):
    assert((algorithm=="pc") or (algorithm=="adaptive_pc"))
    dataset.restart()
    n_classes = len(np.unique(dataset.y))

    clustering = MyKMeans(dataset, n_classes)
    if algorithm=="adaptive_pc":
        if args.radius==0:
            # Initialize the radius as in original Probcover
            learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
        elif args.radius>0:
            learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering, radius= args.radius)
    elif algorithm=="pc":
        learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
    scores = []
    degrees = []
    options = []
    for _ in range(dataset.n_points):
        # Query new points
        if algorithm == "pc":
            deg, opt = learner.query(1)
        elif algorithm=="adaptive_pc":
            deg, opt = learner.adaptive_query(1, K=args.neighbours_rad, deg= args.gauss)
            print(deg, opt)
        degrees.append(deg)
        options.append(opt)
        if plot and len(dataset.queries)%5==0 and len(dataset.queries)<=100:
            dataset.plot_al(True)
        # Train a 1-NN classifier and get the test accuracy
        if len(dataset.queries)%args.eval_freq==0:
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = test_dataset.get_all_data()
            model = KNeighborsClassifier(1)
            model.fit(x_train, y_train)
            perf = model.score(x_test, y_test)
            scores.append(perf)
    return scores, dataset.queries, dataset.radiuses, degrees, options

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
    if args.sd is not None:
        np.random.seed(args.sd)

    # get the training and testing data
    dataset, dataset_test = get_data("clouds", args)
    run_path= f"./{args.run}/{args.separable}_{args.n_points}_{args.std}_{args.gauss}_{args.tsh}_{args.radius}_{args.eval_freq}_{args.sd}"

    if not os.path.exists(run_path):
        os.makedirs(run_path)
    print(run_path)
    # simulate the runs
    if args.radius==0:
        print("Running the benchmarks")
        start_time= time.time()
        # only needs to be run once per dataset (does not depend on gauss or radius) (depends on std)
        random_scores, random_queries = simulate_random(dataset, dataset_test, args)
        random_time = time.time()
        print(f"Random ran in {(random_time-start_time)/60} minutes")
        pc_scores, pc_queries, pc_radiuses, pc_degrees, pc_options = simulate_PC("pc", dataset, dataset_test, args)
        pc_time= time.time()
        print(f"ProbCover ran in {(pc_time-random_time)/60} minutes")
        adpc_scores, adpc_queries, adpc_radiuses, adpc_degrees, adpc_options = simulate_PC("adaptive_pc", dataset, dataset_test, args)
        adpc_time= time.time()
        print(f"Adaptive ProbCover ran in {(adpc_time-pc_time)/60} minutes")
        real_score = get_fullsup_accuracy(dataset, dataset_test)
        final_time= time.time()
        print(f"Total run time {(final_time-start_time)/60} minutes")

        random_df = pd.DataFrame({'random_scores': np.repeat(random_scores, args.eval_freq), 'random_queries': random_queries})
        pc_df = pd.DataFrame({'pc_scores': np.repeat(pc_scores, args.eval_freq),
                              'pc_queries': pc_queries, 'pc_radiuses': pc_radiuses,
                              'pc_degrees': pc_degrees, 'pc_options': pc_options})
        adpc_df = pd.DataFrame({'adpc_scores': np.repeat(adpc_scores, args.eval_freq),
                                'adpc_queries': adpc_queries, 'adpc_radiuses': adpc_radiuses,
                                'adpc_degrees': adpc_degrees, 'adpc_options': adpc_options})
        full_df = pd.DataFrame({'full_scores': np.repeat(real_score, len(pc_scores))})

        random_df.to_csv(f'{run_path}/random.csv')
        pc_df.to_csv(f'{run_path}/pc.csv')
        adpc_df.to_csv(f'{run_path}/adaptive_pc.csv')
        full_df.to_csv(f'{run_path}/full.csv')
        dataset.plot_dataset(save=True, path=f'{run_path}/train.png')
        dataset_test.plot_dataset(save=True, path=f'{run_path}/test.png')
    elif args.radius>0:
        print("Running adaptive only")
        start_time=time.time()
        adpc_scores,  adpc_queries, adpc_radiuses, adpc_degrees, adpc_options = simulate_PC("adaptive_pc", dataset, dataset_test, args)
        stop_time= time.time()
        print(f"Adaptive ProbCover ran in {(stop_time-start_time)/60} minutes")
        adpc_df = pd.DataFrame({'adpc_scores': adpc_scores, 'adpc_queries': adpc_queries, 'adpc_radiuses': adpc_radiuses,
                                'adpc_degrees': adpc_degrees, 'adpc_options': adpc_options})
        adpc_df.to_csv(f'{run_path}/adaptive_pc.csv')
    print("done")
