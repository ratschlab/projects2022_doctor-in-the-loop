import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from activelearners import ProbCoverSampler_Faiss, RandomSampler
from clustering import MyKMeans
from datasets import FourWays, PointClouds, CenteredCircles, TwoMoons, CIFAR_simclr
import argparse
import sys
from IPython import embed
import os
import pandas as pd
import time
import warnings
import pickle
from helper import get_cover

warnings.filterwarnings(action='ignore', category=FutureWarning)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_parser():
    parser = argparse.ArgumentParser()
    ### parameters to set for all runs ###
    parser.add_argument('--dataset', type= str, required= True,
                        help='"cifar10" or "cifar100" or "toy"')
    parser.add_argument('--radius', type=float, required=True,
                        help="initial radius, if equal to 0 the radius will be initialized using kmeans and the default purity threshold")
    parser.add_argument('--run', type=str, required=False, default="runs",
                        help='folder name to save the run')
    parser.add_argument('--sd', type=int, default=None, required=True,
                        help="run seed")
    parser.add_argument('--n_epochs', type=int, required=True,
                        help="number of training epochs for extracted features")

    ### parameters to set for cifar experiments ###
    parser.add_argument('--algorithm', type=str, required=True,
                        help="Wether to run 'benchmark', 'adpc', 'pc' or 'coverpc'")
    parser.add_argument('--warm_start', type= str2bool, default= False)

    ### extra parameters to set for cover-based PC method ###

    ### parameters for toy dataset runs ###
    parser.add_argument('--separable', type= str, required= False, default="not",
                        help='Whether to make problem separable') # others are "linear" and "nonlinear"
    parser.add_argument('--std', type=float, required=False,
                        help="std of clusters")
    parser.add_argument('--n_points', type= int, default= 1000,
                        help="number of points per class")
    parser.add_argument('--dilat', type= float, default= 1,
                        help= 'multiplicative term to drive the cluster centers further apart')

    ### default parameters for all runs ####
    parser.add_argument('--gauss', type=int, required=False, default=4,
                        help='norm for the gaussian weighting')
    parser.add_argument('--tsh', type=float, default=0.95,
                        help="purity threshold")
    parser.add_argument('--savefreq', type=float, default=100,
                        help="how often to save the progress")
    parser.add_argument('--neighbours_rad', type=int, default= 5,
                        help="number of neighbours for radius update using Knn average")

    return parser

def saving_run(algorithm: "str", run_path, scores, queries, radiuses=None, degrees=None, options=None, covers=None):
    pd.DataFrame({f'{algorithm}_scores': scores}).to_csv(f'{run_path}/{algorithm}_scores.csv')
    pd.DataFrame({f'{algorithm}_queries': queries}).to_csv(f'{run_path}/{algorithm}_queries.csv')
    if algorithm == "pc" or algorithm == "adpc" or algorithm=="coverpc":
        pd.DataFrame({f'{algorithm}_degrees': degrees, f'{algorithm}_options': options}).to_csv(f'{run_path}/{algorithm}_degrees.csv')
        pd.DataFrame({f'{algorithm}_covers': covers}).to_csv(f'{run_path}/{algorithm}_covers.csv')
        np.save(f'{run_path}/{algorithm}_radiuses.npy', radiuses)
        state = np.random.get_state()
        with open(run_path + f'/random_state_{algorithm}.pickle', 'wb') as handle:
            pickle.dump(state, handle)

def fetching_run(algorithm: "str", run_path):
    scores = pd.read_csv(run_path + f"/{algorithm}_scores.csv", index_col=0)[f"{algorithm}_scores"].to_numpy()
    queries = pd.read_csv(run_path + f"/{algorithm}_queries.csv", index_col=0)[f"{algorithm}_queries"].to_numpy()
    if algorithm=="pc" or algorithm=="adpc" or algorithm=="coverpc":
        # radiuses = pd.read_csv(run_path + f"/{algorithm}_radiuses.csv", index_col=0)[f"{algorithm}_radiuses"].to_numpy()
        radiuses= np.load(f'{run_path}/{algorithm}_radiuses.npy')
        degrees = pd.read_csv(run_path + f"/{algorithm}_degrees.csv", index_col=0)[f"{algorithm}_degrees"].to_numpy()
        options = pd.read_csv(run_path + f"/{algorithm}_degrees.csv", index_col=0)[f"{algorithm}_options"].to_numpy()
        covers = pd.read_csv(run_path + f"/{algorithm}_covers.csv", index_col=0)[f"{algorithm}_covers"].to_numpy()
        with open(run_path + f'/random_state_{algorithm}.pickle', 'rb') as handle:
            state = pickle.load(handle)
    else:
        radiuses, degrees, options, covers= None, None, None, None
        state= None
    return scores, queries, radiuses, degrees, options, covers, state


def simulate_random(run_path, dataset, dataset_test, args, eval_points= None):
    dataset.restart()
    scores=[]
    learner = RandomSampler(dataset)
    if eval_points is None:
        eval_points = np.arange(1, dataset.n_points + 1)
    # for _ in range(eval_points.max()):
    while len(dataset.queries)<eval_points.max():
        learner.query(1)
        # Train a 1-NN classifier and get the test accuracy
        if len(dataset.queries) in eval_points:
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = dataset_test.get_all_data()
            model = KNeighborsClassifier(1)
            model.fit(x_train, y_train)
            perf = model.score(x_test, y_test)
            scores.append(perf)
        if len(dataset.queries)%args.savefreq==0:
            saving_run("random", run_path, scores, dataset.queries)
    saving_run("random", run_path, scores, dataset.queries)


def simulate_PC(run_path, algorithm: "str", dataset, dataset_test, args, eval_points= None, plot=False):
    assert((algorithm=="pc") or (algorithm=="adpc") or (algorithm=="coverpc"))
    reinitialize=False
    dataset.restart()
    if args.warm_start:
        print("Fetching data and initializing dataset and learner for warm start")
        scores, queries, radiuses, degrees, options, covers, state = fetching_run(algorithm, run_path)
    else:
        print("Starting from scratch")
        scores, degrees, options, covers= [], [], [], []
        radiuses = np.empty(shape=(dataset.n_points, 0))
    scores, degrees, options, covers = list(scores), list(degrees), list(options), list(covers)
    n_classes = len(np.unique(dataset.y))
    clustering = MyKMeans(dataset, n_classes)

    if eval_points is None:
        eval_points = np.arange(1, dataset.n_points + 1)

    if args.radius==0:
        # Initialize the radius as in original Probcover
        learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering)
        print(learner.radius)
    elif args.radius>0:
        learner = ProbCoverSampler_Faiss(dataset, args.tsh, clustering, radius= args.radius)
        if args.warm_start:
            dataset.observe(queries)
            dataset.radiuses= radiuses[:,-1].squeeze()
            np.random.set_state(state)
            reinitialize= True

    while len(dataset.queries)< eval_points.max():
        # Query new points
        if algorithm == "pc" or algorithm=="coverpc":
            deg, opt = learner.query(1, reinitialize= reinitialize)
            reinitialize= False
        elif algorithm=="adpc":
            deg, opt = learner.adaptive_query(1, K=args.neighbours_rad, deg= args.gauss, reinitialize=reinitialize)
            reinitialize= False
        degrees.append(deg)
        options.append(opt)
        if plot and len(dataset.queries)%5==0 and len(dataset.queries)<=100:
            dataset.plot_al(True)
        if len(dataset.queries) in eval_points:
            cover= get_cover(algorithm, dataset, learner.lims_ref, learner.D_ref, learner.I_ref)
            covers.append(cover)
            print(len(dataset.queries), dataset.queries[-1], cover*100)
            x_train, y_train = dataset.get_labeled_data()
            x_test, y_test = dataset_test.get_all_data()
            model = KNeighborsClassifier(1)
            model.fit(x_train, y_train)
            perf = model.score(x_test, y_test)
            scores.append(perf)
            radiuses= np.concatenate((radiuses, dataset.radiuses.reshape(-1,1)), axis=1)
        if len(dataset.queries)%args.savefreq==0:
            saving_run(algorithm, run_path, scores, dataset.queries, radiuses, degrees, options, covers)


    state = np.random.get_state()
    with open(run_path + f"/random_state_{algorithm}.pickle", 'wb') as handle:
        pickle.dump(state, handle)
    saving_run(algorithm, run_path, scores, dataset.queries, radiuses, degrees, options, covers)




def simulate_fullsupervised(run_path, length, dataset, dataset_test):
    x_train, y_train = dataset.get_all_data()
    x_test, y_test = dataset_test.get_all_data()

    model = KNeighborsClassifier(1)
    model.fit(x_train, y_train)
    perf = model.score(x_test, y_test)

    full_df = pd.DataFrame({'full_scores': np.repeat(perf, length)})
    full_df.to_csv(f'{run_path}/full.csv')

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
    if args.dataset=="cifar10" or args.dataset=="cifar100":
        dataset= CIFAR_simclr(args.dataset, args.n_epochs, train=True)
        dataset_test= CIFAR_simclr(args.dataset, args.n_epochs, train=False)
        run_path= f"./{args.run}/{args.dataset}/{args.n_epochs}_{args.gauss}_{args.radius}_{args.sd}"
        eval_freq = np.concatenate((np.repeat(1, 100), np.repeat(2, 50),
                                    np.repeat(5, 20), np.repeat(10, 20), np.repeat(20, 25),
                                   np.repeat(50, 20), np.repeat(100,40),
                                    np.repeat(200,20), np.repeat(500, 10)))
        # eval_freq = np.concatenate((np.repeat(1, 100), np.repeat(2, 50),
        #                             np.repeat(5, 20), np.repeat(10, 20)))
        # eval_freq = np.concatenate((np.repeat(1, 100), np.repeat(2, 50)))

        if args.dataset=="cifar100":
            eval_freq= np.concatenate((eval_freq, np.repeat(500, 20)))
        eval_points= np.cumsum(eval_freq)
    elif args.dataset=="toy":
        dataset, dataset_test = get_data("clouds", args)
        run_path = f"./{args.run}/{args.dataset}/{args.separable}_{args.n_points}_{args.std}_{args.gauss}_{args.radius}_{args.sd}"

    if not os.path.exists(run_path):
        os.makedirs(run_path)


    # simulate the runs
    if args.algorithm=="benchmark":
        print("Running the benchmarks")
        assert(args.radius==0.0)
        start_time= time.time()
        # only needs to be run once per dataset (does not depend on gauss or radius)
        # Running and saving random sampling
        simulate_random(run_path, dataset, dataset_test, args, eval_points)
        random_time = time.time()
        print(f"Random ran in {(random_time-start_time)/60} minutes")
        # Getting the full score
        real_score = simulate_fullsupervised(run_path, len(eval_points), dataset, dataset_test)
        final_time = time.time()
        print(f"Total run time {(final_time - start_time) / 60} minutes")
    elif args.algorithm=="pc":
        print("Running pc only")
        start_time= time.time()
        # Running and saving pc sampling
        simulate_PC(run_path, "pc", dataset, dataset_test, args, eval_points)
        pc_time= time.time()
        print(f"ProbCover ran in {(pc_time-start_time)/60} minutes")
    elif args.algorithm=="adpc":
        print("Running adaptive only")
        start_time=time.time()
        simulate_PC(run_path, "adpc", dataset, dataset_test, args, eval_points)
        stop_time= time.time()
        print(f"Adaptive ProbCover ran in {(stop_time-start_time)/60} minutes")
    elif args.algorithm == "coverpc":
        print("Running cover-based PC only")
        start_time = time.time()
        simulate_PC(run_path, "coverpc", dataset, dataset_test, args, eval_points)
        stop_time = time.time()
        print(f"CoverBased ProbCover ran in {(stop_time - start_time) / 60} minutes")
    if args.dataset == "toy":
        dataset.plot_dataset(save=True, path=f'{run_path}/train.png')
        dataset_test.plot_dataset(save=True, path=f'{run_path}/test.png')

    print("done")

