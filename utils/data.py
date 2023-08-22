from datasets import CIFAR_simclr, PointClouds, CHEXPERT_remedis
import pandas as pd 
import numpy as np
import pickle


def get_data(args):
    if args.dataset=="toy":
        assert(args.std is not None)
        cluster_centers = [[0.5, 2], [1.5, 2], [3, 2.5], [3, 5], [5, 3], [5, 2], [3, 4], [4.5, 4.5], [1.5, 3.5], [0, 4]]
        cluster_centers = np.array(cluster_centers)
        cluster_std = np.repeat(args.std, len(cluster_centers))
        cluster_samples = np.repeat(args.n_points, len(cluster_centers))

        dataset = PointClouds(cluster_centers, cluster_std, cluster_samples, random_state=1)
        dataset_test = PointClouds(cluster_centers, cluster_std, cluster_samples // 2, random_state=2)

        if args.budget=="low":
            eval_freq= np.concatenate((np.repeat(1, 100), np.repeat(5, 20),
                                       np.repeat(10, 10), np.repeat(20,5),
                                       np.repeat(50, 8), np.repeat(100,2)
                                       ))
        elif args.budget=="high":
            eval_freq = np.concatenate((np.repeat(1, 50), np.repeat(2, 25),
                                        np.repeat(5, 20), np.repeat(10, 20),
                                        np.repeat(50, 6), np.repeat(100, 3),
                                        np.repeat(200, 20),
                                        ))
        elif args.budget=="full":
            eval_freq = np.concatenate((np.repeat(1, 50), np.repeat(2, 25),
                                        np.repeat(5, 20), np.repeat(10, 20),
                                        np.repeat(50, 6), np.repeat(100, 3),
                                        np.repeat(200, 20), np.repeat(500, 10),
                                        ))

    elif args.dataset=="cifar10" or args.dataset=="cifar100":
        dataset= CIFAR_simclr(args.dataset, args.n_epochs, train=True)
        dataset_test= CIFAR_simclr(args.dataset, args.n_epochs, train=False)
        
        eval_freq = np.concatenate((np.repeat(1, 100), np.repeat(2, 50),
                                    np.repeat(5, 20), np.repeat(10, 20), np.repeat(20, 25),
                                   np.repeat(50, 20), np.repeat(100,40),
                                    np.repeat(200,20), np.repeat(500, 10))) #15000
        if args.dataset=="cifar100":
            eval_freq= np.concatenate((eval_freq, np.repeat(500, 20))) #25000
        if args.budget=="high":
            eval_freq= np.concatenate((eval_freq, np.repeat(500, 30))) # 30000 or 40000
        if args.budget=="full":
            eval_freq= np.concatenate((np.repeat(1, 100), np.repeat(2, 50),
                                    np.repeat(5, 20), np.repeat(10, 20), np.repeat(20, 25),
                                       np.repeat(50, 20), np.repeat(100,40),
                                    np.repeat(200,20), np.repeat(500, 10),
                                       np.repeat(500,70))) # full 50000

    elif args.dataset=="chexpert":
        dataset= CHEXPERT_remedis(type="train")
        dataset_test= CHEXPERT_remedis(type="test")
        eval_freq= np.repeat(1, dataset.n_points)

    eval_points= np.cumsum(eval_freq)
        
    if args.separable != "not":
        dataset.make_separable(linear=(args.separable == True))
        dataset_test.make_separable(linear=(args.separable == True))

    run_path= get_run_path(args)

    return dataset, dataset_test, run_path, eval_points

def get_run_path(args):
    if args.dataset=="toy":
        path_root= f"./{args.run}/{args.dataset}_{args.separable}_{args.std}"
    else:
        path_root= f"./{args.run}/{args.dataset}"
    if args.algorithm =="adpc":
        run_path= f"{path_root}/{args.n_epochs}_{args.gauss}_{args.tsh}_{args.hard_thresholding}_gamma{args.gamma}_{args.reduction_method}_{args.sd}"
    elif args.algorithm =="partialadpc": 
        run_path= f"{path_root}/{args.n_epochs}_{args.gauss}_{args.tsh}_{args.hard_thresholding}_gamma{args.gamma}_{args.reduction_method}_{args.sd}"
    elif args.algorithm == "benchmark":
        run_path= f"{path_root}/{args.n_epochs}_{args.sd}"
    elif args.algorithm == "pc":
        run_path= f"{path_root}/{args.n_epochs}_{args.tsh}_{args.sd}"
    elif args.algorithm == "coverpc":
        run_path= f"{path_root}/{args.n_epochs}_{args.gauss}_{args.tsh}_{args.hard_thresholding}_eps{args.eps}_cover{args.cover_threshold}_{args.sd}"
    return run_path




def saving_run(algorithm: "str", run_path, scores, queries, radiuses=None, degrees=None, options=None, covers=None, aucs=[]):
    pd.DataFrame({f'{algorithm}_scores': scores}).to_csv(f'{run_path}/{algorithm}_scores.csv')
    pd.DataFrame({f'{algorithm}_queries': queries}).to_csv(f'{run_path}/{algorithm}_queries.csv')
    if algorithm in ["pc", "adpc", "partialadpc", "coverpc"]:
        pd.DataFrame({f'{algorithm}_degrees': degrees, f'{algorithm}_options': options}).to_csv(f'{run_path}/{algorithm}_degrees.csv')
        pd.DataFrame({f'{algorithm}_covers': covers}).to_csv(f'{run_path}/{algorithm}_covers.csv')
        np.save(f'{run_path}/{algorithm}_radiuses.npy', radiuses)
    if len(aucs)>0:
        pd.DataFrame({f'{algorithm}_aucs': aucs}).to_csv(f'{run_path}/{algorithm}_aucs.csv')


def fetching_run(algorithm: "str", run_path):
    assert(algorithm in ["full", "random", "pc", "adpc", "partialadpc", "coverpc"])
    if algorithm=="full":
        scores = pd.read_csv(run_path + f"/full.csv", index_col=0)[f"full_scores"].to_numpy()
        queries= None
    else:
        scores = pd.read_csv(run_path + f"/{algorithm}_scores.csv", index_col=0)[f"{algorithm}_scores"].to_numpy()
        queries = pd.read_csv(run_path + f"/{algorithm}_queries.csv", index_col=0)[f"{algorithm}_queries"].to_numpy()
        if algorithm in ["pc", "adpc", "partialadpc", "coverpc"]:
            radiuses= np.load(f'{run_path}/{algorithm}_radiuses.npy')
            degrees = pd.read_csv(run_path + f"/{algorithm}_degrees.csv", index_col=0)[f"{algorithm}_degrees"].to_numpy()
            options = pd.read_csv(run_path + f"/{algorithm}_degrees.csv", index_col=0)[f"{algorithm}_options"].to_numpy()
            covers = pd.read_csv(run_path + f"/{algorithm}_covers.csv", index_col=0)[f"{algorithm}_covers"].to_numpy()
        else:
            radiuses, degrees, options, covers= None, None, None, None

    return scores, queries, radiuses, degrees, options, covers