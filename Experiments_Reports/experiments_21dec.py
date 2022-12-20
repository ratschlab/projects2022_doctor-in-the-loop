import numpy as np
from datasets import CenteredCircles, PointClouds, MixedClusters, CIFAR10_simclr, TwoMoons
from activelearners import ProbCoverSampler, ProbCoverSampler_Faiss, active_learning_algo, RandomSampler
from clustering import MySpectralClustering, MyKMeans, OracleClassifier, ClusteringAlgo
from IPython import embed
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from models import Classifier1NN
from helper import check_cover, get_radius, cover, get_radius_faiss
from helper import adjacency_graph, adjacency_graph_faiss, get_purity, get_purity_faiss, get_nearest_neighbour, get_nn_faiss

# ========================Initializing all the datasets =======================

n_train= 800
n_test= 200

## Point clouds dataset
#Baseline

clouds={
    "baseline":{
        "centers": [(-5, -5), (-3, 1), (5, -2), (5, 4)],
        "std": [0.3, 0.8, 0.5, 0.7],
        "p": np.array([0.25, 0.25, 0.25, 0.25])
    },
    "unbalanced": {
        "centers": [(-5, -5), (-3, 1), (5, -2), (5, 4)],
        "std": [0.3, 0.8, 0.5, 0.7],
        "p": np.array([0.8, 0.05, 0.05, 0.1])
    },
    "margins": {
        "centers": [(-4, -4), (-1, 1), (5, -2), (5, 4)],
        "std": [1, 1.1, 1, 1.1],
        "p": np.array([0.25, 0.25, 0.25, 0.25])
    }
}

circles= {
    "baseline": {
        "center": [4,5],
        "r": [0.5, 3.5, 6],
        "std": [0.4, 0.3, 0.4],
        "p": np.array([0.3, 0.35, 0.35])
    },
    "unbalanced": {
        "center": [4,5],
        "r": [0.5, 3.5, 6],
        "std": [0.4, 0.3, 0.4],
        "p": np.array([0.05, 0.9, 0.05])
    },
    "margins": {
        "center": [4,5],
        "r": [0.5, 3.5, 6],
        "std": [0.6, 0.5, 0.55],
        "p": np.array([0.3, 0.35, 0.35])
    }
}

moons= {
    "baseline": {
        "centers": [(0, -1), (3, 0)],
        "r": [(3, 3), (2.5, 4)],
        "std": [0.2, 0.2],
        "p": np.array([0.5, 0.5])
    },
    "unbalanced": {
        "centers": [(0, -1), (3, 0)],
        "r": [(3, 3), (2.5, 4)],
        "std": [0.2, 0.2],
        "p": np.array([0.85, 0.15])
    },
    "margins": {
        "centers": [(0, -2), (3, 0)],
        "r": [(3, 4), (2.5, 4)],
        "std": [0.45, 0.45],
        "p": np.array([0.5, 0.5])
    }
}

##CIFAR10 dataset
cifar10_100epochs= CIFAR10_simclr(n_epochs=100)
cifar10_200epochs= CIFAR10_simclr(n_epochs=200)
cifar10_400epochs= CIFAR10_simclr(n_epochs=400)
cifar10_800epochs= CIFAR10_simclr(n_epochs=800)
cifar10_1000epochs= CIFAR10_simclr(n_epochs=1000)


## List containing all the datasets
dataset_parameters= {"circles": circles, "moons": moons, "clouds": clouds}

# ========================Active learning accuracy=======================
def accuracy_probcover(train_dataset, test_dataset,
                       n_queries: np.array, k,
                       clustering_method: str,
                       purity_threshold, gamma,
                       clustering_seed= None,
                       plot_clustering= False):

    train_dataset.restart()
    accuracy_train= []
    accuracy_test= []
    radiuses= []
    covers= []
    if clustering_method=="spectral":
        clustering= MySpectralClustering(train_dataset, k, gamma, random_clustering=clustering_seed)
    elif clustering_method=="kmeans":
        clustering = MyKMeans(train_dataset, k, random_clustering=clustering_seed)
    elif clustering_method=="oracle":
        clustering= OracleClassifier(train_dataset, k, random_clustering=clustering_seed)

    activelearner = ProbCoverSampler_Faiss(train_dataset, purity_threshold,
                                     clustering, [False, False],
                                     search_range=[0, 10], search_step=0.01)


    model = Classifier1NN(train_dataset)

    for B in n_queries:
        activelearner.query(B)
        activelearner.update_labeled(plot_clustering)
        model.update()
        covers.append(cover(train_dataset.x, train_dataset.queries, activelearner.radius))
        radiuses.append(activelearner.radius)
        accuracy_train.append(model.accuracy)
        accuracy_test.append(model.get_accuracy(test_dataset))
    return np.array(covers), np.array(radiuses), np.array(accuracy_train), np.array(accuracy_test)

def accuracy_random_sampling(train_dataset, test_dataset, n_queries):
    train_dataset.restart()
    accuracy_train= []
    accuracy_test= []

    activelearner = RandomSampler(train_dataset)
    model = Classifier1NN(train_dataset)

    for B in n_queries:
        activelearner.query(B)
        model.update()
        accuracy_test.append(model.get_accuracy(test_dataset))
        accuracy_train.append(model.accuracy)
    return np.array(accuracy_train), np.array(accuracy_test)

def accuracy_simulation_average(train_dataset, test_dataset,
                                n_queries, k,
                                purity_threshold, gamma,
                                n_iter,
                                case:str):

    acc_spectral= np.zeros(shape=(n_iter, len(n_queries)))
    acc_kmeans= np.zeros(shape=(n_iter, len(n_queries)))
    acc_oracle= np.zeros(shape=(n_iter, len(n_queries)))
    acc_random= np.zeros(shape=(n_iter, len(n_queries)))


    for n in range(n_iter):
        _, _, acc_spectral[n,:], acc_spectral_test= accuracy_probcover(train_dataset, test_dataset, n_queries, k, "spectral", purity_threshold, gamma)
        _, _, acc_kmeans[n,:], acc_kmeans_test= accuracy_probcover(train_dataset, test_dataset, n_queries, k, "kmeans", purity_threshold, gamma)
        _, _, acc_oracle[n,:], acc_oracle_test= accuracy_probcover(train_dataset, test_dataset, n_queries, k, "oracle", purity_threshold, gamma)
        acc_random[n,:], acc_random_test= accuracy_random_sampling(train_dataset, test_dataset, n_queries)

    pd.DataFrame(np.concatenate((acc_spectral, acc_kmeans, acc_oracle, acc_random), axis=0)).to_csv(f"trainaccuracy_{dataset_name}_{case}_{n_iter}.csv")
    pd.DataFrame(np.concatenate((acc_spectral_test, acc_kmeans_test, acc_oracle_test, acc_random_test), axis=0)).to_csv(f"testaccuracy_{dataset_name}_{case}_{n_iter}.csv")

    mean_spectral, std_spectral = np.mean(acc_spectral, axis=0), np.std(acc_spectral, axis=0)
    mean_kmeans, std_kmeans = np.mean(acc_kmeans, axis=0), np.std(acc_kmeans, axis=0)
    mean_oracle, std_oracle = np.mean(acc_oracle, axis=0), np.std(acc_oracle, axis=0)
    mean_random, std_random = np.mean(acc_random, axis=0), np.std(acc_random, axis=0)

    x = np.cumsum(n_queries)
    plt.plot(x, mean_spectral, color= 'blue', label='Spectral ProbCover')
    plt.plot(x, acc_spectral_test, color= 'blue', linestyle='dashdot', label='Spectral ProbCover')
    plt.fill_between(x, mean_spectral - std_spectral, mean_spectral + std_spectral, color='blue', alpha=0.2)

    plt.plot(x, mean_kmeans, color= 'red', label='Kmeans ProbCover')
    plt.plot(x, acc_kmeans_test, color= 'red', linestyle='dashdot', label='KMeans ProbCover')
    plt.fill_between(x, mean_kmeans - std_kmeans, mean_kmeans + std_kmeans, color='red', alpha=0.2)

    plt.plot(x, mean_oracle, color= 'green', label='Oracle ProbCover')
    plt.plot(x, acc_oracle_test, color= 'green', linestyle='dashdot', label='Oracle ProbCover')
    plt.fill_between(x, mean_oracle - std_oracle, mean_oracle + std_oracle, color='green', alpha=0.2)

    plt.plot(x, mean_random, color= 'black', label='Random Sampling')
    plt.plot(x, acc_random_test, color= 'black', linestyle='dashdot', label='Random Sampling')
    plt.fill_between(x, mean_random - std_random, mean_random + std_random, color='black', alpha=0.2)

    plt.legend(title="Active learning method")
    plt.title(f"Accuracy for {case} {train_dataset.name} dataset")
    plt.savefig(f'{train_dataset.name}_{n_iter}iterations_{case}.png')
    plt.show()



## Accuracy plots for all toy datasets
n_queries= np.concatenate((np.repeat(5, 10), np.repeat(10, 4), np.repeat(20,3)))
purity_threshold=0.95
gamma=3
n_iter=10

for dataset_name, dataset_setting in dataset_parameters.items():
    for dataset_setting_name, params in dataset_setting.items():
        print(dataset_name, dataset_setting_name, params)
        # Creating the train test datasets
        if dataset_name=="clouds":
            train_dataset= PointClouds(params["centers"], params["std"], (params["p"]*n_train).astype(int), random_state=1)
            test_dataset= PointClouds(params["centers"], params["std"], (params["p"]*n_test).astype(int), random_state=2)
        elif dataset_name=="circles":
            train_dataset = CenteredCircles(params["center"], params["r"], (params["p"]*n_train).astype(int), params["std"], random_state=1)
            test_dataset = CenteredCircles(params["center"], params["r"], (params["p"]*n_test).astype(int), params["std"], random_state=2)
        elif dataset_name=="moons":
            train_dataset = TwoMoons(params["centers"], params["r"], params["std"], (params["p"]* n_train).astype(int), random_state=1)
            test_dataset = TwoMoons(params["centers"], params["r"], params["std"], (params["p"]* n_test).astype(int), random_state=2)
        train_dataset.plot_dataset()
        test_dataset.plot_dataset()
        k= len(np.unique(train_dataset.y))
        accuracy_simulation_average(train_dataset, test_dataset, n_queries, k, purity_threshold, gamma, n_iter, dataset_setting_name)



embed()


# ========================Initializing all the datasets =======================