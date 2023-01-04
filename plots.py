import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed
from clustering import MyKMeans, OracleClassifier, MySpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datasets import CIFAR_simclr
from sklearn.model_selection import train_test_split
from helper import coclust

## plotting for the toy dataset only up until budget 250
n_queries_toy = np.concatenate((np.repeat(5, 10), np.repeat(10, 4), np.repeat(20, 3)))
n_queries_cifar10 = np.concatenate((np.repeat(50, 4), np.repeat(100, 5), np.repeat(200,5)))
n_queries_cifar100= np.concatenate((np.repeat(50, 6), np.repeat(100, 4)))
n_queries_cifar10_unbalanced= np.concatenate((np.repeat(50, 4), np.repeat(100, 5)))

labels=["Spectral ProbCover", "Kmeans ProbCover", "Oracle ProbCover", "Random Sampler"]
colors=["blue", "red", "green", "black"]
n_iter=10
toy=False
cifar10= True
cifar100= False
cifar10_unbalanced= False


##CIFAR10 dataset
n_train, p_train= 800, 0.8
n_test, p_test= 200, 0.2

cifar10_100epochs= CIFAR_simclr(n_classes=10, n_epochs=100)
cifar10_200epochs= CIFAR_simclr(n_classes=10, n_epochs=200)
cifar10_400epochs= CIFAR_simclr(n_classes=10, n_epochs=400)
cifar10_800epochs= CIFAR_simclr(n_classes=10, n_epochs=800)
cifar10_1000epochs= CIFAR_simclr(n_classes=10, n_epochs=1000)
train_idx10, test_idx10= train_test_split(np.arange(10000), test_size=p_test, random_state=1)


##CIFAR100 dataset
n_train, p_train= 800, 0.8
n_test, p_test= 200, 0.2

cifar100_100epochs= CIFAR_simclr(n_classes=100, n_epochs=100)
cifar100_200epochs= CIFAR_simclr(n_classes=100, n_epochs=200)
cifar100_400epochs= CIFAR_simclr(n_classes=100, n_epochs=400)
cifar100_800epochs= CIFAR_simclr(n_classes=100, n_epochs=800)
cifar100_1000epochs= CIFAR_simclr(n_classes=100, n_epochs=1000)
train_idx100, test_idx100= train_test_split(np.arange(10000), test_size=p_test, random_state=2)

cifar10= [cifar10_100epochs, cifar10_200epochs, cifar10_400epochs,
              cifar10_800epochs, cifar10_1000epochs]
cifar100= [cifar100_100epochs, cifar100_200epochs, cifar100_400epochs,
              cifar100_800epochs, cifar100_1000epochs]

train_dataset, test_dataset = cifar10_100epochs.split(train_idx10, test_idx10)

cifar10_labels= cifar10_100epochs.y


if cifar10:
    x= np.cumsum(n_queries_cifar10)
    for i, n_epochs in enumerate(np.array([100,200,400,800,1000])):
        path = "/Users/victoriabarenne/projects2022_doctor-in-the-loop/Experiments_Reports/CIFAR10_1iteration/"
        train = pd.read_csv(path + f"csv/train_{n_epochs}epochs_1iteration.csv", skiprows=1, header=None, index_col=0)
        test = pd.read_csv(path + f"csv/test_{n_epochs}epochs_1iteration.csv", skiprows=1, header=None, index_col=0)
        train= train.values
        test= test.values.reshape(4, -1)
        unsupervised= pd.read_csv(path+ f"unsupervised_CIFAR10_{n_epochs}epochs.csv", skiprows=1, header=None, index_col=0).values

        for i in range(4):
            plt.plot(x, train[i,:], color=colors[i], label=labels[i])
            plt.plot(x, test[i,:], color=colors[i], linestyle='dashdot', label=labels[i])
        print(unsupervised[0], unsupervised[1], unsupervised[2], unsupervised[3])
        plt.axhline(y=unsupervised[0], color='red')
        plt.axhline(y=unsupervised[1], color='blue')
        plt.axhline(y=unsupervised[2], color='red', linestyle='dashdot')
        plt.axhline(y=unsupervised[3], color='blue', linestyle='dashdot')

        plt.ylim(0.05, 0.6)
        plt.xlim(25,250)
        plt.legend(title="Active learning method")
        plt.title(f"Accuracy for CIFAR10 features at {n_epochs} epochs")
        plt.savefig(f'CIFAR10_{n_epochs}_modified.png')
        plt.show()

if cifar100:
    x= np.cumsum(n_queries_cifar100)
    for n_epochs in[100,200,400,800,1000]:
        path = "/Users/victoriabarenne/projects2022_doctor-in-the-loop/Experiments_Reports/CIFAR100_1iteration/"
        train = pd.read_csv(path + f"csv/train_{n_epochs}epochs_1iteration.csv", skiprows=1, header=None, index_col=0)
        test = pd.read_csv(path + f"csv/test_{n_epochs}epochs_1iteration.csv", skiprows=1, header=None, index_col=0)
        train= train.values
        test= test.values.reshape(4, -1)
        unsupervised= pd.read_csv(path+ f"unsupervised_CIFAR100_{n_epochs}epochs.csv", skiprows=1, header=None, index_col=0).values

        print(train.shape, test.shape)
        for i in range(4):
            plt.plot(x, train[i,:], color=colors[i], label=labels[i])
            plt.plot(x, test[i,:], color=colors[i], linestyle='dashdot', label=labels[i])
        plt.axhline(y=unsupervised[0], color='red')
        plt.axhline(y=unsupervised[1], color='blue')
        plt.axhline(y=unsupervised[2], color='red', linestyle='dashdot')
        plt.axhline(y=unsupervised[3], color='blue', linestyle='dashdot')

        print(x)
        plt.ylim(0, 0.3)
        plt.xlim(0,700)
        plt.legend(title="Active learning method")
        plt.title(f"Accuracy for CIFAR100 features at {n_epochs} epochs")
        plt.savefig(f'CIFAR100_{n_epochs}_modified.png')
        plt.show()


if toy:
    x= np.cumsum(n_queries_toy)
    path = "/Users/victoriabarenne/projects2022_doctor-in-the-loop/Experiments_Reports/Toy_10iterations/"
    for name in ["circles", "clouds", "moons"]:
        for case in ["baseline", "unbalanced", "margins"]:
            train= pd.read_csv(path+f"csv/trainaccuracy_{name}_{case}_10.csv", skiprows=1, header=None, index_col=0)
            test= pd.read_csv(path+f"csv/testaccuracy_{name}_{case}_10.csv", skiprows=1, header=None, index_col=0)
            train= train.values
            test= test.values.reshape(4, -1)
            unsupervised = pd.read_csv(path + f"unsupervised_{name}_{case}.csv")

            for i in range(4):
                mean_train= np.mean(train[i*n_iter:(i+1)*n_iter, :], axis=0)
                std_train= np.std(train[i*n_iter:(i+1)*n_iter, :], axis=0)
                mean_test= test[i,:]

                plt.plot(x, mean_train, color=colors[i], label=labels[i])
                plt.plot(x, mean_test, color=colors[i], linestyle='dashdot', label=labels[i])
                plt.fill_between(x, mean_train - std_train, mean_train + std_train, color=colors[i], alpha=0.2)

            plt.legend(title="Active learning method")
            plt.title(f"Accuracy for {case} {name} dataset")
            plt.savefig(f'{name}_{n_iter}iterations_{case}_modified.png')
            plt.show()

# if cifar10_unbalanced:
#
#     x= np.cumsum(n_queries_cifar10_unbalanced)
#     for n_epochs in[100,200,400,800,1000]:
#         path = "/Users/victoriabarenne/projects2022_doctor-in-the-loop/Experiments_Reports/CIFAR10_unbalanced/csv/"
#         train = pd.read_csv(path + f"train_{n_epochs}epochs_unbalanced_1.csv", skiprows=1, header=None, index_col=0)
#         test = pd.read_csv(path + f"test_{n_epochs}epochs_unbalanced_1.csv", skiprows=1, header=None, index_col=0)
#         train= train.values
#         test= test.values.reshape(4, -1)
#         print(train.shape, test.shape)
#         for i in range(4):
#             plt.plot(x, train[i,:], color=colors[i], label=labels[i])
#             plt.plot(x, test[i,:], color=colors[i], linestyle='dashdot', label=labels[i])
#         plt.ylim(0.35, 0.70)
#         plt.xlim(25,600)
#         plt.legend(title="Active learning method")
#         plt.title(f"Accuracy for CIFAR10 features at {n_epochs} epochs")
#         plt.savefig(f'CIFAR10_{n_epochs}_modified.png')
#         plt.show()