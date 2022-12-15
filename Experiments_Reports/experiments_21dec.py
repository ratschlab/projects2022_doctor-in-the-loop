import numpy as np
from datasets import CenteredCircles, PointClouds, MixedClusters, CIFAR10_simclr, TwoMoons
from activelearners import ProbCoverSampler, active_learning_algo
from clustering import MySpectralClustering, MyKMeans, ClusteringAlgo
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt
from models import Classifier1NN
from helper import check_cover, get_radius, cover

## Initialize the two moons dataset

centers = [[1, -0.5], [2.5, 0]]
r = [[1.5, 2], [1, 1]]
samples = [100, 100]
cluster_std = [0.4, 0.1]

if (centers[1][0] + r[1][0] >= centers[0][0] + r[0][0] >= centers[1][0] - r[1][0] >= centers[0][0] - r[0][0]):
    # create the dataset
    moons_data = TwoMoons(centers, r, cluster_std, samples)
    moons_data.plot_dataset()
else:
    print("Wrong parameters to create the moons dataset")

embed()
