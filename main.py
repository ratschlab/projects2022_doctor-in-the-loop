from datasets import MixedClusters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed
from clustering import OracleClassifier, MyKMeans, MySpectralClustering
from activelearners import ProbCoverSampler_Faiss
from copy import deepcopy
from activelearners import ProbCoverSampler



cluster_centers= np.array([[0,0]])
cluster_std= [0.3]
cluster_samples= np.array([400])
dataset= MixedClusters(3, cluster_centers, cluster_std, cluster_samples, random_state=1)

clustering= MyKMeans(dataset, 3)
learner= ProbCoverSampler(dataset, 0.95, clustering)
learner.update_radius(0.3)

learner.adaptive_query(5)
dataset.plot_al(plot_circles=True)


embed()

