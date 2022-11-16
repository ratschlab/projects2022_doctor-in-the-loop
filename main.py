import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import seaborn as sns
from activelearners import ActiveLearner, RandomSampler, TypiclustSampler, ProbCoverSampler
from datasets import PointClouds
from helper import get_emperical_radius


m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
# cluster_std=[0.2, 0.5, 0.5, 0.5]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)
M =15
B=5
K=5
show_all_clusters=True

## Initializing the dataset
clustered_2d_data= PointClouds(cluster_centers, cluster_std, cluster_samples)
clustered_2d_data.plot2d()

## Initializing the active learners
random_learner= RandomSampler(dataset=clustered_2d_data.dataset)
typiclust_learner= TypiclustSampler(dataset=clustered_2d_data.dataset, n_neighbours=K)
probcover_learner_1= ProbCoverSampler(dataset=clustered_2d_data.dataset, purity_radius=2,purity_threshold=None, k=4,  plot=[True, True])
probcover_learner_2= ProbCoverSampler(dataset=clustered_2d_data.dataset, purity_radius=None, purity_threshold=0.95, k=4,plot=[True, True],
                                     search_range=[2.3,3], search_step=0.1)
## ProbCover demo
def figure1():
    probcover_learner_1.demo_2dplot(10,5, final_plot=True)
    probcover_learner_2.demo_2dplot(10, 5, final_plot=True)




embed()

