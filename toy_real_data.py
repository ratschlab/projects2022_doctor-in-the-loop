import numpy as np
from datasets import PointClouds, CIFAR10_simclr, CenteredCircles
from scipy.spatial import ConvexHull
from IPython import embed


## Initialize the point clouds dataset
m = 400
cluster_centers = [(-5, -5), (-6, 0), (5, -1), (5, 4)]
cluster_std = [0.2, 1.2, 1.2, 1.2]
p=np.array([0.8, 0.1, 0.05, 0.05])
cluster_samples = p*m
cluster_samples=cluster_samples.astype(int)

clouds_data= PointClouds(cluster_centers, cluster_std, cluster_samples, random_state=1)
dataset= clouds_data

## Initialize the circles dataset
center=[4,5]
radiuses=[0.5, 3, 5]
samples= [30, 100 ,150]
std=[0.5, 0.5, 0.55]

circles_data=CenteredCircles(center, radiuses, samples, std)

## Initialize CIFAR10 extracted features dataset
cifar10_features= CIFAR10_simclr(n_epochs=100)


def point_in_hull(points, hull, tolerance=1e-12):
    in_hull=[np.dot(eq[:-1], point) + eq[-1] <= tolerance for eq in hull.equations for point in points]
    in_hull= np.array(in_hull).reshape(len(hull.equations), len(points))
    in_hull= np.transpose(in_hull)
    return np.all(in_hull, axis=1)

def convex_hull_purity(dataset):
    #for each class, find its convex hull and the amount of points in the convex hull that are not from the class
    classes= np.unique(dataset.y)
    class_purity={}
    for c in classes:
        hull= ConvexHull(dataset.x[dataset.y==c], incremental=False)
        in_hull= point_in_hull(dataset.x[dataset.y!=c], hull)
        class_purity[c]= 1-(np.sum(in_hull)/len(in_hull)) # how should I normalise this? Maybe divide by the length of the entire dataset?
    return class_purity


a= convex_hull_purity(clouds_data)
b= convex_hull_purity(circles_data)
c= convex_hull_purity(cifar10_features)
embed()
