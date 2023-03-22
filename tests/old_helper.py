from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from sympy.utilities.iterables import multiset_permutations
import scipy
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

def accuracy_clustering(true_labels, pseudo_labels):
    assert(true_labels.shape==pseudo_labels.shape)
    true_labels= true_labels.astype(int)
    pseudo_labels= pseudo_labels.astype(int)
    true = np.unique(true_labels)
    pseudo= np.unique(pseudo_labels)
    permutations = [p for p in multiset_permutations(true)]
    accuracy=[]
    for i, p in enumerate(permutations):
        true_labels_copy= true_labels.copy()
        for label in true:
            true_labels_copy[true_labels==label]= p[label]
        accuracy.append(accuracy_score(true_labels_copy, pseudo_labels))
    return np.max(accuracy)

def accuracy_clustering(true_labels, pseudo_labels):
    assert(true_labels.shape==pseudo_labels.shape)
    true_labels = true_labels.astype(int)
    pseudo_labels = pseudo_labels.astype(int)
    classes= np.unique(true_labels)
    print(classes, np.unique(pseudo_labels))

    corresp= np.zeros(shape=(len(classes), 2), dtype=int)
    corresp[:, 0]= classes

    for c in classes:
        idx= np.where(true_labels==c)[0]
        label, count= np.unique(pseudo_labels[idx], return_counts=True)

        print(c, ":","label", label, "count", count)
        id= np.argmax(count)
        print(id, label[id])
        corresp[c, 1]= label[id]
        print(corresp)
    new_labels= change_labels(pseudo_labels, corresp, reverse=True)
    accuracy= accuracy_score(new_labels, true_labels)
    print(accuracy)
    return accuracy

def change_labels(labels, label_corresp, reverse=False):
    """
    label_corresp: np.array of shape n_classes x 2, first column is current label and second column is the replacement label
    """
    new_labels= labels.copy()
    for perm in label_corresp:
        if reverse==False:
            old, new= perm
        elif reverse==True:
            new, old= perm
        idx= np.where(labels==old)[0]
        new_labels[idx]= new
    return new_labels



def typicality(X, K):
    """
    Args:
        K: hyperparameter deciding on the number of nearest neighbours to use
        X: array (n_samples, n_features)
    Returns:
        t: typicality array of shape (n_samples, 1)
    """
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    distances, _ = knn.kneighbors(X)
    t = 1 / (np.mean(distances, axis=1) + 0.000001)
    return t.reshape(-1, 1)
