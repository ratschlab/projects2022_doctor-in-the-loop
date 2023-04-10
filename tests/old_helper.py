from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from sympy.utilities.iterables import multiset_permutations
import scipy
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from activelearners import ActiveLearner, ProbCoverSampler_Faiss
from models import Classifier1NN


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



def update_adjacency_radius_faiss(dataset, new_radiuses, lims, D, I):
    n_vertices= len(dataset.x)
    #replace lims, D, I using the new radiuses
    if (dataset.radiuses!=new_radiuses).any():
        for u in np.where(dataset.radiuses!=new_radiuses)[0]:
            #keep initial lims, I, D because it will be always smaller
            new_distances= np.linalg.norm(dataset.x[u, :] - dataset.x[np.arange(n_vertices), :], axis=1)
            old_covered_points= I[lims[u]:lims[u+1]]
            new_covered_points= np.where(new_distances < new_radiuses[u])[0]  # indices are already sorted
            new_distances= new_distances[new_covered_points]
            length_diff= lims[u+1]- lims[u]- len(new_covered_points)

            #update the neighbours and distances
            I= np.concatenate((I[:lims[u]], new_covered_points, I[lims[u+1]:]), axis=0)
            D= np.concatenate((D[:lims[u]], new_distances, D[lims[u+1]:]), axis=0)
            # update the indices
            lims[u + 1:] = lims[u + 1:] - length_diff
            # construct r (similar to lims lims[i]:lims[i+1] is ri)
            #D[D<r]
            # if the points whose radius we're changing were labeled, we also need to remove/recover incoming edges
            if (new_radiuses[u]>dataset.radiuses[u])&(dataset.labeled[u]==1):
                # remove all incoming edges of the newly covered points
                for v in new_covered_points[np.invert(np.isin(new_covered_points, old_covered_points))]:
                    # find points covered by v
                    covered_by_v= I[lims[v]:lims[v+1]]
                    # remove them from the list of neighbours I and update lims, D accordingly
                    covered_by_v_I_bool= np.isin(I,covered_by_v)
                    covered_by_v_I_id= np.where(covered_by_v_I_bool)[0]
                    # n_covered = np.zeros(len(lims))
                    # for l in range(1, len(lims)):
                    #     n_covered[l] = np.sum(lims[l - 1] <= covered_by_v_I_id < lims[l])
                    n_covered= np.array([np.sum(lims[l - 1] <= covered_by_v_I_id < lims[l]) for l in range(1, len(lims))])
                    lims = lims - np.cumsum(n_covered)
                    lims = lims.astype(int)
                    I= I[np.invert(covered_by_v_I_bool)]
                    D= D[np.invert(covered_by_v_I_bool)]

            elif (new_radiuses[u]< dataset.radiuses[u])&(dataset.labeled[u]==1):
                # need to recover incoming edges of points that are not covered anymore (if they are not covered by other labeled points!!)
                for v in old_covered_points[np.invert(np.isin(old_covered_points, new_covered_points))]:
                    # get all points that have an edge going to v and add v back to their list of neighbours
                    incoming_edge_vertices_v_distances= np.linalg.norm(dataset.x[v, :] - dataset.x[np.arange(n_vertices), :], axis=1)
                    incoming_edge_vertices_v_bool= (incoming_edge_vertices_v_distances<new_radiuses[v])
                    incoming_edge_vertices_v_id= np.where(incoming_edge_vertices_v_bool)[0]
                    for l in incoming_edge_vertices_v_id:
                        # recover only if incoming_edge_vertices_v_id is not covered by a labeled point
                        covered_by_other_labeled= np.max(np.linalg.norm(dataset.x[l, :] - dataset.x[dataset.queries[dataset.queries!=u], :], axis=1)<dataset.radiuses[dataset.queries])
                        if not covered_by_other_labeled:
                            I= np.insert(I, lims[l+1], v)
                            D= np.insert(D, lims[l+1], incoming_edge_vertices_v_distances[l])
                            lims[l+1:]= lims[l+1:]+1
        dataset.radiuses= new_radiuses
    return lims, D, I


#################################### Old ProbCover and its helper functions #############################################


######## Old ProbCover function #########
class ProbCoverSampler(ActiveLearner):
    def __init__(self, dataset, purity_threshold,
                 clustering:ClusteringAlgo,
                 plot=[False, False],
                 search_range=[0,100], search_step=0.1,
                 adaptive=False,
                 model=None):

        super().__init__(dataset, model)
        self.name = "ProbCover Sampler"
        self.clustering= clustering

        plot_clustering = plot[0]
        plot_unpure_balls = plot[1]

        # Get pseudo labels
        # clustering= MySpectralClustering(self.dataset, k , gamma=self.gamma)
        self.pseudo_labels = self.clustering.pseudo_labels
        if plot_clustering:
            self.clustering.plot()

        self.purity_threshold= purity_threshold
        # Get radius for given purity threshold
        self.radius = get_radius(self.purity_threshold, self.dataset.x, self.pseudo_labels, search_range, search_step,
                                                  plot_unpure_balls)
        print(f"ProbCover Sampler initialized for threshold {self.purity_threshold} with radius {round(self.radius, 2)}")

        #TODO: I ADDED THIS
        self.dataset.radiuses= np.repeat(self.radius, len(self.dataset.x))
        # Initialize the graph
        self.graph = adjacency_graph(self.dataset.x, self.dataset.radiuses)
        self.adaptive=adaptive

    def update_radius(self, new_radius):
        self.radius= new_radius
        #TODO: I added this
        self.dataset.radiuses= np.repeat(self.radius, len(self.dataset.x))
        self.graph= adjacency_graph(self.dataset.x, self.dataset.radiuses)
        print(f"Updated ProbCover Sampler radius: {round(self.radius, 2)}")

    def update_labeled(self, plot_clustering=False):
        if self.adaptive==False:
            raise ValueError("This active learner's radius and pseudo-labels are not adaptive")
        self.clustering.fit_labeled(self.dataset.queries)
        self.pseudo_labels= self.clustering.pseudo_labels
        self.radius = get_radius(self.purity_threshold, self.dataset.x, self.pseudo_labels, [0,10], 0.01)
        self.graph= adjacency_graph(self.dataset.x, self.radius)
        print(f"Updated ProbCover Sampler using the new acquired labels, new radius: {round(self.radius, 2)}")
        if plot_clustering:
            self.clustering.plot()

    def query(self, M, B=1, n_initial=1):
        graph = self.graph.copy()

        # Initialize the labels
        n_pool = self.dataset.n_points

        graph= update_adjacency_graph_labeled(self.dataset)

        for m in range(M):
            # get the unlabeled point with highest out-degree
            out_degrees = np.sum(graph, axis=1)
            max_out_degree = np.argmax(out_degrees[self.dataset.labeled == 0])
            c_id = np.arange(n_pool)[self.dataset.labeled == 0][max_out_degree]
            assert ((out_degrees[c_id] == np.max(out_degrees)) & (self.dataset.labeled[c_id] == 0))

            covered_vertices = np.where(graph[c_id, :] == 1)[0]
            graph[:, covered_vertices] = 0
            self.dataset.observe(c_id, self.radius)

    def adaptive_query(self, M, K=3, B=1, n_initial=1):
        graph = self.graph.copy()
        # Initialize the labels
        n_pool = self.dataset.n_points

        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        graph= update_adjacency_graph_labeled(self.dataset)

        for m in range(M):
            print(f"querying point {len(self.dataset.queries)+1}")
            if len(self.dataset.queries)>0:
                n_neighbours = K if len(self.dataset.queries) >= K else len(self.dataset.queries)
                nbrs = NearestNeighbors(n_neighbors=n_neighbours).fit(self.dataset.x[self.dataset.queries])
                distances, indices = nbrs.kneighbors(self.dataset.x[self.dataset.labeled==0])
                # set its radius as a weighted radius of its K-nn
                weights= 1/distances
                weights= weights/(weights.sum(axis=1).reshape(len(weights), -1))
                new_radiuses= (weights*self.dataset.radiuses[self.dataset.queries[indices]]).sum(axis=1)
                self.dataset.radiuses[self.dataset.labeled==0]= new_radiuses
                graph= update_adjacency_graph_labeled(self.dataset)
            # get the unlabeled point with highest out-degree
            out_degrees = np.sum(graph, axis=1)
            max_out_degree = np.argmax(out_degrees[self.dataset.labeled == 0])
            c_id = np.arange(n_pool)[self.dataset.labeled == 0][max_out_degree]
            assert ((out_degrees[c_id] == np.max(out_degrees[self.dataset.labeled == 0])) & (self.dataset.labeled[c_id] == 0))
            rc= self.dataset.radiuses[c_id]

            # Add point, adapting its radius and the radius of all points with conflicting covered regions
            if len(self.dataset.queries)>0:
                d = np.linalg.norm(self.dataset.x[c_id, :] - self.dataset.x[self.dataset.queries, :], axis=1)
                diff_radiuses, new_radiuses= np.zeros(shape= self.dataset.radiuses.shape), np.zeros(shape= self.dataset.radiuses.shape)
                diff_radiuses[self.dataset.queries]= self.dataset.radiuses[self.dataset.queries]+rc-d
                new_radiuses[self.dataset.queries]= np.maximum(0, 0.5*(self.dataset.radiuses[self.dataset.queries]-rc+d))
                mask= (diff_radiuses>0)* (self.dataset.y[c_id]!= self.dataset.y)*(self.dataset.labeled==1)
                self.dataset.radiuses[mask]= new_radiuses[mask]
                if mask.any():
                    rc= rc-0.5*np.max(diff_radiuses[mask])
                new_radiuses[c_id]= rc

            graph = update_adjacency_graph_labeled(self.dataset)


########## old helper functions for ProbCover

def adjacency_graph(x: np.array, radiuses):
    n_vertices = len(x)
    if isinstance(radiuses, float) or radiuses.ndim == 0:
        radiuses= np.repeat(radiuses, repeats= len(x))
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        covered_u= (np.linalg.norm(x[u,:]- x[np.arange(n_vertices),:], axis=1)< radiuses[u])
        graph[u, covered_u]=1
    return graph

def update_adjacency_graph_labeled(dataset):
    n_vertices = len(dataset.x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    # Update all points covered by u with radius ru
    for u in range(n_vertices):
        covered_u= (np.linalg.norm(dataset.x[u,:]- dataset.x[np.arange(n_vertices),:], axis=1)< dataset.radiuses[u])
        graph[u, covered_u]=1
    # Remove all incoming edges of all points covered by labeled points
    if len(dataset.queries)>0:
        all_covered= np.where(np.max(graph[dataset.queries,:], axis=0)==1)[0]
        graph[:,all_covered]=0
    return graph

def get_purity(x, pseudo_labels, nn_idx, delta, plot_unpure_balls=False):
    """
    Args:
        x: np.array
        pseudo_labels: labels predicted by some clustering algorithm
        nn_idx: id of the nearest neighbour of each point
        delta: radius

    Returns:
        purity(delta)
    """
    purity = np.zeros(len(x), dtype=int)
    graph = adjacency_graph(x, delta)

    for u in range(len(x)):
        covered_vertices = np.where(graph[u, :] == 1)[0]
        purity[u] = int(np.all(pseudo_labels[nn_idx[covered_vertices]] == pseudo_labels[nn_idx[u]]))

    if plot_unpure_balls:
        fig, ax = plt.subplots()
        ax.axis('equal')
        sns.scatterplot(x=x[:,0], y=x[:,1], hue=pseudo_labels, palette="Set2")
        for u in range(len(x)):
            if purity[u] == 0:
                plt.plot(x[u, 0], x[u, 1], color="red", marker="P")
                ax.add_patch(plt.Circle((x[u, 0], x[u, 1]), delta, color='red', fill=False, alpha=0.5))
        plt.title(f"Points that are not pure and their covering balls of radius {delta}")
        plt.show()

    return np.mean(purity)

def get_radius(alpha: float, x: np.array, pseudo_labels,
                         search_range=[0.5, 3.5], search_step=0.05, plot_unpure_balls=False):


    # Get the indices of the 1 nearest neighbours for each point
    nn_idx = get_nearest_neighbour(x)
    radiuses = np.arange(search_range[0], search_range[1], search_step)

    while len(radiuses)>1:
        id = int(len(radiuses) / 2)
        purity= get_purity(x, pseudo_labels, nn_idx, radiuses[id])
        radiuses= radiuses[id:] if purity>= alpha else radiuses[:id]
    purity_radius = radiuses[0]
    if plot_unpure_balls:
        get_purity(x, pseudo_labels, nn_idx, purity_radius, True)

    return purity_radius


def weighted_graph(dataset: pd.DataFrame, kernel_fn, kernel_hyperparam, sampled_labels=None):
    n_vertices = len(dataset.x)
    graph = np.zeros(shape=(n_vertices, n_vertices))
    for u in range(n_vertices):
        for v in range(u):
            # if sampled_labels is not None:
            if len(dataset.queries)>=1:
                if (u in dataset.queries)&(v in dataset.queries):
                    if (dataset.y[u] == dataset.y[v]) & (u != v):
                        graph[u,v]= 1000
                        graph[v,u]= 1000
                        # otherwise the weight is kept as zero
                else:
                    w= kernel_fn(dataset.x[u,:], dataset.x[v,:], kernel_hyperparam)
                    graph[u, v] = w
                    graph[v, u] = w
            else:
                w = kernel_fn(dataset.x[u,:], dataset.x[v,:], kernel_hyperparam)
                graph[u, v] = w
                graph[v, u] = w
    return graph

def get_nearest_neighbour(x: np.array):
    knn = NearestNeighbors(n_neighbors=2).fit(X=x)
    _, idx = knn.kneighbors(x)
    idx = idx[:, 1]  # contains the nearest neighbour for each point
    return idx


#################################### Old Typiclust and its helper functions #############################################

class TypiclustSampler(ActiveLearner):
    def __init__(self, dataset, n_neighbours=5, model=None):
        super().__init__(dataset, model)
        self.n_neighbours= n_neighbours
        self.name="Typiclust Sampler"


    def query(self, M, B=1, plot=[False, False],  n_initial=1):
        super().query(M,B,n_initial)

        assert(sum(plot)<=1) # You can only choose one type of plot output
        show_clusters= plot[0]
        show_all_clusters=plot[1]
        count=0

        while count<M:
            n_labeled=np.sum(self.dataset.labeled)

            # Cluster the data using both labeled and unlabeled samples
            kmeans = KMeans(n_clusters=B+n_labeled, random_state=0).fit(self.dataset.x)
            cluster_id= kmeans.predict(self.dataset.x)
            self.dataset["cluster_id"]= cluster_id

            # Extract the B cluster_ids for the B largest uncovered clusters (ie containing only unlabeled data)
            covered= np.unique(self.dataset[self.dataset.queries]["cluster_id"])
            uncovered=np.array(list(set(np.arange(B+n_labeled))-set(covered)))
            uc_sizes=np.zeros(len(uncovered))

            for i,c in enumerate(uncovered):
                uc_sizes[i]= len(self.dataset[self.dataset["cluster_id"]==c])
            cluster_indexes= np.argpartition(uc_sizes, -B)[-B:]
            clusters= uncovered[cluster_indexes]

            # Label the most typical sample from each of the B largest uncovered clusters
            if show_all_clusters:
                fig, ax = plt.subplots()
                ax.axis('equal')
                sns.scatterplot(data=self.dataset, x="x1", y="x2", hue="cluster_id", palette=sns.color_palette("husl", B+n_labeled), ax=ax)
                plt.title(f"Demo of {self.name} with all clusters shown")
            if show_clusters:
                fig, ax = plt.subplots()
                ax.axis('equal')
                sns.scatterplot(data=self.dataset.iloc[np.isin(self.dataset["cluster_id"], clusters)],
                                x="x1", y="x2", hue="cluster_id", palette="Set2")
                sns.scatterplot(data=self.dataset.iloc[np.isin(self.dataset["cluster_id"], clusters, invert=True)],
                                x="x1", y="x2", color="grey")
                plt.title(f"Demo of {self.name} with only the {B} largest uncovered clusters shown")
            if show_all_clusters or show_clusters:
                sns.scatterplot(data=self.dataset[self.dataset.queries], x="x1", y="x2", color="black", marker="P", s=100,
                                label="Points sampled in the previous iteration", linewidths=3)
            for b in clusters:
                data_cluster= self.dataset[self.dataset["cluster_id"]==b]
                if (self.n_neighbours<=len(data_cluster)) & (len(data_cluster)>=1):
                    t=typicality(data_cluster.iloc[:,:-1], self.n_neighbours)
                elif len(data_cluster)>=1:
                    t = typicality(data_cluster.iloc[:,:-1], len(data_cluster))
                idx_c= np.argmax(t) #id in data_cluster
                idx= data_cluster.iloc[idx_c].name

                self.dataset.observe(idx)
                count+=1

            if show_all_clusters or show_clusters:
                sns.scatterplot(data=self.dataset.iloc[self.dataset.queries[count-B:count]], x="x1", y="x2", s=100, color="red", marker="P")
                ax.get_legend().remove()
                plt.show()

        self.dataset= self.dataset.drop(columns=["cluster_id"])



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




################################# Active Learning algo and cover functions ############################################

def active_learning_algo(dataset, clustering, M, B, initial_purity_threshold, p_cover=1):
    dataset.restart()
    activelearner= ProbCoverSampler_Faiss(dataset, initial_purity_threshold,
                                     clustering, [True, False],
                                     search_range=[0,10], search_step=0.01)

    purity_threshold= initial_purity_threshold
    pseudo_labels= activelearner.pseudo_labels
    radius= activelearner.radius
    covered= check_cover(dataset.x, dataset.queries, radius, p_cover)

    model= Classifier1NN(dataset)

    while len(dataset.queries)<M:
        if ((not covered) & (len(dataset.queries)+B<=M)):
            print("Dataset not covered yet")
            activelearner.query(B)
            covered= check_cover(dataset.x, dataset.queries, radius, p_cover)
        elif covered:
            print(f"Dataset covered with {len(dataset.queries)} queries")
            model.update()
            print(f"NN Classifier accuracy: {model.accuracy}")

            # clustering= MySpectralClustering(dataset, k, gamma)
            clustering.fit_labeled(dataset.queries)
            clustering.plot()

            old_labels= pseudo_labels
            old_radius= radius
            pseudo_labels= clustering.pseudo_labels
            while covered:
                changed=(accuracy_clustering(old_labels, pseudo_labels)<1)
                if changed:
                    radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    print(f"The labels were changed with {100*accuracy_clustering(old_labels, pseudo_labels)}% overlap and the new radius is {radius}")
                else:
                    purity_threshold+=(1-purity_threshold)/2
                    radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    print(f"The labels were not changed so we set a higher purity threshold {purity_threshold} with radius {radius}")

                if radius > old_radius:
                    print(f"Radius is bigger, so we raise purity threshold and start the process again")
                    purity_threshold+=(1-purity_threshold)/2
                    radius= get_radius(purity_threshold, dataset.x, pseudo_labels, search_range=[0,10], search_step=0.01)
                    activelearner.update_radius(radius)
                    print(f'Changed active learners purity threshold to {purity_threshold} and radius to {radius}')
                    if radius< old_radius:
                        covered= check_cover(dataset.x, dataset.queries, radius, p_cover)
                    else:
                        print(f'Radius is still bigger or equal than previously so we know the dataset is covered')

                elif radius< old_radius:
                    covered= check_cover(dataset.x, dataset.queries, radius, p_cover)
                    print(f"Radius is smaller")
                    if not covered:
                        activelearner.update_radius(radius)
                        print(f'Changed active learners purity threshold to {purity_threshold} '
                              f'and radius to {radius}')

                elif radius==old_radius:
                    print("Radius is the same")
                    return radius

                old_radius= radius
                old_labels= pseudo_labels
def coclust(true_labels, pseudo_labels):
    cm = confusion_matrix(true_labels, pseudo_labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js= indexes[1]
    cm2 = cm[:, js]
    accuracy= np.trace(cm2) / np.sum(cm2)

    return accuracy
def check_P_cover(x, labeled_idx, radius: float, P=None):
    assert(isinstance(P, int))
    n_vertices= len(x)
    unlabeled_idx = np.array(list(set(np.arange(n_vertices)) - set(labeled_idx)))
    adjacency= adjacency_graph(x, radius)
    dist_matrix= shortest_path(csgraph= adjacency, directed=False, indices= unlabeled_idx, return_predecessors=False)
    paths_length= np.partition(dist_matrix, 1, axis=1)[:, 1]
    max_length= np.max(shortest_path)
    return max_length<=P

def cover(x, labeled_idx, radius:float):
    n_vertices = len(x)
    unlabeled_idx = np.array(list(set(np.arange(n_vertices)) - set(labeled_idx)))
    adjacency = adjacency_graph(x, radius)
    covered= adjacency[unlabeled_idx.reshape(-1,1), labeled_idx.reshape(1,-1)]
    if ((covered.shape[0]==0) or (covered.shape[1]==0)):
        covered_points=0
    else:
        covered_points= np.sum(np.max(covered, axis=1))/(len(unlabeled_idx))
    return covered_points

def check_cover(x, labeled_idx, radius: float, p_cover:float):
    covered_points= cover(x, labeled_idx, radius)
    return covered_points>=p_cover


def get_covered_points(dataset, radius, id):
    n_vertices= len(dataset.x)
    distances= np.linalg.norm(dataset.x[id,:]- dataset.x[np.arange(n_vertices), :], axis=1)
    covered= (distances<= radius)
    return covered

def get_all_covered_points(dataset):
    n_vertices= len(dataset.x)
    all_covered= np.array([], dtype=int)
    for i, id in enumerate(dataset.queries):
        distances = np.linalg.norm(dataset.x[id, :] - dataset.x[np.arange(n_vertices), :], axis=1)
        covered= np.where(distances<= dataset.radiuses[i])[0]
        all_covered = np.concatenate((all_covered, covered), axis=0).astype(int)
    return np.unique(all_covered)
