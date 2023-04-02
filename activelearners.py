from helper import adjacency_graph
from clustering import ClusteringAlgo, MyKMeans, MySpectralClustering
from IPython import embed
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from helper import check_cover
from helper import get_purity_faiss, get_radius_faiss, get_nn_faiss, adjacency_graph_faiss, get_covered_points
from models import Classifier1NN
from helper import get_nearest_neighbour, update_adjacency_graph_labeled
from sklearn.neighbors import NearestNeighbors



class ActiveLearner:
    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model
        self.name= None

    def query(self, M, B, n_initial=1):
        assert((M%B==0)&(B<=M))

    def demo_2dplot(self, M, B, all_plots=False, final_plot=False, n_initial=1):
        self.dataset.restart()
        self.query(M,B)

        if all_plots:
            for i in range(int(M/B)):
                fig, ax = plt.subplots()
                ax.axis('equal')
                sns.scatterplot(x= self.dataset.x[:,0], y=self.dataset.x[:,1], hue=self.dataset.y, palette="Set2")
                sns.scatterplot(x=self.dataset.x[self.dataset.queries[0: i*B],0], y=self.dataset.x[self.dataset.queries[0: i*B],1], color="black", marker="P", s=150)
                sns.scatterplot(x=self.dataset.x[self.dataset.queries[i*B:(i+1)*B], 0], y=self.dataset.x[self.dataset.queries[i*B:(i+1)*B],1], color="red", marker="P", s=150)
                if (self.name=="ProbCover Sampler"):
                    for u in self.dataset.queries[i*B:(i+1)*B]:
                        ax.add_patch(plt.Circle((self.dataset.x[u, 0], self.dataset.x[u, 1]),
                                                self.radius, color='red', fill=False, alpha=0.5))
                plt.title(f"{self.name} with {B*(i+1)} sampled points and batch size {B}")
                plt.show()
        if final_plot:
            fig, ax = plt.subplots()
            ax.axis('equal')
            sns.scatterplot(x=self.dataset.x[:, 0], y=self.dataset.x[:, 1], hue=self.dataset.y, palette="Set2")
            sns.scatterplot(x=self.dataset.x[self.dataset.queries,0], y=self.dataset.x[self.dataset.queries,1], color="red", marker="P", s=150)
            if (self.name == "ProbCover Sampler"):
                for u in self.dataset.queries:
                    ax.add_patch(plt.Circle((self.dataset.x[u, 0], self.dataset.x[u, 1]),
                                            self.radius, color='red', fill=False, alpha=0.5))
            plt.title(f"{self.name} with {M} sampled points and batch size {B}")
            plt.show()


class RandomSampler(ActiveLearner):
    def __init__(self, dataset, model=None):
        super().__init__(dataset, model)
        self.name="Random Sampler"

    def query(self, M, B=1, n_initial=1):
        super().query(M,B,n_initial)

        for i in range(int(M/B)):
            idx= np.random.choice(np.where(self.dataset.labeled==0)[0], B, replace=False)
            self.dataset.observe(idx)

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




# TODO: once all the points are covered the indices are being sampled not randomly but by increasing index order

class ProbCoverSampler_Faiss(ActiveLearner):
    def __init__(self, dataset, purity_threshold,
                 clustering:ClusteringAlgo,
                 plot=[False, False],
                 search_range=[0,10], search_step=0.01,
                 adaptive= False,
                 model=None):

        super().__init__(dataset, model)
        self.name = "ProbCover Sampler"
        self.clustering= clustering

        plot_clustering = plot[0]
        plot_unpure_balls = plot[1]

        # Get pseudo labels
        self.pseudo_labels = self.clustering.pseudo_labels
        if plot_clustering:
            self.clustering.plot()


        self.purity_threshold= purity_threshold

        # Get radius for given purity threshold
        self.radius = get_radius_faiss(self.purity_threshold, self.dataset.x, self.pseudo_labels, search_range, search_step,
                                                  plot_unpure_balls)
        print(f"ProbCover Sampler initialized for threshold {self.purity_threshold} with radius {self.radius}")

        # Initialize the graph
        self.lims, _, self.I = adjacency_graph_faiss(self.dataset.x, self.radius)
        self.adaptive= adaptive

    def update_radius(self, new_radius):
        self.radius= new_radius
        self.lims, _, self.I = adjacency_graph_faiss(self.dataset.x, self.radius)
        print(f"Updated ProbCover Sampler radius: {self.radius}")

    def update_labeled(self, plot_clustering=False):
        if self.adaptive==False:
            raise ValueError("This active learner's radius and pseudo-labels are not adaptive")
        self.clustering.fit_labeled(self.dataset.queries)
        self.pseudo_labels= self.clustering.pseudo_labels
        self.radius = get_radius_faiss(self.purity_threshold, self.dataset.x, self.pseudo_labels, [0,10], 0.01)
        self.lims, _, self.I= adjacency_graph_faiss(self.dataset.x, self.radius)
        print(f"Updated ProbCover Sampler using the new acquired labels, new radius: {self.radius}")
        if plot_clustering:
            self.clustering.plot()

    def query(self, M, B=1, n_initial=1):
        lims, I = self.lims.copy(), self.I.copy()
        # Initialize the labels
        n_pool = self.dataset.n_points

        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if np.sum(self.dataset.labeled) >= 1:
            # Recover all the covered vertices
            covered= np.array([], dtype=int)
            for u in self.dataset.queries:
                covered= np.concatenate((covered, I[lims[u]:lims[u + 1]]), axis=0)
            covered= np.unique(covered)
            I_covered_idx = np.sort(np.where(np.isin(I, covered))[0])

            #Remove all incoming edges to the covered vertices
            n_covered = np.zeros(len(lims))
            for l in range(1, len(lims)):
                n_covered[l] = np.sum(((lims[l - 1] <= I_covered_idx) & (I_covered_idx < lims[l])))
            lims = lims - np.cumsum(n_covered)
            lims= lims.astype(int)
            I = np.delete(I, I_covered_idx)


        for _ in range(M):
            # get the unlabeled point with highest out-degree
            out_degrees= lims[1:] - lims[:-1]
            max_out_degree = np.argmax(out_degrees[self.dataset.labeled == 0])
            c_id = np.arange(n_pool)[self.dataset.labeled == 0][max_out_degree]
            assert ((out_degrees[c_id] == np.max(out_degrees)) & (self.dataset.labeled[c_id] == 0))

            # get all the points covered by c_id
            covered_vertices = I[lims[c_id]:lims[c_id + 1]]
            I_covered_idx= np.sort(np.where(np.isin(I, covered_vertices))[0])

            # Remove all incoming edges to the points covered by c_id
            n_covered = np.zeros(len(lims))
            for l in range(1, len(lims)):
                n_covered[l] = np.sum(((lims[l - 1] <= I_covered_idx) & (I_covered_idx < lims[l])))
            lims = lims - np.cumsum(n_covered)
            lims= lims.astype(int)

            I = np.delete(I, I_covered_idx)
            self.dataset.observe(c_id, self.radius)





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


# Old ProbCover function

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
            if len(self.dataset.queries)>K:
                # set radius of all unlabeled points using average radius ok K-nn neighbours and update the covered points accordingly
                for i in np.where(self.dataset.labeled == 0)[0]:
                    # get K nearest neighbours of i
                    nbrs = NearestNeighbors(n_neighbors=K).fit(self.dataset.x[self.dataset.queries])
                    distances, indices = nbrs.kneighbors(self.dataset.x[i].reshape(1, -1))
                    # set its radius as a weighted radius of its K-nn
                    weights= np.squeeze((1/distances)/((1/distances).sum()))
                    self.dataset.radiuses[i]= (weights*self.dataset.radiuses[indices[0]]).sum()
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
                # Change the points that are considered as covered in the graph
            self.dataset.observe(c_id, rc)
            print(self.dataset.radiuses)
            graph = update_adjacency_graph_labeled(self.dataset)


# old helper functions

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

