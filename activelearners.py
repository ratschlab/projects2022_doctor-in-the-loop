
import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from clustering import ClusteringAlgo
from helper import get_radius_faiss, adjacency_graph_faiss, remove_incoming_edges_faiss, update_adjacency_radius_faiss, \
    reduce_intersected_balls_faiss
from sklearn.cluster import KMeans
from metrics import kl_divergence
from IPython import embed

class ActiveLearner:
    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model
        self.name = None

    def query(self, M, B, n_initial=1):
        assert ((M % B == 0) & (B <= M))

    def demo_2dplot(self, M, B, all_plots=False, final_plot=False, n_initial=1):
        self.dataset.restart()
        self.query(M, B)

        if all_plots:
            for i in range(int(M / B)):
                fig, ax = plt.subplots()
                ax.axis('equal')
                sns.scatterplot(x=self.dataset.x[:, 0], y=self.dataset.x[:, 1], hue=self.dataset.y, palette="Set2")
                sns.scatterplot(x=self.dataset.x[self.dataset.queries[0: i * B], 0],
                                y=self.dataset.x[self.dataset.queries[0: i * B], 1], color="black", marker="P", s=150)
                sns.scatterplot(x=self.dataset.x[self.dataset.queries[i * B:(i + 1) * B], 0],
                                y=self.dataset.x[self.dataset.queries[i * B:(i + 1) * B], 1], color="red", marker="P",
                                s=150)
                if (self.name == "ProbCover Sampler"):
                    for u in self.dataset.queries[i * B:(i + 1) * B]:
                        ax.add_patch(plt.Circle((self.dataset.x[u, 0], self.dataset.x[u, 1]),
                                                self.radius, color='red', fill=False, alpha=0.5))
                plt.title(f"{self.name} with {B * (i + 1)} sampled points and batch size {B}")
                plt.show()
        if final_plot:
            fig, ax = plt.subplots()
            ax.axis('equal')
            sns.scatterplot(x=self.dataset.x[:, 0], y=self.dataset.x[:, 1], hue=self.dataset.y, palette="Set2")
            sns.scatterplot(x=self.dataset.x[self.dataset.queries, 0], y=self.dataset.x[self.dataset.queries, 1],
                            color="red", marker="P", s=150)
            if (self.name == "ProbCover Sampler"):
                for u in self.dataset.queries:
                    ax.add_patch(plt.Circle((self.dataset.x[u, 0], self.dataset.x[u, 1]),
                                            self.radius, color='red', fill=False, alpha=0.5))
            plt.title(f"{self.name} with {M} sampled points and batch size {B}")
            plt.show()


class RandomSampler(ActiveLearner):
    def __init__(self, dataset, model=None):
        super().__init__(dataset, model)
        self.name = "Random Sampler"

    def query(self, M, B=1, n_initial=1):
        super().query(M, B, n_initial)

        for i in range(int(M / B)):
            idx = np.random.choice(np.where(self.dataset.labeled == 0)[0], B, replace=False)
            self.dataset.observe(idx)

class ProbCoverSampler_Faiss(ActiveLearner):
    def __init__(self, dataset, purity_threshold,
                 clustering: ClusteringAlgo,
                 plot=[False, False],
                 search_range=[0, 20], search_step=0.01,
                 radius= None,
                 adaptive=False,
                 model=None):

        super().__init__(dataset, model)
        self.name = "ProbCover Sampler"
        self.clustering = clustering

        plot_clustering = plot[0]
        plot_unpure_balls = plot[1]

        # Get pseudo labels
        self.pseudo_labels = self.clustering.pseudo_labels
        if plot_clustering:
            self.clustering.plot()

        self.purity_threshold = purity_threshold

        # Get radius for given purity threshold
        if radius is None:
            self.radius = get_radius_faiss(self.purity_threshold, self.dataset.x, self.pseudo_labels, search_range,
                                           search_step,
                                           plot_unpure_balls)
        else:
            self.radius= radius

        # Initialize the graph
        self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
        self.lims, self.D, self.I = self.lims_ref.copy(), self.D_ref.copy(), self.I_ref.copy()
        self.adaptive = adaptive
        self.dataset.radiuses = np.repeat(self.radius, len(self.dataset.x))

    def update_radius(self, new_radius):
        self.radius = new_radius
        self.dataset.radiuses = np.repeat(self.radius, len(self.dataset.x))
        self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
        self.lims, self.D, self.I = self.lims_ref.copy(), self.D_ref.copy(), self.I_ref.copy()

    def update_labeled(self, plot_clustering=False):
        if self.adaptive == False:
            raise ValueError("This active learner's radius and pseudo-labels are not adaptive")
        self.clustering.fit_labeled(self.dataset.queries)
        self.pseudo_labels = self.clustering.pseudo_labels
        self.radius = get_radius_faiss(self.purity_threshold, self.dataset.x, self.pseudo_labels, [0, 10], 0.01)
        self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
        if plot_clustering:
            self.clustering.plot()

    def query(self, M, B=1, n_initial=1):
        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)
        for _ in range(M):
            # get the unlabeled point with highest out-degree
            out_degrees = self.lims[1:] - self.lims[:-1]
            if np.any(out_degrees > 0):
                max_out_degree= np.max(out_degrees)
                options = np.where(out_degrees*(self.dataset.labeled==0)==max_out_degree)[0]
                n_options= len(options)
                c_id= np.random.choice(options)
            else:
                c_id = np.random.choice(np.where(self.dataset.labeled == 0)[0])
                max_out_degree= 0
                n_options= len(np.where(self.dataset.labeled == 0)[0])
            # Remove all incoming edges to the points covered by c_id
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I, c_id)
            self.dataset.observe(c_id)
            return max_out_degree, n_options

    @profile
    def adaptive_query(self, M, deg, K=3, B=1, n_initial=1, reinitialize= False):
        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if len(self.dataset.queries)==0 or reinitialize:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)
        for m in range(M):
            # Update initial radiuses using k-means
            # TODO: could also be optimized using old distances? but would require some sort function? might not be faster
            if len(self.dataset.queries) > 0:
                index_knn = faiss.IndexFlatL2(self.dataset.d)  # build the index
                index_knn.add(self.dataset.x[self.dataset.queries].astype("float32"))  # fit it to the labeled data
                n_neighbours = K if len(self.dataset.queries) >= K else len(self.dataset.queries)
                D_neighbours, I_neighbours = index_knn.search(
                    self.dataset.x[self.dataset.labeled == 0].astype("float32"), n_neighbours)  # find K-nn for all
                # set new radiuses as a weighted radius of the labeled K-nn (for all non labeled points)
                # TODO: weigh this as inverse of distances but can cause issue, use yourself as a points, apply Gaussian kernel with (1:laplace, 2: gaussian, 8: flat )
                new_radiuses = self.dataset.radiuses.copy()

                gauss_distances = np.exp(-D_neighbours**deg / new_radiuses[self.dataset.labeled == 0].reshape(-1,1)**deg)
                use_self = True

                if use_self:
                    norm = (gauss_distances.sum(axis=1, keepdims=True) + 1)
                    alpha = (1 / norm)[:, 0]
                    weights = gauss_distances / norm
                    # new_radiuses[self.dataset.labeled == 0] = (weights * self.dataset.radiuses[
                    #     self.dataset.queries[I_neighbours]]).sum(axis=1) + alpha * self.dataset.radiuses[
                    #                                               self.dataset.labeled == 0]
                    new_radiuses[self.dataset.labeled == 0] = (weights * self.dataset.radiuses[
                        self.dataset.queries[I_neighbours]]).sum(axis=1) + alpha * self.radius
                else:
                    alpha = 1 / 2
                    weights = gauss_distances / (gauss_distances.sum(axis=1).reshape(-1, 1))

                    new_radiuses[self.dataset.labeled == 0] = alpha * (
                            weights * self.dataset.radiuses[self.dataset.queries[I_neighbours]]).sum(axis=1) + (
                                                                      1 - alpha) * self.dataset.radiuses[
                                                                  self.dataset.labeled == 0]

                self.lims, self.D, self.I = update_adjacency_radius_faiss(self.dataset, new_radiuses, self.lims_ref,
                                                                          self.D_ref, self.I_ref, self.lims, self.D,
                                                                          self.I)
            # get the unlabeled point with highest out-degree
            out_degrees = self.lims[1:] - self.lims[:-1]
            if np.any(out_degrees > 0):
                max_out_degree= np.max(out_degrees)
                options = np.where(out_degrees*(self.dataset.labeled==0)==max_out_degree)[0]
                n_options= len(options)
                c_id= np.random.choice(options)
            else:
                c_id = np.random.choice(np.where(self.dataset.labeled == 0)[0])
                max_out_degree = 0
                n_options = len(np.where(self.dataset.labeled == 0)[0])

            # Add point, adapting its radius and the radius of all points with conflicting covered regions

            self.lims, self.D, self.I = reduce_intersected_balls_faiss(self.dataset, c_id, self.lims_ref, self.D_ref,
                                                                       self.I_ref, self.lims, self.D, self.I)

        return max_out_degree, n_options


class BALDSampler(ActiveLearner):
    def __init__(self, dataset, n_classes, model=None):

        super().__init__(dataset, model)
        self.name = "BALD Sampler"
        self.n_classes= n_classes
    def query(self, M, K=5, B=1, n_initial=1):

        assert((M>=self.n_classes)or(len(self.dataset.queries)>0))
        m=M
        while m>0:
            if len(self.dataset.queries) == 0:
                # cluster the data into self.n_classes clusters and query points closest to each centroid
                # hope to discover all classes that way
                self.random_seed_init=1
                kmeans = KMeans(n_clusters=self.n_classes, random_state=self.random_seed_init).fit(self.dataset.x)
                index_kmeans= faiss.IndexFlatL2(self.dataset.d)
                index_kmeans.add(self.dataset.x.astype("float32"))
                _, I_centroids = index_kmeans.search(kmeans.cluster_centers_.astype("float32"), 1)
                self.dataset.observe(I_centroids.squeeze())
                m=m-self.n_classes

            elif len(self.dataset.queries)>0:
                # Compute predicted probabilities as distance to each of the classes: does not treat the case where not all classes were discovered yet in the selection of the initial pool
                if len(np.unique(self.dataset.y[self.dataset.queries]))==self.n_classes:
                    P= np.zeros(shape=(self.dataset.n_points,self.n_classes))
                    # set all discovered points predicted probabilities as deterministic
                    P[self.dataset.queries,self.dataset.y[self.dataset.queries]]= 1
                    # for all other points calculate the distance to each class
                    for c in range(self.n_classes):
                        index_to_c= faiss.IndexFlatL2(self.dataset.d)
                        idx_in_c= self.dataset.queries[self.dataset.y[self.dataset.queries]==c]
                        index_to_c.add(self.dataset.x[idx_in_c].astype("float32"))
                        D_to_c, _ = index_to_c.search(self.dataset.x[self.dataset.labeled==0].astype("float32"), 1)
                        P[self.dataset.labeled==0, c]= 1/np.sqrt(D_to_c.squeeze())
                    # Normalize to get a distribution
                    P= P/P.sum(axis=1).reshape(-1,1)

                    #Compute score as the mean Kl_divergence between itself and its K-nn
                    index_knn = faiss.IndexFlatL2(self.dataset.d)  # build the index
                    index_knn.add(self.dataset.x[self.dataset.queries].astype("float32"))  # fit it to the labeled data
                    n_neighbours = K if len(self.dataset.queries) >= K else len(self.dataset.queries)
                    _, I_neighbours = index_knn.search(self.dataset.x.astype("float32"), n_neighbours)  # find K-nn for all

                    scores=np.array([kl_divergence(P[self.dataset.queries[I_neighbours[p].squeeze()],:], P[p,:].reshape(1,-1)).mean() for p in np.where(self.dataset.labeled==0)[0]])
                    scores[self.dataset.queries]=0

                    if np.max(scores)>0:
                        c_id= np.argmax(scores)
                    else:
                        c_id= np.random.choice(np.where(self.dataset.labeled==0)[0], size=1)
                    self.dataset.observe(c_id)
                    m= m-1
