from helper import typicality, estimate_emperical_purity, get_emperical_radius, adjacency_graph, get_knn_labels
from IPython import embed
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt



class ActiveLearner:
    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model
        self.name= None

    def query(self, M, B, n_initial=1):
        assert((M>=B)&(M%B==0))

    def demo_2dplot(self, M, B, all_plots=False, final_plot=False, n_initial=1):
        queries= self.query(M,B)

        if all_plots:
            for i in range(int(M/B)):
                fig, ax = plt.subplots()
                ax.axis('equal')
                sns.scatterplot(data=self.dataset, x="x1", y="x2", hue="y", palette="Set2")
                sns.scatterplot(data=self.dataset.iloc[queries[0: i*B]], x="x1", y="x2", color="black", marker="P", s=150)
                sns.scatterplot(data=self.dataset.iloc[queries[i*B:(i+1)*B]], x="x1", y="x2", color="red", marker="P", s=150)
                if self.name=="ProbCover Sampler":
                    for u in queries[i*B:(i+1)*B]:
                        ax.add_patch(plt.Circle((self.dataset.iloc[u, 0], self.dataset.iloc[u, 1]),
                                                self.purity_radius, color='red', fill=False, alpha=0.5))
                plt.title(f"{self.name} with {B*(i+1)} sampled points and batch size {B}")
                plt.show()
        if final_plot:
            fig, ax = plt.subplots()
            ax.axis('equal')
            sns.scatterplot(data=self.dataset, x="x1", y="x2", hue="y", palette="Set2")
            sns.scatterplot(data=self.dataset.iloc[queries], x="x1", y="x2", color="red", marker="P", s=150)
            if self.name == "ProbCover Sampler":
                for u in queries:
                    ax.add_patch(plt.Circle((self.dataset.iloc[u, 0], self.dataset.iloc[u, 1]),
                                            self.purity_radius, color='red', fill=False, alpha=0.5))
            plt.title(f"{self.name} with {M} sampled points and batch size {B}")
            plt.show()


class RandomSampler(ActiveLearner):
    def __init__(self, dataset, model=None):
        super().__init__(dataset, model)
        self.name="Random Sampler"

    def query(self, M, B, n_initial=1):
        super().query(M,B,n_initial)
        labeled = np.zeros(len(self.dataset), dtype=int)
        queries = np.zeros(M, dtype=int)

        for i in range(int(M/B)):
            idx= np.random.choice(np.where(labeled==0)[0], B, replace=False)
            labeled[idx]=1
            queries[i*B:(i+1)*B]=idx
        return queries


class TypiclustSampler(ActiveLearner):
    def __init__(self, dataset, n_neighbours=5, model=None):
        super().__init__(dataset, model)
        self.n_neighbours= n_neighbours
        self.name="Typiclust Sampler"


    def query(self, M, B, plot=[False, False],  n_initial=1):
        super().query(M,B,n_initial)
        assert(sum(plot)<=1) # You can only choose one type of plot output
        labeled=np.zeros(len(self.dataset), dtype=int)
        queries=np.zeros(M, dtype=int)
        show_clusters= plot[0]
        show_all_clusters=plot[1]
        count=0

        while count<M:
            n_labeled=np.sum(labeled)

            # Cluster the data using both labeled and unlabeled samples
            kmeans = KMeans(n_clusters=B+n_labeled, random_state=0).fit(self.dataset.iloc[:,:-1])
            cluster_id= kmeans.predict(self.dataset.iloc[:,:-1])
            self.dataset["cluster_id"]= cluster_id

            # Extract the B cluster_ids for the B largest uncovered clusters (ie containing only unlabeled data)
            covered= np.unique(self.dataset[labeled==1]["cluster_id"])
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
                sns.scatterplot(data=self.dataset[labeled == 1], x="x1", y="x2", color="black", marker="P", s=100,
                                label="Points sampled in the previous iteration", linewidths=3)
            for b in clusters:
                data_cluster= self.dataset[self.dataset["cluster_id"]==b]
                if (self.n_neighbours<=len(data_cluster)) & (len(data_cluster)>=1):
                    t=typicality(data_cluster.iloc[:,:-1], self.n_neighbours)
                elif len(data_cluster)>=1:
                    t = typicality(data_cluster.iloc[:,:-1], len(data_cluster))
                idx_c= np.argmax(t) #id in data_cluster
                idx= data_cluster.iloc[idx_c].name
                labeled[idx]=1
                queries[count]=idx
                count+=1
            if show_all_clusters or show_clusters:
                sns.scatterplot(data=self.dataset.iloc[queries[count-B:count]], x="x1", y="x2", s=100, color="red", marker="P")
                ax.get_legend().remove()
                plt.show()

        self.dataset= self.dataset.drop(columns=["cluster_id"])
        return queries


class ProbCoverSampler(ActiveLearner):
    def __init__(self, dataset, purity_radius, purity_threshold, k, plot=[False, False], search_range=None, search_step=None,
                 model=None):
        super().__init__(dataset, model)
        self.name = "ProbCover Sampler"

        plot_kmeans= plot[0]
        plot_unpure_balls=plot[1]

        self.pseudo_labels= get_knn_labels(self.dataset, k, plot_kmeans=plot[0])

        #Get the purity for given radius
        #TODO: assert that either threshold or radius is set to None, and that search_range, search_step are initialied
        if purity_radius is not None:
            self.purity_radius=purity_radius
            purity_estimate=estimate_emperical_purity(self.purity_radius, self.dataset, self.pseudo_labels, plot_unpure_balls)
            print(f"With radius {self.purity_radius}, purity({self.purity_radius})={purity_estimate}")

        if purity_threshold is not None:
            #Get radius for given purity threshold
            self.purity_radius= get_emperical_radius(0.95, self.dataset, self.pseudo_labels, search_range, search_step, plot_unpure_balls)
        print(f"ProbCover Sampler initialized with radius {self.purity_radius}")
        #Initialize the graph
        self.graph=adjacency_graph(self.dataset, self.purity_radius)

    def query(self, M, B=1, n_initial=1):
        graph=self.graph.copy()
        #Initialize the labels
        n_pool= len(self.dataset)
        queries=np.zeros(M, dtype=int)
        labeled = np.zeros(n_pool, dtype=int)

        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if np.sum(labeled)>=1:
            covered_edges=graph[(labeled==1), :]
            covered_vertices= np.where(np.max(covered_edges, axis=0)==1)[0]
            graph[:,covered_vertices]=0

        for m in range(M):
            #get the unlabeled point with highest out-degree
            out_degrees=np.sum(graph, axis=1)
            max_out_degree=np.argmax(out_degrees[labeled==0])
            c_id=np.arange(n_pool)[labeled==0][max_out_degree]
            assert((out_degrees[c_id]==np.max(out_degrees))&(labeled[c_id]==0))

            covered_vertices=np.where(graph[c_id, :]==1)[0]
            graph[:,covered_vertices]=0
            queries[m]=c_id
            labeled[c_id]=1

        return queries










