
import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from clustering import ClusteringAlgo
from utils.faiss_graphs import adjacency_graph_faiss, remove_incoming_edges_faiss, update_adjacency_radius_faiss, adjacency_graph_faiss_fast
from utils.reduction_methods import shift_covering_centroids, reduce_all_interesected_balls_faiss, reduce_intersected_balls_faiss, get_knn_weighted_radiuses, get_cover
from utils.hyperparameters import get_radius_faiss, get_init_radiuses, floor_twodecimals
from sklearn.cluster import KMeans
from metrics import kl_divergence
import os

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
                 algorithm: str,
                 sd: int, 
                 plot_clustering=False,
                 radius= None,
                 hard_thresholding= False,
                 model=None):

        super().__init__(dataset, model)
        self.name = "ProbCover Sampler"
        self.clustering = clustering
        self.algorithm= algorithm
        self.purity_threshold = purity_threshold
        self.sd= sd

        # Get pseudo labels
        self.pseudo_labels = self.clustering.pseudo_labels
        if plot_clustering:
            self.clustering.plot()
            
        # Initialize the graph
        
        if self.dataset.C==5: #IF THE DATASET IS CHEXPERT
            self.save_path = f"/cluster/work/grlab/projects/projects2022_doctor-in-the-loop/chexpert_graphs/{self.purity_threshold}_{self.sd}/"
            if os.path.exists(self.save_path): 
                self.initialize_radiuses(radius, hard_thresholding)
                self.load()
                print(f"Active Learner Initialized with radius {self.radius} and hardthreshold {self.hard_threshold}")
            else:
                # Initialize self.radius and self.hard_threshold
                os.makedirs(self.save_path)
                self.initialize_radiuses(radius, hard_thresholding)
                self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
                self.save()
                
        else:
            # Initialize self.radius and self.hard_threshold
            self.initialize_radiuses(radius, hard_thresholding)
            self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
            
        self.lims, self.D, self.I = self.lims_ref.copy(), self.D_ref.copy(), self.I_ref.copy()
        self.dataset.radiuses = np.repeat(self.radius, len(self.dataset.x))

    def initialize_radiuses(self, radius, hard_thresholding):
        if radius is not None:
            self.radius = radius
        else:
            self.radius, _ = get_init_radiuses(self.purity_threshold, self.dataset, self.pseudo_labels)

        self.hard_threshold = floor_twodecimals(self.radius/np.sqrt(2)) if hard_thresholding else 0
        print(f"Active Learner Initialized with radius {self.radius} and hardthreshold {self.hard_threshold}")


    def update_radius(self, new_radius):
        self.radius = new_radius
        self.dataset.radiuses = np.repeat(self.radius, len(self.dataset.x))
        self.lims_ref, self.D_ref, self.I_ref = adjacency_graph_faiss(self.dataset.x, self.radius)
        self.lims_ref, self.D_ref, self.I_ref  = remove_incoming_edges_faiss(self.dataset, self.lims_ref, self.D_ref, self.I_ref)
        self.lims, self.D, self.I = self.lims_ref.copy(), self.D_ref.copy(), self.I_ref.copy()

    def save(self):
        pd.DataFrame({f'radius': self.radius, 
                      f'hard_threshold': self.hard_threshold}, index=[0]).to_csv(f'{self.save_path}initial.csv')
        np.save(f"{self.save_path}lims.npy", self.lims_ref)
        np.save(f"{self.save_path}D.npy", self.D_ref)
        np.save(f"{self.save_path}I.npy", self.I_ref)

    def load(self):
        self.lims_ref = np.load(f"{self.save_path}lims.npy")
        self.D_ref= np.load(f"{self.save_path}D.npy")
        self.I_ref= np.load(f"{self.save_path}I.npy")

        
    # def update_labeled(self, plot_clustering=False):
    #     self.clustering.fit_labeled(self.dataset.queries)
    #     self.pseudo_labels = self.clustering.pseudo_labels
    #     new_radius = get_radius_faiss(self.purity_threshold, self.dataset.x, self.pseudo_labels, [0, 10], 0.01)
    #     self.update_radius(new_radius)

    def get_highest_out_degree(self):
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
        return c_id, n_options, max_out_degree
    
    def query(self, M, args, reinitialize=False):
        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if len(self.dataset.queries)==0 or reinitialize:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)
        for _ in range(M):           
            # get the unlabeled point with highest out-degree
            c_id, n_options, max_out_degree = self.get_highest_out_degree()
            # Remove all incoming edges to the points covered by c_id
            self.dataset.observe(c_id)
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I, c_id)
        return max_out_degree, n_options

    def adpc_query(self, M, args, reinitialize=False):
        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if len(self.dataset.queries)==0 or reinitialize:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)

        for _ in range(M):
            # Update initial radiuses using some weighted average of the radiuses of the knn 
            if len(self.dataset.queries) > 0:
                new_radiuses= get_knn_weighted_radiuses(self.dataset, args.K, args.gauss, self.radius)
                if self.hard_threshold>0:
                    new_radiuses[new_radiuses<= self.hard_threshold]= self.hard_threshold

                self.lims, self.D, self.I = update_adjacency_radius_faiss(self.dataset, new_radiuses, self.lims_ref,
                                                                          self.D_ref, self.I_ref, self.lims, self.D,
                                                                          self.I)
            
            # get the unlabeled point with highest out-degree
            c_id, n_options, max_out_degree = self.get_highest_out_degree()

            # Add point, adapting its radius and the radius of all points with conflicting covered regions
            self.lims, self.D, self.I = reduce_intersected_balls_faiss(self.dataset, c_id, self.lims_ref, self.D_ref,
                                                                       self.I_ref, self.lims, self.D, self.I,
                                                                       args.gamma, args.reduction_method)
            

            if self.hard_threshold>0:
                self.dataset.radiuses[self.dataset.radiuses<= self.hard_threshold]= self.hard_threshold
            self.dataset.observe(c_id, self.dataset.radiuses[c_id])

        return max_out_degree, n_options
    
    def coverpc_query(self, M, args, reinitialize=False):
        if len(self.dataset.queries)==0 or reinitialize:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)

        for _ in range(M):
            cover= get_cover("coverpc", self.dataset, self.lims_ref, self.D_ref, self.I_ref)
            if cover < args.cover_threshold:
                # sample without reducing radiuses
                max_out_degree, n_options = self.query(1, args)
            elif cover >= args.cover_threshold:
                # reduce the radius until cover condition is no longer satisfied
                print(f"Reducing the radius until the cover >= {args.cover_threshold} is no longer satisfied")
                while cover >= args.cover_threshold:
                    self.update_radius(self.radius*args.eps)
                    cover= get_cover("coverpc", self.dataset, self.lims_ref, self.D_ref, self.I_ref)
                print(f"New radius is {self.radius} and cover is {cover*100}")

                # query a point with this new radius
                max_out_degree, n_options= self.query(1, args)

        return max_out_degree, n_options
    
    def partialadpc_query(self, M, args, current_threshold, reinitialize= False):
        thresholds= np.arange(0.1, 0.85, 0.05)
        if len(self.dataset.queries)==0:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)
        
        for _ in range(M):
            cover= get_cover("partialadpc", self.dataset, self.lims_ref, self.D_ref, self.I_ref)     
            
            if cover < current_threshold:
                # sample without reducing radiuses
                max_out_degree, n_options = self.query(1, args)
            elif cover >= current_threshold:
                print("reducing the radiuses for intersections")
                # reduce the intersection
                self.lims, self.D, self.I = reduce_all_interesected_balls_faiss(self.dataset, self.lims_ref, self.D_ref, self.I_ref, 
                                                                                self.lims, self.D, self.I)
                # get the query using knn and intersection updated radiuses
                if len(self.dataset.queries) > 0:
                    new_radiuses= get_knn_weighted_radiuses(self.dataset, args.K, args.gauss, args.radius)
                    self.lims, self.D, self.I = update_adjacency_radius_faiss(self.dataset, new_radiuses, 
                                                                            self.lims_ref, self.D_ref, self.I_ref, 
                                                                            self.lims, self.D, self.I)
                c_id, n_options, max_out_degree = self.get_highest_out_degree()

                # observe but don't reduce for intersections
                self.dataset.observe(c_id, self.dataset.radiuses[c_id])
                self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I, c_id)
                                                                    
                # update the next threshold that will do the update
                i = np.where(thresholds==current_threshold)[0]
                current_threshold= thresholds[i+1]
        return max_out_degree, n_options, current_threshold 
        

class ShiftedProbCoverSampler(ProbCoverSampler_Faiss):
    def __init__(self, dataset, purity_threshold,
                 clustering: ClusteringAlgo,
                 plot=[False, False],
                 search_range=[0, 1.0], search_step=0.01,
                 radius= None,
                 model=None):

        super().__init__(dataset, purity_threshold, clustering, plot,
                         search_range, search_step,
                         radius, model)
        
        #initialize the centroids associated to each query as the query itself
        self.centroids= self.dataset.x
        #self.lims, self.D, self.I represent the graph associated with the centroids
        #self.lims, self.D, self.I represent the graph associated with the original points
        self.class_centroids= np.empty(shape=(self.dataset.n_classes, self.dataset.d))

    def update_class_centroids(self, new_query_id):
        c = self.dataset.y[new_query_id]
        n_observed_c= ((self.dataset.labeled==1)*(self.dataset.y==c)).sum()
        self.class_centroids[c]= (n_observed_c*self.class_centroids[c]+ self.dataset.x[new_query_id])/(n_observed_c+1)
        return self.class_centroids
    
    def shifted_query(self, M, deg, K=5, reinitialize=False, hard_threshold= 0.0):
        # Remove the incoming edges to covered vertices (vertices such that there exists labeled with graph[labeled,v]=1)
        if len(self.dataset.queries)==0 or reinitialize:
            self.lims, self.D, self.I = remove_incoming_edges_faiss(self.dataset, self.lims, self.D, self.I)

        for _ in range(M):
            # get the unlabeled point with highest out-degree
            c_id, n_options, max_out_degree = self.get_highest_out_degree()

            # Add point, shift points centroids that conflict with this points in the direction of class centroids (if they exist)
            self.lims, self.D, self.I = shift_covering_centroids(self.dataset, c_id, self.lims_ref, self.D_ref,
                                                                       self.I_ref, self.lims, self.D, self.I)
            
            # update class centroids
            self.class_centroids = self.update_class_centroids(c_id)

            # observe the newest points
            self.dataset.observe(c_id, self.dataset.radiuses[c_id])

        return max_out_degree, n_options



class BALDSampler(ActiveLearner):
    def __init__(self, dataset, model=None):

        super().__init__(dataset, model)
        self.name = "BALD Sampler"
        self.n_classes= self.dataset.n_classes
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
