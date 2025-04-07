import numpy as np

class kMeansClustering:
    def __init__(self, k:int):
        self.k = k
        self.cluster_centers = None
        self.clustering_cost = None

    def fit(self,features:np.ndarray):
        min_bound = np.min(features, axis=0)
        max_bound = np.max(features, axis=0)
        # self.cluster_centers = np.random.uniform(min_bound, max_bound, size=(self.k, features.shape[1]))
        random_indices = np.random.choice(features.shape[0], self.k, replace=False)
        self.cluster_centers = features[random_indices]
        cluster_map = np.ones(features.shape[0], dtype=int)
        previous_cluster_map = np.zeros(features.shape[0], dtype=int)

        converged = False
        while not converged:
            converged = True
            for index, x in enumerate(features):
                distance = np.sum((self.cluster_centers - x) ** 2, axis=1)
                new_cluster = np.argmin(distance)
                if previous_cluster_map[index] != new_cluster:
                    converged = False
                previous_cluster_map[index] = cluster_map[index]
                cluster_map[index] = new_cluster

            for index in range(len(self.cluster_centers)):
                mask = (cluster_map == index)
                cluster_features = features[mask]
                if cluster_features.shape[0] != 0:
                    self.cluster_centers[index] = np.mean(features[mask], axis=0) 
        
        indices = np.arange(0, len(features))
        assigned_clusters = self.cluster_centers[cluster_map[indices]]
        self.clustering_cost = np.sum((assigned_clusters - features) ** 2)

    def predict(self, data_features:np.ndarray) -> np.ndarray:
        assigned_clusters = np.empty(data_features.shape[0])

        for index, x in enumerate(data_features):
            distance = np.sum((self.cluster_centers - x) ** 2, axis=1)
            assigned_clusters[index] = np.argmin(distance)

        return assigned_clusters

    def getCost(self):
        return self.clustering_cost