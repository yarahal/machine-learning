import numpy as np
from utils import euclidean_distance

class KMeansClustering:
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self,X,itr=100):
        m = X.shape[0]
        self.cluster_centroids = X[np.random.choice(np.arange(0,m,1),size=self.n_clusters,replace=False),:]
        c = np.zeros(m)
        for _ in range(itr):
            for i in range(m):
                min_distance = euclidean_distance(self.cluster_centroids[0,:],X[i,:]) 
                for j in range(self.n_clusters):
                    curr_distance = euclidean_distance(self.cluster_centroids[j,:],X[i,:])
                    if curr_distance < min_distance:
                        c[i] = j
                        min_distance = curr_distance
            for j in range(self.n_clusters):
                self.cluster_centroids[j,:] = np.mean(X[(c==j),:],axis=0)
            yield c, self.cluster_centroids

    def predict(self,X):
        m = X.shape[0]
        c = np.zeros(m)
        for i in range(m):
            min_distance = euclidean_distance(self.cluster_centroids[0,:],X[i,:]) 
            for j in range(self.n_clusters):
                curr_distance = euclidean_distance(self.cluster_centroids[j,:],X[i,:])
                if curr_distance < min_distance:
                    c[i] = j
                    min_distance = curr_distance
        return c