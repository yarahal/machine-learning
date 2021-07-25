import numpy as np

class PCA:
    def __init__(self,k):
        self.k = k

    def fit(self,X):
        m, n = X.shape[0], X.shape[1]
        for j in range(n):
            X[:,j] = (X[:,j] - np.mean(X[:,j]))/(np.std(X[:,j]))
        cov_mat = np.cov(X.transpose())
        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        self.principal_components = np.array([v for _, v in sorted(zip(eig_values, eig_vectors), key=lambda eigen_pair: eigen_pair[0])][:self.k])


    def transform(self,X):
        y = np.zeros((X.shape[0],self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                y[i,j] = np.dot(X[i,:],self.principal_components[j])
        return y