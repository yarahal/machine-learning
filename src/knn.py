import numpy as np
from utils import euclidean_distance

class KNNClassification:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.points = list(zip(X,y))

    def predict(self,X):
        m = X.shape[0]
        y_pred = []
        for i in range(m):
            sorted_points = sorted(self.points,key=lambda point: euclidean_distance(point[0],X[i,:]))
            k_closest_points = sorted_points[:self.k]
            counts = {}
            for point in k_closest_points:
                counts[point[1]] = counts[point[1]] + 1 if point[1] in counts else 1
            y_pred.append(max(counts,key=lambda label: counts[label]))
        y_pred = np.array(y_pred).reshape(-1,1)
        return y_pred

class KNNRegression:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.points = list(zip(X,y))

    def predict(self,X):
        m = X.shape[0]
        y_pred = []
        for i in range(m):
            sorted_points = sorted(self.points,key=lambda point: euclidean_distance(point[0],X[i,:]))
            k_closest_points = sorted_points[:self.k]
            y_pred.append(np.mean(k_closest_points,axis=0)[1])
        y_pred = np.array(y_pred).reshape(-1,1)
        return y_pred