import numpy as np
from utils import sigmoid

class LogisticRegression:
    def __init__(self,l2_parm=0):
        self.l2_parm = l2_parm
    
    def fit(self,X,y,learning_rate=0.001,epochs=100):
        m, n = X.shape[0], X.shape[1]
        # add ones for bias term
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        if self.n_classes == 2:
            # initialize weights
            self.theta = np.random.rand(n+1,1)
            # initialize gradient vector
            gradients = np.zeros((n+1,1))
            for _ in range(epochs):
                gradients = 1/m * (-X.transpose() @ (y-sigmoid(X @ self.theta)) + self.l2_parm * np.sum(self.theta[1:,0]))
                self.theta = self.theta - learning_rate * gradients
        else:
            self.theta = np.random.rand(n+1,self.n_classes)
            for k in range(self.n_classes):
                gradients = np.zeros(n+1)
                for _ in range(epochs):
                    gradients = -1/m * (X.transpose() @ (sigmoid(X @ self.theta[:,k]) - y) + + self.l2_parm * np.sum(self.theta[1:,k]))
                    self.theta[:,k] = self.theta[:,k] - learning_rate * gradients

    def predict(self,X):
        m = X.shape[0]
        # add ones for bias term
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        y_pred = sigmoid(X @ self.theta)
        return y_pred