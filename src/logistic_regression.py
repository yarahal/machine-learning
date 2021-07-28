import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self,lamda=0):
        self.lamda = lamda

    def __cost_gradient(self,X,j):
        m = X.shape[0]
        return np.sum((sigmoid(X @ self.theta)-y.reshape(m,1))*np.expand_dims(X[:,j],-1),axis=0)/m + self.lamda/m * self.theta[j]
    
    def fit(self,X,y,alpha=0.001,epochs=100):
        m, n = X.shape[0], X.shape[1]
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        if self.n_classes == 2:
            self.theta = np.random.rand(n+1,1)
            gradients = np.zeros((n+1,1))
            for epoch in range(epochs):
                for j in range(n+1):
                    gradients[j] = self.__cost_gradient(X,j)
                self.theta = self.theta - alpha * gradients
        else:
            self.theta = np.random.rand(n+1,n_classes)
            for k in range(n_classes):
                gradients = np.zeros(n+1)
                for epoch in range(epochs):
                    for j in range(n+1):
                        gradients[j] = self.__cost_gradient(X,j)
                    self.theta[:,k] = self.theta[:,k] - alpha * gradients

    def predict(self,X):
        m = X.shape[0]
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        n_classes = len(self.classes)
        if n_classes == 2:
            return np.array(sigmoid(X @ self.theta) >= 0.5, dtype=int).flatten()
        return np.argmax(sigmoid(X @ self.theta),axis=1).flatten()