import numpy as np

class LinearRegression:
    def __init__(self,lamda=0):
        self.lamda = lamda

    def _cost_gradient(self,X,y,j):
        m = X.shape[0]
        return np.sum(((X @ self.theta)-y.reshape(m,1))*np.expand_dims(X[:,j],-1),axis=0)/m + self.lamda/m * self.theta[j]
        
    def fit(self,X,y,alpha=0.001,epochs=100):
        m, n = X.shape[0], X.shape[1]
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        self.theta = np.random.rand(n+1,1)
        gradients = np.zeros((n+1,1))
        for _ in range(epochs):
            for j in range(n+1):
                gradients[j] = self._cost_gradient(X,y,j)
            self.theta = self.theta - alpha * gradients
            yield X @ self.theta, self.theta
            
    def predict(self,X):
        m = X.shape[0]
        X = np.concatenate([np.ones((m,1)),X],axis=1)
        return (X @ self.theta).flatten()