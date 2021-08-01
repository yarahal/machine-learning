import numpy as np

class Linear:
    def apply_activation(self,x):
        return x
    
    def derivative(self,x):
        return np.ones(x.shape)

class ReLU:
    def apply_activation(self,x):
        return np.maximum(x,0)
    
    def derivative(self,x):
        d = np.zeros(x.shape)
        d[x >= 0] = 1
        return d

class Sigmoid:
    def apply_activation(self,x):
        return 1/(1+np.exp(-x))
    
    def derivative(self,x):
        return (np.exp(-x))/((1+np.exp(-x))**2)