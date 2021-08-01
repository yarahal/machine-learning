import numpy as np 

class BinaryCrossEntropy:
    def compute_cost(self,y,y_hat):
        m = y.shape[1]
        return np.sum(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))/m
    
    def compute_cost_gradient(self,y,y_hat):
        m = y.shape[1]
        return (-y/y_hat + (1-y)/(1-y_hat))/m