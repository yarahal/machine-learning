import numpy as np

class SGD:
    def __init__(self,batch_size=None):
        self.batch_size = batch_size
        
    def run(self,X,y,alpha,layers,forward_prop_function,backward_prop_function):
        m = X.shape[1]
        nb_batches = int(np.ceil(m/self.batch_size))
        for t in range(nb_batches):
            Xt, yt = X[:,t*self.batch_size:(t+1)*self.batch_size], y[t*self.batch_size:(t+1)*self.batch_size]
            y_hat = forward_prop_function(Xt)
            backward_prop_function(yt,y_hat)
            for layer in layers:
                layer.W = layer.W - alpha * layer.dW
                layer.b = layer.b - alpha * layer.db  

