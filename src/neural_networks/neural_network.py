import numpy as np
from optimizers import BatchGradientDescent

class NeuralNetwork:
    def __init__(self,loss=None,optimizer=BatchGradientDescent()):
        self.layers = []
        self.n = []
        self.loss = loss
        self.optimizer = optimizer
    
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def forward_propagation(self,X):
        L = len(self.layers)
        A_lm1 = X
        for l in range(L):
            A_lm1 = self.layers[l].layer_forward_propagation(A_lm1)
        return A_lm1
    
    def backward_propagation(self,y,y_hat):
        L = len(self.layers)
        dA_l = self.loss.compute_cost_gradient(y,y_hat)
        for l in range(L-1,-1,-1):
            dA_l = self.layers[l].layer_backward_propagation(dA_l)
    
    def fit(self,X,y,epochs=10,alpha=0.01,batch_size=None):
        for epoch in range(epochs):
            self.optimizer.run(X,y,alpha,self.layers,self.forward_propagation,self.backward_propagation) 

    def predict(self,X):
        return self.__forward_propagation(X).flatten()   
        
                

                

            
