import numpy as np
from neural_networks.activations import Linear

from math import sqrt

class FC:
    def __init__(self,n,activation=Linear()):
        self.n = n
        self.activation = activation
        self.is_initialized = False
    
    def _initialize(self,shape):
        self.W = np.random.randn(self.n,shape[0])*(1/sqrt(shape[0]))
        self.b = np.zeros((self.n,1))
        self.is_initialized = True

    def _layer_forward_propagation(self,A_lm1):
        if not self.is_initialized:
            self._initialize(A_lm1.shape)
        self.A_lm1 = A_lm1
        self.Z = self.W @ A_lm1 + self.b
        A_l = self.activation.apply_activation(self.Z)
        return A_l
    
    def _layer_backward_propagation(self,dA_l):
        m = dA_l.shape[1]
        dZ = self.activation.derivative(self.Z) * dA_l
        self.dW =  (dZ @ (self.A_lm1).transpose())/m
        self.db = np.sum(dZ,axis=1).reshape(-1,1)/m
        dA_lm1 = self.W.transpose() @ dZ
        return dA_lm1

class Conv2D:
    def __init__(self,n_filters,f,activation=Linear()):
        self.n_filters = n_filters
        self.f = f
        self.activation = activation
        self.is_initialized = False
    
    def _initialize(self,shape):
        self.W = np.random.randn(self.f,self.f,shape[2],self.n_filters)
        self.b = np.zeros((1,1,1,self.n_filters))
        self.is_initialized = True
    
    def _convolve_2d(self,x,y):
        f = y.shape[0]
        padding = (f-1)//2 
        z = np.zeros(x.shape[0],x.shape[1])
        x = np.concatenate([np.zeros(padding,1),x,np.zeros(padding,1)],axis=1)  
        x = np.concatenate([np.zeros(1,padding),x,np.zeros(1,padding)],axis=0)  
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                z[i,j] = np.sum(x[i:i+f,j,j+f,:]*y)
        return z

    def _layer_forward_propagation(self,A_lm1):
        if not self.is_initialized:
            self._initialize(A_lm1.shape)
        self.A_lm1 = A_lm1
        self.Z = np.zeros(A_lm1.shape[0],A_lm1.shape[1],A_lm1.shape[2],self.n_filters)
        for i in range(A_lm1.shape[0]):
            for k in range(self.n_filters):
                self.Z[i,:,:,k] = self._convolve_2d(A_lm1[i,:,:,:],self.W[:,:,:,k]) 
        self.Z += self.b
        A_l = self.activation.apply_activation(self.Z)
        return A_l
    
    def _layer_backward_propagation(self,dA_l):
        m = dA_l.shape[1]
        dZ = self.activation.derivative(self.Z) * dA_l
        self.dW =  (dZ @ (self.A_lm1).transpose())/m
        self.db = np.sum(dZ,axis=1).reshape(1,1,1,-1)/m
        dA_lm1 = self.W.transpose() @ dZ
        return dA_lm1


