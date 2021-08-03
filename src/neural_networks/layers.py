import numpy as np
from activations import Linear

class FullyConnectedLayer:
    def __init__(self,n,activation=Linear()):
        self.n = n
        self.activation = activation
        self.is_initialized = False
    
    def _initialize(self,shape):
        self.W = np.random.randn(self.n,shape[0])*0.01
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