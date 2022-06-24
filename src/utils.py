import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sign(x):
    x[x < 0] = -1
    x[x >= 0] = 1
    return x

def euclidean_distance(p1,p2):
    return np.sum((p1-p2)**2)