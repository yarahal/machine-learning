import numpy as np
from decision_trees import ClassificationDecisionTree, RegressionDecisionTree

class RandomForestClassification:
    def __init__(self,n_trees=100,max_depth=None,min_samples_split=1,max_features=None,p_samples=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.p_samples = p_samples
        self.trees = []

    def fit(self,X,y):
        m, n = X.shape[0], X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n))
        for b in range(self.n_trees):
            cols = sorted(np.random.choice(np.arange(0,n,1),replace=False,size=self.max_features))
            samples = sorted(np.random.choice(np.arange(0,m,1),replace=False,size=int(self.p_samples*m)))
            X_b = X[:,cols]
            X_b = X_b[samples,:]
            tree = ClassificationDecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            tree.fit(X_b,y)
            self.trees.append(tree)
    
    def predict(self,X):
        m = X.shape[0]
        y_pred =  np.zeros((m,self.n_trees))
        for b in range(self.n_trees):
            y_pred[:,b] = self.trees[b].predict(X)
        for i in range(m):
            counts = {}
            for b in range(self.n_trees):
                counts[y_pred[i,b]] = counts[y_pred[i,b]] + 1 if y_pred[i,b] in counts else 0
            y_pred[i,0] = max(counts,key=lambda label: counts[label])
        return y_pred[:,0]     

class RandomForestRegression:
    def __init__(self,n_trees=100,max_depth=None,min_samples_split=1,n_features=None,p_samples=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = n_features
        self.p_samples = p_samples
        self.trees = []

    def fit(self,X,y):
        m, n = X.shape[0], X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n))
        for b in range(self.n_trees):
            cols = sorted(np.random.choice(np.arange(0,n,1),replace=False,size=self.max_features))
            samples = sorted(np.random.choice(np.arange(0,m,1),replace=False,size=int(self.p_samples*m)))
            X_b = X[:,cols]
            X_b = X_b[samples,:]
            tree = RegressionDecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            tree.fit(X_b,y)
            self.trees.append(tree)
    
    def predict(self,X):
        m = X.shape[0]
        y_pred =  np.zeros((m,self.n_trees))
        for b in range(self.n_trees):
            y_pred[:,b] = self.trees[b].predict(X)
        y_pred = np.mean(y_pred,axis=1)
        return y_pred  