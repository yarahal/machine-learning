import numpy as np
from tree import TreeNode

class ClassificationDecisionTree:
    def __init__(self,depth=5):
        self.depth = depth
        self.classes = []
        
    def __proportions(self,y):
        p = np.zeros(len(self.classes))
        for i in range(len(p)):
            p[i] = np.mean(y==self.classes[i])
        return p

    def __criterion(self,j,s,X,y):
        p1 = self.__proportions(y[X[:,j]<s])
        p2 = self.__proportions(y[X[:,j]>=s])
        return np.sum(p1*(1-p1)) + np.sum(p2*(1-p2))
        
    def __split(self,X_js,y_js,d,current_node):
        m, n = X_js.shape[0], X_js.shape[1]
        j_best, s_best = 0, np.random.choice(np.unique(X[:,0]))
        min_criterion = self.__criterion(j_best,s_best,X_js,y_js)
        for j in range(n):
            for s in np.unique(X[:,j]):
                criterion = self.__criterion(j,s,X_js,y_js)
                if  criterion < min_criterion:
                    j_best, s_best = j, s
                    min_criterion = criterion
        current_node.j = j_best
        current_node.s = s_best
        if(d < self.depth-1):
            if(len(y_js[X_js[:,j_best]<s_best])>0):
                p1 = self.__proportions(y_js[X_js[:,j_best]<s_best])
                current_node.left = TreeNode(np.argmax(p1))
                self.__split(X_js[X_js[:,j_best]<s_best],y_js[X_js[:,j_best]<s_best],d+1,current_node.left)
            if(len(y_js[X_js[:,j_best]>=s_best])>0):
                p2 = self.__proportions(y_js[X_js[:,j_best]>=s_best])
                current_node.right = TreeNode(np.argmax(p2))
                self.__split(X_js[X_js[:,j_best]>=s_best],y_js[X_js[:,j_best]>=s_best],d+1,current_node.right)

    def fit(self,X,y):
        self.classes = np.unique(y)
        p = self.__proportions(y)
        self.tree = TreeNode(np.argmax(p))
        self.__split(X,y,0,self.tree)
    
    def predict(self,X):
        m = X.shape[0]
        y_pred = []
        for i in range(m):
            current_node = self.tree
            y = current_node.y
            while(current_node is not None):
                y = current_node.y
                if(X[i,current_node.j]<current_node.s):
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            y_pred.append(y)
        return np.array(y_pred)