class TreeNode:
    def __init__(self,y_pred=0,j=None,s=None):
        self.j = j
        self.s = s
        self.y = y_pred
        self.left = None
        self.right = None