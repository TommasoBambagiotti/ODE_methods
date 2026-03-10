import numpy as np

class InfiniteWell:
    
    def __init__(self):
        pass
               
    def __call__(self, x):
        """ p = 0*x
        q = -(self.k**2)*np.ones(self.N)
        r = 0*x """
        return 0*x
