import numpy as np


class Exercise:
    def __init__(self):
        pass
      
    def __call__(self, x):
        p = -2/x
        q = 2/x**2
        r = np.sin(np.log(x))/x**2

        return p, q, r