import numpy as np


class ModelExample:
    """Example 1 from Burden "Numerical Analysis", chapter 11.3
    """
    def __init__(self):
        pass

    def __call__(self, x):
        self.p = -2/x
        self.q = 2/x**2
        self.r = np.sin(np.log(x))/x**2
        return self.p, self.q, self.r
