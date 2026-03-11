import numpy as np


class InfiniteWell:

    def __init__(self, L=np.pi):
        self.L = L

    def __call__(self, x):
        return 0*x

    def exact_sol(self, x, n=1):
        return np.sqrt(2/self.L)*np.sin(n*np.pi*x/self.L)
