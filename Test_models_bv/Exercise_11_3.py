# Model for the exercise 11.3 from Burden "Numerical analysis"
import numpy as np


class Exercise:
    def __init__(self):
        pass
      
    def __call__(self, x):
        p = -2/x
        q = 2/x**2
        r = np.sin(np.log(x))/x**2

        return p, q, r
    
    def exactsol(self, x):
        c2 = (1/70)*(8 - 12*np.sin(np.log(2))-4*np.cos(np.log(2)))
        return (11/10-c2)*x + \
                c2/x**2 - \
                (3/10)*np.sin(np.log(x)) - \
                (1/10)*np.cos(np.log(x))