import numpy as np


class Pendulum:

    def __init__(self, l, g=9.81):

        self.l = l
        self.g = g

    def __call__(self, t, u):

        theta, omega = u

        dtheta = omega
        domega = -(self.g / self.l) * np.sin(theta)
        return [dtheta, domega]
