import numpy as np


class ForwardEuler_v0:

    def __init__(self, f):
        self.f = f

    def set_initial_condition(self, u0):
        self.u0 = u0

    def solve(self, t_span, N):

        t0, tN = t_span
        self.dt = tN/N
        self.t = np.zeros(N+1)
        self.u = np.zeros(N+1)

        # Assert statement: if false (no attribute u0) an assertion error is triggered
        msg = "Please set initial conditions before calling solve"
        assert hasattr(self, "u0"), msg

        self.t[0] = t0
        self.u[0] = self.u0

        #Euler forward method
        for i in range(N):
            self.i = i
            self.t[i+1] = self.t[i] + self.dt
            self.u[i+1] = self.u[i] + self.dt*self.f(self.t[i],self.u[i])
        return self.t, self.u


class ForwardEuler:

    def __init__(self, f):
        self.f = lambda t, u: np.array(f(t,u),float)

    def set_iniital_condition(self,u0):
        if np.isscalar(u0):
            self.neq = 1
            u0 = float(u0)
        else:
            self.neq = np.size(u0)
            u0 = np.asarray(u0)
        self.u0 = u0

    def solve(self, t_span, N):
        """Compute solution for t_span[0] <= t <= t_span[1]

        Args:
            t_span (list or nparray): time axis
            N (integer): number of steps
        """        
        t0, t1 = t_span
        self.dt = (t1-t0)/N
        self.t = np.zeros(N+1)

        if self.neq == 1:
            self.u = np.zeros(N+1)
        else:
            self.u = np.zeros((N+1,self.neq)) #self.neq dimensional numpy array
 
        msg = "Please set initial condition before calling solve"
        assert hasattr(self, "u0"), msg

        self.t[0] = t0
        self.u[0] = self.u0

        for n in range(N):
            self.n = n
            self.t[n + 1] = self.t[n] + self.dt
            self.u[n + 1] = self.advance()
        return self.t, self.u

    def advance(self):
        u, dt, t, f, n = self.u, self.dt, self.t, self.f, self.n 
        return u[n] + dt*f(t[n],u[n])

