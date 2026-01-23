import numpy as np


class FiniteDifference:
    """Finite difference method to solve the linear boundary-value problem

    u''=p(x)u'+q(x)u+r(x), a<=x<=b
    u(a)=u_a
    u(b)u_b

    The linear differential equation is reduced to a linear system

    Au=b

    where A is a tridiagonal matrix and b is the coefficients vector.
    The system is solved using np.linalg.solver.
    """

    # Get coefficients linear ODE
    def __init__(self, p, q, r):
        """Construct an instance of the model

        Args:
            p (_type_): p=p(x) coefficient function
            q (_type_): q=q(x) coefficient function
            r (_type_): r=r(x) coefficient function
        """        
        self.p = p
        self.q = q
        self.r = r

    def set_boundary_conditions(self, u_a, u_b):
        """Set the boundary condition

        Args:
            u_a (array): left boundary condition
            u_b (array): right boundary condition
        """        
        # boundary values at x=a and x=b
        self.u_a = u_a
        self.u_b = u_b

    def solve(self, x_span, N):
        """Solve the second order linera ODE using finite difference method

        Args:
            x_span (tuple): left and right values x_a <= x <= x_b
            N (integer): number of interior mesh points

        Returns:
            x (array): x axis
            u (array): approximate solution
        """

        # compute h=(xb-xa)/(N+1) with N+2 points
        x_a, x_b = x_span
        self.dx = (x_b-x_a)/(N+1)

        # create coordinate array and approximate solution array
        self.x = np.linspace(x_a, x_b, N+2)
        self.u = np.zeros(N+2)

        # set boundary values
        # .x[0] = x_a
        # self.x[N+1] = x_b
        self.u[0] = self.u_a
        self.u[N+1] = self.u_b

        # *** WORK IN PROGRESS ***

        # need only interior mesh points
        x_int = self.x[1:-1]  # remove first and last elements

        # define coefficients vector
        b = np.zeros(N)  # same dimension of x_int
        b[0] = -(self.dx**2)*self.r(x_int[0]) + \
                (1+(self.dx/2)*self.p(x_int[0]))*self.u_a

        b[N-1] = -(self.dx**2)*self.r(x_int[N-1]) + \
                  (1-(self.dx/2)*self.p(x_int[N-1]))*self.u_b

        b[1:-1] = -(self.dx**2)*self.r(x_int[1:-1])

        # tridiagonal matrix -> sparse matrix...?
        # first compute p(x_i) for i=1,...,N then remove p(x_N)
        sup_diag = -1 + (self.dx/2)*self.p(x_int)[:-1]
        # compute q(x_i) for i=1,...,N
        princ_diag = 2 + (self.dx**2)*self.q(x_int)
        # first compute p(x_i) for i=1,...,N then remove p(x_1)
        inf_diag = -1 - (self.dx/2)*self.p(x_int)[1:]

        A = np.diag(princ_diag, k=0) + \
            np.diag(sup_diag, k=1) + \
            np.diag(inf_diag, k=-1)   

        # Solve the matrix using numpy

        self.u[1:-1] = np.linalg.solve(A,b)

        return self.x, self.u
