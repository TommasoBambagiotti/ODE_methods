import numpy as np
from scipy import linalg, integrate
import matplotlib.pyplot as plt
from ode_solver_bv import FiniteDifferenceHomo
from infinite_well import InfiniteWell

# *** INFINITE POTENTIAL WELL ***
# Dirichlet boundary conditions at x=0 and x=L=pi
L = np.pi
N = 38  
u_boundary = [0, 0] # list
x_boundary = (0, L) # tuple -> unordered

# interior mesh points
x = np.linspace(x_boundary[0], x_boundary[1], N)

# Create an infinite well within [0,L]
infiniteWell = InfiniteWell(L)

# define ode
ode_infinite_well = FiniteDifferenceHomo(infiniteWell)

#set boundary conditions u=[u[0],u[1]]
ode_infinite_well.set_boundary_conditions(u_boundary[0], u_boundary[1])

# solve
dx, xSol, w, H = ode_infinite_well.solve(x_boundary, N)

# exact solution 
# linalg.norm returns the matrix L^2 norm (=Frobenius norm for matrix)
u0exact = infiniteWell.exact_sol(xSol, n=1)  # discretisation breaks normalisation!
u0exactN = u0exact / integrate.simpson(u0exact, xSol)  # normalised in the right measure space

# Fundamental mode
u0 = np.concatenate([[u_boundary[0]], H[:,0],[u_boundary[1]]])
print(f"norm numerical={linalg.norm(u0)}")

# Right normalisation computed integrating in [0,L]
norm = linalg.norm(u0)/integrate.simpson(u0, xSol)
u0N = u0 * norm

w0 = np.sqrt(w[0])/dx
plt.plot(xSol, u0N, color='#1f77b4', label=fr"$\omega$={w0:.5}")
plt.plot(xSol, u0exactN, color="#f10707", label=f"exact")
plt.title("Fundamental mode")
plt.legend()
plt.show()