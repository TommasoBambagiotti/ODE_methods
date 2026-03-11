import numpy as np
import matplotlib.pyplot as plt
from ode_solver_bv import FiniteDifference
from Test_models.Exercise_11_3 import Exercise

# *** BOUNDARY VALUE PROBLEM - Exercise 11.3 from Burden - TEST EXAMPLE ***
N = 9
u_boundary = [1, 2] # list
x_boundary = (1, 2) # tuple -> unordered

# interior mesh points
x = np.linspace(x_boundary[0], x_boundary[1], N)

#define model
model = Exercise()

#define ode
odeModel = FiniteDifference(model)

#set boundary conditions u=[u[0],u[1]]
odeModel.set_boundary_conditions(u_boundary[0], u_boundary[1])

#solve
xEx, uEx, A, b = odeModel.solve_sparse(x_boundary, N)

""" # Write everything into a file
with open('solution.txt', 'w') as f:
    f.write("# Matrix A\n")
    np.savetxt(f, A, fmt='%.4f')
    f.write("\n# Vector b\n")
    np.savetxt(f, b, fmt='%.4f')
    f.write("\n# x values\n")
    np.savetxt(f, xEx, fmt='%.4f')
    f.write("\n# u values\n")
    np.savetxt(f, uEx, fmt='%.10f')
    f.write("\n# |y(x_i) - u_i|\n")
    np.savetxt(f, np.abs(exactSol(xEx) - uEx), fmt='%.10f')

print("File solution.txt created!") """

plt.plot(xEx, uEx, color='#1f77b4', label=f"numerical")
plt.plot(xEx, model.exactsol(xEx), color="#f10707", label=f"exact")
plt.title("Finite difference method")
plt.legend()
plt.show()