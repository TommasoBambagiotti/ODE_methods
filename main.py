import numpy as np
import matplotlib.pyplot as plt
from OdeSolverBV import FiniteDifference, FiniteDifferenceHomo
from infinite_well import InfiniteWell
# from Exercise_11_3 import Exercise
# from fin_diff_model import ModelExample
# from forward_euler_class import ForwardEuler_v0, ForwardEuler
# from logistic import Logistic
# from pendulum import Pendulum


# *** INFINITE POTENTIAL WELL ***
# Dirichlet boundary conditions at x=0 and x=L=pi
L = np.pi
N = [8, 18, 38, 78]
u_boundary = [0, 0] # list
x_boundary = (0, L) # tuple -> unordered

# interior mesh points
x = np.linspace(x_boundary[0], x_boundary[1], N)

infiniteWell = InfiniteWell()

# define ode
ode_infinite_well = FiniteDifferenceHomo(infiniteWell)

#set boundary conditions u=[u[0],u[1]]
ode_infinite_well.set_boundary_conditions(u_boundary[0], u_boundary[1])

dx, w, H = ode_infinite_well.solve(x_boundary, n)

""" print(v)
plt.plot([1,2,3,4],v)
plt.show() """

""" print("eigenvalues",np.sqrt(w)/dx)
v = H[:,0]
plt.plot(x, v)
plt.xlim(0,L)
plt.show() """

# print("eigenvalues = ", k)
# print(H)

""" # Write everything into a file
with open('solutionInfiniteWell.txt', 'w') as f:
    f.write("# Matrix A\n")
    np.savetxt(f, A, fmt='%.4f')
    f.write("\n# Vector b\n")
    np.savetxt(f, b, fmt='%.4f')
    f.write("\n# x values\n")
    np.savetxt(f, xEx, fmt='%.4f')
    f.write("\n# u values\n")
    np.savetxt(f, uEx, fmt='%.10f')
    f.write("\n# |y(x_i) - u_i|\n")
    #np.savetxt(f, np.abs(exactSol(xEx) - uEx), fmt='%.10f')

plt.plot(xEx, uEx, color='#1f77b4', label=f"numerical")
plt.title("Finite difference method")
plt.legend()
plt.show()
 """
""" # *** BOUNDARY VALUE PROBLEM - Exercise 11.3 from Burden - TEST EXAMPLE ***
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
xEx, uEx, A, b = odeModel.solve(x_boundary, N)

# Write everything into a file
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

print("File solution.txt created!")

plt.plot(xEx, uEx, color='#1f77b4', label=f"numerical")
plt.plot(xEx, model.exactsol(xEx), color="#f10707", label=f"exact")
plt.title("Finite difference method")
plt.legend()
plt.show() """

""" 
# Example: harmonic oscillator with angular frequency omega=1

# angular frequency
omega = 1
N = 50
u_boundary = [0, 1]  # array
x_boundary = (0, np.pi/2)  # tuple


def qCoef(x):
    return -(omega**2)

 """

""" ode = FiniteDifference(lambda x: 0*x, lambda x: -(omega**2)*x, lambda x: 0*x)
ode.set_boundary_conditions(u_boundary[0], u_boundary[1])
x, u = ode.solve(x_boundary, N)
 """
""" plt.plot(x, np.sin(x), color='orange', label=f"exact")
plt.plot(x, u, color='#1f77b4', label=f"numerical")
plt.title("Finite difference method")
plt.legend()
plt.show()
 """

# *** INITIAL VALUE PROBLEM ***

# Problem to solve
# def f(t,u):
#    return u

# Parameters
# u0 = 1
# t0 = 0
# T = 5
# N = 5000

# Solve ODE
# ode = ForwardEuler_v0(f)
# ode.set_initial_condition(u0)
# t, u = ode.solve((t0,T),N)

# plt.plot(t,u)
# plt.show()

# Solve pendulum problem
# model = Pendulum(l=1)
# solver = ForwardEuler(model)
# solver.set_iniital_condition([np.pi/4,0])
# T = 10
# N = 1000
# t, u = solver.solve(t_span=(0,T), N=N)

# plt.plot(t, u[:, 0], label=r'$\theta$')
# plt.plot(t, u[:, 1], label=r'$\omega$')
# plt.xlabel('t')
# plt.ylabel(r'Angle ($\theta$) and angular velocity ($\omega$)')
# plt.legend()
# plt.show()