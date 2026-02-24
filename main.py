import numpy as np
import matplotlib.pyplot as plt
from finite_difference import FiniteDifference
# from fin_diff_model import ModelExample
# from forward_euler_class import ForwardEuler_v0, ForwardEuler
# from logistic import Logistic
# from pendulum import Pendulum

# *** BOUNDARY VALUE PROBLEM ***


# Example chap. 11.3 Burden's "Numerical Analysis"
N = 9
u_boundary = [1, 2]
x_boundary = (1, 2)


def pCoef(x):
    return -2/x


def qCoef(x):
    return 2/x**2


def rCoef(x):
    return np.sin(np.log(x))/x**2


def exactSol(x):
    c2 = (1/70)*(8 - 12*np.sin(np.log(2))-4*np.cos(np.log(2)))
    return (11/10-c2)*x + \
            c2/x**2 - \
            (3/10)*np.sin(np.log(x)) - \
            (1/10)*np.cos(np.log(x))


x = np.linspace(1, 2, N)

print("p(x)= ", pCoef(x))
print("q(x)= ", qCoef(x))
print("r(x)= ", rCoef(x))

odeExample = FiniteDifference(pCoef, qCoef, rCoef)
odeExample.set_boundary_conditions(u_boundary[0], u_boundary[1])
xEx, uEx, A, b = odeExample.solve(x_boundary, N)

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
plt.title("Finite difference method")
plt.legend()
plt.show()

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