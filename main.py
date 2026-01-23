import numpy as np
import matplotlib.pyplot as plt
from finite_difference import FiniteDifference

# from forward_euler_class import ForwardEuler_v0, ForwardEuler
# from logistic import Logistic
# from pendulum import Pendulum

# *** BOUNDARY VALUE PROBLEM ***

# Example: harmonic oscillator with angular frequency omega=0.1

# angular frequency
omega = 1
N = 50
u_boundary = [0, 1]  # array
x_boundary = (0, np.pi/2)  # tuple


def qCoef(x):
    return -(omega**2)


ode = FiniteDifference(lambda x: 0*x, lambda x: -(omega**2)*x, lambda x: 0*x)
ode.set_boundary_conditions(u_boundary[0], u_boundary[1])
x, u = ode.solve(x_boundary, N)

plt.plot(x, np.sin(x), color='orange', label=f"exact")
plt.plot(x, u, color='#1f77b4', label=f"numerical")
plt.title("Finite difference method")
plt.legend()
plt.show()

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