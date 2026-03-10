# Solve pendulum problem - from "Solving ODE in Python", J. Sundnes, Chap. 1.2
model = Pendulum(l=1)
solver = ForwardEuler(model)
solver.set_iniital_condition([np.pi/4,0])
T = 10
N = 1000
t, u = solver.solve(t_span=(0,T), N=N)

plt.plot(t, u[:, 0], label=r'$\theta$')
plt.plot(t, u[:, 1], label=r'$\omega$')
plt.xlabel('t')
plt.ylabel(r'Angle ($\theta$) and angular velocity ($\omega$)')
plt.legend()
plt.show()