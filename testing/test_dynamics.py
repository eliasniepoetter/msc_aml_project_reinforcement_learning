from dynamics.flightdynamics import Flightdynamics
import matplotlib.pyplot as plt
import numpy as np

id = 0
initial_state = np.array([[0.],[0.],[0.],[0.],[0.],[500.]])
testDynamics = Flightdynamics(id, initial_state)

V0 = 51.4
dt = 0.01
t0 = 0
tf = 30
nsteps = int(tf/dt)
time = np.linspace(t0, tf-dt, nsteps)

results = []
for i in time:
    input = np.array([[np.deg2rad(-10)],[0.]])
    results.append(testDynamics.timestep(input, dt))

# print results
alpha = [result[0] for result in results]
q = [result[1] for result in results]
V = [result[2]+V0 for result in results]
teta = [result[3] for result in results]
x = [result[4] for result in results]
z = [result[5] for result in results]

fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(time, alpha)
axs[0, 0].set_title('Alpha')

axs[0, 1].plot(time, q)
axs[0, 1].set_title('q')

axs[1, 0].plot(time, V)
axs[1, 0].set_title('V')

axs[1, 1].plot(time, teta)
axs[1, 1].set_title('Teta')

axs[2, 0].plot(time, x)
axs[2, 0].set_title('x')

axs[2, 1].plot(time, z)
axs[2, 1].set_title('z')
#axs[2, 1].set_ylim([0, 2*initial_state[5][0]])

for ax in axs.flat:
    ax.set(xlabel='time', ylabel='value')
    ax.grid(True)

plt.suptitle('Results for open loop simulation of dynamics')
plt.tight_layout()
plt.show()