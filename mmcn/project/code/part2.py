from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from . import config


# Définir les équations différentielles du système
def hindmarsh_rose(state, t, I, c):
    x, y, z = state

    return [
      (y - x**3 + 3*x**2 + z + I) / c,
      1 - 5*x**2 - y,
      0.01 * (0.3*z - 1 - x)
    ]

I = 1.3
c = 2
initial_state = [0, 0, 0]
t = np.linspace(0, 2000, 20000)

# Simulation du système
state = integrate.odeint(hindmarsh_rose, initial_state, t, args=(I, c))
# x, y, z = state[:, 0], state[:, 1], state[:, 2]


fig, axs = plt.subplots(nrows=3, sharex=True)

for i in range(3):
  axs[i].plot(t, state[:, i])
  axs[i].grid()
  axs[i].yaxis.set_major_locator(MaxNLocator(5))

  if i < 3:
     axs[i].tick_params(bottom=False)

axs[2].axhline(-0.07 - I, color='C1', linewidth=0.5)
axs[2].axhline(-0.93 - I, color='C2', linewidth=0.5)


with (config.output_path / 'part2a.png').open('wb') as file:
  fig.savefig(file)


fig, ax = plt.subplots()

ax.xaxis.set_major_locator(MaxNLocator(10))
ax.plot(state[:, 0], state[:, 1], color='k', linewidth=0.2)
ax.grid()


with (config.output_path / 'part2b.png').open('wb') as file:
  fig.savefig(file)
