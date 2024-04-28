import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke

from . import comp, config, utils


stat_points = comp.get_stat_points(np.linspace(-2, 14, 100000))
eig = comp.get_eignvalues(stat_points)


fig, ax = plt.subplots()

# ax.set_xlim(-0.02, 0.1)
ax.set_ylim(-2, 2)

ax.axhline(0, alpha=0.5, color='gray', linewidth=0.8)

eig_real = utils.isclosereal(eig[:, 0])

ax.plot(stat_points[eig_real, 0], eig[eig_real, 0].real, color='C0', label='Valeurs propres réelles', path_effects=[withStroke(linewidth=3, foreground='white')])
ax.plot(stat_points[eig_real, 0], eig[eig_real, 1].real, color='C0', path_effects=[withStroke(linewidth=3, foreground='white')])
ax.plot(stat_points[~eig_real, 0], eig[~eig_real, 0].real, color='C1', label='Valeurs propres complexes conjuguées', path_effects=[withStroke(linewidth=3, foreground='white')])
ax.grid()

ax.set_xlabel('x')
ax.set_ylabel('Composante réelle des valeurs propres')

ax.legend()

with (config.output_path / 'eigenvalues.png').open('wb') as file:
  fig.savefig(file)
