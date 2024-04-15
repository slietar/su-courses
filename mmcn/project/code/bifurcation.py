import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from . import comp, config, utils


zs = np.linspace(-2, 12, 10000)


stat_points_x_unflattened = comp.get_stat_points(zs)
stat_points_x = stat_points_x_unflattened.ravel()
stat_points_z = np.tile(zs, stat_points_x_unflattened.shape[0])

mask = utils.isclosereal(stat_points_x)
order = np.argsort(stat_points_x[mask])

stat_points_x = stat_points_x[mask][order].real
stat_points_z = stat_points_z[mask][order].real

stat_points_trace = comp.get_trace(stat_points_x)
stat_points_stable = stat_points_trace.real < 0.0


fig, ax = plt.subplots()

current_pos = 0

for index, (stable, count) in enumerate(utils.group(stat_points_stable)):
  sl = slice(current_pos, current_pos + count)
  current_pos += count

  ax.plot(
    stat_points_z[sl],
    stat_points_x[sl],
    color='C0',
    label=(('Point fixe stable' if stable else 'Point fixe instable') if index < 2 else None),
    linestyle=('solid' if stable else 'dotted')
  )

# ax.plot(stat_points_z, stat_points_x)

bifurcations1_z = [5/27, -1]
bifurcations1_x = [
  comp.get_stat_points(5/27)[0].real,
  comp.get_stat_points(-1)[0].real
]

bifurcations2_x = [
  1 - np.sqrt(6) / 3,
  1 + np.sqrt(6) / 3
]

bifurcations2_z = interp1d(stat_points_x, stat_points_z)(bifurcations2_x)

ax.scatter(
  [*bifurcations1_z, *bifurcations2_z],
  [*bifurcations1_x, *bifurcations2_x],
  color='C1',
  label='Bifurcation'
)

# print(interp1d(stat_points_x, stat_points_z)(1 - np.sqrt(6)/3))


ax.set_xlabel('z')
ax.set_ylabel('x')

ax.grid()
ax.legend(loc='lower right')


with (config.output_path / 'bifurcation.png').open('wb') as file:
  fig.savefig(file)
