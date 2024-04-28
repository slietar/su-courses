import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from . import comp, config, utils
from .cycle import limit_cycle, stat_points as limit_cycle_sp


stat_points = comp.get_stat_points(np.linspace(-2, 12, 10000))


fig, ax = plt.subplots()

ax.plot(
  limit_cycle_sp[:, 2],
  limit_cycle[:, 0],
  color='C5',
  label='Cycle limite'
)

ax.plot(
  limit_cycle_sp[:, 2],
  limit_cycle[:, 1],
  color='C5'
)

ax.scatter([limit_cycle_sp[0, 2]] * 2, limit_cycle[0, :], color='C5', label='Connexion homocline', marker='*', zorder=5)


current_pos = 0

for index, (stable, count) in enumerate(utils.group(comp.is_stable(stat_points))):
  sl = slice(current_pos, current_pos + count)
  current_pos += count

  ax.plot(
    stat_points[sl, 2],
    stat_points[sl, 0],
    color='C0',
    label=(('Point fixe stable' if stable else 'Point fixe instable') if index < 2 else None),
    linestyle=('solid' if stable else 'dotted')
  )

# ax.plot(stat_points_z, stat_points_x)

# bifurcations1_z = [5/27, -1]
# bifurcations1_x = [
#   comp.get_stat_points(5/27)[0].real,
#   comp.get_stat_points(-1)[0].real
# ]

bifurcations1 = comp.get_stat_points(np.array([5/27, -1]))
print(bifurcations1)

bifurcations2_x = [
  1 - np.sqrt(6) / 3,
  1 + np.sqrt(6) / 3
]

bifurcations2_z = interp1d(stat_points[:, 0], stat_points[:, 2])(bifurcations2_x)

ax.scatter(
  [*bifurcations1[:, 2], *bifurcations2_z],
  [*bifurcations1[:, 0], *bifurcations2_x],
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
