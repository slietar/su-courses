from pathlib import Path
import pickle
from pprint import pprint
import sys
from matplotlib import pyplot as plt, transforms
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
import numpy as np
from scipy import integrate

from . import comp, config, utils


def run_z(xp: float, yp: float, zp: float):
  integr = integrate.solve_ivp(lambda t, y: [
    (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + zp) / comp.c,
    1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
  ], method='RK23', t_span=(0, 1000 if zp < (11 if comp.c == 1 else 7) else 10000), y0=[xp + 1e-14, yp], max_step=0.1, vectorized=True)

  # return integr.y.T

  return [
    np.min(integr.y[0, :]),
    np.max(integr.y[0, :])
  ]


def smooth(xs: np.ndarray, /, width: int, *, axis: int = 0):
  assert width % 2 == 1
  half_width = width // 2

  box = np.ones(width)

  def concolve(arr: np.ndarray):
    a = np.convolve(arr, box, mode='same')
    a[:half_width] /= np.arange(half_width + 1, width)
    a[half_width:-half_width] /= width
    a[-half_width:] /= np.arange(half_width + 1, width)[::-1]

    return a

  return np.apply_along_axis(concolve, arr=xs, axis=axis)


# fig, ax = plt.subplots()

# u = comp.get_stat_points(np.array([6]))
# r = run_z(*u[0, :])

# ax.plot(r[:, 0], r[:, 1])
# plt.show()


# sys.exit()




stat_points_all = comp.get_stat_points(np.linspace(-2, 14, 100))

# stat_points_ = comp.get_stat_points(np.linspace(-1.1, 0, 1000))
# stat_points = stat_points_[(stat_points_[:, 0] > 0)] # & ~comp.is_stable(stat_points_)]
mask = (stat_points_all[:, 0] > 0) & ~comp.is_stable(stat_points_all)
# mask = np.ones(stat_points_.shape[0], dtype=bool)
stat_points = stat_points_all[mask, :]

cache_path = Path() / f'tmp/limit_cycle{comp.c}.pkl'

if cache_path.exists():
  with cache_path.open('rb') as file:
    limit_cycle = pickle.load(file)
else:
  limit_cycle = np.array([run_z(*stat_points[index, :]) for index in range(stat_points.shape[0])])

  cache_path.parent.mkdir(exist_ok=True, parents=True)

  with cache_path.open('wb') as file:
    pickle.dump(limit_cycle, file)

if comp.c == 1:
  i = limit_cycle.shape[0] - np.argmax(limit_cycle[::-1, 0] < -1.5) - 1 + 1

  limit_cycle = limit_cycle[i:, :]
  stat_points = stat_points[i:, :]

  # limit_cycle = smooth(limit_cycle, 9, axis=0)


if __name__ == '__main__':
  print('Homoclinic connection:')
  print(f'x={limit_cycle[0, 0]}')
  print(f'z={stat_points[0, 2]}')
