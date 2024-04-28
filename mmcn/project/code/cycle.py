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


c = 1.0

def run_z(xp: float, yp: float, zp: float):
  integr = integrate.solve_ivp(lambda t, y: [
    (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + zp) / c,
    1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
  ], method='RK23', t_span=(0, 200), y0=[xp + 1e-15, yp], max_step=0.05, vectorized=True)

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


stat_points_ = comp.get_stat_points(np.linspace(-2, 14, 1000))
# stat_points_ = comp.get_stat_points(np.linspace(-1.1, 0, 1000))
# stat_points = stat_points_[(stat_points_[:, 0] > 0)] # & ~comp.is_stable(stat_points_)]
mask = (stat_points_[:, 0] > 0) & ~comp.is_stable(stat_points_)
# mask = np.ones(stat_points_.shape[0], dtype=bool)
stat_points = stat_points_[mask, :]

limit_cycle = pickle.load(open('cycle.pkl', 'rb'))[mask, :]

i = limit_cycle.shape[0] - np.argmax(limit_cycle[::-1, 0] < -1.5) - 1 + 1

limit_cycle = limit_cycle[i:, :]
stat_points = stat_points[i:, :]

limit_cycle = smooth(limit_cycle, 9, axis=0)

# cycle_limits = np.array([run_z(*stat_points[index, :]) for index in range(stat_points.shape[0])])

# with open('cycle.pkl', 'wb') as file:
#   pickle.dump(cycle_limits, file)


if __name__ == '__main__':
  fig, ax = plt.subplots()

  # ax.scatter(stat_points[:, 2], cycle_limits[:, 0], label='Min', s=4.0, alpha=0.6)
  # ax.scatter(stat_points[:, 2], cycle_limits[:, 1], label='Max', s=4.0, alpha=0.6, color='red')
  ax.plot(stat_points_[:, 2], stat_points_[:, 0], '--')

  ax.plot(stat_points[:, 2], limit_cycle[:, 0])
  ax.plot(stat_points[:, 2], limit_cycle[:, 1])

  ax.grid()

  plt.show()


if __name__ == '__main_':
  fig, ax = plt.subplots()

  ax.axvline(11.59)
  ax.set_yscale('log')
  ax.plot(stat_points[:, 2], limit_cycle[:, 1] - limit_cycle[:, 0])
  ax.grid()

  plt.show()
