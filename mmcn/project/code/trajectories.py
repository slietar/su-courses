from matplotlib.axes import Axes
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from . import comp, config, utils


def system(y: np.ndarray, *, c: float, z: float):
  return np.array([
    (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + z) / c,
    1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
  ])

def nullclines(x: np.ndarray, *, z: float):
  return np.array([
    (x**3 - 3.0 * x**2) - z,
    (1.0 - 5.0 * x**2)
  ])



c = 1.0

fig, axs = plt.subplots(ncols=2, nrows=2)
ax: Axes

for z, ax in zip([-1.5, -0.5, 6, 12], axs.ravel()):
# for z, ax in zip([-.95, -0.90, 0, 6], axs.ravel()):
  ax.set_title(f'z = {z}')

  ylim = -25, 5
  x = np.linspace(-3, 4, 100)
  # ylim = 0.6, 1.0
  # x = np.linspace(0.1, 0.3, 50)
  y = np.linspace(*ylim, 100)
  X, Y = np.meshgrid(x, y)

  ncl = nullclines(x, z=z)

  XY = np.c_[X.flat, Y.flat].T
  UV = system(XY, c=c, z=z)
  UV_ = system(np.array([X, Y]), c=c, z=z)
  UV_norm = np.linalg.norm(UV, axis=0)
  UV /= UV_norm

  # q = ax.quiver(XY[0, :], XY[1, :], UV[0, :], UV[1, :], UV_norm, cmap='gray_r')
  # plt.colorbar(q, ax=ax)

  sp = ax.streamplot(X, Y, UV_[0, :], UV_[1, :], color=np.linalg.norm(UV_, axis=0), cmap='viridis', density=0.5, broken_streamlines=True)
  # ax.streamplot(X, Y, UV_[0, :], UV_[1, :], color=np.linalg.norm(UV_, axis=0), cmap='gray_r') #, color=UV_norm.reshape(X.shape), cmap='gray_r')

  sp.lines.set_alpha(0.5)
  # sp.arrows.set_alpha(0.0)

  ax.plot(x, ncl[0, :], label='x-nullcline', linestyle='dashed')
  ax.plot(x, ncl[1, :], label='y-nullcline', linestyle='dashed')

  ax.legend()
  ax.set_ylim(*ylim)


plt.show()
