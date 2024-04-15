from matplotlib import pyplot as plt
import numpy as np

from . import config as _



def nullclines(x: np.ndarray, *, z: np.ndarray | float):
  z_ = np.asarray(z)

  return np.array([
    (x**3 - 3.0 * x**2)[(..., *((None,) * len(z_.shape)))] - z_,
    np.broadcast_to((1.0 - 5.0 * x**2)[(..., *((None,) * len(z_.shape)))], x.shape + z_.shape)
  ])


xs = np.linspace(-2.5, 2.5, 100)
zs = np.array([
  -2.0, -1.5, -1.0,
  -0.5, 0.0, 0.5,
  2.0, 2.5, 3.0
])

ncls = nullclines(xs, z=zs)
print(ncls.shape)

fig, axs = plt.subplots(ncols=3, nrows=3)

for index, (ax, z) in enumerate(zip(axs.flat, zs)):
  ax.plot(xs, ncls[0, :, index], 'r', label=r'$\dot{x} = 0$')
  ax.plot(xs, ncls[1, :, index], 'b', label=r'$\dot{y} = 0$')

  ax.grid()
  ax.set_title(f'z = {z}')

# fig.legend()


plt.show()
