from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def system(y: np.ndarray, *, c: float, z: float):
  return np.array([
    (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + z) / c,
    1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
  ])


c = 1.0
z = -0.5


fig, ax = plt.subplots()

x = np.linspace(-2, 4, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)

XY = np.c_[X.flat, Y.flat].T
UV = system(XY, c=c, z=z)
UV_ = system(np.array([X, Y]), c=c, z=z)
UV_norm = np.linalg.norm(UV, axis=0)
UV /= UV_norm

# q = ax.quiver(XY[0, :], XY[1, :], UV[0, :], UV[1, :], UV_norm, cmap='gray_r')
# plt.colorbar(q, ax=ax)

ax.streamplot(X, Y, UV_[0, :], UV_[1, :], color=np.linalg.norm(UV_, axis=0), cmap='gray_r') #, color=UV_norm.reshape(X.shape), cmap='gray_r')

# ax.streamplot
# ax.contour(XY[0, :], XY[1, :], UV[0, :], levels=[0])

# print(system(np.array([X, Y]), c=1.0, z=-0.5).shape)
# UV_ = system(np.array([X, Y]), c=c, z=z)
# ax.contour(X, Y, UV_[0, :], levels=[0])

ax.plot(x, (x**3 - 3 * x**2 - z) / c, 'r', label=r'$\dot{x} = 0$')
ax.plot(x, 1 - 5 * x**2, 'b', label=r'$\dot{y} = 0$')

ax.legend()
ax.set_xlim(-2, 4)
ax.set_ylim(-10, 10)

plt.show()
