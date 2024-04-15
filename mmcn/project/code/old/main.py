from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy import integrate


# fig, axs = plt.subplots(ncols=2, nrows=2)
# ax: Axes

# x = np.linspace(-10, 5, 100)
# x = np.linspace(-2, 2, 100)
# c = 1.0
# zs = [-0.5, 0.5, 1.0, 10.0]

# for z, ax in zip(zs, axs.flat):
#   ax.plot(x, (x**3 + 3 * x**2 + z) / c, 'r', label=r'$\dot{x} = 0$')
#   ax.plot(x, 1 - 5 * x**2, 'b', label=r'$\dot{y} = 0$')

#   ax.legend()
#   ax.set_title(f'$z = {z:.2f}$')


c = 1.0
dt = 1e-2

xp = []
yp = []
zp = []

tmax = 100

for z in np.linspace(-2, 12, 40):
  print(f'z = {z:.2f}')

  # z = 2.0

  for _ in range(10):
    x0 = np.random.random() * 20 - 10
    y0 = np.random.random() * 20 - 10

    result = integrate.solve_ivp(lambda t, y: [
      (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + z) / c,
      1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
    ], t_eval=[tmax], t_span=(0, tmax), y0=[x0, y0], vectorized=True)

    # print(result.y[:, 0])

    xp.append(result.y[0, 0])
    yp.append(result.y[1, 0])
    zp.append(z)

  # break

  # for _ in range(20):
  #   x = np.random.random() * 20 - 10
  #   y = np.random.random() * 20 - 10

  #   for _ in range(10000):
  #     dx = (y - x**3 + 3 * x**2 + z) / c
  #     dy = 1 - 5 * x**2 - y

  #     x += dx * dt
  #     y += dy * dt

  #   xp.append(x)
  #   yp.append(y)
  #   zp.append(z)

# print(xp)

fig, ax = plt.subplots()

ax.plot(zp, xp, '.')


plt.show()
