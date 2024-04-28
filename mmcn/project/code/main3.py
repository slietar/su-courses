import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate

from . import comp, config, utils


zs = np.linspace(-0.5, 12, 100)
zs = np.linspace(10.75, 12, 100)
# zs = np.array([11])

# z = 2 doesn't work without a perturbation


stat_points_x_unflattened = comp.get_stat_points(zs)
stat_points_x = stat_points_x_unflattened.ravel()
stat_points_z = np.tile(zs, stat_points_x_unflattened.shape[0])

mask = utils.isclosereal(stat_points_x)
order = np.argsort(stat_points_x[mask])

stat_points_x = stat_points_x[mask][order].real
stat_points_z = stat_points_z[mask][order].real

# stat_points_trace = comp.get_trace(stat_points_x)
# stat_points_det = comp.get_det(stat_points_x)
# stat_points_stable = (stat_points_trace.real < 0.0) & (stat_points_det.real > 0.0)

m = stat_points_x > 0.3

xn = stat_points_x[m]
yn = comp.get_y(xn)
zn = stat_points_z[m]

# print(xn.shape)
# print(yn.shape)
# print(zn.shape)

# print(xn[0], yn[0], zn[0])
# print(comp.system.subs({comp.x: xn[0], comp.y: yn[0], comp.z: zn[0]}))


c = 1.0
tmax = 200

def run_z(xp: float, yp: float, zp: float):
  integr = integrate.solve_ivp(lambda t, y: [
    (y[1, ...] - y[0, ...]**3 + 3.0 * y[0, ...]**2 + zp) / c,
    1.0 - 5.0 * y[0, ...]**2 - y[1, ...]
  ], method='RK23', t_span=(0, tmax), y0=[xp + 1e-10, yp], max_step=0.2, vectorized=True)

  t_index = np.argmax(integr.t > 100)
  t_index = 0

  if 0:
    fig, ax = plt.subplots()

    # ax.plot(integr.t, integr.y[0, :])
    # print(integr.y.T)
    ax.plot(integr.y[0, :], integr.y[1, :])
    ax.plot([xp + 1e-10], [yp], 'ro')
    ax.grid()

    print([
      np.min(integr.y[0, t_index:]),
      np.max(integr.y[0, t_index:])
    ])

    plt.show()
    sys.exit()

  return [
    np.min(integr.y[0, t_index:]),
    np.max(integr.y[0, t_index:])
  ]

results = np.array([run_z(x, y, z) for x, y, z in zip(xn, yn, zn)])

# print(results.shape)

# ], method='RK23', t_span=(0, tmax), y0=[xn[-1] + 1e-10, yn[-1]], max_step=0.2) #, vectorized=True)

# print(result.y.shape)
# print(result.y.T)

# print(results)

fig, ax = plt.subplots()

ax.scatter(zn, results[:, 0], label='Min', s=1.0)
ax.scatter(zn, results[:, 1], label='Max', color='red', s=1.0)
ax.grid()
# ax.legend()
# ax.plot(result.y[0, :], result.y[1, :])

plt.show()
