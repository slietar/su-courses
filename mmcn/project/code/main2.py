import sys
from matplotlib import pyplot as plt
import numpy as np
import sympy

from . import comp, config as config, utils


# zs = np.linspace(-1.1, -0.5, 10000, dtype=complex)
zs = np.linspace(-2, 12, 100000, dtype=complex)


fig, ax = plt.subplots()

# ax.plot(zs, np.real(l1a), alpha=0.5, color='r', label='Re')
# ax.plot(zs, np.imag(l1a), alpha=0.5, color='b', label='Im')

# ax.plot(zs, np.real(l1b), alpha=0.5, color='r')
# ax.plot(zs, np.imag(l1b), alpha=0.5, color='b')

# l1s = [sympy.lambdify(z, l, 'numpy')(zs) for l in lambdas[0]]
# l2s = [sympy.lambdify(z, l, 'numpy')(zs) for l in lambdas[1]]

# for l in l1s:
#   ax.plot(np.real(l), np.imag(l), color='b', label=r'$\lambda_1$')

# for l in l2s:
#   ax.plot(np.real(l), np.imag(l), color='r', label=r'$\lambda_2$')


# stat_points = np.asarray([sympy.lambdify(comp.z, stat_point, 'numpy')(zs) for stat_point in comp.stat_points])
stat_points = np.asarray(sympy.lambdify(comp.z, comp.stat_points, 'numpy')(zs))

# trace = sympy.lambdify(comp.x, comp.jacobian.trace(), 'numpy')(stat_points)
# stable = trace.real > 0

sp = np.array([
  stat_points,
  np.tile(zs, (stat_points.shape[0], 1))
]).reshape(2, -1)

# tr = sympy.lambdify(jacobian.trace(), 'numpy')(zs)

# print(np.isreal(sp[0, :]).sum())
# print(np.isclose(sp[0, :].imag, 0).sum())


# mask = isreal(sp[0, :])
sp = sp[:, utils.isclosereal(sp[0, :])].real

# sort = np.argsort(sp[0, :])
sp = sp[:, np.argsort(sp[0, :])]

# print(sp.shape)

# print(sx.shape)
# print(zs[None, :].shape)
# print(np.tile(zs, (sx.shape[0], 1)).shape)

# print(np.c_[sx, zs].shape)

# s = np.argsort(sx, axis=1)
# print(np.sort(sp, axis=1).shape)

# print(sp.T)
# print(sp.shape)

trace = sympy.lambdify(comp.x, comp.jacobian.trace(), 'numpy')(sp[0, :])
stable = trace.real < 0

# print(list(group(stable)))

# ax.set_xlim(-3, -2)
# ax.plot(sp[1, :], sp[0, :])

# ax.scatter(sp[1, :], sp[0, :], c=stable.ravel()[mask][sort])
# ax.scatter(sp[1, :], sp[0, :], c=stable, s=2.0)


if False:
  current_pos = 0

  for item, count in group(stable):
    sl = slice(current_pos, current_pos + count)
    current_pos += count

    ax.plot(sp[1, sl], sp[0, sl], color='C0', linestyle=('solid' if item else 'dashed'))
    # ax.scatter(sp[1, sl], sp[0, sl], c=('b' if item else 'r'), s=2.0)


if True:
  l1 = sympy.lambdify(comp.x, comp.eigenval1, 'numpy')(sp[0, :].astype(complex))
  l2 = sympy.lambdify(comp.x, comp.eigenval2, 'numpy')(sp[0, :].astype(complex))

  # ax.set_xlim(-1, 0)

  # ax.scatter(sp[0, :], l1.real, color='b', s=2.0)
  ax.scatter(sp[0, :], l1.imag, color='r', s=2.0)

  ax.scatter(sp[0, :], l2.real, color='b', s=2.0)
  ax.scatter(sp[0, :], l2.imag, color='r', s=2.0)

  # ax.scatter(l2.real, l2.imag, color='b', s=2.0)

  # ax.plot(sp[1, :], l1.real, color='r', linestyle='solid')
  # ax.plot(sp[1, :], l1.imag, color='r', linestyle='dashed')

  # ax.plot(sp[1, :], l2.real, color='b', linestyle='solid')
  # ax.plot(sp[1, :], l2.imag, color='b', linestyle='dashed')


# print(stable.ravel()[mask][sort].astype(int).shape)
# ax.plot(sp[0, :], '.')

# for i in range(sx.shape[0]):
#   mask = np.isreal(sx[i, :])
#   ax.plot(zs[mask], sx[i, :][mask].real)


ax.grid()
# ax.legend()

plt.show()
