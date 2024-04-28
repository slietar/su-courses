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
  ], method='RK23', t_span=(0, 200), y0=[xp + 1e-15, yp], max_step=0.2, vectorized=True)

  if 0:
    fig, ax = plt.subplots()

    # ax.plot(integr.t, integr.y[0, :])
    # print(integr.y.T)
    ax.plot(integr.y[0, :], integr.y[1, :])
    ax.plot([xp + 1e-15], [yp], 'ro')
    ax.grid()

    print([
      np.min(integr.y[0, :]),
      np.max(integr.y[0, :])
    ])

    plt.show()
    sys.exit()

  return [
    np.min(integr.y[0, :]),
    np.max(integr.y[0, :])
  ]


# Limit cycle apparition: z = -0.125

stat_points_ = comp.get_stat_points(np.linspace(-2, 12, 1000))
# stat_points = stat_points_[(stat_points_[:, 0] > 0)] # & ~comp.is_stable(stat_points_)]
stat_points = stat_points_

if False:
  cycle_limits = np.array([run_z(*stat_points[index, :]) for index in range(stat_points.shape[0])])

  fig, ax = plt.subplots()

  ax.scatter(stat_points[:, 2], cycle_limits[:, 0], label='Min', s=4.0, alpha=0.6)
  ax.scatter(stat_points[:, 2], cycle_limits[:, 1], label='Max', s=4.0, alpha=0.6, color='red')
  ax.plot(stat_points[:, 2], stat_points[:, 0], '--')

  ax.grid()
  # ax.legend()

if True:
  eig = comp.get_eignvalues(stat_points)
  fig, axs = plt.subplots(ncols=2)
  ax: Axes

  for eig_index, ax in enumerate(axs.flat):
    eig_real = utils.isclosereal(eig)
    ax.plot(stat_points[eig_real[:, eig_index], 0], eig[eig_real[:, eig_index], eig_index].real, color='C0', label=rf'$Im(\lambda_{eig_index + 1}) = 0$', linestyle='solid')
    ax.plot(stat_points[~eig_real[:, eig_index], 0], eig[~eig_real[:, eig_index], eig_index].real, color='C0', label=rf'$Im(\lambda_{eig_index + 1}) \neq 0$', linestyle='dotted')
    ax.grid()

    # ax.autoscale()
    # print(ax.get_xlim(), ax.get_ylim())
    _ = ax.get_xlim(), ax.get_ylim()
    # print((ax.transData + ax.transAxes.inverted()).transform([-2, -25]))
    # print(ax.transData)
    # print(ax.transAxes)

    # print((fig.dpi_scale_trans + ax.transData).transform_affine([0, 0]))

    break

  # print(ax.get_xlim(), ax.get_ylim())
  # trans = (fig.dpi_scale_trans + ax.transData) # transforms.ScaledTranslation(10, 10, ax.transData))
  # trans = (fig.dpi_scale_trans + transforms.ScaledTranslation(0, 0, ax.transData))
  # trans = fig.dpi_scale_trans + ax.transData
  # print(trans.transform([0, 0]))
  # print(trans.transform_affine([0, 0]))

  # print(fig.dpi_scale_trans)
  # print(ax.transData.trans)

  # print(fig.dpi_scale_trans)
  # print(fig.dpi)

  # print(fig.transFigure)

  # pt = (fig.transFigure.inverted() + ax.transData).transform([0, -10])
  # print(pt)

  pt = np.array([-0.5, -5])
  trans = fig.dpi_scale_trans + transforms.ScaledTranslation(pt[0], pt[1], ax.transData)
  c = Circle((0.0, 0.0), clip_on=False, edgecolor='black', linewidth=1, facecolor='none', path_effects=[withStroke(linewidth=3, foreground='white')], radius=0.05, transform=trans, zorder=10)

  # c = Circle((0.5, 0.5), clip_on=False, edgecolor='red', facecolor='blue', path_effects=[withStroke(linewidth=7, foreground='white')], radius=0.01, transform=fig.transFigure)
  # c = Circle((0.5, 0.5), edgecolor='red', facecolor='blue', path_effects=[withStroke(linewidth=7, foreground='white')], radius=0.5, transform=ax.transAxes)
  ax.add_artist(c)

  # ptp = pt + (fig.dpi_scale_trans)
  # ax.text(pt[0], pt[1], 'Bifurcation', ha='left', va='center', fontsize=8, color='black', path_effects=[withStroke(linewidth=3, foreground='white')])
  ax.text(0.1, -0.005, 'Bifurcation', ha='left', va='center', fontsize=8, color='black', path_effects=[withStroke(linewidth=2, foreground='white')], transform=trans)

  # ax.legend()

  # ax.scatter(stat_points[:, 2], eig[:, 1].real, alpha=0.6, label=r'$\lambda_2$', s=4.0)
  # c=comp.is_stable(stat_points)
  # ax.legend(loc='upper right')
  # ax.set_ylim(-5, 1)

  with (config.output_path / 'eigenvalues.png').open('wb') as file:
    fig.savefig(file)


  # plt.show()
