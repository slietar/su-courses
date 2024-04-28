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


# Limit cycle apparition: z = -0.125

stat_points_ = comp.get_stat_points(np.linspace(-2, 14, 100000))
# stat_points_ = comp.get_stat_points(np.linspace(-1.1, 0, 1000))
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

  fig, axs = plt.subplots(nrows=2)
  fig.subplots_adjust(left=0.2, right=0.8, top=0.85)

  ax: Axes

  for eig_index, ax in enumerate(axs.flat):
    eig_real = utils.isclosereal(eig)

    ax.axhline(0, alpha=0.5, color='gray', linewidth=0.8)
    # ax.set_xlim(0, 0.2)
    # pprint(plt.rcParams)
    # , path_effects=[withStroke(linewidth=3, foreground='white')]

    # ax.plot(stat_points[eig_real[:, eig_index], 0], eig[eig_real[:, eig_index], eig_index].imag, color='C0', label=('Valeur propre réelle' if eig_index == 0 else None), linestyle='none', marker='.')
    # ax.plot(stat_points[eig_real[:, 1], 0], eig[eig_real[:, 1], 1].real, color='C0', linestyle='solid', path_effects=[withStroke(linewidth=3, foreground='white')])
    ax.plot(stat_points[eig_real[:, eig_index], 0], eig[eig_real[:, eig_index], eig_index].real, color='C0', label=('Valeur propre réelle' if eig_index == 0 else None), linestyle='solid', path_effects=[withStroke(linewidth=3, foreground='white')])
    ax.plot(stat_points[~eig_real[:, eig_index], 0], eig[~eig_real[:, eig_index], eig_index].real, color='C1', label=('Valeur propre complexe' if eig_index == 0 else None), linestyle='solid', path_effects=[withStroke(linewidth=3, foreground='white')])
    ax.grid()

    # _ = ax.get_xlim(), ax.get_ylim()

    if eig_index == 1:
      pt = np.array([1.82, 0])
      trans = fig.dpi_scale_trans + transforms.ScaledTranslation(pt[0], pt[1], ax.transData)
      c = Circle((0.0, 0.0), clip_on=False, edgecolor='black', linewidth=1, facecolor='none', path_effects=[withStroke(linewidth=3, foreground='white')], radius=0.05, transform=trans, zorder=10)

      # c = Circle((0.5, 0.5), clip_on=False, edgecolor='red', facecolor='blue', path_effects=[withStroke(linewidth=7, foreground='white')], radius=0.01, transform=fig.transFigure)
      # c = Circle((0.5, 0.5), edgecolor='red', facecolor='blue', path_effects=[withStroke(linewidth=7, foreground='white')], radius=0.5, transform=ax.transAxes)
      ax.add_artist(c)

      # ptp = pt + (fig.dpi_scale_trans)
      # ax.text(pt[0], pt[1], 'Bifurcation', ha='left', va='center', fontsize=8, color='black', path_effects=[withStroke(linewidth=3, foreground='white')])
      ax.text(0.1, -0.005, 'Bifurcation', ha='left', va='center', fontsize=8, color='black', path_effects=[withStroke(linewidth=2, foreground='white')], transform=trans)

  axs[0].xaxis.set_tick_params(bottom=False, labelbottom=False)
  axs[0].set_ylim(-5, 2)
  axs[1].set_xlabel('x')

  axs[0].set_ylabel('λ₁')
  axs[1].set_ylabel('λ₂')

  fig.legend()

  # ax.scatter(stat_points[:, 2], eig[:, 1].real, alpha=0.6, label=r'$\lambda_2$', s=4.0)
  # c=comp.is_stable(stat_points)
  # ax.legend(loc='upper right')
  # ax.set_ylim(-5, 1)

  with (config.output_path / 'eigenvalues.png').open('wb') as file:
    fig.savefig(file)


  # plt.show()
