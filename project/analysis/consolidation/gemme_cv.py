from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from .. import plots as _, shared
from ..gemme import gemme_all, gemme_orthologs
from ..cv import cv
from ..mutations import all_mutation_info, known_mutation_info


fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
fig.set_figheight(3.2)
fig.subplots_adjust(top=0.95)

df = all_mutation_info.loc[:, ['pathogenic']].join(cv.loc[:, 10.0].rename('cv')).join(gemme_all.mean.rename('gemme_mean'))

ax1.scatter(df.gemme_mean[~df.pathogenic], df.cv[~df.pathogenic], color='C0', marker='.', s=1, alpha=0.4, label='Résidu non pathogène')

ax1.set_ylabel('Variance circulaire (seuil 10 Å)')

for ax in (ax1, ax2):
  ax.scatter(df.gemme_mean[df.pathogenic], df.cv[df.pathogenic], color='C1', marker='.', s=1, label='Résidu pathogène', zorder=2)
  ax.set_xlabel('GEMME moyen')
  ax.grid()

ax1.legend()
ax2.yaxis.set_tick_params(left=False)


ax.add_artist(Rectangle((-7.4, 0.54), 5.0, 0.42, alpha=0.3, facecolor='C0', zorder=1))
ax.add_artist(Rectangle((-2.3, 0.54), 2.5, 0.3, alpha=0.3, facecolor='C2', zorder=1))
ax.add_artist(Rectangle((-3.5, 0.35), 3.5, 0.17, alpha=0.3, facecolor='C3', zorder=1))


with (shared.output_path / 'gemme_cv.png').open('wb') as file:
  plt.savefig(file)
