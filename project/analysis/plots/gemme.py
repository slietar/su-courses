from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from .. import data, shared
from ..gemme import gemme_all, gemme_mutations, gemme_variants


def discrete_colobar(im, *, boundary_labels: Optional[tuple[str, str]] = None, label: Optional[str] = None):
  value_count = int(im.norm.vmax - im.norm.vmin) + 1

  cmap = LinearSegmentedColormap.from_list('a', im.cmap(im.norm(np.linspace(im.norm.vmin, im.norm.vmax, value_count))), value_count)
  cbar = fig.colorbar(ScalarMappable(Normalize(im.norm.vmin - 0.5, im.norm.vmax + 0.5), cmap), ax=im.axes, label=label, location='bottom')

  if boundary_labels is not None:
    cbar.set_ticks(
      ticks=[im.norm.vmin - 0.5, im.norm.vmax + 0.5],
      labels=boundary_labels
    )

    cbar.ax.xaxis.set_tick_params(bottom=False)

    left_tick, right_tick = cbar.ax.xaxis.get_majorticklabels()
    left_tick.set_horizontalalignment('left')
    right_tick.set_horizontalalignment('right')

  else:
    cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))


fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
fig.set_figheight(4.0)
fig.subplots_adjust(
  bottom=0.01,
  left=0.05,
  top=0.95,
  right=0.98
)


mutations_mean = data.mutations.apply(lambda mut: gemme_all.mean.at[mut.position], axis='columns')
variants_mean = data.variants.apply(lambda mut: gemme_all.mean.at[mut.position], axis='columns')

mutation_effects = data.mutations.loc[:, [
  'effect_cardio',
  'effect_cutaneous',
  'effect_ophtalmo',
  'effect_neuro',
  'effect_pneumothorax',
  'effect_severe',
  'effect_sk'
]]

mutation_effect_count = (mutation_effects > 0).sum(axis='columns')


axs[0].set_title('Mutations faux sens de l\'hôpital Bichat')

im1 = axs[0].scatter(gemme_mutations.gemme_all, mutations_mean, c=mutation_effect_count, s=1.0)
discrete_colobar(im1, label='Nombre d\'effets')


axs[1].set_title('Mutations faux sens de gnomAD')

mask = data.variants.pathogenicity > 0
im2 = axs[1].scatter(gemme_variants.gemme_all[mask], variants_mean[mask], c=data.variants.pathogenicity[mask], s=1.0)

discrete_colobar(im2, boundary_labels=('Bénin', 'Pathogène'), label='Effets')


axs[0].set_ylabel('Score GEMME de la mutation')
axs[1].yaxis.set_tick_params(left=False)

for ax in axs.flat:
  ax.set_xlabel('Score GEMME moyen')
  ax.grid()


with (shared.output_path / 'gemme.png').open('wb') as file:
  plt.savefig(file)
