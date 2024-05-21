from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np

from .. import shared
from .residues import descriptor_names, native_descriptors
from ..plots import utils


df = native_descriptors.rename(columns=descriptor_names)

fig, ax = plt.subplots()
fig.subplots_adjust(
  bottom=0.2,
  top=0.95
)

corr = df.corr().to_numpy()
corr[np.triu_indices_from(corr)] = np.nan
corr = corr[1:, :-1]
im = ax.matshow(corr, cmap='RdYlBu_r', vmin=-1.0, vmax=1.0)

for (i, j), z in np.ndenumerate(corr):
  if not np.isnan(z):
    ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', path_effects=[withStroke(alpha=0.7, linewidth=1, foreground='white')])

cbar = fig.colorbar(im)
utils.set_colobar_label(cbar, 'Corrélation')

ax.set_xticks(labels=df.columns[:-1], ticks=range(len(df.columns) - 1), ha='right', rotation=45)
ax.set_yticks(labels=df.columns[1:], ticks=range(0, len(df.columns) - 1))
ax.xaxis.set_tick_params(bottom=False, top=False, labelbottom=True, labeltop=False)
ax.yaxis.set_tick_params(left=False)


with (shared.output_path / 'residues_correlations.png').open('wb') as file:
  fig.savefig(file)
