from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from .. import data, shared
from ..gemme import gemme_all, gemme_mutations, gemme_variants


fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(bottom=0.15, top=0.98)


mutations_mean = data.mutations.apply(lambda mut: gemme_all.mean.at[mut.position], axis='columns')

print(gemme_all.dataframe)

ax.scatter(gemme_all.array.flat, list(gemme_all.mean) * len(gemme_all.dataframe.columns), alpha=0.01, color='gray', marker='.', s=5.5)
ax.scatter(gemme_mutations.gemme_all, mutations_mean, color='C1', marker='.', s=5.5)

ax.set_xlabel('Score GEMME moyen')
ax.set_ylabel('Score GEMME de la mutation')
ax.grid()


with (shared.output_path / 'gemme_full.png').open('wb') as file:
  plt.savefig(file)
