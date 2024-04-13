import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .. import shared
from ..plddt import plddt
from .utils import highlight_domains


df = pd.concat(plddt.values(), axis='columns').sort_index()

fig, ax = plt.subplots(figsize=(25, 8))

im = ax.imshow(df.values.T, cmap='plasma', aspect='auto', interpolation='none', extent=(0, len(df.values), 0, len(plddt)), vmin=0, vmax=100)
highlight_domains(ax, len(plddt))

x_ticks = np.arange(199, len(df.values), 200)
ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels([str(x + 1) for x in x_ticks])

labels = {
  'alphafold_global': 'AlphaFold global',
  'alphafold_pruned': 'AlphaFold per domain with context',
  'esmfold_pruned': 'ESMFold per domain with context',
  'esmfold_isolated': 'ESMFold per domain without context'
}

ax.set_yticks(np.arange(len(plddt)) + 0.5)
ax.set_yticklabels(labels[key] for key in reversed(plddt.keys()))
ax.tick_params('y', left=False)
ax.set_ylim(0, len(plddt) + 1)

fig.subplots_adjust(left=0.15)
cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('pLDDT', rotation=270)

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file)
