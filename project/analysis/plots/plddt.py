import sys
from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .. import data, shared
from ..esmfold import plddt as plddt_esmfold_contextualized


def highlight_domains(ax: Axes, y: float):
  ax1 = ax.twiny()
  ax1.set_xticks(
    labels=[domain.Index for domain in data.domains.itertuples()],
    ticks=[(domain.start_position - 0.5 + domain.end_position + 0.5) * 0.5 for domain in data.domains.itertuples()],
    rotation='vertical'
  )

  ax1.set_xlim(ax.get_xlim())
  ax1.tick_params('x', top=False)

  colors = {
    'EGF': 'r',
    'EGFCB': 'g',
    'TB': 'b'
  }

  for domain in data.domains.itertuples():
    rect = patches.Rectangle([domain.start_position - 0.5, y], domain.end_position - domain.start_position + 1, 1, color=colors[domain.kind], alpha=0.5, linewidth=0)
    ax.add_artist(rect)



df = pd.concat([data.plddt_alphafold_global, plddt_esmfold_contextualized], axis='columns').sort_index()

fig, ax = plt.subplots(figsize=(25, 8))

im = ax.imshow(df.values.T, cmap='bwr', aspect='auto', interpolation='none', extent=(0, len(df.values), 0, 2), vmin=0, vmax=100)
highlight_domains(ax, 2)

x_ticks = np.arange(199, len(df.values), 200)
ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels([str(x + 1) for x in x_ticks])

ax.set_yticks([0.5, 1.5])
ax.set_yticklabels([ 'ESMFold contextualized', 'AlphaFold global'])
ax.tick_params('y', left=False)
ax.set_ylim(0, 3)

fig.subplots_adjust(left=0.15)
fig.colorbar(im, ax=ax)

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file, dpi=300)
