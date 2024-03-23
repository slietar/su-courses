import sys
from matplotlib import patches, pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .. import data, shared
from ..plddt import plddt


def highlight_domains(ax: Axes, y: float):
  ax1 = ax.twiny()
  ax1.set_xticks(
    labels=[f'{domain.kind} {domain.number}' for domain in data.domains.itertuples()],
    ticks=[(domain.start_position - 0.5 + domain.end_position + 0.5) * 0.5 for domain in data.domains.itertuples()],
    rotation='vertical'
  )

  ax1.set_xlim(ax.get_xlim())
  # ax1.tick_params('x', top=False)

  colors = {
    'EGF': 'r',
    'EGFCB': 'g',
    'TB': 'b'
  }

  for domain in data.domains.itertuples():
    rect = patches.Rectangle(
      [domain.start_position - 0.5, y],
      domain.end_position - domain.start_position + 1, 1,
      alpha=0.5,
      edgecolor='white',
      facecolor=colors[domain.kind],
      linewidth=1
    )

    ax.add_artist(rect)



df = pd.concat(plddt.values(), axis='columns').sort_index()

fig, ax = plt.subplots(figsize=(25, 8))

im = ax.imshow(df.values.T, cmap='plasma', aspect='auto', interpolation='none', extent=(0, len(df.values), 0, len(plddt)), vmin=0, vmax=100)
highlight_domains(ax, len(plddt))

x_ticks = np.arange(199, len(df.values), 200)
ax.set_xticks(x_ticks + 0.5)
ax.set_xticklabels([str(x + 1) for x in x_ticks])

ax.set_yticks(np.arange(len(plddt)) + 0.5)
ax.set_yticklabels(reversed(plddt.keys()))
ax.tick_params('y', left=False)
ax.set_ylim(0, len(plddt) + 1)

fig.subplots_adjust(left=0.15)
cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('pLDDT', rotation=270)

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file, dpi=300)
