import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from .. import data, plots, shared
from ..dssp import dssp, dssp_labels
from .utils import highlight_domains


df = dssp.reindex(index=range(1, data.protein_length + 1), fill_value=np.nan)

fig, ax = plt.subplots(figsize=(25, 8))
# ax.set_xlim(50, 500)
# ax.set_xlim(2675, 2700)

im = ax.imshow(df.values.T, aspect='auto', cmap='plasma', extent=((0.5, data.protein_length + 0.5, 0, 2)), interpolation='none')
highlight_domains(ax, 2)

ax.set_yticks(
  labels=df.columns,
  ticks=(np.arange(2) + 0.5)
)

# ax.set_ylabel('Cutoff (Å)')
ax.set_ylim(0, 3)

ax.tick_params('y', left=False)


colors = [im.cmap(im.norm(value)) for value in range(len(dssp_labels))]
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, dssp_labels)]
ax.legend(handles=patches, loc='lower right')


with (shared.output_path / 'dssp.png').open('wb') as file:
  fig.savefig(file)
