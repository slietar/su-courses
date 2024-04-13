import numpy as np
from matplotlib import pyplot as plt

from .. import data, shared
from ..cv import cv
from .utils import highlight_domains


df = cv.reindex(index=range(1, data.protein_length + 1), fill_value=np.nan).loc[:, [10.0, 20.0, 30.0, 40.0, 50.0]]


fig, ax = plt.subplots(figsize=(25, 8))

im = ax.imshow(df.values.T, aspect='auto', cmap='plasma', extent=((0.5, data.protein_length + 0.5, 0, len(df.columns))), interpolation='none', vmin=0.0, vmax=1.0)
highlight_domains(ax, len(df.columns))

ax.set_yticks(
  labels=reversed(df.columns),
  ticks=(np.arange(len(df.columns)) + 0.5)
)

ax.set_ylabel('Cutoff (Å)')
ax.set_ylim(0, len(df.columns) + 1)

ax.tick_params('y', left=False)

# fig.subplots_adjust(left=0.15)
cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Circular variance', rotation=270)

with (shared.output_path / 'cv.png').open('wb') as file:
  fig.savefig(file, dpi=300)
