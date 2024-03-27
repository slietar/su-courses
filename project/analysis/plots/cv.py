import math
from matplotlib import pyplot as plt
import numpy as np

from .plddt import highlight_domains
from .. import data, shared
from ..cv import cv


df = cv.unstack()
df = df.reindex(columns=(('cv', position) for position in range(1, data.protein_length + 1)), fill_value=math.nan)


fig, ax = plt.subplots(figsize=(25, 8))

im = ax.imshow(df.values, aspect='auto', cmap='plasma', extent=((0, data.protein_length, 0, df.shape[0])), interpolation='none', vmin=0.0, vmax=1.0)
highlight_domains(ax, df.shape[0])

ax.set_yticks(
  labels=reversed(df.index.values),
  ticks=(np.arange(df.shape[0]) + 0.5)
)

ax.set_ylabel('Cutoff (Å)')
ax.set_ylim(0, df.shape[0] + 1)

ax.tick_params('y', left=False)

# fig.subplots_adjust(left=0.15)
cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Circular variance', rotation=270)

with (shared.output_path / 'cv.png').open('wb') as file:
  fig.savefig(file, dpi=300)
