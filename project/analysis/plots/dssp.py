from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from .. import shared
from ..dssp import dssp, dssp_labels
from .utils import ProteinMap


fig, ax = plt.subplots(figsize=(25, 8))

map = ProteinMap(ax)
im = map.plot_dataframe(
  dssp.rename(columns=dict(
    ss_global='Global',
    ss_contextualized='With context',
    ss_pruned='With context removed'
  ))
)
map.finish()

colors = [im.cmap(im.norm(value)) for value in range(len(dssp_labels))]
patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, dssp_labels)]
ax.legend(handles=patches, loc='lower right')


with (shared.output_path / 'dssp.png').open('wb') as file:
  fig.savefig(file)
