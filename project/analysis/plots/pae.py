from matplotlib import pyplot as plt

from .. import shared
from ..pae import pae_mean_by_position
from .utils import ProteinMap, set_colobar_label


fig, ax = plt.subplots(figsize=(25, 8))

map = ProteinMap(ax)
im = map.plot_dataframe(
  pae_mean_by_position.rename(columns=dict(
    pae1='Mean over columns',
    pae2='Mean over rows'
  ))
)
map.finish()

cbar = fig.colorbar(im, ax=ax)
set_colobar_label(cbar, 'PAE')


with (shared.output_path / 'pae.png').open('wb') as file:
  fig.savefig(file)
