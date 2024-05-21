from matplotlib import pyplot as plt

from .. import shared
from ..pae import pae_mean_by_position
from .utils import ProteinMap


map = ProteinMap()
im = map.plot_dataframe(
  pae_mean_by_position.rename(columns=dict(
    pae_inter='Moyenne sur les\nrésidus des\ndomaines adjacents',
    pae_intra='Moyenne sur les\nrésidus du domaine'
  )),
  label='PAE (Å)',
  vmin=0.0
)
map.finish()


with (shared.output_path / 'pae.png').open('wb') as file:
  map.fig.savefig(file)
