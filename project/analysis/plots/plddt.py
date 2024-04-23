from matplotlib import pyplot as plt

from .. import shared
from ..plddt import plddt
from .utils import ProteinMap, set_colobar_label


fig, ax = plt.subplots(figsize=(25, 8))

map = ProteinMap(ax)
im = map.plot_dataframe(
  plddt.rename(columns=dict(
    alphafold_global='AlphaFold global',
    alphafold_pruned='AlphaFold per domain with context',
    esmfold_pruned='ESMFold per domain with context',
    esmfold_isolated='ESMFold per domain without context'
  )
))
map.finish()

cbar = fig.colorbar(im, ax=ax)
set_colobar_label(cbar, 'pLDDT')

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file)
