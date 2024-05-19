from matplotlib import pyplot as plt

from .. import shared
from ..plddt import plddt
from .utils import ProteinMap, get_transform_linear_component, set_colobar_label


fig, ax = plt.subplots()

map = ProteinMap(ax)
im = map.plot_dataframe(
  plddt.loc[:, ['alphafold_global', 'alphafold3_global', 'alphafold_pruned']].rename(columns=dict(
    alphafold_global='AlphaFold 2\n(protéine entière)',
    alphafold3_global='AlphaFold 3\n(protéine entière)',
    alphafold_pruned='AlphaFold 2\n(par domaine)',
    # esmfold_pruned='ESMFold per domain with context',
    # esmfold_isolated='ESMFold per domain without context'
  )
), vmin=0.0, vmax=100.0)

map.add_colorbar(im, 'pLDDT')
map.finish()

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file)
