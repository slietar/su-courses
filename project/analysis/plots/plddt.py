from matplotlib import pyplot as plt

from .. import shared
from ..plddt import plddt
from .utils import ProteinMap, get_transform_linear_component, set_colobar_label


fig, ax = plt.subplots()
fig.set_figheight(2.0)
fig.subplots_adjust(
  top=0.9,
  left=0.12,
  right=1.0
)

map = ProteinMap(ax)
im = map.plot_dataframe(
  plddt.loc[:, ['alphafold_global', 'alphafold3_global', 'alphafold_pruned']].rename(columns=dict(
    alphafold_global='AlphaFold 2\n(protéine entière)',
    alphafold3_global='AlphaFold 3\n(protéine entière)',
    alphafold_pruned='AlphaFold 2\n(par domaine)',
    # esmfold_pruned='ESMFold per domain with context',
    # esmfold_isolated='ESMFold per domain without context'
  )
))

inch_size = get_transform_linear_component(fig.dpi_scale_trans + fig.transFigure.inverted())

cbar = fig.colorbar(im, ax=ax, fraction=(0.6 * inch_size[0]), pad=(0.2 * inch_size[0]))
set_colobar_label(cbar, 'pLDDT')

map.finish()

with (shared.output_path / 'plddt.png').open('wb') as file:
  fig.savefig(file)
