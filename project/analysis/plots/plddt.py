from .. import shared
from ..plddt import plddt
from .utils import ProteinMap


# map = ProteinMap(6, max_ax_length=3)

# map.plot_dataframe(
#   pd.Series([1, 2, 3, 4], index=pd.Series([2, 3, 4, 5], name='position'), name='foo'),
#   label='Aaa'
# )

map = ProteinMap()
map.plot_dataframe(
  plddt.loc[:, ['alphafold_global', 'alphafold3_global', 'alphafold_pruned']].rename(columns=dict(
    alphafold_global='AlphaFold 2\n(protéine entière)',
    alphafold3_global='AlphaFold 3\n(protéine entière)',
    alphafold_pruned='AlphaFold 2\n(par domaine)',
    # esmfold_pruned='ESMFold per domain with context',
    # esmfold_isolated='ESMFold per domain without context'
  )
), label='pLDDT', vmin=0.0, vmax=100.0)

map.finish()

with (shared.output_path / 'plddt.png').open('wb') as file:
  map.fig.savefig(file)
