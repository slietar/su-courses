from .. import shared
from ..plots.utils import ProteinMap
from .domain_pca import pc_training_dataframes


pc_training_dataframe = pc_training_dataframes[1]

pmap = ProteinMap()
pmap.plot_dataframe(
  pc_training_dataframe.PC1,
  cmap='cool',
  label='PC1'
)

pmap.plot_dataframe(
  pc_training_dataframe.PC2,
  cmap='cool',
  label='PC2'
)

pmap.finish()


with (shared.output_path / 'pca_map.png').open('wb') as file:
  pmap.fig.savefig(file)
