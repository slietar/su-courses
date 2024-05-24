from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import pandas as pd

from ..polymorphism import polymorphism_score
from .utils import ProteinMap
from .. import data, shared
from ..mutations import all_mutation_info, mutation_effects, mutation_pathogenic


pmap = ProteinMap()

# def group(df):
#   return pd.Series(dict(
#     pathogenicity=df.pathogenicity.max()
#   ))

# variant_info = data.variants.groupby('position').apply(group)
# # variant_info = data.variants[data.variants.pathogenicity != 0].groupby('position').apply(group)

# print(variant_info)

# pmap.plot_dataframe(
#   polymorphism_score
#   (variant_info.pathogenicity >= 4).rename('Mutations'),
#   cmap=LinearSegmentedColormap.from_list('a', ['C0', 'C1'], 3)
# )

pmap.plot_dataframe(
  polymorphism_score.rename('Score de\npolymorphisme'),
  cmap='viridis',
  label='Score'
)

pmap.finish()


with (shared.output_path / 'mutations_gnomad.png').open('wb') as file:
  pmap.fig.savefig(file)
