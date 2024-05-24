from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from .utils import ProteinMap
from .. import data, shared
from ..mutations import all_mutation_info, mutation_effects, mutation_pathogenic


# print((mutation_effects == 1).sum())
# print(data.mutations.effect_severe.sum())

# print(mutation_pathogenic.sum())
# print(all_mutation_info.pathogenic.sum())

pmap = ProteinMap()

for region_index, region in enumerate(data.interest_regions.itertuples()):
  pmap.axs[region_index].add_patch(Rectangle(
    (region.start_position - 0.5, -1),
    region.end_position - region.start_position + 1,
    1,
    alpha=0.5,
    facecolor='gray',
    zorder=-1
  ))

pmap.plot_dataframe(
  (all_mutation_info.pathogenic.astype(int) + all_mutation_info.severe).rename('Présence de\nmutations pathogènes'),
  cmap=LinearSegmentedColormap.from_list('a', [(0, 0, 0, 0), 'C0', 'C1'], 3)
)

pmap.finish()


with (shared.output_path / 'mutations.png').open('wb') as file:
  pmap.fig.savefig(file)
