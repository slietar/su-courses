from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from .. import data, gemme, shared
from ..aa import amino_acids


variants_filtered = data.variants[data.variants.alternate_aa.isin(amino_acids.letter) & (data.variants.pathogenicity > 0)]

a = variants_filtered.apply(lambda variant: gemme.gemme.loc[variant.position, variant.alternate_aa], axis='columns') # type: ignore
b = variants_filtered.apply(lambda variant: gemme.gemme_mean[variant.position], axis='columns')


fig, ax = plt.subplots(figsize=(10, 8))

ax.axline((0, data.gemme_threshold), color='gray', linestyle='--', slope=0)
ax.axline((data.gemme_threshold, 0), (data.gemme_threshold, 1), color='gray', linestyle='--')

scatter = ax.scatter(b, a, c=variants_filtered.pathogenicity, cmap='RdYlGn', s=5.5)

ax.set_xlabel('Mean')
ax.set_ylabel('Mutation')

legend = ax.legend(
  *scatter.legend_elements(
    num=range(len(data.pathogenicity_labels)),
    fmt=FuncFormatter(lambda x, i: data.pathogenicity_labels[x])
  ),
  loc='upper left',
  title='Pathogenicity'
)

ax.add_artist(legend)
ax.set_title('gnomAD variants')


with (shared.output_path / 'gemme_gnomad.png').open('wb') as file:
  plt.savefig(file, dpi=300)
