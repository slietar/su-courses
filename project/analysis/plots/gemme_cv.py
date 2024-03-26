from matplotlib import pyplot as plt

from .. import data, gemme, shared
from ..cv import cv


a = data.mutations.apply(lambda mutation: gemme.gemme.loc[mutation.position, mutation.alternate_aa], axis='columns') # type: ignore
b = data.mutations.apply(lambda mutation: gemme.gemme_mean[mutation.position], axis='columns')


fig, ax = plt.subplots(figsize=(10, 8))

ax.axline((0, data.gemme_threshold), color='gray', linestyle='--', slope=0)
ax.axline((data.gemme_threshold, 0), (data.gemme_threshold, 1), color='gray', linestyle='--')

scatter = ax.scatter(b, a, c=cv.loc[data.mutations.position], cmap='RdYlGn', s=5.5)

ax.set_xlabel('Mean')
ax.set_ylabel('Mutation')

cbar = fig.colorbar(scatter, ax=ax)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Circular variance', rotation=270)


with (shared.output_path / 'gemme_cv.png').open('wb') as file:
  plt.savefig(file, dpi=300)
