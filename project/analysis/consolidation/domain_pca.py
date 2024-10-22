from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from .. import data, plots as _, shared
from .residues import classification_descriptors, descriptor_names, native_descriptors
from ..mutations import all_mutation_info


# full_df = native_descriptors.join(all_mutation_info.drop(['pathogenic', 'severe'], axis='columns').rename(columns=dict(
#   gemme_all='gemme_all_mut',
#   gemme_orthologs='gemme_orthologs_mut'
# )))

full_df = native_descriptors

# print(full_training_df)


scatter_fig = plt.figure()
scatter_fig.set_figheight(4.5)
scatter_fig.subplots_adjust(
  bottom=0.10,
  left=0.08,
  right=0.92,
  top=0.90,
  hspace=0.3
)

bar_fig, bar_axs = plt.subplots(ncols=2, nrows=len(data.domain_kinds), sharex=True, sharey=True)
bar_fig.set_figheight(8.0)
bar_fig.subplots_adjust(
  top=0.97,
  hspace=0.05,
  wspace=0.05
)

pc_training_dataframes = list[pd.DataFrame]()

for domain_kind_index, domain_kind in enumerate(data.domain_kinds):
  ax = scatter_fig.add_subplot(2, 2, domain_kind_index + 1)
  target_domains = data.domains[data.domains.kind == domain_kind]
  target_residues_mask = classification_descriptors.domain.isin(target_domains.name)
  training_residues_mask = target_residues_mask & all_mutation_info.pathogenic

  # training_df = full_training_df[target_residues_mask.reindex(full_training_df.index)]
  target_df = full_df[target_residues_mask.reindex(full_df.index)]
  training_df = full_df[training_residues_mask.reindex(full_df.index)].rename(columns=descriptor_names)

  # print(domain_kind, np.unique(training_residues_mask, return_counts=True))

  # print(target_df)

  model = PCA(n_components=5)
  pc_training = model.fit_transform(scale(training_df))

  pcs_index = pd.Series([f'PC{pc_index + 1}' for pc_index in range(model.components_.shape[0])], name='component')
  pc_training_df = pd.DataFrame(pc_training, columns=pcs_index, index=training_df.index)

  pc_training_dataframes.append(pc_training_df)

  # scale(target_df) model.components_
  # print(target_df.to_numpy().shape, model.components_.shape)

  pc = scale(target_df) @ model.components_.T
  pathogenic_mask = all_mutation_info.pathogenic.reindex(target_df.index)

  components = pd.DataFrame(model.components_, columns=training_df.columns, index=pcs_index)

  # ax.scatter(pc_training[:, 0], pc_training[:, 1], cmap='RdYlBu', s=1.0)
  # ax.scatter(pc[:, 0], pc[:, 1], c=training_residues_mask.reindex(target_df.index), cmap='rainbow', s=0.5)
  ax.scatter(pc[:, 0][~pathogenic_mask], pc[:, 1][~pathogenic_mask], color='C0', marker='.', s=1.0, alpha=0.4, label=('Résidu non pathogène' if domain_kind_index == 0 else None))
  ax.scatter(pc[:, 0][pathogenic_mask], pc[:, 1][pathogenic_mask], color='C1', marker='.', s=1.0, label=('Résidu pathogène' if domain_kind_index == 0 else None))

  ax.set_title(f'Domaines de type {domain_kind}')

  ax.grid()

  if domain_kind_index > 0:
    ax.set_xlabel('PC1')

  if domain_kind_index % 2 == 0:
    ax.set_ylabel('PC2')

  print(model.explained_variance_ratio_)
  # print(model.components_.shape)


  bar_axs[domain_kind_index, 0].set_ylabel(f'{domain_kind}', rotation=0)

  for pc_index in range(2):
    ax = bar_axs[domain_kind_index, pc_index]

    ax.bar(range(len(training_df.columns)), model.components_[pc_index, :])
    ax.grid()

for pc_index, ax in enumerate(bar_axs[0, :].flat):
  ax.set_title(f'PC{pc_index + 1}')

for ax in bar_axs[-1, :].flat:
  ax.set_xticks(
    range(len(training_df.columns)),
    labels=training_df.columns,
    rotation=45,
    ha='right'
  )

for ax in bar_axs.flat:
  ax.xaxis.set_tick_params(bottom=False)

for ax in bar_axs[:, 1:].flat:
  ax.yaxis.set_tick_params(left=False)

scatter_fig.legend()


pc_training_dataframe = pd.concat(pc_training_dataframes)


if __name__ == '__main__':
  with (shared.output_path / 'residues_pca.png').open('wb') as file:
    scatter_fig.savefig(file)

  with (shared.output_path / 'residues_pca_components.png').open('wb') as file:
    bar_fig.savefig(file)
