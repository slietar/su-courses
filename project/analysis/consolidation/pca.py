import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from .. import plots as _
from .. import shared
from .residues import native_descriptors, phenotypes


training_df = native_descriptors
phenotypes_df = phenotypes.loc[training_df.index]
# training_df = native_descriptors[phenotypes.loc[native_descriptors.index].any(axis='columns')]
pathogenic = phenotypes_df.any(axis='columns')
# phenotypes_df = phenotypes_df[mask]

model = PCA(n_components=3)
pc = model.fit_transform(scale(training_df))

comp = pd.DataFrame(model.components_, columns=training_df.columns, index=[f'PC{pc_index + 1}' for pc_index in range(model.components_.shape[0])])

for pc_index, (pc_name, row) in enumerate(comp.iterrows()):
  print(f'PC{pc_index + 1}')
  print(f'  Explained variance ratio: {(model.explained_variance_ratio_[pc_index] * 100):.2f}%')
  print(f'  Most significant features:')

  for feature_name, value in row.sort_values(ascending=False, key=abs)[:10].items():
    print(f'    {feature_name.ljust(16)} {value:+.4f}')

  print()


output_path = shared.output_path / 'pca'
output_path.mkdir(exist_ok=True)

def plot1():
  fig, ax = plt.subplots()

  scatter = ax.scatter(pc[:, 0], pc[:, 1], c=pathogenic, s=1.0, cmap=ListedColormap(['C0', 'C1']))

  ax.legend(handles=scatter.legend_elements()[0], labels=['Non-pathogenic', 'Pathogenic'])

  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')

  ax.grid()


  with (output_path / 'pathogenic.png').open('wb') as file:
    plt.savefig(file)


def plot2():
  fig, axs = plt.subplots(figsize=(12, 12), ncols=3, nrows=3, sharex=True, sharey=True)
  # fig.subplots_adjust(hspace=0.5, wspace=0.3)

  for index, (ax, (label, effect)) in enumerate(zip(axs.flat, phenotypes_df.rename(columns=dict(
    effect_cardio='Cardio',
    effect_cutaneous='Cutaneous',
    effect_neuro='Neuro',
    effect_opthalmo='Opthalmo',
    effect_pneumothorax='Pneumothorax',
    effect_severe='Severe',
    effect_sk='SK'
  )).items())):
    mask = ~effect & ~pathogenic
    ax.scatter(pc[mask][:, 0], pc[mask][:, 1], alpha=0.5, c='C0', s=3.0, label=('No effect' if index < 1 else None))
    ax.scatter(pc[effect][:, 0], pc[effect][:, 1], c='C1', s=3.0, label=('Effect' if index < 1 else None))

    ax.set_title(label)
    ax.grid()

  fig.legend()

  with (output_path / 'effects.png').open('wb') as file:
    plt.savefig(file)


plot1()
plot2()
