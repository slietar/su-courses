from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from .. import plots as _, shared
from .residues import consolidated_residues


df = consolidated_residues.dropna()
pathogenic = df.effect_cardio | df.effect_ophtalmo | df.effect_sk
df_train = df.drop(['effect_cardio', 'effect_ophtalmo', 'effect_sk'], axis='columns')

model = PCA(n_components=2)
pc = model.fit_transform(scale(df_train))

# ang = 32 / 180 * np.pi
# rot = np.array([
#   [np.cos(ang), -np.sin(ang)],
#   [np.sin(ang), np.cos(ang)]
# ])

# comp = pd.DataFrame(rot.T @ model.components_, columns=df_train.columns, index=[f'PC{pc_index + 1}' for pc_index in range(model.components_.shape[0])])
# pc = pc @ rot

comp = pd.DataFrame(model.components_, columns=df_train.columns, index=[f'PC{pc_index + 1}' for pc_index in range(model.components_.shape[0])])

for pc_index, (pc_name, row) in enumerate(comp.iterrows()):
  print(f'PC{pc_index + 1}')
  print(f'  Explained variance ratio: {(model.explained_variance_ratio_[pc_index] * 100):.2f}%')
  print(f'  Most significant features:')

  for feature_name, value in row.abs().sort_values(ascending=False)[:5].items():
    print(f'    {feature_name}\t{value:.4f}')

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
  effects = pd.concat([
    df.effect_cardio,
    df.effect_neuro,
    df.effect_ophtalmo,
    df.effect_pneumothorax,
    df.effect_severe,
    df.effect_sk
  ], axis='columns')

  fig, axs = plt.subplots(figsize=(12, 8), ncols=3, nrows=2)
  # fig.subplots_adjust(hspace=0.5, wspace=0.3)

  for index, (ax, (effect_name, effect), label) in enumerate(zip(axs.flat, effects.items(), ['Cardio', 'Neuro', 'Opthalmo', 'Pneumothorax', 'Severe', 'SK'])):
    # scatter = ax.plot(pc[effect][:, 0], pc[effect][:, 1], color='C1', linestyle=None)
    # scatter = ax.scatter(pc[pathogenic][:, 0], pc[pathogenic][:, 1], s=1.0, c=effect[pathogenic], cmap='coolwarm')
    ax.scatter(pc[~effect & pathogenic][:, 0], pc[~effect & pathogenic][:, 1], alpha=0.5, c='C0', s=3.0)
    ax.scatter(pc[effect][:, 0], pc[effect][:, 1], c='C1', s=3.0, label=('Effect' if index < 1 else None))

    ax.set_title(label)
    ax.grid()

  fig.legend()

  with (output_path / 'effects.png').open('wb') as file:
    plt.savefig(file)


plot2()
