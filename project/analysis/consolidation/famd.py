import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from prince import FAMD, PCA
# from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from .. import plots as _, shared
from .residues import native_descriptors as df, phenotypes as effects


pathogenic = effects.effect_cardio | effects.effect_neuro | effects.effect_ophtalmo | effects.effect_pneumothorax | effects.effect_severe | effects.effect_sk
# df_train = df.drop(['effect_cardio', 'effect_ophtalmo', 'effect_sk'], axis='columns')

model = FAMD(n_components=4)
# model = PCA(n_components=4)
pc = model.fit_transform(df)

# print(model.column_contributions_)
# print(model.percentage_of_variance_)

# comp = pd.DataFrame(model.column_coordinates_, columns=df.columns, index=[f'PC{pc_index + 1}' for pc_index in range(model.components_.shape[0])])

for pc_index in model.column_contributions_.columns:
  print(f'PC{pc_index + 1}')
  print(f'  Explained variance ratio: {(model.percentage_of_variance_[pc_index]):.2f}%')
  print(f'  Most significant features:')

  for feature_name, value in model.column_contributions_.loc[:, pc_index].sort_values(ascending=False, key=abs)[:].items():
    print(f'    {feature_name: <16} {value:+.4f}')

  print()

print(model.eigenvalues_summary)


fig, ax = plt.subplots()

scatter = ax.scatter(pc.values[:, 0], pc.values[:, 2], c=pathogenic, s=1.0, cmap=ListedColormap(['C0', 'C1']))

ax.legend(handles=scatter.legend_elements()[0], labels=['Non-pathogenic', 'Pathogenic'])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

ax.grid()


fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(pc.values[:, 0], pc.values[:, 2], pc.values[:, 3], c=pathogenic, s=1.0, cmap=ListedColormap(['C0', 'C1']))
ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')


plt.show()
