import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from .. import plots as _
from .residues import consolidated_residues


df = consolidated_residues.dropna()
df_pathogenic = df.effect_cardio | df.effect_ophtalmo | df.effect_sk
# df = df[df_pathogenic]
df_train = df.drop(['effect_cardio', 'effect_ophtalmo', 'effect_sk'], axis='columns')

def proc(row):
  effects = [row.effect_cardio, row.effect_ophtalmo, row.effect_sk]

  if np.sum(effects) == 0:
    return 0
  if np.sum(effects) > 1:
    return 1

  return np.argmax(effects) + 2

# effect = df.apply(proc, axis='columns')

# print(df)

model = PCA(n_components=2)
pc = model.fit_transform(df_train)
print(model.explained_variance_ratio_)

fig, ax = plt.subplots()

ax.scatter(pc[:, 0], pc[:, 1], c=df_pathogenic, s=3.0)
# ax.scatter(pc[:, 0], pc[:, 1], c=effect, s=2.0)

# im = ax.scatter(pc[:, 0], pc[:, 1], c=df.index, s=3.0)
# fig.colorbar(im, ax=ax)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.show()
