from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .. import data, shared
from . import utils


print(data.mutations)

df = pd.concat([
  data.mutations.effect_cardio.rename('Cardio') > 0,
  data.mutations.effect_cutaneous.rename('Cutaneous'),
  data.mutations.effect_ophtalmo.rename('Ophtalmo') > 0,
  data.mutations.effect_pneumothorax.rename('Pneumothorax'),
  data.mutations.effect_severe.rename('Severe'),
  data.mutations.effect_sk.rename('SK') > 0,
], axis='columns')

# print(data.mutations)
# print(df.max(axis='index'))
# print(df.corr())

def compute_cond(df: pd.DataFrame):
  mat = df.values # to_numpy(dtype=float)
  print(mat.shape)

  result = np.zeros((len(df.columns), len(df.columns), 2))

  for column_index in range(len(df.columns)):
    mask = mat[:, column_index]
    # print(mask.sum(), mat[mask, :].shape)
    result[column_index, :, 0] = mat[mask, :].sum(axis=0) / mask.sum()
    result[column_index, :, 1] = mat[~mask, :].sum(axis=0) / (~mask).sum()

  return result

# im = ax.matshow(df.corr())

fig, (ax1, ax2) = plt.subplots(figsize=(12, 8), ncols=2)
ax1: Axes
ax2: Axes

cond = compute_cond(df)
im1 = ax1.matshow(cond[:, :, 0], vmin=0.0, vmax=1.0)
im2 = ax2.matshow(cond[:, :, 1], vmin=0.0, vmax=1.0)

cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6)
utils.set_colobar_label(cbar, 'Probability')

for ax in (ax1, ax2):
  ax.set_xlabel('B')
  ax.xaxis.set_label_position('top')

  ax.set_xticks(labels=df.columns, ticks=range(len(df.columns)), rotation='vertical')
  ax.set_yticks(labels=df.columns, ticks=range(len(df.columns)))
  ax.xaxis.set_tick_params(bottom=False, top=False)
  ax.yaxis.set_tick_params(left=False)

ax1.set_ylabel('A')
ax2.yaxis.set_tick_params(labelleft=False, left=False)

ax1.set_title('$P(B|A)$')
ax2.set_title(r'$P(B|\overline{A})$')

with (shared.output_path / 'phenotype_condprob.png').open('wb') as file:
  fig.savefig(file)


fig, ax = plt.subplots(figsize=(10, 8))
fig.subplots_adjust(top=0.85)

corr = df.corr()
im = ax.matshow(corr, cmap='RdYlBu_r', vmin=-1.0, vmax=1.0)

for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center')

cbar = fig.colorbar(im)
utils.set_colobar_label(cbar, 'Correlation')

ax.set_xticks(labels=df.columns, ticks=range(len(df.columns)), rotation='vertical')
ax.set_yticks(labels=df.columns, ticks=range(len(df.columns)))
ax.xaxis.set_tick_params(bottom=False, top=False)
ax.yaxis.set_tick_params(left=False)


with (shared.output_path / 'phenotype_correlations.png').open('wb') as file:
  fig.savefig(file)
