from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .gemme import gemme_all, gemme_mutations


mutation_effects = data.mutations.loc[:, [
  'effect_cardio',
  'effect_cutaneous',
  'effect_ophtalmo',
  'effect_neuro',
  'effect_pneumothorax',
  'effect_sk'
]]

mutation_effect_count = ((mutation_effects == True) | (mutation_effects == 2)).sum(axis='columns')
mutation_pathogenic = mutation_effect_count > 0

mutation_consequences = pd.concat([
  mutation_pathogenic.rename('pathogenic'),
  (mutation_pathogenic & data.mutations.effect_severe).rename('severe'),
  mutation_effect_count.rename('effect_count')
], axis='columns')


def map_mutation_group(group: pd.DataFrame, /):
  worst_mutation = group.iloc[group.effect_count.argmax()]

  return pd.Series(dict(
    gemme_all=worst_mutation.gemme_all,
    gemme_orthologs=worst_mutation.gemme_orthologs,
    pathogenic=group.pathogenic.any(),
    severe=group.severe.any()
  ))

known_mutation_info = mutation_consequences.join(data.mutations.position).join(gemme_mutations).groupby('position').apply(map_mutation_group)

all_mutation_info = known_mutation_info.reindex(data.position_index, fill_value=np.nan).fillna(dict(
  gemme_all=known_mutation_info.gemme_all.mean(),
  gemme_orthologs=known_mutation_info.gemme_orthologs.mean(),
  pathogenic=False,
  severe=False
))


if __name__ == '__main__':
  print(all_mutation_info)
