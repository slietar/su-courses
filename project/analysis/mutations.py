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

mutation_pathogenic = ((mutation_effects == True) | (mutation_effects == 2)).sum(axis='columns') > 0

mutation_consequences = pd.concat([
  mutation_pathogenic.rename('pathogenic'),
  (mutation_pathogenic & data.mutations.effect_severe).rename('severe')
], axis='columns')


def map_mutation_group(group: pd.DataFrame, /):
  return pd.Series(dict(
    gemme_all=group.gemme_all.min(),
    gemme_orthologs=group.gemme_orthologs.min(),
    pathogenic=group.pathogenic.any(),
    severe=group.severe.any()
  ))

known_mutation_info = mutation_consequences.join(data.mutations.position).join(gemme_mutations).groupby('position').apply(map_mutation_group)


# def map_residue(residue: pd.Series, /):
#   return pd.Series(dict(
#     gemme_all=np.nan, # gemme_all.dataframe.at[residue.position, residue.amino_acid]
#     gemme_orthologs=np.nan,
#     pathogenic=False,
#     severe=False
#   ))

# y = pd.DataFrame(data.amino_acids).reset_index(drop=False).apply(map_residue, axis='columns')
# print(pd.concat([data.amino_acids, data.amino_acids.index.to_series()]))
# y = pd.concat([data.amino_acids, data.amino_acids.index.to_series()], axis='columns').apply(map_residue)
# print(y)

# default_mutation_info = pd.DataFrame(index=data.position_index).assign(
#   gemme_all=0.0,
#   gemme_orthologs=0.0,
#   pathogenic=False,
#   severe=False
# )

all_mutation_info = known_mutation_info.reindex(data.position_index, fill_value=np.nan).fillna(dict(
  gemme_all=0.0,
  gemme_orthologs=0.0,
  pathogenic=False,
  severe=False
))


# print(df)


# fig, ax = plt.subplots()

# ax.histogram(df)



# # mutations = data.mutations[data.mutations.position.isin(cv.index)]

# df = pd.concat([
#   mutations.apply(lambda mutation: gemme_all.dataframe.loc[mutation.position, mutation.alternate_aa], axis='columns').rename('gemme_all'),
#   mutations.apply(lambda mutation: gemme_orthologs.dataframe.loc[mutation.position, mutation.alternate_aa], axis='columns').rename('gemme_orthologs')
# ], axis='columns')
# # b = mutations.apply(lambda mutation: gemme.gemme_mean[mutation.position], axis='columns')

# print(df)

# # print((~np.isclose(a.gemme_orthologs.to_numpy(), 0)).sum())
# print(np.isclose(df.to_numpy(), 0).sum(axis=0) / len(df))
# print(np.isclose(gemme_orthologs.array, 0).sum() / gemme_orthologs.array.size)

# # fig, ax = plt.subplots()

# # ax.scatter(a.gemme_all, a.gemme_orthologs)

# # plt.show()

# print(mutations)
