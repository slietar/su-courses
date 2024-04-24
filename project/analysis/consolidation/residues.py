import pickle
import numpy as np
import pandas as pd

from .. import shared, utils


# @utils.cache
def compute_consolidated_residues():
  from .. import data
  from ..cv import cv
  from ..dssp import dssp
  from ..gemme import gemme_mean
  from ..pae import pae_mean_by_position
  from ..plddt import plddt
  from ..rmsf import rmsf_by_position


  def get_domains():
    domains = list[int]()
    positions = list[int]()

    for index, (_, domain) in enumerate(data.domains.iterrows()):
      domains += [index] * (domain.end_position - domain.start_position + 1)
      positions += range(domain.start_position, domain.end_position + 1)

    series = pd.Series(domains, index=positions, name='domain')
    return pd.get_dummies(series, prefix='domain')

  mutation_effects = data.mutations.loc[:, ['effect_cardio', 'effect_ophtalmo', 'effect_sk', 'position']].groupby('position').aggregate(np.max) > 1

  shared_index = pd.Index(range(1, data.protein_length + 1), name='position')

  residues = pd.concat([
    cv.loc[:, 10.0].rename('cv_10'),
    cv.loc[:, 20.0].rename('cv_20'),
    get_domains().reindex(shared_index, fill_value=False),
    pd.get_dummies(dssp.ss_contextualized, prefix='dssp').reindex(shared_index, fill_value=False),
    gemme_mean,
    rmsf_by_position,
    plddt['alphafold_pruned'].rename('plddt'),
    pae_mean_by_position,
    mutation_effects.reindex(shared_index, fill_value=False)
  ], axis=1)

  return residues


consolidated_residues = compute_consolidated_residues()


__all__ = [
  'consolidated_residues'
]


if __name__ == '__main__':
  print(consolidated_residues)
