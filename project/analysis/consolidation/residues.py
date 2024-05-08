import numpy as np
import pandas as pd

from .. import shared, utils


# SecondaryStructureDtype = pd.CategoricalDtype(['helix', 'loop', 'strand'], ordered=True)

@utils.cache()
def compute_consolidated_residues():
  from .. import data
  from ..cv import cv
  from ..dssp import dssp
  from ..gemme import gemme_mean
  from ..pae import pae_mean_by_position
  from ..plddt import plddt
  from ..polymorphism import polymorphism_score
  from ..rmsf import rmsf_by_position
  from ..sasa import sasa


  def get_domains():
    domains = list[int]()
    positions = list[int]()

    for index, (_, domain) in enumerate(data.domains.iterrows()):
      domains += [index] * (domain.end_position - domain.start_position + 1)
      positions += range(domain.start_position, domain.end_position + 1)

    return pd.Series(domains, index=positions, name='domain')

  phenotypes = (data.mutations.loc[:, [
    'effect_cardio',
    'effect_cutaneous',
    'effect_neuro',
    'effect_ophtalmo',
    'effect_pneumothorax',
    'effect_severe',
    'effect_sk',
    'position'
  ]].groupby('position').aggregate(np.max) > 0).reindex(data.position_index, fill_value=False)

  native_descriptors = pd.concat([
    cv.loc[:, 10.0].rename('cv_10'),
    cv.loc[:, 20.0].rename('cv_20'),
    # get_domains().astype('category'),
    ## get_domains().reindex(data.position_index, fill_value=False),
    ## pd.get_dummies(dssp.ss_contextualized, prefix='dssp').reindex(data.position_index, fill_value=False),
    ## pd.Series(pd.Categorical.from_codes(dssp.ss_contextualized.dropna(), dtype=SecondaryStructureDtype)).reindex(data.position_index).rename('dssp'),
    ## dssp.ss_contextualized.astype(SecondaryStructureDtype.dtype).reindex(data.position_index),
    # dssp.ss_contextualized.astype('category').reindex(data.position_index).rename('dssp'),
    gemme_mean,
    rmsf_by_position,
    sasa.total.rename('sasa'),
    (polymorphism_score + 1).apply(np.log),
    # pd.get_dummies(dssp.ss_contextualized, prefix='dssp').reindex(data.position_index, fill_value=False),
  ], axis='columns').dropna()

  classification_descriptors = pd.concat([
    plddt.alphafold_pruned.rename('plddt'),
    pae_mean_by_position,
    # get_domains(),
    # get_domains().astype('category'),
    # pd.get_dummies(get_domains(), prefix='domain'),
    # dssp.ss_contextualized.rename('dssp')
  ], axis='columns').dropna()

  return native_descriptors, classification_descriptors, phenotypes


native_descriptors, classification_descriptors, phenotypes = compute_consolidated_residues()
all_descriptors = native_descriptors.join(classification_descriptors)


__all__ = [
  'all_descriptors',
  'classification_descriptors',
  'native_descriptors',
  'phenotypes'
]


if __name__ == '__main__':
  print(all_descriptors)
  print(phenotypes)
