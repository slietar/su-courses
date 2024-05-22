import numpy as np
import pandas as pd

from .. import shared, utils


# SecondaryStructureDtype = pd.CategoricalDtype(['helix', 'loop', 'strand'], ordered=True)

# @utils.cache()
def compute_consolidated_residues():
  from .. import data
  from ..cv import cv
  from ..dssp import dssp
  from ..gemme import gemme_all, gemme_orthologs
  from ..pae import pae_mean_by_position
  from ..plddt import plddt
  from ..polymorphism import polymorphism_score
  from ..rmsf import rmsf_by_position
  from ..sasa import sasa


  def get_domains():
    domains = list[str]()
    positions = list[int]()

    for domain in data.domains.itertuples():
      domains += [domain.name] * (domain.end_position - domain.start_position + 1)
      positions += range(domain.start_position, domain.end_position + 1)

    return pd.Series(domains, index=pd.Index(positions, name='position'), name='domain')

  # phenotypes = (data.mutations.loc[:, [
  #   'effect_cardio',
  #   'effect_cutaneous',
  #   'effect_neuro',
  #   'effect_ophtalmo',
  #   'effect_pneumothorax',
  #   'effect_severe',
  #   'effect_sk',
  #   'position'
  # ]].groupby('position').aggregate(np.max)).astype(int).reindex(data.position_index, fill_value=0)
  # # ]].groupby('position').aggregate(np.max) > 0).reindex(data.position_index, fill_value=False)

  native_descriptors = pd.concat([
    cv.loc[:, 10.0].rename('cv_10'),
    cv.loc[:, 20.0].rename('cv_20'),
    gemme_all.mean.rename('gemme_all'),
    gemme_orthologs.mean.rename('gemme_orthologs'),
    rmsf_by_position,
    plddt.alphafold_pruned.rename('plddt'),
    sasa.total.rename('sasa'),
    polymorphism_score
  ], axis='columns', join='inner')

  classification_descriptors = pd.concat([
    data.amino_acids,
    pae_mean_by_position,
    get_domains(),
    # get_domains().astype('category'),
    # pd.get_dummies(get_domains(), prefix='domain'),
    dssp.ss_contextualized.rename('dssp')
  ], axis='columns', join='inner')

  return native_descriptors, classification_descriptors


native_descriptors, classification_descriptors = compute_consolidated_residues()
all_descriptors = native_descriptors.join(classification_descriptors)

descriptor_names = dict(
  cv_10='Variance circulaire\n(seuil 10 Å)',
  cv_20='Variance circulaire\n(seuil 20 Å)',
  gemme_all='GEMME moyen',
  gemme_orthologs=r'$\Delta$ GEMME',
  rmsf='Écart à la  structure\nconsensus',
  plddt='pLDDT',
  sasa='SASA',
  polymorphism='Polymorphisme'
)


__all__ = [
  'all_descriptors',
  'descriptor_names',
  'classification_descriptors',
  'native_descriptors'
]


if __name__ == '__main__':
  print(native_descriptors)
  # print(phenotypes)
