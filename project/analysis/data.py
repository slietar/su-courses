import pickle
from pathlib import Path

import pandas as pd

from . import shared


with (shared.root_path / 'structure/output/data.pkl').open('rb') as file:
  pkl_data = pickle.load(file)


gemme_threshold = -0.779
sequence = ''.join(pkl_data['sequence'])
protein_length = len(sequence)

position_index = pd.Index(range(1, protein_length + 1), name='position')

amino_acids = pd.Series(pkl_data['sequence'], index=position_index, name='amino_acid')
domain_kinds = pkl_data['domain_kinds']
domains = pd.DataFrame.from_records(pkl_data['domains']).set_index('name', drop=False)
exons = pd.DataFrame.from_records(pkl_data['exons']).set_index('name', drop=False)
mutations = pd.DataFrame.from_records(pkl_data['mutations']).set_index('name', drop=False)
pathogenicity_labels = pkl_data['pathogenicity_labels']
structures = pd.DataFrame.from_records(pkl_data['structures']).set_index('id', drop=False)

all_variants = pd.DataFrame.from_records(pkl_data['variants']).set_index('name', drop=False)
variants = all_variants[all_variants.annotation == 'missense_variant']

all_mutations = pd.concat([
  mutations.assign(source='hospital', pathogenicity=1),
  variants.assign(source='gnomad')
], join='inner')

interest_regions = pd.DataFrame.from_records([
  dict(name='Neonat', start_position=951, end_position=1362),
  dict(name='TB5', start_position=1689, end_position=1762)
]).set_index('name', drop=False)


if __name__ == '__main__':
  for df, label in [
    (amino_acids, 'Amino acids'),
    (domains, 'Domains'),
    (exons, 'Exons'),
    (interest_regions, 'Interest regions'),
    (mutations, 'Mutations'),
    (structures, 'Experimental structures'),
    (variants, 'Variants')
  ]:
    print(label.upper())
    print(df)
    print()
    print()


__all__ = [
  'all_mutations',
  'amino_acids',
  'domain_kinds',
  'domains',
  'exons',
  'gemme_threshold',
  'interest_regions',
  'mutations',
  'pathogenicity_labels',
  'protein_length',
  'sequence',
  'structures',
  'variants'
]
