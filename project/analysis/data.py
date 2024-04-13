import pickle
from pathlib import Path

import pandas as pd

from . import shared


with (shared.root_path / 'structure/output/data.pkl').open('rb') as file:
  pkl_data = pickle.load(file)


domain_kinds = pkl_data['domain_kinds']
domains = pd.DataFrame.from_records(pkl_data['domains'], index='name')
effect_labels = pkl_data['effect_labels']
exons = pd.DataFrame.from_records(pkl_data['exons'], index='name')
mutations = pd.DataFrame.from_records(pkl_data['mutations'], index='name')
pathogenicity_labels = pkl_data['pathogenicity_labels']
sequence = ''.join(pkl_data['sequence'])
structures = pd.DataFrame.from_records(pkl_data['structures'], index='id')
variants = pd.DataFrame.from_records(pkl_data['variants'], index='name')

all_mutations = pd.concat([
  mutations.assign(source='hospital', pathogenicity=1),
  variants.assign(source='gnomad')
], join='inner')

gemme_threshold = -0.779
protein_length = len(sequence)

interest_regions = {
  'Neonat': (951, 1362),
  'TB5': (1689, 1762)
}


if __name__ == '__main__':
  for df, label in [
    (domains, 'Domains'),
    (exons, 'Exons'),
    (mutations, 'Mutations'),
    (structures, 'Experimental structures'),
    (variants, 'Variants')
  ]:
    print(label.upper())
    print(df.head())
    print()
    print()


__all__ = [
  'all_mutations',
  'domain_kinds',
  'domains',
  'effect_labels',
  'exons',
  'gemme_threshold',
  'mutations',
  'pathogenicity_labels',
  'protein_length',
  'sequence',
  'structures',
  'variants'
]
