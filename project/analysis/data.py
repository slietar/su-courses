import pickle
from pathlib import Path

import pandas as pd


with (Path(__file__).parent / '../structure/output/data.pkl').open('rb') as file:
  pkl_data = pickle.load(file)


domains = pd.DataFrame.from_records(pkl_data['domains'], index='name')
effect_labels = pkl_data['effect_labels']
exons = pd.DataFrame.from_records(pkl_data['exons'], index='name')
mutations = pd.DataFrame.from_records(pkl_data['mutations'], index='name')
pathogenicity_labels = pkl_data['pathogenicity_labels']
sequence = pkl_data['sequence']
variants = pd.DataFrame.from_records(pkl_data['variants'], index='name')

all_mutations = pd.concat((
  mutations.assign(source='hospital', pathogenicity=1),
  variants.assign(source='gnomad')
), join='inner')


if __name__ == '__main__':
  for df, label in [
    (domains, 'Domains'),
    (exons, 'Exons'),
    (mutations, 'Mutations'),
    (variants, 'Variants')
  ]:
    print(label.upper())
    print(df.head())
    print()
    print()
