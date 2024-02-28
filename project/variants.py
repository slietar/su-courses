from pathlib import Path
import pickle

import pandas as pd


with Path('structure/output/data.pkl').open('rb') as file:
  data = pickle.load(file)

mutations_df = pd.DataFrame(data['mutations'])
variants_df = pd.DataFrame(data['variants'])
# variants_df = variants_df.sort_values(by='protein_position')

grouped = variants_df.groupby('protein_position') # .size()

x = grouped['clinical_effect'].mean() # .reset_index(name='avg_clinical_effect')
print(x)

y = mutations_df.join(x, on='position', how='left')

print(y)
print(y['clinical_effect'].mean())

# m = x.idxmax()

# print(variants_df)
# print(variants_df[variants_df['protein_position'] == m])

# print(mutations_df)
