import csv
import pickle
from pathlib import Path
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def conv(x: bytes):
  return x if x != b'NA' else np.nan

length = 2871
threshold = -0.779

with Path('../drive/P35555/P35555_normPred_evolCombi.txt').open() as file:
  file.readline()
  reader = csv.reader(file, delimiter=' ')

  items = [[cell if cell != 'NA' else 'nan' for cell in row] for row in reader]

arr = np.array(items)
residues = [res.upper() for res in arr[:, 0]]

gemme = arr[:, 1:].astype(float)
gemme_mean = np.nanmean(gemme, axis=0)

  # gemme = np.loadtxt(file, converters=conv, skiprows=1, usecols=range(1, length + 1))
  # file.seek(0)
  # residues = np.loadtxt(file, dtype=str, skiprows=1, usecols=[0])

# print(gemme.shape) # => (20, 2871)
# print(residues)

residues_inv = {res: index for index, res in enumerate(residues)}


with Path('../structure/output/data.pkl').open('rb') as file:
  data = pickle.load(file)

variants = pd.DataFrame(data['variants'])
variants_filtered = variants[variants['alternate_residue'].isin(residues) & (variants['clinical_effect'] > 0)]

a = pd.Series(gemme[[residues_inv[res] for res in variants_filtered['alternate_residue']], variants_filtered['protein_position'] - 1], index=variants_filtered.index)
b = pd.Series(gemme_mean[variants_filtered['protein_position'] - 1], index=variants_filtered.index)

# print(pd.DataFrame([a, b]).T)

# print(pd.Series(a, index=variants.index[variants_mask]))
# print(a)


fig, ax = plt.subplots()

print(variants)

ax.axline((0, threshold), color='gray', linestyle='--', slope=0)
ax.axline((threshold, 0), (threshold, 1), color='gray', linestyle='--')

scatter = ax.scatter(b, a, c=variants_filtered['clinical_effect'], cmap='RdYlGn', s=3.5)

ax.set_xlabel('Mean')
ax.set_ylabel('Mutation')

legend = ax.legend(
  *scatter.legend_elements(
    num=range(len(data['pathogenicity_labels'])),
    fmt=FuncFormatter(lambda x, i: data['pathogenicity_labels'][x])
  ),
  loc="upper left",
  title='Pathogenicity'
)

ax.add_artist(legend)
ax.set_title('gnomAD variants')


plt.show()
