import csv

import numpy as np
import pandas as pd

from . import data, shared


def conv(x: bytes):
  return x if x != b'NA' else np.nan

threshold = -0.779

with (shared.root_path / 'drive/P35555/P35555_normPred_evolCombi.txt').open() as file:
  file.readline()
  reader = csv.reader(file, delimiter=' ')

  items = [[cell if cell != 'NA' else 'nan' for cell in row] for row in reader]

full_arr = np.array(items)
aas = [res.upper() for res in full_arr[:, 0]]

gemme_arr = full_arr[:, 1:].astype(float)

gemme = pd.DataFrame(gemme_arr.T, columns=aas, index=pd.Series(np.arange(data.protein_length) + 1, name='position'))
gemme_mean = gemme.mean(axis='columns').rename('gemme_mean')


__all__ = [
  'gemme',
  'gemme_arr',
  'gemme_mean'
]


if __name__ == '__main__':
  print('GEMME')
  print(gemme)

  print('\n\nGEMME mean')
  print(gemme_mean)
