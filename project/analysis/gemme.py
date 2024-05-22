import csv
from dataclasses import dataclass
import functools
from typing import IO

import numpy as np
import pandas as pd

from . import data, shared, utils


@dataclass
class GEMMEData:
  array: np.ndarray
  dataframe: pd.DataFrame

  @functools.cached_property
  def mean(self):
    return self.dataframe.mean(axis='columns').rename('gemme_mean')

def load_gemme(file: IO[str], /):
  file.readline()
  reader = csv.reader(file, delimiter=' ')

  items = [[cell if cell != 'NA' else 'nan' for cell in row] for row in reader]
  full_arr = np.array(items)
  aas = [res.upper() for res in full_arr[:, 0]]

  arr = full_arr[:, 1:].astype(float)
  dataframe = pd.DataFrame(arr.T, columns=aas, index=data.position_index)

  return GEMMEData(arr, dataframe)


@utils.cache()
def compute_gemme():
  with (shared.root_path / 'drive/P35555/P35555_normPred_evolCombi.txt').open() as file:
    gemme_all = load_gemme(file)

  with (shared.root_path / 'sources/gemme_orthologs.txt').open() as file:
    gemme_orthologs = load_gemme(file)
    # gemme_orthologs.array = gemme_all.array - gemme_orthologs.array
    # gemme_orthologs.dataframe = gemme_all.dataframe - gemme_orthologs.dataframe

  def get_mutations_gemme(mutations: pd.DataFrame, /):
    return pd.concat([
      mutations.apply(lambda mutation: gemme_all.dataframe.at[mutation.position, mutation.alternate_aa], axis='columns').rename('gemme_all'),
      mutations.apply(lambda mutation: gemme_orthologs.dataframe.at[mutation.position, mutation.alternate_aa], axis='columns').rename('gemme_orthologs')
    ], axis='columns')

  return gemme_all, gemme_orthologs, get_mutations_gemme(data.mutations), get_mutations_gemme(data.variants)


gemme_all, gemme_orthologs, gemme_mutations, gemme_variants = compute_gemme()

# Deprecated
gemme = gemme_all.dataframe
gemme_arr = gemme_all.array
gemme_mean = gemme_all.mean


__all__ = [
  'gemme_all',
  'gemme_orthologs'
]


if __name__ == '__main__':
  print('GEMME all')
  print(gemme_all.dataframe.join(gemme_all.mean))

  print('\n\nGEMME orthologs')
  print(gemme_orthologs.dataframe.join(gemme_orthologs.mean))
