import sys
import numpy as np
import pandas as pd

from .. import data, shared
from ..pdb import read_pdb


with (shared.root_path / 'drive/FBN1_AlphaFold.pdb').open() as file:
  exp, resol, nummdl, chains, ca_atom_list = read_pdb(file, chain='A')


ca_atoms = pd.DataFrame.from_records([vars(target) for target in ca_atom_list], index='residue_number')
ca_atoms.index.rename('position', inplace=True)
ca_atom_coords = np.asarray([ca_atoms.x, ca_atoms.y, ca_atoms.z]).T


def normalize(arr: np.ndarray):
  norm = np.linalg.norm(arr, axis=-1)
  return np.divide(arr, norm[..., None], out=np.zeros_like(arr), where=(norm[..., None] != 0.0))


x = normalize(ca_atom_coords[:, None, :] - ca_atom_coords).sum(axis=1)
cv_arr = 1.0 - np.sqrt((x ** 2).sum(axis=1)) / (len(ca_atom_coords) - 1)

cv = pd.Series(cv_arr, index=ca_atoms.index, name='cv')

del x


__all__ = [
  'cv'
]
