from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .pdb import read_pdb, write_pdb


with Path('../drive/FBN1_AlphaFold.pdb').open('rt') as file:
  exp, resol, nummdl, chains, atom_list = read_pdb(file, chain='A')


atoms = pd.DataFrame.from_records([vars(target) for target in atom_list], index='residue_number')
atom_coords = np.asarray([atoms.x, atoms.y, atoms.z]).T


def normalize(arr: np.ndarray):
  norm = np.linalg.norm(arr, axis=-1)
  return np.divide(arr, norm[..., None], out=np.zeros_like(arr), where=(norm[..., None] != 0.0))

# target = atom_coords[5]
# x = normalize(atom_coords - target).sum(axis=0)
# cv = 1.0 - np.sqrt((x ** 2).sum()) / (len(atom_coords) - 1)


x = normalize(atom_coords[:, None, :] - atom_coords).sum(axis=1)
cv = 1.0 - np.sqrt((x ** 2).sum(axis=1)) / (len(atom_coords) - 1)

del x

# print(x.shape)
# print(y.shape)
# print(cv)

for index, atom in enumerate(atom_list):
  atom.temp_factor = cv[index]


output_path = Path('output')
output_path.mkdir(exist_ok=True)

# with (output_path / 'structure.pdb').open('wt') as file:
  # write_pdb(file, atom_list)


fig, ax = plt.subplots()
ax.hist(cv)

with (output_path / 'histogram.png').open('wb') as file:
  fig.savefig(file, dpi=300)
