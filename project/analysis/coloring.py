from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from pymol import cmd

import pandas as pd

from . import data, shared
from .cv import cv
from .rmsf import pdb, rmsf_arr_by_domain_kind


def set_temp_factor(atoms: pd.DataFrame, values: pd.Series, /, fill_value: Optional[float] = None):
  for atom in atoms.itertuples():
    if atom.residue_seq_number in values:
      val = values.at[atom.residue_seq_number]
    elif fill_value is not None:
      val = fill_value
    else:
      raise ValueError(f'No value for residue {atom.residue_seq_number}')

    atoms.at[atom.atom_serial_number, 'temp_factor'] = val


# with (shared.output_path / 'structures/alphafold3-global/structure.pdb').open() as file:
#   atoms = pdb.load(file)

# domain = data.domains.loc['EGFCB 8']

# with (shared.output_path / f'structures/alphafold-contextualized/{domain.global_index:04}.pdb').open() as file:
#   atoms = pdb.load(file)

# print(domain)
# print(cv.index)
# # set_temp_factor(atoms, 0.5 + cv[(cv.index >= domain.start_position) & (cv.index <= domain.end_position)].loc[:, 10.0], fill_value=0.0)
# set_temp_factor(atoms, cv.loc[:, 30.0], fill_value=0.0)

# with NamedTemporaryFile('w') as file:
#   pdb.dump(atoms, file)

#   cmd.load(file.name, format='pdb')
#   cmd.color('green')
#   cmd.spectrum('b', 'blue_white_red', selection=f'resi {domain.start_position}-{domain.end_position}')
#   cmd.show('surface')
#   cmd.bg_color('white')
#   cmd.set_view ([
#   #    0.863741755,   -0.486851722,    0.130102411,
#   #    0.103257090,    0.423676074,    0.899910510,
#   #   -0.493243873,   -0.763856411,    0.416216344,
#   #    0.000000000,    0.000000000, -240.927978516,
#   #    0.142234802,    2.008576393,   -4.232963562,
#   # -783.610412598, 1265.466186523,  -20.000000000 ])

#        0.618179739,    0.151973933,   -0.771205008,\
#      0.276815474,   -0.960368812,    0.032638110,\
#     -0.735681832,   -0.233656779,   -0.635749280,\
#      0.000000000,    0.000000000, -240.927978516,\
#      0.142234802,    2.008576393,   -4.232963562,\
#   -783.610412598, 1265.466186523,  -20.000000000 ])
#   cmd.save('x.pse')

domain = data.domains.loc['EGFCB 11']
# domain = data.domains.loc['TB 9']
print(domain.global_index)

with (shared.output_path / f'structures/alphafold-pruned/{domain.global_index:04}.pdb').open() as file:
  atoms = pdb.load(file)

# set_temp_factor(atoms, cv.loc[:, 10.0], fill_value=0.0)
rmsf = rmsf_arr_by_domain_kind[domain.kind]
# mean = np.nanmean(rmsf, axis=0)
mean = rmsf[0, :]
filtered_mean = mean[~np.isnan(rmsf[0, :])]
set_temp_factor(atoms, pd.Series(filtered_mean * 10.0, index=pd.Index(range(domain.start_position, domain.end_position + 1), name='position')))

# # atoms.temp_factor = mean[~np.isnan(rmsf[0, :])]

# # print(
# #   len(range(domain.start_position, domain.end_position + 1)),
# #   len(filtered_mean)
# # )

# print(pd.Series(filtered_mean).describe())

# # fig, ax = plt.subplots()
# # ax.plot(filtered_mean)
# # plt.show()


with NamedTemporaryFile('w') as file:
  pdb.dump(atoms, file)

  cmd.load(file.name, format='pdb')
  cmd.spectrum('b', 'blue_white_red')
  cmd.bg_color('white')
#   cmd.set_view ([
#      0.863741755,   -0.486851722,    0.130102411,
#      0.103257090,    0.423676074,    0.899910510,
#     -0.493243873,   -0.763856411,    0.416216344,
#      0.000000000,    0.000000000, -240.927978516,
#      0.142234802,    2.008576393,   -4.232963562,
#   -783.610412598, 1265.466186523,  -20.000000000 ])
  cmd.save('b.pse')



with Path('a.pdb').open('w') as file:
  pdb.dump(atoms, file)
