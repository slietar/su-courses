from pathlib import Path
from tempfile import NamedTemporaryFile
from pymol import cmd

import pandas as pd

from . import data, shared
from .cv import cv
from .rmsf import pdb


def set_temp_factor(atoms: pd.DataFrame, values: pd.Series, /, fill_value: float):
  for atom in atoms.itertuples():
    atoms.at[atom.atom_serial_number, 'temp_factor'] = values.at[atom.residue_seq_number] if atom.residue_seq_number in values else fill_value


# with (shared.output_path / 'structures/alphafold3-global/structure.pdb').open() as file:
#   atoms = pdb.load(file)

domain = data.domains.loc['EGFCB 8']

with (shared.output_path / f'structures/alphafold-contextualized/{domain.global_index:04}.pdb').open() as file:
  atoms = pdb.load(file)

# set_temp_factor(atoms, 0.5 + cv[(cv.index >= domain.start_position) & (cv.index <= domain.end_position)].loc[:, 10.0], fill_value=0.0)
set_temp_factor(atoms, cv.loc[:, 10.0], fill_value=0.0)

with NamedTemporaryFile('w') as file:
  pdb.dump(atoms, file)

  cmd.load(file.name, format='pdb')
  cmd.spectrum('b', 'blue_white_red')
  cmd.show('surface')
  cmd.bg_color('white')
  cmd.set_view ([
     0.863741755,   -0.486851722,    0.130102411,
     0.103257090,    0.423676074,    0.899910510,
    -0.493243873,   -0.763856411,    0.416216344,
     0.000000000,    0.000000000, -240.927978516,
     0.142234802,    2.008576393,   -4.232963562,
  -783.610412598, 1265.466186523,  -20.000000000 ])
  cmd.save('a.pse')


with Path('a.pdb').open('w') as file:
  pdb.dump(atoms, file)
