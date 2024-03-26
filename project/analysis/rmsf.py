from typing import IO
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymol import cmd

from . import data, shared
from .msa import msa
from .pymol import PymolTransformation


def parse_pdb_atoms(file: IO[str], /):
  def process_line(line: str):
    return [item for item in [
      int(line[6:11]),
      line[12:16].strip(),
      line[16].strip(),
      line[17:20].strip(),
      line[21],
      int(line[22:26]),
      line[26].strip(),
      float(line[30:38]),
      float(line[38:46]),
      float(line[46:54]),
      float(line[54:60]),
      float(line[60:66]),
      line[72:76].strip(),
      line[76:78].strip(),
      line[78:80].strip()
    ]]

  lines = (process_line(line) for line in file.readlines() if line.startswith('ATOM'))

  return pd.DataFrame(lines, columns=['atom_serial_number', 'atom_name', 'alt_loc_ind', 'residue_name', 'chain_id', 'residue_seq_number', 'code', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'segment_id', 'element_symbol', 'charge'])




all_residue_coords = np.zeros((*msa['EGFCB'].shape, 3)) * np.nan # (domains, residues, dimension)

for relative_domain_index, (domain_index, domain_name, domain) in enumerate((domain_index, domain_name, domain) for domain_index, (domain_name, domain) in enumerate(data.domains.iterrows()) if domain.kind == 'EGFCB'):
  sequence_alignment = msa['EGFCB'].loc[domain_name]

  name = ('R' if relative_domain_index < 1 else 'M')
  path = shared.root_path / f'output/structures/alphafold-pruned/{domain_index:04}.pdb'

  # if relative_domain_index == 1:
  #   path = Path('/Users/simon/Downloads/p.pdb')


  # Get structure alignment

  cmd.load(path, name)

  if relative_domain_index > 0:
    cmd.align('M', 'R', cutoff=1000.0, transform=1)

  transformation = PymolTransformation(cmd.get_object_matrix(name))

  if relative_domain_index > 0:
    cmd.delete('M')


  # Get atom positions

  with path.open() as file:
    atoms = parse_pdb_atoms(file)

  first_residue_seq_number = atoms.residue_seq_number.min()

  for global_residue_index in range(all_residue_coords.shape[1]):
    domain_residue_index = sequence_alignment[global_residue_index + 1]

    if domain_residue_index > 0:
      # -1 because domain_residue_index starts at 1
      residue_mask = atoms.residue_seq_number == first_residue_seq_number + domain_residue_index - 1

      if 0: # Use CA
        residue_coords = atoms.loc[residue_mask & (atoms.atom_name == 'CA')].loc[:, ['x', 'y', 'z']]
      else: # Average over all atoms
        residue_coords = atoms.loc[residue_mask].loc[:, ['x', 'y', 'z']].mean()

      all_residue_coords[relative_domain_index, global_residue_index, :] = transformation.apply(residue_coords)


diff = np.sqrt((all_residue_coords - np.nanmean(all_residue_coords, axis=0)).sum(axis=2))
rmsf = np.sqrt(np.nanmean(diff ** 2, axis=0))

print(rmsf)

fig, ax = plt.subplots()

ax.plot(rmsf)
ax.set_xlabel('Position')
ax.set_ylabel('RMSF (Å)')


output_path = shared.output_path / 'rmsf'
output_path.mkdir(exist_ok=True)

with (output_path / 'EGFCB.png').open('wb') as file:
  fig.savefig(file)
