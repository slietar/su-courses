from typing import IO, Literal

import numpy as np
import pandas as pd

from . import data, shared, utils
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


def get_aligned_residue_coords(domain_kind: str, *, mode: Literal['ca', 'mean'] = 'mean'):
  from pymol import cmd

  all_residue_coords = np.zeros((*msa[domain_kind].shape, 3)) * np.nan # (domains, residues, xyz)
  domains = data.domains[data.domains.kind == domain_kind]

  for relative_domain_index, domain in enumerate(domains.itertuples()):
    sequence_alignment = msa[domain_kind].loc[domain.Index]

    name = ('R' if relative_domain_index < 1 else 'M')
    path = shared.root_path / f'output/structures/alphafold-pruned/{domain.global_index:04}.pdb'


    # Get structure alignment

    cmd.load(path, name)

    # If this domain is not the first, align it with respect to the first
    if relative_domain_index > 0:
      cmd.align('M', 'R', cutoff=1000.0, transform=1)

    transformation = PymolTransformation(cmd.get_object_matrix(name))

    # If this domain is not the first, delete it
    if relative_domain_index > 0:
      cmd.delete('M')


    # Get atom positions

    with path.open() as file:
      atoms = parse_pdb_atoms(file)

    first_residue_seq_number = atoms.residue_seq_number.min()

    for aln_residue_index in range(all_residue_coords.shape[1]):
      domain_residue_index = sequence_alignment[aln_residue_index + 1]

      if domain_residue_index > 0:
        # -1 because domain_residue_index starts at 1
        residue_mask = atoms.residue_seq_number == (first_residue_seq_number + domain_residue_index - 1)

        match mode:
          case 'ca': # Use CA
            residue_coords = atoms.loc[residue_mask & (atoms.atom_name == 'CA')].loc[:, ['x', 'y', 'z']]
          case 'mean': # Average over all atoms
            residue_coords = atoms.loc[residue_mask].loc[:, ['x', 'y', 'z']].mean()

        all_residue_coords[relative_domain_index, aln_residue_index, :] = transformation.apply(residue_coords)

  all_residue_coords_mean = np.nanmean(all_residue_coords, axis=0)
  all_residue_rmsf = np.sqrt(((all_residue_coords - all_residue_coords_mean) ** 2).sum(axis=2))


  residue_positions = [position for domain in domains.itertuples() for position in range(domain.start_position, domain.end_position + 1)]
  residue_rmsf = all_residue_rmsf.ravel()
  residue_rmsf = residue_rmsf[~np.isnan(residue_rmsf)]

  rmsf_by_position = pd.Series(residue_rmsf, index=pd.Index(residue_positions, name='position'), name='rmsf')

  return all_residue_rmsf, rmsf_by_position


@utils.cache()
def compute_rmsf():
  result = { domain_kind: get_aligned_residue_coords(domain_kind) for domain_kind in data.domain_kinds }
  rmsf_arr_by_domain_kind = { domain_kind: rmsf_arr for domain_kind, (rmsf_arr, _) in result.items() }
  rmsf_by_position = pd.concat((rmsf_by_position for _, rmsf_by_position in result.values()), axis=0)

  return rmsf_arr_by_domain_kind, rmsf_by_position

rmsf_arr_by_domain_kind, rmsf_by_position = compute_rmsf()


__all__ = [
  'rmsf_arr_by_domain_kind',
  'rmsf_by_position'
]


if __name__ == '__main__':
  print(rmsf_by_position)
