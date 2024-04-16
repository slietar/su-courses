from typing import IO, Literal

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymol import cmd

from . import data, plots, shared
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
  all_residue_coords = np.zeros((*msa[domain_kind].shape, 3)) * np.nan # (domains, residues, xyz)
  domains = data.domains[data.domains.kind == domain_kind]

  for relative_domain_index, domain in enumerate(domains.itertuples()):
    sequence_alignment = msa[domain_kind].loc[domain.Index]

    name = ('R' if relative_domain_index < 1 else 'M')
    path = shared.root_path / f'output/structures/alphafold-pruned/{domain.Index:04}.pdb'


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

  return domains, all_residue_rmsf

  positions = list[int]()
  rmsf = list[float]()

  for relative_domain_index, (_, _, domain) in enumerate(domains_):
    offset = 0

    for aln_residue_index in range(all_residue_coords.shape[1]):
      domain_residue_index = sequence_alignment[aln_residue_index + 1]

      if domain_residue_index > 0:
        diff = np.sqrt((all_residue_coords_mean[aln_residue_index, :] - all_residue_coords[relative_domain_index, aln_residue_index, :]).mean())
        positions.append(domain.start_position + offset)
        rmsf.append(diff)

        offset += 1

  print(pd.Series(rmsf, index=positions))

  return all_residue_coords


if __name__ == '__main__':
  output_path = shared.output_path / 'rmsf'
  output_path.mkdir(exist_ok=True)

  for domain_kind in data.domain_kinds:
    domains, rmsf_arr = get_aligned_residue_coords(domain_kind)
    # rmsf_arr = rmsf_arr[:, :5]
    # rmsf_arr[0, :] = 100

    fig, ax = plt.subplots()
    fig.set_figheight(8.0)

    divider = make_axes_locatable(ax)

    im = ax.imshow(rmsf_arr, extent=(0.5, rmsf_arr.shape[1] + 0.5, 0, rmsf_arr.shape[0]), aspect='auto', cmap='hot')

    ax1: Axes = divider.append_axes('top', 1.2, pad=0.1, sharex=ax)
    ax1.plot(range(1, rmsf_arr.shape[1] + 1), np.nanmean(rmsf_arr, axis=0))
    ax1.xaxis.set_tick_params(bottom=False, labelbottom=False)
    ax1.grid()

    cbar = fig.colorbar(im, ax=ax)

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('RMSF (Å)', rotation=270)

    ax.set_yticks(
      labels=reversed([str(number) for number in domains['number']]),
      ticks=(np.arange(len(domains)) + 0.5)
    )

    # ax.yaxis.set_tick_params(left=False)


    with (output_path / f'{domain_kind}.png').open('wb') as file:
      fig.savefig(file)
