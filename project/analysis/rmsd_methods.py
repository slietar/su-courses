from matplotlib import pyplot as plt
import numpy as np
from pymol import cmd

from . import data, shared
from .pymol_align import PymolAlignment


alignment_data = np.empty((len(data.domains), 3))

cmd.load(shared.root_path / 'drive/FBN1_AlphaFold.pdb', '0')

for domain_index in range(len(data.domains)):
  cmd.load(shared.root_path / f'esmfold-output/isolated/domains/{domain_index:04}/structure.pdb', '1')
  cmd.load(shared.root_path / f'esmfold-pruning/output/{domain_index:04}.pdb', '2')

  alignment_data[domain_index, 0] = PymolAlignment(cmd.align('0', '1', cutoff=1000)).rmsd
  alignment_data[domain_index, 1] = PymolAlignment(cmd.align('0', '2', cutoff=1000)).rmsd
  alignment_data[domain_index, 2] = PymolAlignment(cmd.align('1', '2', cutoff=1000)).rmsd

  cmd.delete('1')
  cmd.delete('2')


fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(alignment_data.T, cmap='hot', interpolation='none')

ax.set_yticks(
  labels=[
    'AlphaFold global vs ESMFold isolated',
    'AlphaFold global vs ESMFold contextualized',
    'ESMFold isolated vs ESMFold contextualized'
  ],
  ticks=np.arange(3)
)

ax.set_xticks(
  labels=[domain.Index for domain in data.domains.itertuples()],
  ticks=np.arange(len(data.domains)),
  rotation='vertical'
)

cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('RMSD (Å)', rotation=270)

fig.subplots_adjust(left=0.3)

with (shared.output_path / 'rmsd_methods.png').open('wb') as file:
  fig.savefig(file, dpi=300)
