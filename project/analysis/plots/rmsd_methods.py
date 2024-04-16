import numpy as np
from matplotlib import pyplot as plt
from pymol import cmd

from .. import data, plots, shared
from ..pymol import PymolAlignment


labels = [
  'AlphaFold global',
  'AlphaFold pruned',
  'ESMFold pruned',
  'ESMFold isolated'
]

alignments = [
  (0, 1),
  (1, 2),
  (2, 3)
]

rmsds = np.empty((len(data.domains), len(alignments)))

cmd.load(shared.output_path / 'structures/alphafold-global/structure.pdb', '0')

for domain_index in data.domains.global_index:
  cmd.load(shared.output_path / f'structures/alphafold-pruned/{domain_index:04}.pdb', '1')
  cmd.load(shared.output_path / f'structures/esmfold-pruned/{domain_index:04}.pdb', '2')
  cmd.load(shared.output_path / f'structures/esmfold-isolated/{domain_index:04}.pdb', '3')

  for alignment_index, (a, b) in enumerate(alignments):
    rmsds[domain_index, alignment_index] = PymolAlignment(cmd.align(f'%{a}', f'%{b}', cutoff=1000, transform=0)).rmsd

  cmd.delete('1')
  cmd.delete('2')
  cmd.delete('3')


fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(rmsds.T, cmap='hot', interpolation='none', vmin=0.0, vmax=10.0)

ax.set_yticks(
  labels=[f'{labels[a]} vs {labels[b]}' for a, b in alignments],
  ticks=np.arange(len(alignments))
)

ax.set_xticks(
  labels=data.domains.name,
  ticks=np.arange(len(data.domains)),
  rotation='vertical'
)

cbar = fig.colorbar(im, ax=ax)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('RMSD (Å)', rotation=270)

fig.subplots_adjust(left=0.3)

with (shared.output_path / 'rmsd_methods.png').open('wb') as file:
  fig.savefig(file)
