import itertools
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pymol import cmd


output_path = Path('output')
output_path.mkdir(exist_ok=True)


with Path('../structure/output/data.pkl').open('rb') as file:
  data = pickle.load(file)

domains = data['domains']


for domain_index, domain in enumerate(domains):
  cmd.load(f'esmfold_output/domains/{domain_index:04}/structure.pdb', f'A{domain_index}')


domain_kinds = ['EGFLike', 'EGFLikeCalciumBinding', 'TB']

for domain_kind in domain_kinds:
  domain_indices = [domain_index for domain_index, domain in enumerate(domains) if domain['kind'] == domain_kind]
  rmsd = np.zeros((len(domain_indices), len(domain_indices)))

  # print(domain_kind, domain_indices[-1], domain_indices[-4])

  for a, b in itertools.combinations(range(len(domain_indices)), 2):
    a_index = domain_indices[a]
    b_index = domain_indices[b]

    # transform=0 -> Do not actually move the atoms
    align_output = cmd.align(f'A{a_index}', f'A{b_index}', transform=0)

    rmsd[a, b] = align_output[0]
    rmsd[b, a] = align_output[0]


  fig, ax = plt.subplots(figsize=(14, 10))

  im = ax.imshow(rmsd, cmap='hot', interpolation='nearest')
  ax.set_title(domain_kind)

  ax.set_xticks(
    labels=[domains[domain_index]['name'] for domain_index in domain_indices],
    ticks=range(len(domain_indices)),
    rotation='vertical'
  )

  ax.set_yticks(
    labels=[domains[domain_index]['name'] for domain_index in domain_indices],
    ticks=range(len(domain_indices))
  )

  ax.tick_params('x', bottom=False, labelbottom=False, labeltop=True)
  ax.tick_params('y', left=False)

  fig.colorbar(im, ax=ax)
  fig.subplots_adjust(top=0.7)

  with (output_path / f'{domain_kind}.png').open('wb') as file:
    fig.savefig(file, dpi=300)


# align() return values
# 1 RMSD
# 2 Atom count in RMSD
# 3 Cycle count
# 4 Initial RMSD?
# 5 Aligned atom count
# 6 Score
# 7 Number of rejected atoms?
