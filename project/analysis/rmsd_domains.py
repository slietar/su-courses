from dataclasses import dataclass
import itertools
import sys

import numpy as np
from matplotlib import pyplot as plt
from pymol import cmd

from .pymol import PymolAlignment

from . import data, shared


@dataclass(frozen=True, kw_only=True)
class Structure:
  kind: int
  label: str
  query: str


structures = list[Structure]()

for domain_index, (_, domain) in enumerate(data.domains.iterrows()):
  name = f'A{domain_index}'
  cmd.load(shared.root_path / f'esmfold-output/contextualized/domains/{domain_index:04}.pdb', name)

  domain_kind_index = data.domain_kinds.index(domain.kind)

  structures.append(Structure(
    kind=(1 << domain_kind_index),
    label=domain.number,
    query=f'%{name}'
  ))

for exp_structure_index, (_, exp_structure) in enumerate(data.structures.iterrows()):
  name = f'B{exp_structure_index}'
  cmd.load(shared.root_path / f'experimental-output/{exp_structure.name}.pdb', name)

  structures.append(Structure(
    kind=sum([
      1 << domain_kind_index
      if any(
        (domain.kind == domain_kind) and
        (domain.start_position <= exp_structure.end_position) and
        (exp_structure.start_position <= domain.end_position)
        for _, domain
        in data.domains.iterrows()
      )
      else 0
      for domain_kind_index, domain_kind in enumerate(data.domain_kinds)
    ]),
    label=exp_structure.name,
    query=f'%{name} and chain A'
  ))

  print(exp_structure.name)

  for _, domain in data.domains.iterrows():
    if (domain.start_position <= exp_structure.end_position) and (exp_structure.start_position <= domain.end_position):
      print(f'  - {domain.kind} {domain.number}')

  # print(exp_structure.name, bin(structures[-1].kind))


# print(structures)
# sys.exit()

output_path = shared.output_path / 'rmsd_domains'
output_path.mkdir(exist_ok=True)

for domain_kind_index, domain_kind in enumerate(data.domain_kinds):
  structs = [struct for struct in structures if (struct.kind & (1 << domain_kind_index)) > 0]
  rmsd = np.zeros((len(structs), len(structs)))

  # print(domain_kind, domain_indices[-1], domain_indices[-4])

  for a, b in itertools.combinations(range(len(structs)), 2):
    struct_a = structs[a]
    struct_b = structs[b]

    # transform=0 -> Do not actually move the atoms
    alignment = PymolAlignment(cmd.align(struct_a.query, struct_b.query, cutoff=1000.0, transform=0))

    rmsd[a, b] = alignment.rmsd
    rmsd[b, a] = alignment.rmsd

  fig, ax = plt.subplots(figsize=(14, 10))

  im = ax.imshow(rmsd, cmap='hot', interpolation='nearest')
  ax.set_title(domain_kind)

  ax.set_xticks(
    labels=[struct.label for struct in structs],
    ticks=range(len(structs)),
    rotation='vertical'
  )

  ax.set_yticks(
    labels=[struct.label for struct in structs],
    ticks=range(len(structs))
  )

  ax.tick_params('x', bottom=False, labelbottom=True, labeltop=True)
  ax.tick_params('y', labelright=True, left=False)

  cbar = fig.colorbar(im, ax=ax)

  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('RMSD (Å)', rotation=270)

  with (output_path / f'{domain_kind}.png').open('wb') as file:
    fig.savefig(file, dpi=300)


# plt.show()
