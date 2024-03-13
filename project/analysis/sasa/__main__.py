from copy import deepcopy
from matplotlib import pyplot as plt

from .. import shared
from ..cv import ca_atom_list
from ..pdb import write_pdb
from ..sasa import sasa


new_ca_atom_list = deepcopy(ca_atom_list)

for atom in new_ca_atom_list:
  atom.temp_factor = sasa.total.loc[atom.residue_number]


output_path = shared.output_path / 'sasa'
output_path.mkdir(exist_ok=True)

with (output_path / 'structure.pdb').open('wt') as file:
  write_pdb(file, new_ca_atom_list)


fig, ax = plt.subplots()

ax.set_xlabel('Relative total SASA')
ax.hist(sasa.total, bins=30)

with (output_path / 'histogram.png').open('wb') as file:
  fig.savefig(file, dpi=300)
