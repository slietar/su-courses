import numpy as np
import pandas as pd
import pydssp

from . import data, shared, utils
from .folding_targets import target_domains


atomnum = {'N':0, 'CA': 1, 'C': 2, 'O': 3}

def read_pdbtext_with_checking(pdbstring: str):
    lines = pdbstring.split("\n")
    coords, atoms, resid_old, check = [], None, None, []
    for l in lines:
        if l.startswith('ATOM'):
            iatom = atomnum.get(l[12:16].strip(), None)
            resid = l[21:26]
            if resid != resid_old:
                if atoms is not None:
                    coords.append(atoms)
                    check.append(atom_check)
                atoms, resid_old, atom_check = [], resid, []
            if iatom is not None:
                xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                atoms.append(xyz)
                atom_check.append(iatom)
    if atoms is not None:
        coords.append(atoms)
        check.append(atom_check)
    coords = np.array(coords)
    # check
    assert len(coords.shape) == 3, "Some required atoms [N,CA,C,O] are missing in the input PDB file"
    check = np.array(check)
    assert np.all(check[:,0]==0), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,1]==1), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,2]==2), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    assert np.all(check[:,3]==3), "Order of PDB line may be broken. It's required to be N->CA->C->O w/o any duplicate or lack"
    # output
    return coords


@utils.cache
def compute_dssp():
  ss_contextualized = list[int]()
  ss_pruned = list[int]()
  ss_positions = list[int]()

  for domain in data.domains.itertuples():
    target_domain = target_domains.loc[domain.name]

    with (shared.output_path / f'structures/alphafold-contextualized/{domain.global_index:04}.pdb').open() as file:
      pdb_contextualized = read_pdbtext_with_checking(file.read())

    with (shared.output_path / f'structures/alphafold-pruned/{domain.global_index:04}.pdb').open() as file:
      pdb_pruned = read_pdbtext_with_checking(file.read())


    domain_ss_contextualized = pydssp.assign(pdb_contextualized, out_type='index')
    domain_ss_pruned = pydssp.assign(pdb_pruned, out_type='index')

    ss_contextualized += list(domain_ss_contextualized[(target_domain.rel_start_position - 1):target_domain.rel_end_position])
    ss_pruned += list(domain_ss_pruned)

    ss_positions += range(domain.start_position, domain.end_position + 1)

  with (shared.root_path / 'sources/dssp.csv').open() as file:
    dssp_global = pd.read_csv(file).set_index('position').loc[:, 'secondary index'].rename('ss_global')

  return pd.DataFrame(dssp_global).join(
     pd.DataFrame.from_dict(dict(
      ss_contextualized=ss_contextualized,
      ss_pruned=ss_pruned,
      position=ss_positions
    )).set_index('position'),
    how='outer'
  )

# ).join(dssp_global, how='outer')

dssp = compute_dssp()

dssp_labels = [
  'loop',
  'alpha-helix',
  'beta-strand'
]


__all__ = [
  'dssp',
  'dssp_labels'
]


if __name__ == '__main__':
   print(dssp)
