import json

import numpy as np
import pandas as pd

from . import data, shared, utils
from .folding_targets import target_domains


@utils.cache
def compute_pae():
  def get_domain_pae(domain_index: int):
    with (shared.root_path / f'sources/alphafold-contextualized/{domain_index:04}/main_scores_rank_001_alphafold2_ptm_model_1_seed_000.json').open() as file:
      return np.array(json.load(file)['pae'])

  pae = pd.Series([get_domain_pae(domain_index) for domain_index in data.domains.global_index], index=data.domains.index, name='pae')

  positions = list[int]()
  residue_pae1 = list[float]()
  residue_pae2 = list[float]()

  for domain_info in target_domains.join(pae).join(data.domains).itertuples():
    sl = slice(domain_info.rel_start_position - 1, domain_info.rel_end_position)

    positions += range(domain_info.start_position, domain_info.end_position + 1)
    residue_pae1 += list(domain_info.pae[sl, :].mean(axis=1))
    residue_pae2 += list(domain_info.pae[:, sl].mean(axis=0))

  pae_by_position = pd.DataFrame.from_dict(
    dict(pae1=residue_pae1, pae2=residue_pae2, position=positions)
  ).set_index('position')

  return pae, pae_by_position

pae, pae_mean_by_position = compute_pae()


__all__ = [
  'pae',
  'pae_mean_by_position'
]


if __name__ == '__main__':
  print(pae_mean_by_position)
