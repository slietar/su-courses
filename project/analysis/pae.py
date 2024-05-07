import json

import numpy as np
import pandas as pd

from . import data, shared, utils
from .folding_targets import target_domains


@utils.cache()
def compute_pae():
  def get_domain_pae(domain_index: int):
    with (shared.root_path / f'sources/alphafold-contextualized/{domain_index:04}/main_scores_rank_001_alphafold2_ptm_model_1_seed_000.json').open() as file:
      return np.array(json.load(file)['pae'])

  pae = pd.Series([get_domain_pae(domain_index) for domain_index in data.domains.global_index], index=data.domains.index, name='pae')

  positions = list[int]()
  residue_inter = list[float]()
  residue_intra = list[float]()

  for domain_info in target_domains.join(pae).join(data.domains).itertuples():
    start = domain_info.rel_start_position - 1
    stop = domain_info.rel_end_position

    positions += range(domain_info.start_position, domain_info.end_position + 1)
    residue_inter += list(np.r_[domain_info.pae[:start, start:stop], domain_info.pae[stop:, start:stop]].mean(axis=0))
    residue_intra += list(domain_info.pae[start:stop, start:stop].mean(axis=0))

  pae_by_position = pd.DataFrame.from_dict(
    dict(pae_inter=residue_inter, pae_intra=residue_intra, position=positions)
  ).set_index('position')

  return pae, pae_by_position

pae, pae_mean_by_position = compute_pae()


__all__ = [
  'pae',
  'pae_mean_by_position'
]


if __name__ == '__main__':
  print(pae_mean_by_position)
