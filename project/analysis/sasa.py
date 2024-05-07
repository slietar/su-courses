import freesasa
import pandas as pd

from . import data, shared, utils
from .folding_targets import target_domains


@utils.cache()
def compute_sasa():
  records = list()

  for domain in data.domains.join(target_domains).itertuples():
    structure = freesasa.Structure(str(shared.root_path / f'output/structures/alphafold-contextualized/{domain.global_index:04}.pdb'))
    result = freesasa.calc(structure)

    # area.residueNumber starts at 1
    records += [dict(
      position=(domain.start_position + rel_position - domain.rel_start_position),

      apolar=area.relativeApolar,
      main_chain=area.relativeMainChain,
      polar=area.relativePolar,
      side_chain=area.relativeSideChain,
      total=area.relativeTotal,
    ) for area in result.residueAreas()['A'].values() if domain.rel_start_position <= (rel_position := int(area.residueNumber)) <= domain.rel_end_position]


  return pd.DataFrame.from_records(records, index='position')

sasa = compute_sasa()


__all__ = [
  'sasa'
]


if __name__ == '__main__':
  print(sasa)
