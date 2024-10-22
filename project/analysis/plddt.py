import pickle

import pandas as pd

from . import data, shared, utils


@utils.cache()
def compute_plddt():
  series = dict[str, pd.Series]()

  for name, kind in [
    ('alphafold-global', 'global'),
    ('alphafold3-global', 'global'),
    ('alphafold-pruned', 'pruned'),
    ('esmfold-pruned', 'pruned'),
    ('esmfold-isolated', 'isolated'),
  ]:
    with (shared.root_path / 'output/structures' / name / 'plddt.pkl').open('rb') as file:
      raw_plddt = pickle.load(file)

    match kind:
      case 'global':
        plddt_positions = range(1, len(raw_plddt) + 1)
        plddt_values = raw_plddt
      case 'isolated' | 'pruned':
        plddt_positions = list[int]()
        plddt_values = list[float]()

        for domain_plddt, domain in zip(raw_plddt, data.domains.itertuples()):
          if domain_plddt:
            plddt_positions += range(domain.start_position, domain.end_position + 1)
            plddt_values += domain_plddt
      case _:
        continue

    name_repl = name.replace('-', '_')
    series[name_repl] = pd.Series(plddt_values, index=pd.Index(plddt_positions, name='position'), name=name_repl)

  return pd.concat(series, axis='columns')

plddt = compute_plddt()


__all__ = [
  'plddt'
]


if __name__ == '__main__':
  print(plddt)
