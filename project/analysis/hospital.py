import pandas as pd

from . import data, shared


with (shared.root_path / 'sources/hospital/structure.csv').open() as file:
  df = pd.read_csv(file)


def map_domain(residues: pd.DataFrame):
  name = residues.iloc[0]['Structure ']

  if isinstance(name, str) and ('#' in name):
    raw_kind, raw_number = name.split('#', maxsplit=1)

    kind = raw_kind.strip()
    number = int(raw_number)
  else:
    kind = pd.NA
    number = pd.NA

  start_position = int(residues.iloc[0]['n°aa'])
  end_position = int(residues.iloc[-1]['n°aa'])

  tolerance = 10

  for unip_domain in data.domains.itertuples():
    if (abs(unip_domain.start_position - start_position) < tolerance) and (abs(unip_domain.start_position - start_position) < tolerance):
      unip_name = unip_domain.name
      break
  else:
    unip_name = pd.NA

  return pd.Series([
    name,
    kind,
    number,
    start_position,
    end_position,
    unip_name
  ], index=[
    'name',
    'kind',
    'number',
    'start_position',
    'end_position',
    'unip_name'
  ])

adj_check = (df['Structure '] != df['Structure '].shift()).cumsum()
hospital_domains = df.groupby(adj_check).apply(map_domain).dropna(subset='name').set_index('name', drop=False)


__all__ = [
  'hospital_domains'
]


if __name__ == '__main__':
  print(hospital_domains)
