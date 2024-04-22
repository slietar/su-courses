import pandas as pd

from . import shared


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

  return pd.Series([
    name,
    kind,
    number,
    residues.iloc[0]['n°aa'],
    residues.iloc[-1]['n°aa']
  ], index=[
    'name',
    'kind',
    'number',
    'start_position',
    'end_position'
  ])

adj_check = (df['Structure '] != df['Structure '].shift()).cumsum()
hospital_domains = df.groupby(adj_check).apply(map_domain).astype(dict(end_position='int', start_position='int')).dropna(subset='name').set_index('name')


__all__ = [
  'hospital_domains'
]


if __name__ == '__main__':
  print(hospital_domains)
