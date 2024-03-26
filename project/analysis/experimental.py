import requests

from . import data, shared


output_path = shared.output_path / 'structures/experimental'
output_path.mkdir(exist_ok=True, parents=True)

for structure in data.structures.itertuples():
  res = requests.get(f'https://files.rcsb.org/download/{structure.Index}.pdb')

  with (output_path / f'{structure.Index}.pdb').open('wb') as file:
    file.write(res.content)
