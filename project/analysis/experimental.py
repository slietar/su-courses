import requests

from . import data, shared


output_path = shared.root_path / 'experimental-output'
output_path.mkdir(exist_ok=True)

for structure in data.structures.itertuples():
  res = requests.get(f'https://files.rcsb.org/download/{structure.Index}.pdb')

  with (output_path / f'{structure.Index}.pdb').open('wb') as file:
    file.write(res.content)
