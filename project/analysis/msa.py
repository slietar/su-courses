import numpy as np
import pandas as pd

from . import data, shared


# for domain_index, (_, domain) in enumerate(data.domains.iterrows()):
#   if domain.kind == 'EGFCB':
#     print(f'> {domain_index:04}')
#     print(data.sequence[(domain.start_position - 1):domain.end_position])


with (shared.root_path / 'sources/alignments/EGFCB.txt').open() as file:
  file.readline()
  lines = [line.split() for line in file.read().splitlines() if line and not line.startswith(' ')]

  offsets = [1] * len(lines)
  length = len(lines[0][1])

  columns = np.zeros((length, len(lines)), dtype=int)

  for position_index in range(length):
    for line_index, line in enumerate(lines):
      if line[1][position_index] != '-':
        columns[position_index, line_index] = offsets[line_index]
        offsets[line_index] += 1

  msa_ = pd.DataFrame(columns.T, columns=range(1, length + 1), index=[data.domains.iloc[int(line[0])].name for line in lines])
  msa_.sort_index(inplace=True, key=(lambda x: data.domains.loc[x].number))

  msa = {
    'EGFCB': msa_
  }


if __name__ == '__main__':
  print(msa)


__all__ = [
  'msa'
]
