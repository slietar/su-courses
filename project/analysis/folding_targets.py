import json

import pandas as pd

from . import data, shared


target_sequences = list[str]()
target_pruning_ranges = list[tuple[int, int]]()

for prev_domain, domain, next_domain in zip(
  [None, *data.domains.iloc[:-1].itertuples()],
  data.domains.itertuples(),
  [*data.domains.iloc[1:].itertuples(), None]
):
  start = prev_domain.start_position - 1 if prev_domain else 0
  end = next_domain.end_position if next_domain else len(data.sequence)

  # Prev domain:    3-5 -> 1-3
  # Current domain: 7-8 -> 5-6
  #
  # 123456789
  # **PPP*CC*
  #   123456

  offset = (prev_domain.start_position - 1) if prev_domain is not None else 0

  target_pruning_ranges.append((
    domain.start_position - offset,
    domain.end_position - offset
  ))

  target_sequences.append(data.sequence[start:end])


target_domains = pd.DataFrame(target_pruning_ranges, columns=['rel_start_position', 'rel_end_position'], index=data.domains.index)

# a = data.domains.end_position - data.domains.start_position
# b = target_domains.rel_end_position - target_domains.rel_start_position

# print(pd.concat([a, b], axis=1))


__all__ = [
  'target_domains'
]


if __name__ == '__main__':
  print(target_domains)

  with (shared.root_path / 'output/target_sequences.json').open('w') as file:
    json.dump(target_sequences, file)
