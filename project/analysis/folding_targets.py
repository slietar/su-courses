import json

from . import data, shared


target_sequences = list[str]()
target_ranges = list[tuple[int, int]]()

for prev_domain, domain, next_domain in zip(
  [None, *data.domains.iloc[:-1].itertuples()],
  data.domains.itertuples(),
  [*data.domains.iloc[1:].itertuples(), None]
):
  start = prev_domain.start_position - 1 if prev_domain else 0
  end = next_domain.end_position if next_domain else len(data.sequence)

  # target_ranges.append(())

  target_sequences.append(data.sequence[start:end])


if __name__ == '__main__':
  with (shared.root_path / 'folding/target_sequences.json').open('w') as file:
    json.dump(target_sequences, file)
