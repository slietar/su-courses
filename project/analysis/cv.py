import pickle

import pandas as pd

from . import data, shared


with (shared.root_path / 'output/cv.pkl').open('rb') as file:
  raw_cv = pickle.load(file)


entries_cv = list[float]()
entries_cutoff = list[float]()
entries_position = list[int]()

for (_, domain), cv_domain in zip(data.domains.iterrows(), raw_cv['data']):
  for (cutoff, cv_cutoff) in zip(raw_cv['cutoffs'], cv_domain):
    entries_cutoff += [cutoff] * (domain.end_position - domain.start_position + 1)
    entries_cv += cv_cutoff
    entries_position += range(domain.start_position, domain.end_position + 1)

# print(len(entries_cv))
# print(len(entries_cutoff))
# print(len(entries_position))

cv = pd.DataFrame.from_dict(dict(
  cutoff=entries_cutoff,
  cv=entries_cv,
  position=entries_position
))

cv = cv.pivot(index='position', columns='cutoff', values='cv')


if __name__ == '__main__':
  print(cv)


__all__ = [
  'cv'
]
