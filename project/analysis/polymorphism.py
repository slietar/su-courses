import functools
import operator

import pandas as pd

from . import data


def map_variants(variants: pd.DataFrame):
  drop_levels = [1, 2]
  mask = ~functools.reduce(operator.or_, (variants.pathogenicity == i for i in drop_levels)).any()

  return (variants.allele_count * mask).sum()

polymorphism_score = data.variants.groupby('position').apply(map_variants).reindex(data.position_index, fill_value=0).rename('polymorphism')


__all__ = [
  'polymorphism_score'
]


if __name__ == '__main__':
  print(polymorphism_score)
