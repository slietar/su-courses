import functools
import operator

import numpy as np
import pandas as pd

from . import data, utils


@utils.cache()
def compute_polymorphism():
  def map_variants(variants: pd.DataFrame):
    drop_levels = [1, 2, 3]
    mask = ~functools.reduce(operator.or_, (variants.pathogenicity == i for i in drop_levels)).any()

    return (variants.allele_count * mask).sum()

  counts = data.variants.groupby('position').apply(map_variants).reindex(data.position_index, fill_value=0).rename('polymorphism')
  score = counts.apply(lambda x: np.log(x + 1))

  return counts, score

polymorphism_counts, polymorphism_score = compute_polymorphism()


__all__ = [
  'polymorphism_counts',
  'polymorphism_score'
]


if __name__ == '__main__':
  print(polymorphism_counts)
  print(polymorphism_score)
