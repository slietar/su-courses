import numpy as np


isclosereal = lambda x: np.isclose(x.imag, 0)

def group(x: np.ndarray, /):
  current_item = x[0, ...]
  current_count = 1

  for index in range(1, x.shape[0]):
    if x[index, ...] != current_item:
      yield current_item, current_count
      current_item = x[index, ...]
      current_count = 1
    else:
      current_count += 1

  yield current_item, current_count
