import math
import random
from typing import Optional


def break_point_reversal_sort(π: list[int]):
  reversals = list[tuple[int, int]]()

  while π != sorted(π):
    k: Optional[int] = None

    current_type = 'inc'
    prev = 0

    l = 0
    r = -1

    for index, a in enumerate([*π, max(π) + 1]):
      if a == prev + 1:
        current_type = 'inc'

        if r < 0:
          l = index - 1
      elif a != prev - 1:
        if current_type == 'dec':
          k = min(k, prev) if k is not None else prev
        elif (r < 0) and (index > 0):
          r = index

        current_type = 'dec'

      prev = a
      # print(a, current_type)

    if k is not None:
      k_index = π.index(k)
      k1_index = π.index(k - 1) if k != min(π) else -1

      l = min(k_index, k1_index) + 1
      r = max(k_index, k1_index) + 1

    # print(π)
    π = π[0:l] + list(reversed(π[l:r])) + π[r:]

    # print(k)
    # print(k, l, r)
    print(π)

    reversals.append((l, r - 1))

  return reversals


# a = list(range(1, 10))
# random.shuffle(a)

# break_point_reversal_sort(a)

# print(break_point_reversal_sort([6, 5, 7, 1, 2, 3, 4, 9, 10, 8]))
print(break_point_reversal_sort([3, 4, 1, 2]))
