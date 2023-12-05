import itertools
from typing import Optional, Sequence

from tqdm import tqdm


def break_point_reversal_sort(input_π: Sequence[int], /):
  π = list(input_π)
  reversals = list[tuple[int, int]]()

  while π != sorted(π):
    k: Optional[int] = None

    current_type = 'inc'
    prev = 0

    l = 0
    r = -1

    for index, a in enumerate([*π, max(π) + 1]):
      if a == prev + 1:
        if (r < 0) and (current_type == 'dec'):
          l = index - 1

        current_type = 'inc'
      elif a != prev - 1:
        if current_type == 'dec':
          k = min(k, prev) if k is not None else prev
        elif (r < 0) and (index > 1) and (prev != index):
          r = index

        current_type = 'dec'

      prev = a

    if k is not None:
      k_index = π.index(k)
      k1_index = π.index(k - 1) if k != min(π) else -1

      l = min(k_index, k1_index) + 1
      r = max(k_index, k1_index) + 1
    else:
      assert r >= 0

    # print(π)
    π = π[0:l] + list(reversed(π[l:r])) + π[r:]

    # print(k, l, r)
    # print(π)
    # print()

    reversals.append((l, r - 1))

  return reversals


# a = list(range(1, 10))
# random.shuffle(a)

# break_point_reversal_sort(a)

# print(break_point_reversal_sort([6, 5, 7, 1, 2, 3, 4, 9, 10, 8]))
# print(break_point_reversal_sort([3, 4, 1, 2]))
# print(break_point_reversal_sort([1, 4, 5, 2, 3]))
# print(break_point_reversal_sort([1, 2, 5, 6, 3, 4]))
# print(break_point_reversal_sort([3, 4, 5, 6, 1, 2]))


def apply_reversals(input_π: Sequence[int], reversals: Sequence[tuple[int, int]], /):
  π = list(input_π)

  for l, r in reversals:
    π = π[0:l] + list(reversed(π[l:(r + 1)])) + π[(r + 1):]

  return π


arr_length = 10
identity_π = list(range(1, arr_length + 1))

i = 0
for π in tqdm(list(itertools.permutations(range(1, arr_length + 1)))):
  i += 1
  # print(π)
  # print(break_point_reversal_sort(π))
  assert apply_reversals(π, break_point_reversal_sort(π)) == identity_π
  # print()

print(i)
