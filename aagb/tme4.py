import itertools
from typing import Optional, Sequence

from tqdm import tqdm


def break_point_reversal_sort(input_π: Sequence[int], /):
  π = list(input_π)
  reversals = list[tuple[int, int]]()
  sorted_π = sorted(π)

  while π != sorted_π:
    k: Optional[int] = None
    increasing = True
    left = 0
    right = -1

    for index, (a, b) in  enumerate(zip([0, *π], [*π, max(π) + 1])):
      if b == a + 1:
        if (right < 0) and not increasing:
          left = index - 1

        increasing = True
      elif b != a - 1:
        if not increasing:
          k = min(k, a) if k is not None else a
        elif (right < 0) and (index > 1) and (a != index):
          right = index

        increasing = False

    if k is not None:
      k_index = π.index(k)
      km1_index = π.index(k - 1) if k != min(π) else -1

      left = min(k_index, km1_index) + 1
      right = max(k_index, km1_index) + 1
    else:
      assert right >= 0

    π = π[0:left] + list(reversed(π[left:right])) + π[right:]
    reversals.append((left, right - 1))

  return reversals


# Test

def apply_reversals(input_π: Sequence[int], reversals: Sequence[tuple[int, int]], /):
  π = list(input_π)

  for l, r in reversals:
    π = π[0:l] + list(reversed(π[l:(r + 1)])) + π[(r + 1):]

  return π


arr_length = 9
identity_π = list(range(1, arr_length + 1))

for π in tqdm(list(itertools.permutations(range(1, arr_length + 1)))):
  assert apply_reversals(π, break_point_reversal_sort(π)) == identity_π
