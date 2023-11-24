import math
import random


def break_point_reversal_sort(π: list[int]):
  reversals = list[tuple[int, int]]()

  while π != sorted(π):
    k = math.inf

    current_type = 'inc'
    prev = 0

    for a in [*π, max(π) + 1]:
      if a == prev + 1:
        current_type = 'inc'
      elif a != prev - 1:
        if current_type == 'dec':
          k = min(k, prev)
          print('calc')

        current_type = 'dec'
      
      prev = a
      print(a, current_type)

    break

    # for a, b in zip(π[:-1], π[1:]):
    #   if a - 1 == b:
    #     k = min(k, b)
    
    if not math.isfinite(k):
      pass
    
    k_index = π.index(k)
    k1_index = π.index(k - 1) if k != min(π) else -1

    l = min(k_index, k1_index) + 1
    r = max(k_index, k1_index) + 1

    π = π[0:l] + list(reversed(π[l:r])) + π[r:]
    print(π)

    reversals.append((l, r - 1))
  
  return reversals


# a = list(range(1, 10))
# random.shuffle(a)

# break_point_reversal_sort(a)

print(break_point_reversal_sort([6, 5, 7, 1, 2, 3, 4, 9, 10, 8]))