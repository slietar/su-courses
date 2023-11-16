import random
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def print_arr(arr: list[int], n: Optional[int] = None):
  n = n if n is not None else len(arr)
  items = set((y, x) for y, x in enumerate(arr))

  for y in range(n):
    for x in range(n):
      if (x, y) in items:
        print('[X]', end=str())
      else:
        print(' · ', end=str())

    print()

  print('---' * n)


def is_cell_free(x: int, y: int, arr: list[int]):
  if x in arr:
    return False

  for ay, ax in enumerate(arr):
    if abs(ay - y) == abs(ax - x):
      return False

  return True


def find_arrangement1(n: int):
  arr = list[int]()
  y = 0

  while 0 <= y < n:
    for x in range(arr.pop() + 1 if arr and (len(arr) > y) else 0, n):
      if is_cell_free(x, y, arr):
        break
    else:
      y -= 1
      continue

    arr.append(x)
    y += 1
    # print_arr(arr, n)

  return arr


def find_arrangement2(n: int):
  attempt = 0

  while True:
    attempt += 1

    arr = list[int]()

    for y in range(n):
      xs = [x for x in range(n) if is_cell_free(x, y, arr)]

      if not xs:
        # print_arr(arr, n)
        break

      arr.append(random.choice(xs))
    else:
      break

  return arr, attempt


print_arr(find_arrangement1(7))


if False:
  ns = list(range(4, 24))
  data = np.zeros(len(ns))
  sample_count = 50

  for n_index, n in enumerate(ns):
    data[n_index] = sum(find_arrangement2(n)[1] for _ in range(sample_count)) / sample_count


  fig, ax = plt.subplots()

  ax.plot(ns, data, label='Average number of attempts')
  ax.grid()

  fig.savefig('plot.png')
