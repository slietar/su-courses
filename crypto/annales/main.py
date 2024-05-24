from typing import Callable


def feistel(f: Callable[[int], int], l: int, r: int):
  return (
    r,
    l ^ f(r)
  )

l = 0b1101
r = 0b0011

# print(f'{(l ^ r ^ 0b1011):04b}')
# print(f'{(l ^ 0b0001):04b}')

l = 0b1010
r = 0b1011

for f in [
  lambda x: x ^ 0b1011,
  lambda x: x ^ 0b0101 ^ 0b1111,
]:
  l, r = feistel(f, l, r)

print(f'{l:04b} {r:04b}')
