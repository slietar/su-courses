import math
from pathlib import Path



with Path('all_flags.txt').open() as file:
  lines = [line for line in file.read().split('\n') if line]

xs = []
ys = []

for line in lines:
  mac = bytes.fromhex(line.split()[0].split('|')[1])
  x = int.from_bytes(mac[0:8])
  a = int.from_bytes(mac[8:16])
  b = int.from_bytes(mac[16:24])
  c = int.from_bytes(mac[24:32])

  xs.append(x)
  ys.append(b)

  # if len(xs) >= 9:
  if len(xs) >= 17:
    break


p = 2 ** 64 - 59

num = [math.prod(x_m for m, x_m in enumerate(xs) if m != j) for j, _ in enumerate(zip(xs, ys))]
den = [math.prod(x_m - x_j for m, x_m in enumerate(xs) if m != j) for j, (x_j, _) in enumerate(zip(xs, ys))]

den1 = math.prod(den)

num1 = sum((n_i * den1 * y_i % p) * pow(den_i, -1, p) for n_i, y_i, den_i in zip(num, ys, den))
sigma = (num1 * pow(den1, -1, p) + p) % p

print(sigma.to_bytes(8).hex())
