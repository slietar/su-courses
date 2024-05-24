import hashlib
from pathlib import Path


with Path('output/prefix').open('rb') as file:
  prefix = file.read()

with Path('output/a').open('rb') as file:
  a = file.read()

with Path('output/b').open('rb') as file:
  b = file.read()

with Path('output/c').open('rb') as file:
  c = file.read()

with Path('output/d').open('rb') as file:
  d = file.read()


print(hashlib.md5(prefix + a).hexdigest())
print(hashlib.md5(prefix + b).hexdigest())
print()

print(hashlib.md5(prefix + c).hexdigest())
print(hashlib.md5(prefix + d).hexdigest())
print()

print(hashlib.md5(prefix + a + prefix + c).hexdigest())
print(hashlib.md5(prefix + a + prefix + d).hexdigest())
