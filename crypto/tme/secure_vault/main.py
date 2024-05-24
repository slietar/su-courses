import hashlib


iv = 0x6fc54dee6256c269d5fd93e3263aad5e


def key_expansion(seed: bytes):
  state = seed
  output = bytes()

  for _ in range(8):
    state = hashlib.sha256(state).digest()
    output += state[:4]

  return output


for seed in range(2 ** 16):
  exp = key_expansion(seed.to_bytes(2))
  if exp[16:32] == iv.to_bytes(16):
    print('Ok')
    break

print('IV', iv.to_bytes(16).hex())
print('Key', exp[0:16].hex())
