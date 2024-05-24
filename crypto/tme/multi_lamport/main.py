import hashlib
import json
from pathlib import Path
import sys

from lamport import LamportPrivateKey, LamportPublicKey


full = (1 << 256) - 1

found0 = 0
found1 = 0

# key = np.empty((256, 2), dtype=int)
# key0 = [0 for _ in range(256)]
# key1 = [0 for _ in range(256)]
key_bytes = [bytes() for _ in range(256 * 2)]

for path in Path('messages').glob('*.json'):
  with path.open() as file:
    data = json.load(file)

  message = int.from_bytes(hashlib.sha256(data['message'].encode()).digest())
  signature = bytes.fromhex(data['signature'])

  found0 |= full ^ message
  found1 |= message

  for bit in range(256):
    # print(int.from_bytes(signature[(bit * 32):(bit * 32 + 32)]))
    # key[bit, int((message & (1 << bit)) > 0)] = int.from_bytes(signature[(bit * 32):(bit * 32 + 32)])

    sig_value = signature[(bit * 32):(bit * 32 + 32)]
    msg_value = int((message & (1 << bit)) > 0)

    key_bytes[bit * 2 + msg_value] = sig_value

    # if (message & (1 << bit)) > 0:
    #   key1[bit] = bit_value
    # else:
    #   key0[bit] = bit_value

  if (found0 == full) & (found1 == full):
    break
else:
  raise RuntimeError

private_key = LamportPrivateKey(bytes().join(key_bytes))

with Path('public_robertenglish.pem').open() as file:
  assert LamportPublicKey.loads(file.read()).key == private_key.pk().key


print(private_key.sign(sys.argv[1].encode()))
