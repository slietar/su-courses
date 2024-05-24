import json
import os
import sys

import requests


encrypted = bytes.fromhex('b555d5f0764ef70c511197fd3addadb2a2ec6cf7dfb1f6406c08cb243d714872509bc9bf42a6f006c78778f6049395e103fa0e6f1a57445dea8c1b0c4466de063ae7cf67abb07891b594888fec4e89fd')
world_id = 'f45988237fe63c0ad5f7916df487ef94'


def oracle(ciphertexts: list[bytes]):
  res = requests.post('http://m1.tme-crypto.fr:8888', data=json.dumps({
    'jsonrpc': '2.0',
    'id': 3,
    'method': 'chip.whisperer',
    'params': {
      'ciphertexts': [c.hex() for c in ciphertexts],
      'world_id': world_id
    }
  }), headers={
    'Content-Type': 'application/json'
  })

  result: list[bool] = res.json()['result']

  try:
    return result.index(True)
  except ValueError:
    return None


blocks = [encrypted[i:(i+16)] for i in range(0, len(encrypted), 16)]
result = bytes()

for block_index in range(1, len(blocks)):
  xs = []

  for byte_index in range(16):
    while True:
      iv_base = bytes().join([(xs[k] ^ (byte_index + 1)).to_bytes(1) for k in range(byte_index)])
      ivs = [os.urandom(16 - len(iv_base)) + iv_base for _ in range(256)]

      print(ivs[0].hex())

      iv_index = oracle([ivs[i] + blocks[block_index] for i in range(256)])

      if iv_index is not None:
        xs.insert(0, ivs[iv_index][-byte_index - 1] ^ (byte_index + 1))
        break

  result += b''.join([(x ^ blocks[block_index - 1][i]).to_bytes(1) for i, x in enumerate(xs)])


print()
print(result)

# b'38354af1dd04d602 (system debug firmware update key)\r\r\r\r\r\r\r\r\r\r\r\r\r'
