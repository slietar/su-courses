import hashlib
import sys

from hlxextend import sha256


ref_mac = '165daddb7e8019461e69bcda99511e2c64a10c3d79c8aa025371e655affbef27'


# prefix = b'''-----BEGIN RML PROGRAM -----
prefix = b'''extern type string.
extern type ROOM.

extern def print (message : string) -> nothing.
extern def here () -> ROOM.
extern def room_name(room : ROOM) -> string.

def main(action : string maybe, direction : string maybe, item : string maybe) -> nothing {
  let location : ROOM = here();
  if room_name(location) != "ISIR_CAFET" {
    print("0xbadCAFE !!!");
    panic                               # nobody steals our hackable coffee pot!
  };
  # ... rest of the coffee-making program
  # ejecting previous roast
  # grounding fresh beans
  # sending water
  # etc...'''


program = b'''
print("Helloooooo")
'''

# Block size: 64 bytes


# prefix = key + ref_program
appended_code = program + b'}'

# prefix = b'a'
# appended_code = b'b'

# appended_code = appended_code.ljust(32, b'\x01')
# out_msg += "0" * (((self._blockSize*7) - len(out_msg) % self._b2) % self._b2) + length

# Test
# key = b'Hello'.ljust(16, b'\x00')
# key = b'Hello world!!!!!'
# ref_mac = hashlib.sha256(key + prefix).hexdigest()

# print('Ref    program MAC', hashlib.sha256(prefix).hexdigest())

sha = sha256()
final_program = sha.extend(appended_code, prefix, 16, ref_mac)
# final_program = prefix + appended_code

# final_program = sha.extend(appended_code, prefix, 16, hashlib.sha256(key + prefix).hexdigest())

# Test
# print('Target program MAC', hashlib.sha256(key + final_program).hexdigest(), file=sys.stderr)

print('Test program MAC', hashlib.sha256(final_program).hexdigest(), file=sys.stderr)
print('Signature MAC', sha.hexdigest(), file=sys.stderr)
print('Length', len(final_program), file=sys.stderr)

# print(final_program, file=sys.stderr)
# print(file=sys.stderr)

# print(final_program, file=sys.stderr)
# print(len(prefix), file=sys.stderr)

suffix = b'\n-----END RML PROGRAM -----'

sys.stdout.buffer.write(final_program + suffix)
# sys.stdout.buffer.write(key + final_program)


# import HashTools

# magic = HashTools.new(algorithm="sha256")
# magic.update(b"Hello World!")
# print(magic.hexdigest())

# new_data, new_sig = magic.extension(
#     secret_length=len(key),
#     original_data=prefix,
#     append_data=appended_code,
#     signature=hashlib.sha256(key + prefix).hexdigest()
# )

# print('>', new_data)
# print('>', new_sig)


import requests
import json

params = {
  'message': prefix.hex(),
  'mac': ref_mac,
  'world_id': '40f8a6f8430c83220bd0ba096726fd24'
}

# params = {
#   'message': final_program.hex(),
#   'mac': sha.hexdigest(),
#   'world_id': '40f8a6f8430c83220bd0ba096726fd24'
# }

# print(repr(prefix))
print(params, file=sys.stderr)

res = requests.post('http://m1.tme-crypto.fr:8888', data=json.dumps({
  'jsonrpc': '2.0',
  'id': 3,
  # 'method': 'string.key',
  # 'params': {
  #   'handle': '7f565f763098ad43d4de84c2488b54aa',
  #   'world_id': 'd8d522549f8977786478e9e63078a638'
  # }
  'method': 'isir.checkmac',
  'params': params
}), headers={
  'Content-Type': 'application/json'
})

result = res.json()
print('>>>', result)
