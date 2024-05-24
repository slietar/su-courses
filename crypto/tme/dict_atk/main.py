from base64 import b64decode, b64encode
import hashlib
import os
from pathlib import Path
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


input_encrypted = b64decode('U2FsdGVkX18ioJl/79whjAXwWvBMIpris//up9uCmbikcEMYpw2dO0ygrKnxyFrP')
input_encrypted = input_encrypted[len('Salted__'):]

# print(len(y))

input_salt = input_encrypted[0:8]
input_encrypted = input_encrypted[8:]

iv = bytes.fromhex('bd0cf8a4850b27f58e31a184991d859d')

padder = padding.PKCS7(128).padder()
padded_data = padder.update(b'turps funny pavan hammy petri') + padder.finalize()

print(padded_data)

with Path('dictionary.txt').open() as file:
  for raw_word in file:
    word = raw_word.strip()

    kdf = PBKDF2HMAC(
      algorithm=hashes.SHA256(),
      length=16,
      salt=input_salt,
      iterations=10000
    )

    key = kdf.derive(word.encode())

    encryptor = Cipher(
      algorithms.AES(key),
      modes.CBC(iv)
    ).encryptor()

    a = encryptor.update(padded_data) + encryptor.finalize()

    print(word)
    print(b64encode(a))
    break

    if a == input_encrypted:
      print(word)
      break
