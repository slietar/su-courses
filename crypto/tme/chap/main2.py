from base64 import b64decode
import subprocess
from pathlib import Path


def decrypt(input_encrypted: bytes, passphrase: str):
  proc = subprocess.run(['openssl', 'enc', '-aes-128-cbc', '-d', '-pass', f'pass:{passphrase}', '-pbkdf2'], input=input_encrypted, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  if proc.returncode > 0:
    return None

  return proc.stdout


input_encrypted = b64decode('U2FsdGVkX18ioJl/79whjAXwWvBMIpris//up9uCmbikcEMYpw2dO0ygrKnxyFrP')

with Path('dictionary.txt').open() as file:
  for raw_word in list(file):
    word = raw_word.strip()

    if word:
      if decrypt(input_encrypted, word) == b'turps funny pavan hammy petri':
        print('>>', word)
        break
