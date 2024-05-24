import hashlib
import random


def egcd(b: int, n: int):
  x0, x1, y0, y1 = (1, 0, 0, 1)

  while n != 0:
    (q, b, n) = (b // n, n, b % n)
    (x0, x1) = (x1, x0 - q * x1)
    (y0, y1) = (y1, y0 - q * y1)

  return b, x0, y0

def modinv(a: int, n: int):
  g, _, x = egcd(n, a)
  assert g == 1
  return x % n


# Part 1

# http://mpqs.free.fr/h11300-pkcs-1v2-2-rsa-cryptography-standard-wp_EMC_Corporation_Public-Key_Cryptography_Standards_(PKCS).pdf#page=40

n = 0x00bed6cdc8f142d61854b6bddc6f9eb36bbbf4e5dab77207240078293c384eb53d4e3a0b2f250d6dd1192448973b250d563517218c90a12c0447f5b31df37410d8a2e21573c0f05a8aa9924114708053b08878d3b53ccd35ba3516c02c3692d048ad46e98b7fbe13a99b4670fcc96dd51e7a04a3da93493ab5b5b0ff7ae77708d74f8c964112523fed59c1bdc949bfea248ff0a39285302b0292a6b8de23f98a920135cf1b5660e16eb4fddbb24b4312ef5c59f4a02a67dff2b28a6d1b0c3e3942c1736faa43a94f0995e04bda6c873a1cbaf6685424196abc6185d3e40fc1cc276fae72de0465d9748e6eb7e165b62800a53f2c67e4693cc92b37a9ee4bc449e9

raw_text = b'I, the lab director, hereby grant bob permission to take the BiblioDrone-NG.'
text = b'\x30\x31\x30\x0d\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02\x01\x05\x00\x04\x20' + hashlib.sha256(raw_text).digest()

output_size = 2048 // 8
padding_len = output_size - len(text) - 3

raw_m = b'\x00\x01' + (b'\xff' * padding_len) + b'\x00' + text

assert len(raw_m) == output_size


# https://crypto.stackexchange.com/questions/35644/chosen-message-attack-rsa-signature

m = int.from_bytes(raw_m)
m1 = random.randint(0, 256 ** output_size)
m1_inv = modinv(m1, n)
m2 = (m * m1_inv) % n

print('m=', m.to_bytes(output_size).hex())
print('m1=', m1.to_bytes(output_size).hex())
print('m2=', m2.to_bytes(output_size).hex())

assert (m * m1_inv) % n == m2


# Part 2

s1 = 0x13eafce0d7df32480eba76d3afb27cfb74ada9a75bb0780dea711d7c7921507a67258317b33aa20ea5293cd3cb0594ea5ab066f1f29ac31dc936dcef39a8ffe76c961acad227e3762b3ca9320b0861e4f796defedaac95005bfd155dc22d9d6f0276a351470e7bc73728855a0f991e6c8d68cf04f1b7cb21a12efec26c05112729b3efe01feea41ec0c376060d76141106f0b3247ce12fde4745be0d97fd713249f642f31a31cad3a8a899aee9dfbaf459a1b263004290d94a7ae1482c78360b2bfad8e30781c89e46d026daeade2ccb369b4164ccbdade44d1bd45c691ffe8300a68250e80034c6ddffb482109ec591c58aa4956ba27673840e4b5aefaa204a
s2 = 0xa0c6f6f03a28bb001c88714afd92cb2e7ccf5dc4cb240ccc752bbb6ee324eb419820e555f28579f56755949e0ce3d1c68f3871b9d3a1e7161e8aca8c29b4581f183424a896fce30f84b8a08f1ac07deeb2f21da7275ce9213afdb28271719a5f62236cdbbc75f7a6015d8c6ef7f06576259cea1325217ffbaf155717759b7a8c254491869fd6c08cad565172af0cc1d84665d758c1ed83af0059daa320b636b33903c33d5e46787ea7b82ed7d7473434bab1739ea848f2eb5a1af43387f4b461dd9da104fa49e77dabc06c86ec07c9bc8fd510b836d48e8a7bbfccabd11716036eca5301cd7c442de6567f83e5c69823f57637c641ff4771397b0cbec5af2945

s = (s1 * s2) % n
print('s=', s.to_bytes(output_size).hex())
