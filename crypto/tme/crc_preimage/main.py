def crc64(m : bytes):
  i = int.from_bytes(m, byteorder='big')
  i <<= 64
  k = i.bit_length()
  Q = 0x1000000000000001b

  while k > 64:
    i ^= Q << (k - 65)
    k = i.bit_length()

  return i.to_bytes(8, byteorder='big')

# x = crc64(b'a')

# print(f'{int.from_bytes(x, byteorder="big"):64b}')
# # print(bin(int.from_bytes(crc64(b'hellowglerhgjlaherkg'))))


inp = 0xa086a5a747b088d5
# print(bin(inp))

# print(bin(0x1000000000000001b))

from sympy import ZZ
from sympy.polys.galoistools import gf_gcdex, gf_mul, gf_rem

# print(bin(inp))
# print([(inp >> (64 - i)) & 1 for i in range(64)])

pack = lambda x: int(''.join([str(a) for a in x]), 2)
unpack = lambda x: [(x >> (64 - i)) & 1 for i in range(65)]


q = unpack(0x1000000000000001b)
q2 = [1] + [0] * 59 + [1, 1, 0, 1, 1]
# v = [0, *q[1:]]
v = [1, 1, 0, 1, 1]
a, b, _ = gf_gcdex(v, q, 2, ZZ)


# print(gf_rem(gf_mul(v, a, 2, ZZ), q, 2, ZZ))


# q2 = [1] + [0] * 59 + [1, 1, 0, 1, 1]

# v = [1, 1, 0, 1, 1]
# a, b, _ = gf_gcdex(v, q, 2, ZZ)
# # x = gf_rem(a, q, 2, ZZ)
# x = a

# # print(x)

# # print(a, b, c)
# s = [(inp >> (64 - i)) & 1 for i in range(64)]
s = unpack(inp)
y = gf_mul(a, s, 2, ZZ)
z = pack(y).to_bytes(16)

# # y = gf_rem(gf_mul(x, s, 2, ZZ), q, 2, ZZ)
# z = int(''.join([str(x) for x in y]), 2).to_bytes(16)

print(z.hex())
print(crc64(z).hex())
