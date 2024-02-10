from base64 import b64encode
import math
import random
import sys


# def is_prime(a: int):
#   return not (a < 2 or any(a % x == 0 for x in range(2, int(math.sqrt(a)) + 1)))

# def is_prime(a: int):
#     if a < 2: return False
#     print(int(math.sqrt(a)) + 1)
#     for x in range(2, int(math.sqrt(a)) + 1):
#         # print('>', x)
#         if a % x == 0:
#             return False
#     return True

def is_prime(n, k = 40):

    # Implementation uses the Miller-Rabin Primality Test
    # The optimal number of rounds for this test is 40
    # See http://stackoverflow.com/questions/6325576/how-many-iterations-of-rabin-miller-should-i-use-for-cryptographic-safe-primes
    # for justification

    # If number is even, it's a composite number

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def egcd(b: int, n: int):
  x0, x1, y0, y1 = (1, 0, 0, 1)

  while n != 0:
    (q, b, n) = (b // n, n, b % n)
    (x0, x1) = (x1, x0 - q * x1)
    (y0, y1) = (y1, y0 - q * y1)

  return b, x0, y0


# init = 0x00c24098688c0c7975f1f229a3457f7730af677f4164fdcbb33b77e2bc32b20d8baccd4234c22ba8266a23d20b802321b436bd324d1b38a3b1721c3377dbaf124c1a41810f39ac2bbab41d9b1367395801e41163f127f6d8c95a1c95db2c8340ede484eda076222233a4182019a522aefe2cad5736ccde02797adea95a50066ce070695304e68eac7dd3eb3e42fd4b96a3e5c94820ce0f8d037d443a032556e4c8179784958a55d681d0b7e7301a6a7657a4a88deb46cecd62a52747e66fd218fe977ff789c215a205fe1190cb7ebd1eb41ae2949136d209eb0c68acffa432c25da5b5a71cdf758c5bb2b48891b9a9ed9dcb8f01d81ae540b331b2d3bf30ee290f
init = 0x00c24098688c0c7975f1f229a3457f7730af677f4164fdcbb33b77e2bc32b20d8baccd4234c22ba8266a23d20b802321b436bd324d1b38a3b1721c3377dbaf124c1a41810f39ac2bbab41d9b1367395801e41163f127f6d8c95a1c95db2c8340ede484eda076222233a4182019a522aefe2cad5736ccde02797adea95a50066ce070695304e68eac7dd3eb3e42fd4b96a3e5c94820ce0f8d037d443a032556e4c8179784958a55d681d0b7e7301a6a7657a4a88deb46cecd62a52747e66fd218fe977ff789000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# print(init)
# init &= (1 << 4097) - (1 << 1000)

# print(init)


# +++ATRIUM+++ in base64
target = 0xfbef804d121433efbe00000000
init += target

# print(target)

# print(target.to_bytes(20))
# print(b64encode(target.to_bytes(12 + 3)))
# # b64encode()


p = 7
q = init // p

q += 1 << 2000
# print(q)


# print(is_prime(q + 7))

# print(x)
# print(target // p * p == target)

# print(math.lcm(p - 1, q - 1))

q += 3206
for i in range(0x10000):
  q += 1

  if is_prime(q):
    print('Prime', i)
    break

n = p * q

# assert '+++ATRIUM+++' in b64encode(n.to_bytes(257)).decode()

assert is_prime(p)
assert is_prime(q)

ln = math.lcm(p - 1, q - 1)
# e = 34573495
e = 65537

assert 2 < e < ln
assert math.gcd(ln, e)

_, _, d = egcd(ln, e)

assert (d * e) % ln == 1


# print('-----BEGIN PUBLIC KEY-----')
# print('MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A')
# print(b64encode(b'\x30\x82\x01\x0a\x02\x82\x01\x01\x00' + n.to_bytes(256) + b'\x02\x03' + e.to_bytes(3)).decode())
# print('-----END PUBLIC KEY-----')


# print('-----BEGIN PRIVATE KEY-----')
# print(b64encode(d.to_bytes(6000)))
# print('-----END PRIVATE KEY-----')

print('N=', n)
print('e=', e)
print('d=', d)
print('p=', p)
print('q=', q)
