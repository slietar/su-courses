from base64 import b64encode
import math
import random


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



# +++ATRIUM+++ in base64
target = 0xfbef804d121433efbe00000000
    #  a = 0x1000000000000000000000

# print(target)

# print(target.to_bytes(20))
# print(b64encode(target.to_bytes(12 + 3)))
# # b64encode()


p = 7
q = target // p

q += 1 << 2000
# print(q)

# print(is_prime(q + 7))

# print(x)
# print(target // p * p == target)

# print(math.lcm(p - 1, q - 1))

for i in range(0x10000):
  q += 1

  if is_prime(q):
    # print('Prime', i)
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


print('-----BEGIN PUBLIC KEY-----')
print('MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A')
print(b64encode(b'\x30\x82\x01\x0a\x02\x82\x01\x01\x00' + n.to_bytes(256) + b'\x02\x03' + e.to_bytes(3)).decode())
print('-----END PUBLIC KEY-----')
