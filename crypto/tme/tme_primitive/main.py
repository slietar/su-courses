import math
import sympy
from sympy.ntheory import residue_ntheory

def isqrt(x: int):
    """Return the integer part of the square root of x, even for very
    large values."""
    if x < 0:
        raise ValueError('square root not defined for negative numbers')
    n = int(x)
    if n == 0:
        return 0
    a, b = divmod(n.bit_length(), 2)
    x = (1 << (a+b)) - 1
    while True:
        y = (x + n//x) // 2
        if y >= x:
            return x
        x = y



a = 0xa55779f94f09d6fe4d1ad0ac599901c3a03bf46341aea79b67a41e8e3f1061ca6ff6ae6c31c6836d1bf9f14dcbf8c71ff08446f74c95910562593c1f91de63c742c480f6d7132d439b428d8745ce2c7fb91b95283e1743116c00d49cfab5e53ec286aff4ab4b89101487d2c0522e736f15e1ecd8bcdc40cae46ae269a0aa4f8172aed87dd06f2f69c72c54db2034bffa003389aec25f57a74e5a723770334cd68a9035343907ed627b7ca895144a721028701188b71fe7813681dd1783ebba232b1abfa79a2c51ba84521010f92e0348b149257e9156197dad7cd983c101e20d55c825a2ea86043360f93dd76d1c2a77eceb2719d9c9d3571fb9e3d9afbc4cfa
b = a + 2**1950
q = 0x37717b51ae7c5eee1f27b3ed2b5f41e05d444a8cb1d92eba32d93b3e60dad9348bea4869e91bd99ba6a9328a92eba4db


# p = sympy.randprime(a, b)
# print(p)

# while True:
#   # k = sympy.randprime(a // q, b // q)
#   k = sympy.randprime(isqrt(a // q // 2), isqrt(b // q // 2))
#   pm1 = q * k ** 2 * 2
#   p = pm1 + 1

#   # print(p)
#   # print(a <= p < b)

#   if sympy.isprime(p):
#     assert a <= p < b
#     print(f'{k=}')
#     print(f'{p=}')
#     break


# https://math.stackexchange.com/questions/124408/finding-a-primitive-root-of-a-prime-number

k=34971004874273109743273794966542254288852031768420935904830263557886104443448941333250877924846343367370847373972448188419653490231723361574856700959549137030068589328187993881476593091438855390684167303749295455282231279434695047450973612507592360993

pm1 = q * k ** 2 * 2
p = pm1 + 1

# assert sympy.isprime(p)
assert a <= p < b

factors = [k, 2, q]

for a in range(2, 100):
  for factor in factors:
    if pow(a, pm1 // factor, p) == 1:
       break
  else:
    print(a)
    break

print(f'2, {k}, {k}')
