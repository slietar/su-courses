from pprint import pprint
from sympy import N, Eq, Rational, im, latex, roots, solve, re

from sympy.abc import x, y, z, c

z = -Rational(1, 2)

xp = (y - x**3 + 3 * x**2 + z) / c
yp = 1 - 5 * x**2 - y

a = solve(Eq(yp, 0), y)
sub = xp.subs(y, a[0])

b = roots(sub, x, filter='R')
b = roots(sub, x)

for sol in b.keys():
  # print(solve(Eq(im(sol), 0)))
  print(latex(sol))
  print(N(sol))
  print(latex(re(sol)))
  print()

# print(a)
# print(sub)
# b = solve(Eq(sub, 0), x)
# print(latex(b[0]))
