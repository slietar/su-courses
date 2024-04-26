from sympy import Eq, Matrix, S, Symbol
from sympy.abc import x, y
from sympy.calculus.util import continuous_domain
import numpy as np
import sympy


c = 1
z = Symbol('z', real=True)

system = Matrix([
  (y - x**3 + 3 * x**2 + z) / c,
  1 - 5*x**2 - y
])

jacobian: Matrix = system.jacobian(Matrix([x, y]))

# delta = jacobian.trace() ** 2 - 4 * jacobian.det()
# sol = sympy.solveset(delta < 0, x, S.Reals)

# print(sol.start.evalf(), sol.end.evalf())
# print(sympy.latex(sol.start))
eigenval1, eigenval2 = iter(jacobian.eigenvals().keys())
# trace = jacobian.trace()

# print(l1)
# print(l2)
# print(sympy.solve(sympy.im(l1) != 0, x))

# print(system[1] == 0)
# print(sympy.solve((
#   Eq(system[0], 0),
#   Eq(system[1], 0)
# ), (x, y)))

a = sympy.solve(Eq(system[1], 0), y)
b = system[0].subs(y, a[0])
stat_points = sympy.solve(Eq(b, 0), x)

get_stat_points = lambda zs: np.asarray(sympy.lambdify(z, stat_points, 'numpy')(np.asarray(zs, dtype=complex)))
get_trace = lambda xs: np.asarray(sympy.lambdify(x, jacobian.trace(), 'numpy')(xs))
get_det = lambda xs: np.asarray(sympy.lambdify(x, jacobian.det(), 'numpy')(xs))
get_y = lambda xs: np.asarray(sympy.lambdify(x, a[0], 'numpy')(xs))


# Solution boundary test
#
# for stat_point in stat_points:
#   print(stat_point)
#   print(continuous_domain(stat_point, z, S.Reals))

# Stable boundary test
#
# p = sympy.solve(Eq(jacobian.trace(), 0), x)
# print(p)
# print([a.evalf() for a in p])

# Stable boundary test
#
# for stat_point in stat_points:
#   print(jacobian.trace().subs(x, stat_point))
#   p = sympy.solve(Eq(jacobian.trace().subs(x, stat_point), 0), z)
#   print(p)
