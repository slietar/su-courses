from sympy import Eq, Matrix, S
import sympy
from sympy.abc import x, y, z


c = 1

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

# lambdas = [
#   [eigenval1.subs(x, c) for c in stat_points],
#   [eigenval2.subs(x, c) for c in stat_points]
# ]
