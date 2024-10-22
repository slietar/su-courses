from sympy import Eq, Matrix, S, Symbol
from sympy.abc import x, y
from sympy.calculus.util import continuous_domain
import numpy as np
import sympy

from . import utils


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
# eigenval1, eigenval2 = iter(jacobian.eigenvals().keys())
# trace = jacobian.trace()

# print(l1)
# print(l2)
# print(sympy.solve(sympy.im(l1) != 0, x))

# print(system[1] == 0)
# print(sympy.solve((
#   Eq(system[0], 0),
#   Eq(system[1], 0)
# ), (x, y)))

eq1_sol = sympy.solve(Eq(system[1], 0), y)[0]
eq0_sols = system[0].subs(y, eq1_sol) # type: ignore
x_sols = sympy.solve(Eq(eq0_sols, 0), x)

# get_stat_points = lambda zs: np.asarray(sympy.lambdify(z, stat_points, 'numpy')(np.asarray(zs, dtype=complex)))
# get_y = lambda xs: np.asarray(sympy.lambdify(x, a[0], 'numpy')(xs))

def get_stat_points(zr: np.ndarray):
  xs_unflattened = np.asarray(sympy.lambdify(z, x_sols, 'numpy')(zr.astype(complex)))
  xs = xs_unflattened.ravel()
  zs = np.tile(zr, xs_unflattened.shape[0])

  mask = utils.isclosereal(xs)
  order = np.argsort(xs[mask])

  xs = xs[mask][order].real
  zs = zs[mask][order].real

  ys = np.asarray(sympy.lambdify(x, eq1_sol, 'numpy')(xs))

  return np.array([xs, ys, zs]).T

get_trace = lambda sp: np.asarray(sympy.lambdify(x, jacobian.trace(), 'numpy')(sp[:, 0]))
get_det = lambda sp: np.asarray(sympy.lambdify(x, jacobian.det(), 'numpy')(sp[:, 0]))
is_stable = lambda sp: (get_trace(sp).real < 0.0) & (get_det(sp).real > 0.0)


eigenval1, eigenval2 = iter(jacobian.eigenvals().keys())

def get_eignvalues(sp: np.ndarray):
  return np.asarray(sympy.lambdify(x, list(jacobian.eigenvals().keys()), 'numpy')(sp[:, 0].astype(complex))).T


format_ex = lambda value: str(value).replace('x', 'v_0').replace('**', '^').replace('*', '')
format_sols = lambda sols: format_ex(' "ou" '.join([f'x = {sol}' for sol in sols]))


# Complex conjugate eigenvalues

complex_eigenval_eq = jacobian.trace() ** 2 - 4 * jacobian.det()

if __name__ == '__main__':
  print('Condition for complex conjugate eigenvalues:')
  print(format_ex(sympy.expand(complex_eigenval_eq)) + ' &< 0 \\')
  print([f'{sol.evalf():.3f}' for sol in sympy.solve(Eq(complex_eigenval_eq, 0), x) if sol.is_real])
  print()
  # print(sympy.solve(complex_eigenval_eq < 0, x))
  # print(sympy.latex(sympy.solve(Eq(complex_eigenval_eq, 0), x)[0]))


# Saddle-node bifurcations

bif1_x = sympy.solve(Eq(jacobian.det(), 0), x)

if __name__ == '__main__':
  print('Condition for saddle-node bifurcation:')
  print(format_ex(jacobian.det()) + ' &= 0 \\')
  print(format_sols(bif1_x))
  print()


# Hopf bifurcations

bif2_x = sympy.solve(Eq(jacobian.trace(), 0))

if __name__ == '__main__':
  print('Condition for Hopf bifurcation:')
  print(format_ex(jacobian.trace()) + ' &= 0 \\')
  print(format_sols(bif2_x))


bifurcations_x = np.array([float(x.evalf()) for x in [*bif1_x, *bif2_x]])


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
