from scipy.optimize import linprog


result = linprog([
  -10, 1
], [
  [2, 5]
], [
  11
], bounds=[
  (0, None)
])

print(result)
print(result.x)
print(result.fun)
