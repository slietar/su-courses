import numpy as np


M1 = np.array([[0,8,7,12], [8,0,9,14], [7,9,0,11], [12,14,11,0]])
M2 = np.array([[0,2,3,8,14,18],[2,0,3,8,14,18],
      [3,3,0,8,14,18],[8,8,8,0,14,18],
      [14,14,14,14,0,18],[18,18,18,18,18,0]])
#UPGMA
M3 = np.array([[0,19,27,8,33,18,13],[19,0,31,18,36,1,13],
          [27,31,0,26,41,32,29],[8,18,26,0,31,17,14],
          [33,36,41,31,0,35,28],[18,1,32,17,35,0,12],
          [13,13,29,14,28,12,0]])
#Neighbor Joining
M4 = np.array([[0,2,4,6,6,8],[2,0,4,6,6,8],
          [4,4,0,6,6,8],[6,6,6,0,4,8],
          [6,6,6,4,0,8],[8,8,8,8,8,0]])


def is_additive(matrix: np.ndarray):
  p = 0

  for a in range(len(matrix)):
    for b in range(a):
      for c in range(b):
        for d in range(c):
          x = matrix[a, b] + matrix[c, d]
          y = matrix[a, d] + matrix[b, c]
          z = matrix[a, c] + matrix[b, d]

          if not (x >= z and y >= z) and not (x >= y and z >= y) and not (y >= x and z >= x):
            return False

          p += 1

  return True


# print(is_additive(M1))
# print(is_additive(M2))
# print(is_additive(M3))
# print(is_additive(M4))


def is_multrametric(matrix: np.ndarray):
  for a in range(len(matrix)):
    for b in range(a):
      for c in range(b):
        if matrix[a, c] > max(matrix[a, b], matrix[b, c]):
          return False

  return True

# print(is_multrametric(M1))
# print(is_multrametric(M2))
# print(is_multrametric(M3))
# print(is_multrametric(M4))


def upgma(matrix: np.ndarray):
  print(np.stack(matrix, axis=-1))

print(upgma(M1))
