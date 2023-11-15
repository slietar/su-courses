from dataclasses import dataclass
from typing import Self
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


@dataclass
class Leaf:
  value: int

@dataclass
class Node:
  children: list[tuple[Self | Leaf, float]]

  def _newick(self) -> str:
    return '(' + ','.join([(chr(ord('A') + child.value) if isinstance(child, Leaf) else child._newick()) + f':{dist:.2f}' for child, dist in self.children]) + ')'

  def newick(self):
    return self._newick() + ';'

# def add_nodes(a: Node | Leaf, b: Node | Leaf, da: float, db: float):
#   match a, b:
#     case (Node(), Node()) | (Leaf(), Leaf()):
#       return Node([
#         (a, da),
#         (b, db)
#       ])
#     case Node(a_children), Leaf():
#       return Node([
#         *a_children,
#         (b, da + db)
#       ])
#     case Leaf(), Node(b_children):
#       return Node([
#         (a, da + db),
#         *b_children
#       ])
#     case _, _:
#       raise RuntimeError


def nj(matrix: np.ndarray):
  m = matrix.astype(np.float64)
  clusters: list[Node | Leaf] = [Leaf(i) for i in range(len(m))]

  while (n := len(m)) > 2:
    q: np.ndarray = (n - 2) * m - m.sum(axis=0) - m.sum(axis=1, keepdims=True)
    q[np.diag_indices_from(q)] = np.inf

    a, b = np.unravel_index(q.argmin(), q.shape)
    a = int(a)
    b = int(b)

    da = 0.5 * m[a, b] + 0.5 / (n - 2) * (m[a, :].sum() - m[b, :].sum())
    db = m[a, b] - da

    clusters.append(Node([
      (clusters[a], da),
      (clusters[b], db)
    ]))

    del clusters[max(a, b)]
    del clusters[min(a, b)]

    new_row = 0.5 * (m[a, :] + m[b, :] - m[a, b])

    indices = [i for i in range(n) if i != a and i != b]
    m = m[indices, :][:, indices]
    m = np.pad(m, (0, 1))

    m[0:-1, -1] = new_row[indices]
    m[-1, 0:-1] = new_row[indices]

  root = Node([
    (clusters[0], m[0, 1] * 0.5),
    (clusters[1], m[0, 1] * 0.5)
  ])

  return root.newick()


x = np.array([
  [0, 5, 9, 9, 8],
  [5, 0, 10, 10, 9],
  [9, 10, 0, 8, 7],
  [9, 10, 8, 0, 3],
  [8, 9, 7, 3, 0]
], dtype=np.float64)

print(nj(x))
