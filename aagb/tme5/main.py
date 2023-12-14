import functools
import sys
from typing import IO
from networkx import shortest_path
import numpy as np
# from matplotlib import pyplot as plt
from dataclasses import dataclass
from pathlib import Path
# from scipy.optimize import curve_fit


@dataclass(frozen=True)
class Graph:
  matrix: np.ndarray

  @property
  def vertex_count(self):
    return len(self.matrix)

  @functools.cached_property
  def clustering_coefficients(self):
    triange_double_count = self.degrees * (self.degrees - 1)

    return np.divide(
      (self.matrix & self.matrix[:, :, None] & self.matrix[:, None, :]).sum(axis=(0, 1)),
      triange_double_count,
      out=np.zeros(self.vertex_count),
      where=(triange_double_count > 0)
    )

  @functools.cached_property
  def degrees(self):
    return self.matrix.sum(axis=0)

  @classmethod
  def parse(cls, file: IO[str], /):
    edges = set[tuple[int, int]]()
    vertex_names = list[str]()
    vertex_count = 0

    def get_vertex(name: str):
      if name in vertex_names:
        return vertex_names.index(name)

      vertex = len(vertex_names)
      vertex_names.append(name)
      return vertex

    for raw_line in file.readlines():
      line = raw_line.strip()

      if line:
        a, b = [get_vertex(x) for x in line.split(',')]
        edges.add((a, b))
        vertex_count = max(vertex_count, a + 1, b + 1)

    matrix = np.zeros((vertex_count, vertex_count), dtype=bool)

    for a, b in edges:
      matrix[a, b] = True
      matrix[b, a] = True

    print(*[f'{n}={vertex_names.index(n)}' for n in ['A', 'B', 'C', 'D', 'E']])

    return Graph(matrix)


# for name in ['reseau1', 'reseau2', 'reseau3']:
#   with Path(f'{name}.txt').open() as file:
#     graph = Graph.parse(file)

#   degrees = graph.degrees

#   fig, ax = plt.subplots()

#   ax.hist(degrees, bins=(degrees.max() - degrees.min()))
#   fig.savefig(f'out_{name}.png')


with Path('reseau3.txt').open() as file:
  graph = Graph.parse(file)

# print(graph.clustering_coefficients[[0, 11, 22, 32, 35]])
# print(g.clustering_coefficients.mean())

# aka Seidel's algorithm
def apd(a: np.ndarray):
  n = len(a)

  if a.sum() >= n ** 2 - n:
    return a

  z = a.astype(int) @ a.astype(int)
  b = (a | (z > 0)) & ~np.eye(n, dtype=bool)

  t = apd(b)
  x = t @ a.astype(int)

  degrees = a.sum(axis=0)
  return 2 * t - (x < t * degrees)


# x = np.array([
#   # [0, 1, 0],
#   # [1, 0, 1],
#   # [0, 1, 0]

#   [0, 1, 0, 0],
#   [1, 0, 1, 0],
#   [0, 1, 0, 1],
#   [0, 0, 1, 0]
# ]).astype(bool)

# np.set_printoptions(threshold=sys.maxsize)

# s = [0, 11, 22, 32, 35]
# print(apd(g.matrix)[0, :])
# # print(g.matrix[0, :].astype(int))
# # print(g.matrix[:, 0].astype(int))
# print(apd(g.matrix)[s, :][:, s])
# # print(apd(g.matrix)[:, s][s, :])

# # print(apd(x))


g = Graph(np.array([
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 1, 0, 1, 0, 0, 0],
  [0, 1, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 1, 1, 0, 1],
  [0, 1, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 0, 0, 1, 0]
]).astype(bool))


dist = apd(g.matrix)

# shortest_path_counts = np.zeros((g.vertex_count, g.vertex_count), dtype=int)

# def walk():
#   # b = np.argmax(g.matrix[a, :])
#   # # c = np.argmax([0 if i == a else n for i, n in enumerate(g.matrix[b, :])])
#   # c = 2
#   # shortest_path_counts[a, b] += 1

#   for a in range(g.vertex_count):
#     for b in range(a):
#       d = dist[a, b]

#       if a != b and shortest_path_counts[a, b] < 1:
#         # if all()
#         for m in range(g.vertex_count):
#           if a != m and b != m and g.matrix[a, m] and g.matrix[b, m] and (dist[a, m] + dist[b, m] == d):
#             shortest_path_counts[a, b] += shortest_path_counts[a, m] * shortest_path_counts[b, m]
#             shortest_path_counts[b, a] += shortest_path_counts[a, m] * shortest_path_counts[b, m]

def walk(graph: Graph):
  shortest_path_counts = graph.matrix.astype(int)
  shortest_path_incl = np.zeros((graph.vertex_count, graph.vertex_count, graph.vertex_count), dtype=int)

  # while True:
  for i in range(1):
    # d = ((shortest_path_counts == 0) & ~np.eye(n, dtype=bool)).sum(axis=0)
    d = (shortest_path_counts == 0).sum(axis=0)
    f = d.argmax()
    f = 0

    # print(shortest_path_counts)
    # print('>', f, d[f])
    # print(graph.matrix[f, :].astype(int))
    # break


    if d[f] <= 1:
      break

    paths = list[tuple[int, set[int]]]()

    # f = degrees.argmin()
    # f = 4
    paths.append((f, set()))

    explored = {f}

    # for _ in range(2):
    while paths:
      # print('---')

      new_explored = set[int]()
      new_paths = list[tuple[int, set[int]]]()

      mask = shortest_path_counts == 0

      for a, ancestors in paths:
        for b in range(graph.vertex_count):
          if not (b in explored) and graph.matrix[a, b]:
            # print(a, b)

            if not b in new_explored:
              new_paths.append((b, ancestors | {a}))
              new_explored.add(b)
            else:
              path = next(path for path in new_paths if path[0] == b)
              path[1].add(a)

            shortest_path_counts[a, b] = 1
            shortest_path_counts[b, a] = 1

            for l in ancestors:
              # print(l, a, b)
              shortest_path_counts[l, b] += shortest_path_counts[l, a] * mask[l, b]
              shortest_path_counts[b, l] += shortest_path_counts[l, a] * mask[l, b]

            # print()

      explored |= new_explored
      paths = new_paths
      # print(paths)

    # distances = apd(graph.matrix)
    # y = distances + distances[..., None]
    # z = y == distances[:, None, :]

    # # print(shortest_path_counts[:, f, None].shape, shortest_path_counts[None, f, :].shape)
    # shortest_path_counts += (shortest_path_counts[:, f, None] @ shortest_path_counts[None, f, :]) * (shortest_path_counts == 0) * z[:, f, :]

  # print(i)
  return shortest_path_counts

  # x, y = np.where((shortest_path_counts == 0) & ~np.eye(n, dtype=bool))
  # print(next(zip(x, y)))

np.set_printoptions(linewidth=160, threshold=sys.maxsize)

# print(walk(graph))


x = np.array([
  [0, 1, 0, 0],
  [1, 0, 1, 0],
  [0, 1, 0, 1],
  [0, 0, 1, 0],
]).astype(bool)

x = np.array([
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 0, 1, 1, 0, 0, 0],
  [0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 0, 1, 1, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
]).astype(bool)

# distances = apd(g.matrix)

# y = distances + distances[..., None]
# z = y == distances[:, None, :]
# # w = x * z

# # print(z[0, :, 7].astype(int))
# s = (z.sum(axis=(0, 2)) - 1) // 2 - len(distances) + 1

# # print(s)
# print(z.sum(axis=1) - 1)
# print(distances)


# y = x.astype(int)
# u = ~np.eye(len(x), dtype=bool)

# print(((y @ y) * u) @ y)


def algo(matrix: np.ndarray):
  distances = apd(matrix)

  y = distances + distances[..., None]
  z = y == distances[:, None, :]
  node_count_in_paths = z.sum(axis=1) - 2

  # print(distances)
  # print(node_count_in_paths[0, 1])

  shortest_path_counts = np.where(node_count_in_paths + 1 == distances, 1, 0)
  shortest_path_counts += node_count_in_paths * (distances == 2) * (shortest_path_counts == 0)
  # shortest_path_counts *= ~np.eye(len(matrix), dtype=bool)

  for dist in range(3, distances.max() + 1):
    # a = (shortest_path_counts[:, :, None] @ shortest_path_counts[:, None, :]) * z
    a = np.einsum('im, mj, imj -> ij', shortest_path_counts, shortest_path_counts, z)
    # print(a[0, 3] // (dist - 1) * (distances == dist))
    # print(a.sum(axis=2) / (dist - 1))
    # shortest_path_counts += (shortest_path_counts[:, :, None] @ shortest_path_counts[:, None, :]) * z[:, :, :] * (distances == dist) * (shortest_path_counts == 0)
    shortest_path_counts += a // (dist - 1) * (distances == dist) * (shortest_path_counts == 0)

  shortest_path_counts *= ~np.eye(len(matrix), dtype=bool)

  print(shortest_path_counts)

# algo(graph.matrix)
# algo(g.matrix)
algo(x)
