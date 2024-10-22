import functools
import itertools
import os
import sys
from time import time_ns
from typing import IO
from matplotlib import pyplot as plt
import numpy as np
# from matplotlib import pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
# from scipy.optimize import curve_fit


@dataclass(frozen=True)
class Graph:
  matrix: np.ndarray

  @property
  def edge_count(self):
    return self.matrix.sum() // 2

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

    # print(*[f'{n}={vertex_names.index(n)}' for n in ['A', 'B', 'C', 'D', 'E']])

    return Graph(matrix)

  @classmethod
  def random(cls, vertex_count: int, edge_probability: float = 0.5):
    matrix = np.random.random((vertex_count, vertex_count)) < edge_probability
    matrix = (matrix & ~np.triu(matrix)) | np.tril(matrix).T
    matrix[np.diag_indices(vertex_count)] = False
    return cls(matrix)


# for name in ['reseau1', 'reseau2', 'reseau3']:
#   with Path(f'{name}.txt').open() as file:
#     graph = Graph.parse(file)

#   degrees = graph.degrees

#   fig, ax = plt.subplots()

#   ax.hist(degrees, bins=(degrees.max() - degrees.min()))
#   fig.savefig(f'out_{name}.png')


with Path('reseau1.txt').open() as file:
  graph1 = Graph.parse(file)

with Path('reseau2.txt').open() as file:
  graph2 = Graph.parse(file)

with Path('reseau3.txt').open() as file:
  graph3 = Graph.parse(file)


def betweenness_centralities(matrix: np.ndarray):
  n = len(matrix)

  shortest_path_counts = np.zeros((n, n), dtype=int)
  shortest_path_incl = np.zeros((n, n, n), dtype=int)

  # for start_node in range(n):
  #   for end_node in range(start_node):
  for start_node, end_node in tqdm(list(itertools.combinations(range(n), 2))):
      queue: list[tuple[int, set[int]]] = [(start_node, set())]
      explored_nodes = set[int]()
      path_count = 0

      while path_count < 1:
        new_explored_nodes = set[int]()
        new_queue = list[tuple[int, set[int]]]()

        for current_node, path in queue:
          for next_node in range(n):
            if not matrix[current_node, next_node]:
              continue

            if next_node == end_node:
              path_count += 1

              for middle_node in path:
                shortest_path_incl[middle_node, start_node, end_node] += 1
                shortest_path_incl[middle_node, end_node, start_node] += 1
            else:
              if next_node in explored_nodes:
                continue

              new_explored_nodes.add(next_node)
              new_queue.append((next_node, path | {next_node}))

        queue = new_queue
        explored_nodes |= new_explored_nodes

        shortest_path_counts[start_node, end_node] = path_count
        shortest_path_counts[end_node, start_node] = path_count

  print(shortest_path_counts)
  return (shortest_path_incl / np.maximum(shortest_path_counts, 1)).sum(axis=(1, 2))


graph0 = Graph(np.array([
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 1, 0, 1, 0, 0, 0],
  [0, 1, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 1, 1, 0, 1],
  [0, 1, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 0, 0, 1, 0]
]).astype(bool))

# g = Graph(np.array([
#   [0, 1, 0, 0, 0],
#   [1, 0, 1, 0, 1],
#   [0, 1, 0, 1, 0],
#   [0, 0, 1, 0, 1],
#   [0, 1, 0, 1, 0],
# ]).astype(bool))

np.set_printoptions(linewidth=160, threshold=sys.maxsize, suppress=True)

s = [0, 11, 22, 32, 35]
# print(betweenness_centralities(graph.matrix)[s] / 2)
# print(betweenness_centralities(g.matrix))


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

def apd2(input_a: np.ndarray, /):
  n = len(input_a)

  mask = ~np.eye(n, dtype=bool)
  stack = [input_a]

  while (a := stack[-1]).sum() < n ** 2 - n:
    if len(stack) > n:
      raise RuntimeError('Graph is not connected')

    stack.append((a | (a @ a)) & mask)

  t = stack[-1]

  for a in stack[-2::-1]: # Equivalent to reversed(stack[:-1])
    x = t @ a.astype(int)
    degrees = a.sum(axis=0)
    t = 2 * t - (x < t * degrees)

  return t


# APD tests

# t1 = time_ns()
# x = apd(graph2.matrix)
# t2 = time_ns()
# y = apd2(graph2.matrix)
# t3 = time_ns()

# print('Old', (t2 - t1) * 1e-9)
# print('New', (t3 - t2) * 1e-9)
# print(np.allclose(x, y))

# sys.exit()


# # x = np.array([
# #   # [0, 1, 0],
# #   # [1, 0, 1],
# #   # [0, 1, 0]

# #   [0, 1, 0, 0],
# #   [1, 0, 1, 0],
# #   [0, 1, 0, 1],
# #   [0, 0, 1, 0]
# # ]).astype(bool)

# # np.set_printoptions(threshold=sys.maxsize)

# # s = [0, 11, 22, 32, 35]
# # print(apd(g.matrix)[0, :])
# # # print(g.matrix[0, :].astype(int))
# # # print(g.matrix[:, 0].astype(int))
# # print(apd(g.matrix)[s, :][:, s])
# # # print(apd(g.matrix)[:, s][s, :])

# # # print(apd(x))


# dist = apd(g.matrix)

# # shortest_path_counts = np.zeros((g.vertex_count, g.vertex_count), dtype=int)

# # def walk():
# #   # b = np.argmax(g.matrix[a, :])
# #   # # c = np.argmax([0 if i == a else n for i, n in enumerate(g.matrix[b, :])])
# #   # c = 2
# #   # shortest_path_counts[a, b] += 1

# #   for a in range(g.vertex_count):
# #     for b in range(a):
# #       d = dist[a, b]

# #       if a != b and shortest_path_counts[a, b] < 1:
# #         # if all()
# #         for m in range(g.vertex_count):
# #           if a != m and b != m and g.matrix[a, m] and g.matrix[b, m] and (dist[a, m] + dist[b, m] == d):
# #             shortest_path_counts[a, b] += shortest_path_counts[a, m] * shortest_path_counts[b, m]
# #             shortest_path_counts[b, a] += shortest_path_counts[a, m] * shortest_path_counts[b, m]

def algo0(matrix: np.ndarray):
  n = len(matrix)
  shortest_path_counts = matrix.astype(int)
  shortest_path_incl = np.zeros((n, n, n), dtype=int)

  while True:
  # for i in range(100000000):
    # d = ((shortest_path_counts == 0) & ~np.eye(n, dtype=bool)).sum(axis=0)
    d = (shortest_path_counts == 0).sum(axis=0)
    f = d.argmax()
    # f = 0

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
        for b in range(n):
          if not (b in explored) and matrix[a, b]:
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

  return shortest_path_counts

  # x, y = np.where((shortest_path_counts == 0) & ~np.eye(n, dtype=bool))
  # print(next(zip(x, y)))

# np.set_printoptions(linewidth=160, threshold=sys.maxsize)

# # print(walk(graph))


# x = np.array([
#   [0, 1, 0, 0],
#   [1, 0, 1, 0],
#   [0, 1, 0, 1],
#   [0, 0, 1, 0],
# ]).astype(bool)

# x = np.array([
#   [0, 1, 0, 0, 0, 0, 0, 0],
#   [1, 0, 1, 0, 0, 0, 0, 0],
#   [0, 1, 0, 1, 1, 0, 0, 0],
#   [0, 0, 1, 0, 0, 1, 0, 0],
#   [0, 0, 1, 0, 0, 1, 0, 0],
#   [0, 0, 0, 1, 1, 0, 1, 0],
#   [0, 0, 0, 0, 0, 1, 0, 1],
#   [0, 0, 0, 0, 0, 0, 1, 0],
# ]).astype(bool)

# # distances = apd(g.matrix)

# # y = distances + distances[..., None]
# # z = y == distances[:, None, :]
# # # w = x * z

# # # print(z[0, :, 7].astype(int))
# # s = (z.sum(axis=(0, 2)) - 1) // 2 - len(distances) + 1

# # # print(s)
# # print(z.sum(axis=1) - 1)
# # print(distances)


# # y = x.astype(int)
# # u = ~np.eye(len(x), dtype=bool)

# # print(((y @ y) * u) @ y)


def algo1(matrix: np.ndarray):
  distances = apd(matrix)
  n = len(matrix)

  y = distances + distances[..., None]
  z = y == distances[:, None, :]
  z1 = z.swapaxes(0, 1) & ~np.eye(n, dtype=bool)[:, None, :] & ~np.eye(n, dtype=bool)[:, :, None]
  node_count_in_paths = z.sum(axis=1) - 2

  # a[i, i, :] = 0
  # a[i, :, i] = 0

  # print(z1.astype(int))

  # print(distances)
  # print(node_count_in_paths[0, 1])

  shortest_path_incl = np.zeros((n, n, n), dtype=int)

  shortest_path_counts = np.where(node_count_in_paths + 1 == distances, 1, 0)
  shortest_path_incl += z1 * (distances == 2)[None, :, :]
  # print(z1[:, 0, 2])
  # print((distances == 2).astype(int)[0, 2])
  shortest_path_counts += node_count_in_paths * (distances == 2) * (shortest_path_counts == 0)
  # shortest_path_counts *= ~np.eye(len(matrix), dtype=bool)

  for dist in range(3, distances.max() + 1):
    # a = (shortest_path_counts[:, :, None] @ shortest_path_counts[:, None, :]) * z
    a = np.einsum('im, mj, imj -> ij', shortest_path_counts, shortest_path_counts, z)
    # b = np.einsum('im, mij -> .', shortest_path_incl, shortest_path_incl, z1)
    # print(a[0, 3] // (dist - 1) * (distances == dist))
    # print(a.sum(axis=2) / (dist - 1))
    # shortest_path_counts += (shortest_path_counts[:, :, None] @ shortest_path_counts[:, None, :]) * z[:, :, :] * (distances == dist) * (shortest_path_counts == 0)
    shortest_path_counts += a // (dist - 1) * (distances == dist) * (shortest_path_counts == 0)

  shortest_path_counts *= ~np.eye(n, dtype=bool)

  # print(shortest_path_counts)
  print(shortest_path_incl[:, 1, 3])

# algo(graph.matrix)
# algo(g.matrix)
# algo(x)


def algo2(matrix: np.ndarray):
  int_dtype = np.uint64
  n = len(matrix)

  matrix_int = matrix.astype(int_dtype)
  distances = matrix_int.copy()
  shortest_path_counts = matrix_int.copy()

  matrix_sq = matrix_int
  mask = ~np.eye(n, dtype=bool)

  for distance in range(2, n):
    matrix_sq = (matrix_sq @ matrix_int) * mask
    shortest_path_counts += matrix_sq * (distances < 1)
    distances += (distance * (matrix_sq > 0) * (distances < 1)).astype(int_dtype)

    if (distances > 0).sum() >= n ** 2 - n:
      break

  result = np.zeros(n)

  for target_node in list(range(n)):
    current_path_count = np.zeros(n, dtype=int_dtype)
    current_path_count[target_node] = 1

    path_count = np.zeros(n, dtype=int_dtype)
    path_count[target_node] = 1

    distances_through_target_node = distances[target_node, :] + distances[target_node, :, None]
    is_node_on_path = (distances_through_target_node == distances)

    while current_path_count.any():
      current_path_count = (current_path_count @ matrix) * (path_count < 1)
      path_count += current_path_count

    path_count[target_node] = 0

    shortest_path_incl = (path_count[:, None] @ path_count[None, :]) * is_node_on_path
    result[target_node] = (shortest_path_incl / np.maximum(shortest_path_counts, 1)).sum()

  return result


# print(abs(betweenness_centralities(graph3.matrix) - algo2(graph3.matrix)).sum())

g = graph1 # Graph.random(100, 0.5)
print(algo2(g.matrix))

# sys.exit()
# # print(g.matrix.astype(int))

# a1 = betweenness_centralities(g.matrix)

# t1 = time_ns()
# a2 = algo2(g.matrix)

# print((time_ns() - t1) * 1e-9)
# print(np.allclose(a1, a2))


# # fig, ax = plt.subplots()
# # ax.hist(bc)
# # fig.savefig('bc_reseau1.png')

# print(os.getpid())

# it = 2
# t1 = time_ns()

# for _ in range(it):
#   algo2(g.matrix)

# print((time_ns() - t1) / it * 1e-9)


# a = np.array([
#   [0, 1, 0, 0, 0, 0, 0, 0],
#   [1, 0, 1, 0, 1, 0, 0, 0],
#   [0, 1, 0, 1, 0, 0, 0, 0],
#   [0, 0, 1, 0, 1, 1, 0, 1],
#   [0, 1, 0, 1, 0, 0, 0, 0],
#   [0, 0, 0, 1, 0, 0, 1, 0],
#   [0, 0, 0, 0, 0, 1, 0, 1],
#   [0, 0, 0, 1, 0, 0, 1, 0]
# ]) # .astype(bool)

# # print(a)
# # print()

# n = len(a)
# mask = ~np.eye(n, dtype=bool)

# # z = (a + a @ a) * mask
# # z = (a + a @ a) * mask

# distances = a.copy()
# count = a.copy() # np.zeros((n, n), dtype=int)

# z = a

# # distances = 1 * (z > 0) * (distances < 1)
# # print(distances)

# for i in range(2, 6):
#   z = (z @ a) * mask
#   count += z * (distances < 1)
#   distances += i * (z > 0) * (distances < 1)

# print(apd2(a.astype(bool)) - distances)
# print(count)
# betweenness_centralities(a.astype(bool))
# # print(distances)

# # print(z.astype(int))
# # print(z)

# # z = a.astype(int) @ a.astype(int)
# # b = (a | (z > 0)) & ~np.eye(n, dtype=bool)

# # t = apd(b)
# # x = t @ a.astype(int)

# # degrees = a.sum(axis=0)
# # return 2 * t - (x < t * degrees)
