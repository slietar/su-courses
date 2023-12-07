import functools
import itertools
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from time import time_ns
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class Graph:
  matrix: np.ndarray

  def __init__(self, matrix: Any):
    self.matrix = np.asarray(matrix, dtype=int)
    assert np.allclose(self.matrix, self.matrix.T)

  @property
  def edge_count(self):
    return np.sum(self.matrix) // 2

  @property
  def vertex_count(self):
    return len(self.matrix)

  @functools.cached_property
  def contracted(self):
    return ContractedGraph(self.matrix, [{vertex} for vertex in range(self.vertex_count)])

  def draw(self):
    output = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">"""
    edge_pos = dict[int, tuple[float, float]]()

    for index, vertex in enumerate(range(self.vertex_count)):
      c = math.cos(index * 2 * math.pi / self.vertex_count)
      s = math.sin(index * 2 * math.pi / self.vertex_count)

      x = 50 + 40 * c
      y = 50 + 40 * s
      edge_pos[vertex] = (x, y)

      output += f"""<circle cx="{x}" cy="{y}" r="1.5" />"""
      output += f"""<text x="{50 + 48 * c}" y="{50 + 48 * s}" dominant-baseline="central" font-family="Helvetica" font-size="10" text-anchor="middle">{vertex}</text>"""

    for a in range(self.vertex_count):
      for b in range(a):
        if self.matrix[a, b] > 0:
          ax, ay = edge_pos[a]
          bx, by = edge_pos[b]

          output += f"""<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="black" />"""

    output += """</svg>"""

    return output

  def karger(self):
    graph = self.contracted.contract_until(2)
    return graph.sources[0], graph.sources[1], graph.matrix[0, 1]

  def karger_repeated(self, it_count: int, /):
    return min((self.karger() for _ in range(it_count)), key=(lambda result: result[2]))

  def karger_stein(self):
    if self.vertex_count <= 6:
      return self.min_cut()

    t = math.ceil(self.vertex_count / math.sqrt(2))
    return min((self.contracted.contract_until(t).karger_stein() for _ in range(2)), key=(lambda result: result[2]))

  def cut_size(self, vertices: set[int], /):
    left_vertices = list(vertices)
    right_vertices = list(set(range(self.vertex_count)) - vertices)

    return self.matrix[left_vertices, :][:, right_vertices].sum()

  def min_cut(self):
    all_vertices = set(range(self.vertex_count))
    min_cut_size = math.inf
    min_cut_left_vertices: Optional[set[int]] = None

    for vertex_set in itertools.chain.from_iterable(itertools.combinations(all_vertices, r) for r in range(1, len(all_vertices))):
      cut_size = self.cut_size(set(vertex_set))

      if cut_size < min_cut_size:
        min_cut_size = cut_size
        min_cut_left_vertices = set(vertex_set)

    assert min_cut_left_vertices is not None
    return min_cut_left_vertices, set(range(self.vertex_count)) - min_cut_left_vertices, min_cut_size

  def pick_random_edge(self):
    assert self.edge_count > 0

    edge_count_per_vertex1_acc = self.matrix.sum(axis=0).cumsum()
    a = random.randint(0, self.edge_count * 2 - 1)

    for vertex1, x1 in enumerate(edge_count_per_vertex1_acc):
      if a < x1:
        break
    else:
      raise RuntimeError

    edge_count_per_vertex2_acc = self.matrix[vertex1, :].cumsum()
    b = random.randint(0, edge_count_per_vertex2_acc[-1] - 1)

    for vertex2, x2 in enumerate(edge_count_per_vertex2_acc):
      if b < x2:
        break
    else:
      raise RuntimeError

    return vertex1, vertex2

  @classmethod
  def cycle(cls, n: int):
    matrix = np.zeros((n, n), dtype=int)

    for index in range(-1, n - 1):
      matrix[index, index + 1] = 1
      matrix[index + 1, index] = 1

    return cls(matrix)

  @classmethod
  def complete(cls, n: int):
    return cls(
      np.ones((n, n)) - np.eye(n)
    )

  @classmethod
  def bipartite(cls, n: int, p: float = 0.5):
    k = n // 2

    while True:
      matrix = np.zeros((n, n), dtype=int)

      for a in range(k):
        for b in range(k, n):
          if random.random() < p:
            matrix[a, b] = 1
            matrix[b, a] = 1

      if matrix.sum(axis=0).all():
        break

    return cls(matrix)

  @classmethod
  def random(cls, n: int, p: float = 0.5):
    while True:
      matrix = np.zeros((n, n), dtype=int)

      for a in range(n):
        for b in range(n):
          if (a != b) and (random.random() < p):
            matrix[a, b] = 1
            matrix[b, a] = 1

      if matrix.sum(axis=0).all():
        break

    return cls(matrix)


@dataclass
class ContractedGraph(Graph):
  sources: list[set[int]]

  @property
  def contracted(self):
    return self

  def contract(self, a_: int, b_: int):
    a = min(a_, b_)
    b = max(a_, b_)

    matrix = self.matrix.copy()
    matrix[a, :] += self.matrix[b, :]
    matrix[:, a] += self.matrix[:, b]
    matrix[a, a] = 0

    matrix = np.delete(matrix, b, axis=0)
    matrix = np.delete(matrix, b, axis=1)

    sources = deepcopy(self.sources)
    sources[a] |= sources[b]
    del sources[b]

    return self.__class__(matrix, sources)

  def contract_until(self, final_vertex_count: int, /):
    graph = self

    while graph.vertex_count > final_vertex_count:
      graph = graph.contract(*graph.pick_random_edge())

    return graph

  def min_cut(self):
    apply_sources = lambda vertices: set.union(*(self.sources[vertex] for vertex in vertices))
    left, right, cut_size = super().min_cut()

    return apply_sources(left), apply_sources(right), cut_size


# random.seed(0)

# g = Graph([
#   [0, 0, 0, 0],
#   [0, 0, 1, 1],
#   [0, 1, 0, 0],
#   [0, 1, 0, 0]
# ])

# print(g)

# g = Graph.random(15, 0.3)
# g = Graph.complete(15)
g = Graph.bipartite(22, 0.2)
# g = g.contract(6, 7)

with Path('out.svg').open('w') as f:
  f.write(g.draw())

# def check(a: set[int], b: set[int], c: float):
#   return g.cut_size(a) == g.cut_size(b) == c

# for _ in range(1000):
#   x = g.karger()

#   if not check(*x):
#     print(x)
#     print(g.matrix)
#     sys.exit(1)

print(g.karger())
print(g.karger_repeated(10))
print(g.karger_stein())
print(g.min_cut())

# print(g.to_contracted().contract_until(3))

# g = g.to_contracted().contract(2, 1)

# graph_sizes = np.arange(5, 200, 5)
# output = np.zeros(len(graph_sizes))
# it_count = 100

# for graph_size_index, graph_size in enumerate(graph_sizes):
#   total_time = 0

#   print(graph_size)
#   for _ in range(it_count):
#     # g = Graph.cycle(graph_size)
#     g = Graph.complete(graph_size)

#     a = time_ns()
#     g.karger()
#     total_time += time_ns() - a

#   output[graph_size_index] = total_time / it_count * 1e-6


# fig, ax = plt.subplots()

# ax.plot(graph_sizes, output, '.-')
# fig.savefig('out.png')
