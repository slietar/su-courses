import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any

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

  def to_contracted(self):
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
    graph = self.to_contracted()

    while graph.vertex_count > 2:
      a, b = graph.pick_random_edge()
      # print('Contract', a, b)
      graph = graph.contract(a, b)

    return graph.sources, graph.matrix[0, 1]

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


@dataclass
class ContractedGraph(Graph):
  sources: list[set[int]]

  def contract(self, a_: int, b_: int):
    a = min(a_, b_)
    b = max(a_, b_)

    matrix = self.matrix.copy()
    matrix[a, :] += self.matrix[b, :]
    matrix[:, a] += self.matrix[:, b]
    matrix[a, a] = 0

    matrix = np.delete(matrix, b, axis=0)
    matrix = np.delete(matrix, b, axis=1)

    sources = self.sources.copy()
    sources[a] |= sources[b]
    del sources[b]

    return self.__class__(matrix, sources)

# random.seed(0)

# g = Graph([
#   [0, 0, 0, 0],
#   [0, 0, 1, 1],
#   [0, 1, 0, 0],
#   [0, 1, 0, 0]
# ])

# print(g)

# g = Graph.cycle(5)
# g = Graph.complete(5)
# g = g.bipartite(4)
# g = g.contract(6, 7)

# with Path('out.svg').open('w') as f:
#   f.write(g.draw())


# print(g.karger())

# g = g.to_contracted().contract(2, 1)

graph_sizes = np.arange(5, 200, 5)
output = np.zeros(len(graph_sizes))
it_count = 100

for graph_size_index, graph_size in enumerate(graph_sizes):
  total_time = 0

  print(graph_size)
  for _ in range(it_count):
    # g = Graph.cycle(graph_size)
    g = Graph.complete(graph_size)

    a = time_ns()
    g.karger()
    total_time += time_ns() - a

  output[graph_size_index] = total_time / it_count * 1e-6


fig, ax = plt.subplots()

ax.plot(graph_sizes, output, '.-')
fig.savefig('out.png')
