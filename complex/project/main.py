import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from random import random
from time import time_ns
from typing import Any, Callable, Optional

import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt


@dataclass
class Graph:
  edges: set[tuple[int, int]] = field(default_factory=set)
  vertices: set[int] = field(default_factory=set)

  def __post_init__(self):
    self.edges = {tuple(sorted(edge)) for edge in self.edges} # type: ignore

  def add_edge(self, a: int, b: int):
    self.edges.add(tuple(sorted((a, b)))) # type: ignore

  def copy(self):
    return self.__class__(self.edges.copy(), self.vertices.copy())

  def degrees(self):
    degrees = { vertex: 0 for vertex in self.vertices }

    for edge in self.edges:
      degrees[edge[0]] += 1
      degrees[edge[1]] += 1

    return degrees

  def remove_vertex(self, vertex: int):
    self.vertices.remove(vertex)
    self.edges = set(filter(lambda edge: vertex not in edge, self.edges))

  def remove_vertices(self, vertices: set[int]):
    for vertex in vertices:
      self.remove_vertex(vertex)

  def draw(self):
    output = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">"""
    edge_pos = dict[int, tuple[float, float]]()

    for index, vertex in enumerate(self.vertices):
      c = math.cos(index * 2 * math.pi / len(self.vertices))
      s = math.sin(index * 2 * math.pi / len(self.vertices))

      x = 50 + 40 * c
      y = 50 + 40 * s
      edge_pos[vertex] = (x, y)

      output += f"""<circle cx="{x}" cy="{y}" r="1.5" />"""
      output += f"""<text x="{50 + 48 * c}" y="{50 + 48 * s}" dominant-baseline="central" font-family="Helvetica" font-size="10" text-anchor="middle">{vertex}</text>"""

    for a, b in self.edges:
      ax, ay = edge_pos[a]
      bx, by = edge_pos[b]

      output += f"""<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="black" />"""

    output += """</svg>"""

    return output

  @classmethod
  def random(cls, vertex_count: int, edge_probability: float):
    vertices = set[int](range(vertex_count))
    edges = set[tuple[int, int]]()

    for a in vertices:
      for b in vertices:
        if b >= a:
          break

        if random() < edge_probability:
          edges.add((a, b))

    return cls(edges, vertices)


def cover_from_coupling(graph: Graph):
  size = 0
  vertices = set[int]()

  for a, b in graph.edges:
    if (not a in vertices) and (not b in vertices):
      size += 1
      vertices |= {a, b}

  return size, vertices

def cover_greedy(input_graph: Graph):
  graph = input_graph.copy()
  vertices = set[int]()

  while graph.edges:
    degrees = graph.degrees()
    max_degree_vertex = max(list(degrees.items()), key=lambda x: x[1])[0]

    vertices.add(max_degree_vertex)
    graph.remove_vertex(max_degree_vertex)

  return vertices

def cover_optimal(input_graph: Graph):
  stack = [(input_graph, set[int]())]
  done = list[set[int]]()

  while stack:
    graph, included_vertices = stack.pop()

    if graph.edges:
      vertex1, vertex2 = next(iter(graph.edges))
      # print(f"{included_edges!r} -> {vertex1}, {vertex2}")

      a = graph.copy()
      a.remove_vertex(vertex1)

      b = graph.copy()
      b.remove_vertex(vertex2)

      # print("  ", a.vertices, b.vertices)
      # print("  ", included_edges | {vertex1}, included_edges | {vertex2})

      stack.append((a, included_vertices | {vertex1}))
      stack.append((b, included_vertices | {vertex2}))
    else:
      done.append(included_vertices)

  return done


@dataclass(frozen=True, slots=True)
class Node:
  graph: Graph
  lower_bound: float
  upper_bound: float
  vertices: set[int]

  @classmethod
  def derive(cls, graph: Graph, vertices: set[int]):
    m = len(graph.edges)
    n = len(graph.vertices)

    max_degree = max(graph.degrees().values())
    b2, coupling_cover = cover_from_coupling(graph)

    b1 = math.ceil(len(graph.edges) / max_degree) if max_degree > 0 else 0
    b3 = 0.5 * (2 * n - 1 - math.sqrt((2 * n - 1) ** 2 - 8 * m))

    lower_bound = len(vertices) + max(b1, b2, b3)
    upper_bound = len(vertices) + len(coupling_cover)

    return Node(
      graph,
      lower_bound,
      upper_bound,
      vertices
    )

def cover_optimal2(input_graph: Graph):
  explored_node_count = 0
  stack = [Node.derive(input_graph, set())]

  best_solution: Optional[set[int]] = None
  best_solution_upper_bound = math.inf

  while stack:
    node = stack.pop()
    best_solution_upper_bound = min(best_solution_upper_bound, node.upper_bound)

    if node.lower_bound > best_solution_upper_bound:
      continue

    if node.graph.edges:
      explored_node_count += 1
      vertex1, vertex2 = next(iter(node.graph.edges))

      a = node.graph.copy()
      a.remove_vertex(vertex1)

      b = node.graph.copy()
      b.remove_vertex(vertex2)

      stack.append(Node.derive(a, node.vertices | {vertex1}))
      stack.append(Node.derive(b, node.vertices | {vertex2}))
    else:
      assert node.upper_bound == node.lower_bound == len(node.vertices)

      best_solution = node.vertices
      best_solution_upper_bound = node.lower_bound

  assert best_solution is not None
  return explored_node_count, best_solution

# Testing

if 0:
  graph = Graph.random(8, 0.3)
  # pickle.dump(graph, open("graph.pickle", "wb"))
  # graph: Graph = pickle.load(open("graph.pickle", "rb"))

  # print(cover_from_coupling(graph))
  # print(cover_greedy(graph))
  print(cover_optimal2(graph))

  with (Path(__file__).parent / "out.svg").open("wt") as file:
    file.write(graph.draw())


if 0:
  sample_count = 100
  total_explored_node_count = 0

  for i in range(sample_count):
    print(f"> {i}")
    graph = Graph.random(20, 0.3)
    explored_node_count, _ = cover_optimal2(graph)
    total_explored_node_count += explored_node_count

  print(total_explored_node_count / sample_count)


# Benchmark

if 1:
  @dataclass
  class Algorithm:
    function: Callable[[Graph], Any]
    label: str
    color: Any

  algorithms = [
    Algorithm(cover_from_coupling, "Coupling", cm.Greens),
    Algorithm(cover_greedy, "Greedy", cm.Blues),
    Algorithm(cover_optimal, "Optimal (4.1)", cm.Reds),
    Algorithm(cover_optimal2, "Optimal (4.2)", cm.Purples),
  ]

  algorithm_count = len(algorithms)
  sample_count = 2
  p_values = [0.25] # [0, 0.25, 0.5, 0.75, 1]
  n_values = np.linspace(10, 20, 5, dtype=int)

  if 1:
    # (coupling, greedy), sample, n, p
    cover_size = np.zeros((algorithm_count, sample_count, len(n_values), len(p_values)), dtype=int)
    exec_time = np.zeros((algorithm_count, sample_count, len(n_values), len(p_values)))

    for sample_index in range(sample_count):
      for p_index, p in enumerate(p_values):
        for n_index, n in enumerate(n_values):
          graph = Graph.random(n, p)

          for algorithm_index, algorithm in enumerate(algorithms):
            t0 = time_ns()
            algorithm.function(graph)
            t1 = time_ns()

            index = algorithm_index, sample_index, n_index, p_index

            # cover_size[*index] = ...
            exec_time[*index] = (t1 - t0) * 1e-6

    with Path("out.pickle").open("wb") as file:
      pickle.dump((cover_size, exec_time), file)
  else:
    with Path("out.pickle").open("rb") as file:
      cover_size, exec_time = pickle.load(file)

  # print(cover_size)

  # avg_cover_size = np.average(cover_size, axis=1)
  avg_exec_time = np.average(exec_time.clip(min=1e-3), axis=1)


  fig1, ax1 = plt.subplots()
  fig2, ax2 = plt.subplots()

  p_normalize = colors.Normalize(vmin=min(p_values), vmax=max(p_values))

  for algorithm_index, algorithm in enumerate(algorithms):
    for p_index, p in enumerate(p_values):
      ax1.plot(n_values, avg_exec_time[algorithm_index, :, p_index], color=algorithm.color(p_normalize(p) * 0.8 + 0.2), label=f"{algorithm.label} (p={p})")
      # ax2.scatter(n_values, avg_cover_size[0, :, p_index], color=cm.autumn(p_normalize(p)), label=f"Coupling (p={p})")

  ax1.set_yscale('log')
  ax1.set_xlabel("Nombre de sommets (n)")
  ax1.set_ylabel("Temps d'exécution (ms)")
  ax1.legend()

  r_cover_size = cover_size.reshape((2, -1))
  j = range(r_cover_size.shape[1])

  print(r_cover_size)
  ax2.scatter(j, r_cover_size[0, :])
  ax2.scatter(j, r_cover_size[1, :])

  # ax2.set_xlabel("Nombre de sommets (n)")
  # ax2.set_ylabel("Nombre de sommets dans la couverture ($|C|$)")
  ax2.legend()

  fig1.savefig("out1.png", dpi=200)
  fig2.savefig("out2.png", dpi=200)
