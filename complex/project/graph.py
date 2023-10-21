from dataclasses import dataclass, field
from random import random
import math


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

  def neighbors(self, vertex: int) -> set[int]:
    return set.union(*[set(edge) for edge in self.edges if vertex in edge]) - {vertex}

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
