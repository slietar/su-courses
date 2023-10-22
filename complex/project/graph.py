from dataclasses import dataclass, field
from random import random
from typing import IO, Optional
import math


@dataclass(slots=True)
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
  def parse(cls, file: IO[str]):
    keys = {
      'Nombre de sommets': 'vertex_count',
      'Nombre d aretes': 'edge_count',
      'Sommets': 'vertices',
      'Aretes': 'edges'
    }

    values: dict[str, Optional[list[str]]] = { key: None for key in keys.values() }
    current_pair: Optional[tuple[str, list[str]]] = None

    for raw_line in file:
      line = raw_line[:-1]

      if not line:
        continue

      if line in keys:
        if current_pair:
          values[current_pair[0]] = current_pair[1]

        if values[keys[line]] is None:
          current_pair = keys[line], list[str]()
          continue
      elif current_pair:
        current_pair[1].append(line)
        continue

      raise ValueError(f'Unexpected line: "{line}"')

    if current_pair:
      values[current_pair[0]] = current_pair[1]

    assert values['vertex_count'] is not None
    assert len(values['vertex_count']) == 1
    vertex_count = int(values['vertex_count'][0])

    assert values['edge_count'] is not None
    assert len(values['edge_count']) == 1
    edge_count = int(values['edge_count'][0])

    assert values['vertices'] is not None
    assert len(values['vertices']) == vertex_count
    vertices = {int(raw_vertex) for raw_vertex in values['vertices']}

    assert values['edges'] is not None
    assert len(values['edges']) == edge_count

    edges = set[tuple[int, int]]()

    for raw_edge in values['edges']:
      raw_edge_vertices = raw_edge.split(' ')
      assert len(raw_edge_vertices) == 2
      edges.add((int(raw_edge_vertices[0]), int(raw_edge_vertices[1])))

    return cls(edges, vertices)

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
