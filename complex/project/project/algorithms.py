from dataclasses import dataclass
from typing import Optional
import math

from .graph import Graph


def cover_from_coupling(graph: Graph):
  """
  Create a suboptimal cover from a coupling.

  Returns
    A tuple containing the size of the coupling and the vertices in the cover.
  """

  size = 0
  vertices = set[int]()

  for a, b in graph.edges:
    if (not a in vertices) and (not b in vertices):
      size += 1
      vertices |= {a, b}

  return size, frozenset(vertices)

def cover_greedy(input_graph: Graph):
  """
  Create a suboptimal cover using a greedy algorithm.

  Returns
    A set with the vertices in the cover.
  """

  graph = input_graph.copy()
  vertices = set[int]()

  while graph.edges:
    degrees = graph.degrees()
    max_degree_vertex = max(list(degrees.items()), key=lambda x: x[1])[0]

    vertices.add(max_degree_vertex)
    graph.remove_vertex(max_degree_vertex)

  return frozenset(vertices)

def cover_optimal1(input_graph: Graph):
  """
  List all optimal covers of a graph using a branch and bound algorithm, without pruning.

  Returns
    A tuple containing the number of explored nodes and a set of a set of vertices in each cover.
  """

  explored_node_count = 0
  stack = [(input_graph, set[int]())]
  covers = set[frozenset[int]]()

  while stack:
    graph, included_vertices = stack.pop()

    if graph.edges:
      explored_node_count += 1
      vertex1, vertex2 = next(iter(graph.edges))

      graph1 = graph.copy()
      graph2 = graph.copy()

      graph1.remove_vertex(vertex1)
      graph2.remove_vertex(vertex2)

      stack.append((graph1, included_vertices | {vertex1}))
      stack.append((graph2, included_vertices | {vertex2}))
    else:
      covers.add(frozenset(included_vertices))

  min_cover_size = min(len(cover) for cover in covers)
  return explored_node_count, {cover for cover in covers if len(cover) == min_cover_size}


@dataclass(slots=True)
class Node:
  """
  Utility class to manage nodes in the branch and bound cover algorithm.
  """

  graph: Graph
  lower_bound: float
  upper_bound: float
  vertices: set[int]

  @classmethod
  def derive(cls, graph: Graph, vertices: set[int]):
    m = len(graph.edges)
    n = len(graph.vertices)

    max_degree = max(graph.degrees().values(), default=0)
    b2, coupling_cover = cover_from_coupling(graph)

    b1 = math.ceil(len(graph.edges) / max_degree) if max_degree > 0 else 0
    b3 = 0.5 * (2 * n - 1 - math.sqrt((2 * n - 1) ** 2 - 8 * m))

    lower_bound = len(vertices) + max(b1, b2, b3)
    upper_bound = len(vertices) + len(coupling_cover)

    return cls(
      graph,
      lower_bound,
      upper_bound,
      vertices
    )

def cover_optimal2(input_graph: Graph):
  """
  Create an optimal cover using a branch and bound algorithm, with pruning.

  Returns
    A tuple containing the number of explored nodes and a set with the vertices in the cover.
  """

  explored_node_count = 0
  stack = [Node.derive(input_graph, set())]

  best_solution: Optional[frozenset[int]] = None
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

      best_solution = frozenset(node.vertices)
      best_solution_upper_bound = node.lower_bound

  assert best_solution is not None
  return explored_node_count, best_solution


def cover_optimal3(input_graph: Graph):
  """
  Create an optimal cover using a branch and bound algorithm, with pruning and branching that avoids redundant solutions.

  Returns
    A tuple containing the number of explored nodes and a set with the vertices in the cover.
  """

  explored_node_count = 0
  stack = [Node.derive(input_graph, set())]

  best_solution: Optional[frozenset[int]] = None
  best_solution_upper_bound = math.inf

  while stack:
    node = stack.pop()
    best_solution_upper_bound = min(best_solution_upper_bound, node.upper_bound)

    if node.lower_bound > best_solution_upper_bound:
      continue

    if node.graph.edges:
      explored_node_count += 1
      vertex1, vertex2 = next(iter(node.graph.edges))

      graph1 = node.graph.copy()
      graph2 = node.graph.copy()

      graph1.remove_vertex(vertex1)
      node1 = Node.derive(graph1, node.vertices | {vertex1})

      # Include all neighbor vertices of vertex1 in the cover as vertex1 is only taken in the first branch
      neighbor_vertices = graph2.neighbors(vertex1)
      assert vertex2 in neighbor_vertices

      # vertex2 is also removed because it is included in neighbor_vertices
      graph2.remove_vertices(neighbor_vertices | {vertex1})
      node2 = Node.derive(graph2, node.vertices | neighbor_vertices)

      # Add node2 first as it is more likely to be pruned because node1 contains less vertices
      stack += [node2, node1]
    else:
      assert node.upper_bound == node.lower_bound == len(node.vertices)

      best_solution = frozenset(node.vertices)
      best_solution_upper_bound = node.lower_bound

  assert best_solution is not None
  return explored_node_count, best_solution


def cover_optimal4(input_graph: Graph):
  """
  Create an optimal cover using a branch and bound algorithm, with pruning, branching that avoids redundant solutions and optimized edge selection.

  Returns
    A tuple containing the number of explored nodes and a set with the vertices in the cover.
  """

  explored_node_count = 0
  stack = [Node.derive(input_graph, set())]

  best_solution: Optional[frozenset[int]] = None
  best_solution_upper_bound = math.inf

  while stack:
    node = stack.pop()
    best_solution_upper_bound = min(best_solution_upper_bound, node.upper_bound)

    if node.lower_bound > best_solution_upper_bound:
      continue

    if node.graph.edges:
      explored_node_count += 1

      # Find the edge with the highest-degree vertex
      graph_degrees = node.graph.degrees()
      vertex1, vertex2 = max(node.graph.edges, key=(lambda edge: max(graph_degrees[edge[0]], graph_degrees[edge[1]])))

      if graph_degrees[vertex2] > graph_degrees[vertex1]:
        vertex1, vertex2 = vertex2, vertex1

      graph1 = node.graph.copy()
      graph2 = node.graph.copy()

      graph1.remove_vertex(vertex1)
      node1 = Node.derive(graph1, node.vertices | {vertex1})

      neighbor_vertices = graph2.neighbors(vertex1)
      assert vertex2 in neighbor_vertices

      graph2.remove_vertices(neighbor_vertices | {vertex1})
      node2 = Node.derive(graph2, node.vertices | neighbor_vertices)

      stack += [node2, node1]
    else:
      assert node.upper_bound == node.lower_bound == len(node.vertices)

      best_solution = frozenset(node.vertices)
      best_solution_upper_bound = node.lower_bound

  assert best_solution is not None
  return explored_node_count, best_solution


def cover_optimal5(input_graph: Graph):
  """
  Create an optimal cover using a branch and bound algorithm, with pruning, branching that avoids redundant solutions, optimized edge selection and elimination of vertices of degree 1.

  Returns
    A tuple containing the number of explored nodes and a set with the vertices in the cover.
  """

  explored_node_count = 0
  stack = [Node.derive(input_graph, set())]

  best_solution: Optional[frozenset[int]] = None
  best_solution_upper_bound = math.inf

  while stack:
    node = stack.pop()
    best_solution_upper_bound = min(best_solution_upper_bound, node.upper_bound)

    if node.lower_bound > best_solution_upper_bound:
      continue

    if node.graph.edges:
      explored_node_count += 1

      graph_degrees = node.graph.degrees()
      vertex1, vertex2 = max(node.graph.edges, key=(lambda edge: max(graph_degrees[edge[0]], graph_degrees[edge[1]])))

      if graph_degrees[vertex2] > graph_degrees[vertex1]:
        vertex1, vertex2 = vertex2, vertex1

      graph1 = node.graph.copy()
      graph2 = node.graph.copy()

      graph1.remove_vertex(vertex1)
      node1 = Node.derive(graph1, node.vertices | {vertex1})

      # Eliminate vertex2 if it has degree 1
      if graph_degrees[vertex2] > 1:
        neighbor_vertices = graph2.neighbors(vertex1)
        assert vertex2 in neighbor_vertices

        graph2.remove_vertices(neighbor_vertices | {vertex1})
        node2 = Node.derive(graph2, node.vertices | neighbor_vertices)

        stack.append(node2)

      stack.append(node1)
    else:
      assert node.upper_bound == node.lower_bound == len(node.vertices)

      best_solution = frozenset(node.vertices)
      best_solution_upper_bound = node.lower_bound

  assert best_solution is not None
  return explored_node_count, best_solution
