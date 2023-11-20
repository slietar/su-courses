# Utilities for building and drawing trees

import re
from dataclasses import dataclass, field
from typing import Optional, Self
from matplotlib import pyplot as plt

from matplotlib.lines import Line2D
import numpy as np


@dataclass
class Leaf:
  value: str

@dataclass
class Node:
  children: list[tuple[Self | Leaf, float]]

  def draw(self):
    fig, ax = plt.subplots()

    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    @dataclass
    class NodeMarker:
      node: Node
      x: float

      current_child_index: int = 0
      y_start: float = 0.0
      y_end: float = 0.0

    pointer = [NodeMarker(self, 0.0)]
    leaf_vertical_padding = 0.1

    current_y = 0.0
    max_x = 0.0

    while pointer:
      marker = pointer[-1]

      if marker.current_child_index >= len(marker.node.children):
        pointer.pop()
        ax.add_line(Line2D([marker.x, marker.x], [marker.y_start, marker.y_end], color='k'))

        if pointer:
          marker_y_mid = (marker.y_start + marker.y_end) * 0.5
          prev_marker = pointer[-1]

          if prev_marker.current_child_index == 0:
            prev_marker.y_start = marker_y_mid

          prev_marker.y_end = marker_y_mid
          prev_marker.current_child_index += 1

          ax.add_line(Line2D([prev_marker.x, marker.x], [marker_y_mid, marker_y_mid], color='k'))
      else:
        child_node, child_dist = marker.node.children[marker.current_child_index]

        match child_node:
          case Leaf():
            current_y += leaf_vertical_padding

            if marker.current_child_index == 0:
              marker.y_start = current_y

            leaf_x = marker.x + child_dist
            max_x = max(max_x, marker.x + child_dist)

            ax.add_line(Line2D([marker.x, marker.x + child_dist], [current_y, current_y], color='k'))
            ax.text(leaf_x + 0.05, current_y, child_node.value[0:3], verticalalignment='center')

            marker.current_child_index += 1
            marker.y_end = current_y
            current_y += leaf_vertical_padding
          case Node():
            pointer.append(NodeMarker(child_node, marker.x + child_dist))

    ax.set_xlim(-0.1, max_x + 0.15)
    ax.set_ylim(current_y, 0.0)

    bottom_height = 0.4
    total_height = current_y * 2.0 + bottom_height

    fig.set_size_inches(5.0, total_height)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=(bottom_height / total_height))

    return fig, ax

  @classmethod
  def parse(cls, contents: str, /):
    index = 0
    stack = list[Node]()

    while index < len(contents):
      ch = contents[index]

      if ch == ')':
        index += 1
        last = stack.pop()

        if stack:
          stack[-1].children.append((last, 1.0))
        else:
          assert index == len(contents)
          return last
      else:
        if stack and stack[-1].children:
          assert ch == ','
          index += 1

        ch = contents[index]

        if ch == '(':
          index += 1
          stack.append(Node([]))
        else:
          match = re.match(r'([a-zA-Z-]+)', contents[index:])
          assert match

          name = match.group(0)
          index += len(name)

          stack[-1].children.append((Leaf(name), 1.0))

    raise RuntimeError

tree = Node.parse('((((Electrode,Magnezone),Porygon-Z),((((Aggron,Bastiodon),Forretress),Ferrothorn),((((Regirock,Regice),Registeel),Metagross),Klinklang),Genesect)),Probopass)')

# fig, _ = tree.draw()
# fig.savefig('out.png')


mutation_matrix = np.array([
  [0, 3, 4, 9],
  [3, 0, 2, 4],
  [4, 2, 0, 4],
  [9, 4, 4, 0]
])


nucleotides = {
  'Probopass': 'A',
  'Aggron': 'T',
  'Bastiodon': 'T',
  'Regirock': 'G',
  'Registeel': 'G',
  'Regice': 'G',
  'Klinklang': 'G',
  'Metagross': 'C',
  'Genesect': 'A',
  'Porygon-Z': 'C',
  'Magnezone': 'C',
  'Forretress': 'T',
  'Electrode': 'A',
  'Ferrothorn': 'G'
}

@dataclass
class NodeMarker:
  node: Node
  arr: list[list[float]] = field(default_factory=list)
  # mat: list[tuple[list[float], list[int]]] = field(default_factory=list)
  current_child_index: int = 0


def sankoff(root: Node):
  stack = [NodeMarker(root)]

  while True:
    head = stack[-1]

    if head.current_child_index >= len(head.node.children):
      stack.pop()

      if stack:
        a = np.array([arr + mutation_matrix for arr in head.arr])
        _ = np.argmin(a, axis=1)
        print(_)
        stack[-1].arr.append(np.min(a, axis=1).sum(axis=0))
        stack[-1].current_child_index += 1

        continue
      else:
        return 0

        # sum([(arr + mutation_matrix).min(axis=0) for arr in head.arr])
        # stack[-1].arr.append(sum([(arr + mutation_matrix).min(axis=0) for arr in head.arr]))

    child_node, child_dist = head.node.children[head.current_child_index]

    match child_node:
      case Leaf():
        nucleotide = ['A', 'C', 'G', 'T'].index(nucleotides[child_node.value])

        arr = [np.inf] * 4
        arr[nucleotide] = 0.0

        head.arr.append(arr)
        head.current_child_index += 1

      case Node():
        stack.append(NodeMarker(child_node))


sankoff(tree)
