from dataclasses import dataclass
from typing import Any, Sequence
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
# from sklearn.tree import Tree


@dataclass
class Range:
  start: int
  end: int


def plot_tree(ax: Axes, tree, *, range_x: tuple[float, float], range_y: tuple[float, float]):
  grid = np.array([[0]])
  splits = [[], []]

  stack: list[tuple[int, tuple[Range, Range]]] = [
    (0, (Range(0, 1), Range(0, 1)))
  ]

  while stack:
    node_id, node_range = stack.pop()

    left_id = tree.children_left[node_id]
    right_id = tree.children_right[node_id]

    if left_id != right_id: # Split node
      feature = tree.feature[node_id]

      split_index = next((index for index, split in enumerate(splits[feature]) if split > tree.threshold[node_id]), len(splits[feature]))
      splits[feature].insert(split_index, tree.threshold[node_id])

      match feature:
        case 0:
          grid = np.r_[
            grid[:split_index, :],
            grid[split_index, :][None, :],
            grid[split_index:, :]
          ]
        case 1:
          grid = np.c_[
            grid[:, :split_index],
            grid[:, split_index][:, None],
            grid[:, split_index:]
          ]

      for _, other_node_range in stack:
        if other_node_range[feature].start > split_index:
          other_node_range[feature].start += 1

        if other_node_range[feature].end > split_index:
          other_node_range[feature].end += 1

      #
      # ---|---*---|---
      #  0     1     2    OLD GRID
      #
      # ---|---|---|---
      #  0   1   2   3    NEW GRID
      #
      #
      stack.append((left_id, (
        Range(node_range[0].start, (split_index + 1 if feature == 0 else node_range[0].end)),
        Range(node_range[1].start, (split_index + 1 if feature == 1 else node_range[1].end))
      )))

      stack.append((right_id, (
        (Range(split_index + 1, node_range[0].end + 1) if feature == 0 else node_range[0]),
        (Range(split_index + 1, node_range[1].end + 1) if feature == 1 else node_range[1])
      )))
    else: # Leaf node
      grid[
        node_range[0].start:node_range[0].end,
        node_range[1].start:node_range[1].end
      ] = tree.value[node_id].argmax()

  for i, (x1, x2) in enumerate(zip([range_x[0], *splits[0]], [*splits[0], range_x[1]])):
    for j, (y1, y2) in enumerate(zip([range_y[0], *splits[1]], [*splits[1], range_y[1]])):
      rect = Rectangle([x1, y1], x2 - x1, y2 - y1, alpha=1.0, color=[(0xb2 / 0xff, 0xb2 / 0xff, 0xff / 0xff), (0xff / 0xff, 0xb2 / 0xff, 0xb2 / 0xff)][grid[i, j]], linewidth=None)
      ax.add_artist(rect)


def filter_axes(axs: np.ndarray, /):
  for ax in axs[:-1, :].flat:
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_xlabel(None)

  for ax in axs[:, 1:].flat:
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.set_ylabel(None)
