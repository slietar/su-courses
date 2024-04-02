from dataclasses import dataclass
from typing import Any, Callable, Sequence
import unicodedata
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np


light_blue = (0xb2 / 0xff, 0xb2 / 0xff, 0xff / 0xff)
light_red = (0xff / 0xff, 0xb2 / 0xff, 0xb2 / 0xff)

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
      rect = Rectangle(
        [x1, y1],
        x2 - x1,
        y2 - y1,
        color=[light_blue, light_red][grid[i, j]],
        linewidth=None
      )

      ax.add_artist(rect)


def filter_axes(axs: np.ndarray, /):
  for ax in axs[:-1, :].flat:
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_xlabel(None)

  for ax in axs[:, 1:].flat:
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.set_ylabel(None)


def plot_boundary(ax: Axes, fn: Callable[[np.ndarray], np.ndarray], *, label: bool = True, x_range: tuple[float, float] = (-2, 2), y_range: tuple[float, float] = (-2, 2)):
  x_values = np.linspace(*x_range, 100)
  y_values = np.linspace(*y_range, 100)

  x, y = np.meshgrid(x_values, y_values)
  g = np.c_[x.ravel(), y.ravel()]

  ax.contour(x, y, fn(g).reshape(len(x_values), len(y_values)), colors='gray', levels=[0], linestyles='dashed')
  ax.plot([], [], color='gray', linestyle='dashed', label=('Frontière de décision' if label else None))


def plot_boundary_contour(ax: Axes, fn: Callable[[np.ndarray], np.ndarray], *, label: bool = True, x_range: tuple[float, float] = (-2, 2), y_range: tuple[float, float] = (-2, 2)):
  x_values = np.linspace(*x_range, 100)
  y_values = np.linspace(*y_range, 100)

  x, y = np.meshgrid(x_values, y_values)
  g = np.c_[x.ravel(), y.ravel()]

  ax.contourf(x, y, fn(g).reshape(len(x_values), len(y_values)), colors=[light_blue, light_red], levels=[-2, 0, 2])

def remove_accents(input_str: str):
  nfkd_form = unicodedata.normalize('NFKD', input_str)
  return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


SUPERSCRIPT_CHARS = {
  '0': '\u2070',
  '1': '\u00b9',
  '2': '\u00b2',
  '3': '\u00b3',
  '4': '\u2074',
  '5': '\u2075',
  '6': '\u2076',
  '7': '\u2077',
  '8': '\u2078',
  '9': '\u2079',
  '-': '\u207b'
}

def format_scientific(value: float, *, precision: int = 2):
  left, right = f'{value:.{precision}e}'.split('e')

  exp_sign = (SUPERSCRIPT_CHARS['-'] if right[0] == '-' else '')
  exp = ''.join(SUPERSCRIPT_CHARS[digit] for digit in right[1:].lstrip('0'))

  return f'{left} \u00b7 10' + exp_sign + exp
