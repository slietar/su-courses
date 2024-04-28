from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
import numpy as np


isclosereal = lambda x: np.isclose(x.imag, 0)

def group(x: np.ndarray, /):
  current_item = x[0, ...]
  current_count = 1

  for index in range(1, x.shape[0]):
    if x[index, ...] != current_item:
      yield current_item, current_count
      current_item = x[index, ...]
      current_count = 1
    else:
      current_count += 1

  yield current_item, current_count


def draw_circle(ax: Axes, point: np.ndarray):
  trans = ax.get_figure().dpi_scale_trans + transforms.ScaledTranslation(pt[0], pt[1], ax.transData)
  circle = Circle((0.0, 0.0), clip_on=False, edgecolor='black', linewidth=1, facecolor='none', path_effects=[withStroke(linewidth=3, foreground='white')], radius=0.05, transform=trans, zorder=10)

  ax.add_artist(circle)
  ax.text(0.1, -0.005, 'Bifurcation', ha='left', va='center', fontsize=8, color='black', path_effects=[withStroke(linewidth=2, foreground='white')], transform=trans)
