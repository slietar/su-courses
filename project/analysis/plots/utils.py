import math
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.transforms import Transform

from .. import data


def get_transform_linear_component(transform: Transform, /):
  return transform.transform((1, 1)) - transform.transform((0, 0))


def set_colobar_label(cbar: Colorbar, label: str):
  cbar.ax.get_yaxis().labelpad = 10
  cbar.ax.set_ylabel(label, rotation=270)


# In inches
colorbar_width = 0.1
domain_row_height = 0.1
ax_gap = 0.1
data_row_height = 0.5
xaxis_height = 0.2

class ProteinMap:
  def __init__(self, map_range: tuple[int, int] = (1, data.protein_length), *, max_ax_length: int = 2000):
    length = map_range[1] - map_range[0] + 1
    ax_count = math.ceil(length / max_ax_length)
    ax_length = math.ceil(length / ax_count)

    self.ax_bounds = [(
      map_range[0] + ax_index * ax_length + 1,
      min(map_range[0] + (ax_index + 1) * ax_length, map_range[1])
    ) for ax_index in range(ax_count)]

    self.fig, axs = plt.subplots(ax_count, 1, squeeze=False)
    self.axs = list[Axes](axs.flat)
    self.length = length

    self.colorbars = list[tuple[AxesImage, str, dict]]()
    self.ylabels = list[str]()

  def add_colorbar(self, im: AxesImage, /, label: str, **kwargs):
    self.colorbars.append((im, label, kwargs))

  def plot_dataframe(self, df_like: pd.DataFrame | pd.Series, /, label: Optional[str] = None, **kwargs):
    df = df_like if isinstance(df_like, pd.DataFrame) else df_like.to_frame()
    df_reindexed = df.reindex(index=data.position_index, fill_value=np.nan)

    updated_kwargs: dict = dict(cmap='plasma') | kwargs

    if not 'vmin' in updated_kwargs:
      updated_kwargs['vmin'] = df.min().min()
    if not 'vmax' in updated_kwargs:
      updated_kwargs['vmax'] = df.max().max()

    for ax, (ax_start, ax_end) in zip(self.axs, self.ax_bounds):
      im = ax.imshow(
        df_reindexed.loc[ax_start:ax_end].to_numpy(dtype=float, na_value=np.nan).T,
        aspect='auto',
        extent=((ax_start - 0.5, ax_end + 0.5, -len(self.ylabels) - len(df.columns), -len(self.ylabels))),
        interpolation='none',
        **updated_kwargs
      )

    if label is not None:
      self.colorbars.append((im, label, {}))

    self.ylabels += list(df.columns)

  def finish(self):
    for ax, (ax_start, ax_end) in zip(self.axs, self.ax_bounds):
      # x_ticks = np.arange(ax_start, ax_end, 200)

      # ax.set_xticks(x_ticks + 0.5)
      # ax.set_xticklabels([str(x + 1) for x in x_ticks])

      ax.set_xlim(ax_start - 0.5, ax_end + 0.5)
      ax.set_ylim(-len(self.ylabels), 0)
      ax.tick_params('y', left=False)
      ax.set_yticks(
        labels=self.ylabels,
        ticks=(-np.arange(len(self.ylabels)) - 0.5)
      )

    ax_count = len(self.axs)
    ax_height = (len(self.ylabels) * data_row_height + xaxis_height)
    legend_height = 0.25
    total_height = legend_height + (domain_row_height + ax_height) * ax_count + ax_gap * (ax_count - 1)
    self.fig.set_figheight(total_height)

    self.fig.subplots_adjust(
      top=(1.0 - (domain_row_height + legend_height) / total_height),
      bottom=(xaxis_height / total_height),
      right=(1.0 if self.colorbars else 0.98),
      hspace=((domain_row_height + ax_gap + xaxis_height) / ax_height)
    )


    # Add color bars

    inch_size = get_transform_linear_component(self.fig.dpi_scale_trans + self.fig.transFigure.inverted())

    for im, label, kwargs in self.colorbars:
      cbar = self.fig.colorbar(
        im,
        aspect=((total_height - domain_row_height - xaxis_height) / colorbar_width),
        ax=self.axs,
        fraction=(0.6 * inch_size[0]),
        pad=(0.2 * inch_size[0]),
        **kwargs
      )

      cbar.ax.get_yaxis().labelpad = 10
      cbar.ax.set_ylabel(label, rotation=270)


    # Add domains

    colors = {
      'EGF': 'C0',
      'EGFCB': 'C1',
      'TB': 'C2'
    }

    for ax, (ax_start, ax_end) in zip(self.axs, self.ax_bounds):
      ax_transform = ax.transData + self.fig.dpi_scale_trans.inverted()
      ax_transform_lin = get_transform_linear_component(ax_transform)

      for domain in data.domains.itertuples():
        if (domain.start_position > ax_end) or (domain.end_position < ax_start):
          continue

        start_position = max(domain.start_position, ax_start)
        end_position = min(domain.end_position, ax_end)

        if (end_position - start_position) < 10:
          continue

        rect = patches.Rectangle(
          ax_transform.transform((start_position - 0.5, 0)),
          ax_transform_lin[0] * (end_position - start_position + 1),
          domain_row_height,
          clip_on=False,
          edgecolor='white',
          facecolor=colors[domain.kind],
          linewidth=1,
          transform=self.fig.dpi_scale_trans
        )

        ax.add_artist(rect)

        ax.text(
          *(ax_transform.transform(((start_position + end_position) * 0.5, 0)) + [0, domain_row_height * 0.5]),
          str(domain.number),
          clip_on=False,
          color='white',
          fontsize=5,
          ha='center',
          transform=self.fig.dpi_scale_trans,
          va='center'
        )

    self.fig.legend(
      handles=[patches.Patch(color=color, label=label) for label, color in colors.items()],
      loc='upper right',
      ncol=len(colors)
    )
