from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import IdentityTransform, Transform
import numpy as np
import pandas as pd

from .. import data
from ..hospital import hospital_domains


def get_transform_linear_component(transform: Transform, /):
  return transform.transform((1, 1)) - transform.transform((0, 0))


def highlight_domains(ax: Axes, y: float):
  # ax1 = ax.twiny()
  # ax1.set_xticks(
  #   labels=data.domains.name,
  #   ticks=[(domain.start_position - 0.5 + domain.end_position + 0.5) * 0.5 for domain in data.domains.itertuples()],
  #   rotation='vertical'
  # )

  # ax1.set_xlim(ax.get_xlim())
  # ax1.tick_params('x', top=False)

  colors = {
    'EGF': 'r',
    'EGFCB': 'g',
    'TB': 'b'
  }

  domain_row_height = 0.2

  ax_transform = ax.transData + ax.figure.dpi_scale_trans.inverted()
  ax_transform_lin = get_transform_linear_component(ax_transform)

  for domain in data.domains.itertuples():
    rect = patches.Rectangle(
      ax_transform.transform((domain.start_position - 0.5, y)),
      ax_transform_lin[0] * (domain.end_position - domain.start_position + 1),
      domain_row_height,
      alpha=0.5,
      clip_on=False,
      edgecolor='white',
      facecolor=colors[domain.kind],
      linewidth=1,
      transform=ax.figure.dpi_scale_trans
    )

    ax.add_artist(rect)

  # for region in data.interest_regions.itertuples():
  #   rect = patches.Rectangle(
  #     [region.start_position - 0.5, y + 0.25],
  #     region.end_position - region.start_position + 1,
  #     0.25,
  #     color='blueviolet',
  #     alpha=0.8,
  #     linewidth=0
  #   )

  #   ax.add_artist(rect)
  #   ax.text((region.start_position + region.end_position) * 0.5, y + 0.375, region.name, ha='center', va='center', fontsize=10, color='white')

  # for domain in data.domains.loc[:, ['start_position', 'end_position']].merge(hospital_domains.loc[:, ['number', 'unip_name']], left_index=True, right_on='unip_name').itertuples():
  for domain in data.domains.itertuples():
    # if not pd.isna(domain.number):
    ax.text(
      *(ax_transform.transform(((domain.start_position + domain.end_position) * 0.5, 0)) + [0, domain_row_height * 0.5]),
      # (domain.start_position + domain.end_position) * 0.5,
      # y + 0.75,
      str(domain.number),
      clip_on=False,
      color='white',
      fontsize=4,
      ha='center',
      transform=ax.figure.dpi_scale_trans,
      va='center'
    )


def set_colobar_label(cbar: Colorbar, label: str):
  cbar.ax.get_yaxis().labelpad = 10
  cbar.ax.set_ylabel(label, rotation=270)


class ProteinMap:
  def __init__(self, ax: Axes, protein_length: int = data.protein_length):
    self.ax = ax
    self.protein_length = protein_length
    self.ylabels = list[str]()


  def plot_dataframe(self, df_like: pd.DataFrame | pd.Series, /, **kwargs):
    df = df_like if isinstance(df_like, pd.DataFrame) else df_like.to_frame()
    df_reindexed = df.reindex(index=data.position_index, fill_value=np.nan)
    im = self.ax.imshow(df_reindexed.to_numpy(dtype=float, na_value=np.nan).T, aspect='auto', cmap='plasma', extent=((0.5, self.protein_length + 0.5, -len(self.ylabels) - len(df.columns), -len(self.ylabels))), interpolation='none', **kwargs)

    self.ylabels += list(df.columns)
    return im

  def finish(self):
    x_ticks = np.arange(199, data.protein_length + 1, 200)
    self.ax.set_xticks(x_ticks + 0.5)
    self.ax.set_xticklabels([str(x + 1) for x in x_ticks])

    self.ax.set_ylim(-len(self.ylabels), 0)
    self.ax.tick_params('y', left=False)
    self.ax.set_yticks(
      labels=self.ylabels,
      ticks=(-np.arange(len(self.ylabels)) - 0.5)
    )

    highlight_domains(self.ax, 0)
