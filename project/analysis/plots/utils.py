from matplotlib import patches
from matplotlib.axes import Axes
import pandas as pd

from .. import data
from ..hospital import hospital_domains


def highlight_domains(ax: Axes, y: float):
  ax1 = ax.twiny()
  ax1.set_xticks(
    labels=data.domains.name,
    ticks=[(domain.start_position - 0.5 + domain.end_position + 0.5) * 0.5 for domain in data.domains.itertuples()],
    rotation='vertical'
  )

  ax1.set_xlim(ax.get_xlim())
  # ax1.tick_params('x', top=False)

  colors = {
    'EGF': 'r',
    'EGFCB': 'g',
    'TB': 'b'
  }

  for domain in data.domains.itertuples():
    rect = patches.Rectangle(
      [domain.start_position - 0.5, y],
      domain.end_position - domain.start_position + 1,
      1,
      alpha=0.5,
      edgecolor='white',
      facecolor=colors[domain.kind],
      linewidth=1
    )

    ax.add_artist(rect)

  for start_position, end_position in data.interest_regions.values():
    rect = patches.Rectangle(
      [start_position - 0.5, y + 0.25],
      end_position - start_position + 1,
      0.25,
      color='blueviolet',
      alpha=0.5,
      linewidth=0
    )

    ax.add_artist(rect)

  for domain in data.domains.loc[:, ['start_position', 'end_position']].merge(hospital_domains.loc[:, ['number', 'unip_name']], left_index=True, right_on='unip_name').itertuples():
    if not pd.isna(domain.number):
      ax.text((domain.start_position + domain.end_position) * 0.5, y + 0.75, str(domain.number), ha='center', va='center', fontsize=10, color='white')
