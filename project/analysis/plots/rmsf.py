from matplotlib.axes import Axes
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from .. import data, plots, shared
from ..rmsf import rmsf


output_path = shared.output_path / 'rmsf'
output_path.mkdir(exist_ok=True)

for domain_kind in data.domain_kinds:
  domains, rmsf_arr = rmsf[domain_kind]
  # rmsf_arr = rmsf_arr[:, :5]
  # rmsf_arr[0, :] = 100

  fig, ax = plt.subplots()
  fig.set_figheight(8.0)

  divider = make_axes_locatable(ax)

  im = ax.imshow(rmsf_arr, extent=(0.5, rmsf_arr.shape[1] + 0.5, 0, rmsf_arr.shape[0]), aspect='auto', cmap='hot')

  ax1: Axes = divider.append_axes('top', 1.2, pad=0.1, sharex=ax)
  ax1.plot(range(1, rmsf_arr.shape[1] + 1), np.nanmean(rmsf_arr, axis=0))
  ax1.xaxis.set_tick_params(bottom=False, labelbottom=False)
  ax1.grid()

  cbar = fig.colorbar(im, ax=ax)

  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('RMSF (Å)', rotation=270)

  ax.set_yticks(
    labels=reversed([str(number) for number in domains['number']]),
    ticks=(np.arange(len(domains)) + 0.5)
  )

  # ax.yaxis.set_tick_params(left=False)

  ax1.set_title(domain_kind)


  with (output_path / f'{domain_kind}.png').open('wb') as file:
    fig.savefig(file)
