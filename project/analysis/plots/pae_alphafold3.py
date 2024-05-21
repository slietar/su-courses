import json

from matplotlib import pyplot as plt
import numpy as np

from . import utils
from .. import shared


with (shared.root_path / 'sources/alphafold3-global/fold_fbn1_full_data_0.json').open() as file:
  pae = np.array(json.load(file)['pae'])


fig, ax = plt.subplots()
fig.set_figheight(3.0)
# fig.subplots_adjust(
#   top=0.9,
#   bottom=0.1
# )

ax.matshow(pae, cmap='YlGn_r')
cbar = plt.colorbar(ax.matshow(pae, cmap='YlGn_r'))

ax.xaxis.set_tick_params(bottom=False)

utils.set_colobar_label(cbar, 'PAE (Å)')


with (shared.output_path / 'pae_alphafold3.png').open('wb') as file:
  plt.savefig(file)
