import numpy as np

from .. import data, shared
from ..cv import cv
from .utils import ProteinMap


df = cv.reindex(index=data.position_index, fill_value=np.nan).loc[:, [10.0, 20.0, 30.0, 40.0, 50.0]]

map = ProteinMap((80, 1000))
map.plot_dataframe(df, label='Variance circulaire', vmin=0.0, vmax=1.0)
map.finish()

with (shared.output_path / 'cv.png').open('wb') as file:
  map.fig.savefig(file, dpi=300)
