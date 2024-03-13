from matplotlib import pyplot as plt
import pandas as pd
from .. import shared
from ..cv import cv
from ..sasa import sasa


df = pd.concat([cv, sasa.total.rename('sasa')], axis='columns')

fig, ax = plt.subplots()

ax.scatter(df.cv, df.sasa, s=4)
ax.set_xlabel('Circular Variance')
ax.set_ylabel('Relative total SASA')

with (shared.output_path / 'cv_sasa.png').open('wb') as file:
  fig.savefig(file, dpi=300)
