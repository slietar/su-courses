import pandas as pd
from matplotlib import pyplot as plt

from .. import data, shared
from ..cv import cv
from ..sasa import sasa


pathogenic = data.all_mutations.groupby('position').pathogenicity.aggregate(lambda x: ((1 <= x) & (x <= 3)).any()).rename('pathogenic')

df = pd.concat([cv, sasa.total.rename('sasa'), pathogenic], axis='columns')
df.pathogenic.fillna(False, inplace=True)


fig, ax = plt.subplots()

ax.scatter(df.cv, df.sasa, c=df.pathogenic, s=4)
ax.set_xlabel('Circular variance')
ax.set_ylabel('Relative total SASA')

with (shared.output_path / 'cv_sasa.png').open('wb') as file:
  fig.savefig(file, dpi=300)
