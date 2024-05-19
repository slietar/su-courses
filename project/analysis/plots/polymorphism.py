from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .. import shared
from ..polymorphism import polymorphism_counts


fig, ax = plt.subplots()
ax: Axes

x = polymorphism_counts[polymorphism_counts > 0]

logbins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 30)
ax.hist(x, bins=logbins)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()

ax.set_xlabel('Nombre d\'observations de chaque résidu mutés sans être connu comme pathogénique')
ax.set_ylabel('Nombre de résidus')


with (shared.output_path / 'polymorphism_histogram.png').open('wb') as file:
  fig.savefig(file)
