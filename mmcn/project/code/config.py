from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.rcsetup import cycler


plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.sf'] = 'Helvetica'
plt.rcParams['figure.figsize'] = 21.0 / 2.54 - 2.0, 4.0
plt.rcParams['font.size'] = 11.0
plt.rcParams['figure.dpi'] = 288
plt.rcParams['grid.color'] = 'whitesmoke'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.prop_cycle'] = cycler(color=[
  '#348abd',
  '#e24a33',
  '#988ed5',
  '#fbc15e',
  '#777777',
  '#8eba42',
  '#ffb5b8'
])
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.edgecolor'] = 'k'


output_path = Path('output')
output_path.mkdir(exist_ok=True)
