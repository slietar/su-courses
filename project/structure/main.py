import pickle
from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

with Path('output/data.pkl').open('rb') as file:
  data = pickle.load(file)

domains = data['domains']
mutations = data['mutations']
length = len(data['protein']['sequence'])

plot_start = 0
plot_end = length

mutations_by_position = [[] for _ in range(length)]

for mutation in mutations:
  mutations_by_position[mutation['position'] // 3].append(mutation)

effect_keys = list(reversed(mutations[0]['effects'].keys()))

effects = np.array([tuple(len([0 for mutation in mutations if mutation['effects'][key]]) for key in effect_keys) for mutations in mutations_by_position])

fig, ax = plt.subplots(figsize=(20, 10))

ax.imshow(effects[plot_start:plot_end, :].T, aspect='auto', interpolation='none')

for domain in domains:
  start, end = domain['range']
  rect = patches.Rectangle([start, len(effect_keys) - 0.5], end - start - 1, 1, color=('r' if 'EGF' in domain['name'] else 'b'), alpha=0.5, linewidth=0)
  # ax.axvline(x=domain['range'][0], color='r')
  # ax.axvline(x=domain['range'][1], color='r')

  ax.add_artist(rect)

# Regions of interest
for start, end in [
  (951, 1362),
  (1689, 1762)
]:
  rect = patches.Rectangle([start, -0.5], end - start - 1, len(effect_keys) + 1, color='g', alpha=0.5, linewidth=0)
  ax.add_artist(rect)

ax1 = ax.twiny()
ax1.set_xticks(
  labels=[domain['name'] for domain in domains],
  ticks=[(domain['range'][0] + domain['range'][1]) * 0.5 for domain in domains],
  rotation='vertical'
)

ax.set_xticks(np.arange(plot_start, plot_end, 100))
ax.set_yticks(np.arange(len(effect_keys) + 1))
ax.set_yticklabels([*effect_keys, 'Domains'])
ax.set_xlim(plot_start - 0.5, plot_end - 0.5)
ax.set_ylim(-0.5, len(effect_keys) + 1 - 0.5)
ax.set_ylabel('Sequence')

ax1.tick_params('x', top=False)
ax.tick_params('y', left=False)

ax1.set_xlim(ax.get_xlim())

fig.subplots_adjust(top=0.7)

fig.savefig('output/map.png')
