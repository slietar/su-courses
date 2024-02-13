from pathlib import Path
import pickle
from pprint import pprint
from matplotlib import pyplot as plt

import numpy as np

with Path('output/mutations.pkl').open('rb') as file:
  mutations = pickle.load(file)


length = mutations[-1]['position'] + 1
mutations_by_position = [[] for _ in range(length)]

for mutation in mutations:
  mutations_by_position[mutation['position']].append(mutation)

effect_keys = list(mutations[0]['effects'].keys())

effects = np.array([tuple(len([0 for mutation in mutations if mutation['effects'][key]]) for key in effect_keys) for mutations in mutations_by_position])

fig, ax = plt.subplots()

ax.imshow(effects[0:1000, :].T, aspect='auto', interpolation='none')
ax.set_yticks(np.arange(len(effect_keys)))
ax.set_yticklabels(effect_keys)

plt.show()
