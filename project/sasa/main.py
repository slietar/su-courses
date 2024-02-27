import pickle
from pathlib import Path

import freesasa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Load processed data

with Path('../structure/output/data.pkl').open('rb') as file:
  data = pickle.load(file)

domains = pd.DataFrame(data['domains'])
mutations = pd.DataFrame([{**{ k: v for k, v in m.items() if k != 'effects' }, **m['effects']} for m in data['mutations']])


# Calculate SASAs

structure = freesasa.Structure('../drive/FBN1_AlphaFold.pdb')
result = freesasa.calc(structure)

residue_areas = [{
  'number': int(area.residueNumber),

  'apolar': area.apolar,
  'main_chain': area.mainChain,
  'polar': area.polar,
  'side_chain': area.sideChain,
  'total': area.total,
} for area in result.residueAreas()['A'].values()]

sasa_df = pd.DataFrame.from_records(residue_areas, index='number')
sorted_df = sasa_df.sort_values('total')

# print('30 residues with lowest SASA')
# print('select nterm, resi', '+'.join(str(a) for a in sorted_df[:30].index))
# print()

# print('30 residues with highest SASA')
# print('select nterm, resi', '+'.join(str(a) for a in sorted_df[-30:].index))
# print()


mutated_residue_mask = sasa_df.index.isin(mutations['residue'])


def plot1():
  fig, ax = plt.subplots()

  ax.scatter(sasa_df['main_chain'], sasa_df['side_chain'], c=np.array(['b', 'r'])[mutated_residue_mask.astype(int)], s=1)

  ax.set_xlabel(r'Main chain ($\AA^2$)')
  ax.set_ylabel(r'Side chain ($\AA^2$)')

  ax.legend((
    ax.scatter([], [], c='b', s=1),
    ax.scatter([], [], c='r', s=1)
  ), ('Non-mutated residue', 'Mutated residue'))

  fig.savefig('output/1.png', dpi=300)


def plot2():
  fig, ax = plt.subplots()

  ax.scatter(sasa_df['apolar'], sasa_df['polar'], c=np.array(['b', 'r'])[mutated_residue_mask.astype(int)], s=1)

  ax.set_xlabel(r'Apolar ($\AA^2$)')
  ax.set_ylabel(r'Polar ($\AA^2$)')

  ax.legend((
    ax.scatter([], [], c='b', s=1),
    ax.scatter([], [], c='r', s=1)
  ), ('Non-mutated residue', 'Mutated residue'))

  fig.savefig('output/2.png', dpi=300)


def plot3():
  fig, ax = plt.subplots()

  ax.boxplot([
    sasa_df[mutated_residue_mask]['total'],
    sasa_df[~mutated_residue_mask]['total']
  ], labels=['Mutated residues', 'Non-mutated residues'])

  ax.set_ylabel('Surface (Å²)')

  fig.savefig('output/3.png', dpi=300)



Path('output').mkdir(exist_ok=True)

plot1()
plot2()
plot3()
# plt.show()
