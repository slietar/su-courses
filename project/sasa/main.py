import pickle
from pathlib import Path
import sys

import freesasa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Prepare output

output_path = Path('output')


# Load processed data

with Path('../structure/output/data.pkl').open('rb') as file:
  core = pickle.load(file)

with Path('../gemme/output/data.pkl').open('rb') as file:
  gemme, residues = pickle.load(file)

domains = pd.DataFrame(core['domains'])
mutations = pd.DataFrame([{**{ k: v for k, v in m.items() if k != 'effects' }, **m['effects']} for m in core['mutations']])


# Calculate SASAs

structure = freesasa.Structure('../drive/FBN1_AlphaFold.pdb')
result = freesasa.calc(structure)

residue_areas = [{
  'number': int(area.residueNumber),

  'apolar': area.relativeApolar,
  'main_chain': area.relativeMainChain,
  'polar': area.relativePolar,
  'side_chain': area.relativeSideChain,
  'total': area.relativeTotal,
} for area in result.residueAreas()['A'].values()]

sasa_df = pd.DataFrame.from_records(residue_areas, index='number')
sorted_df = sasa_df.sort_values('total')

# print('30 residues with lowest SASA')
# print('select nterm, resi', '+'.join(str(a) for a in sorted_df[:30].index))
# print()

# print('30 residues with highest SASA')
# print('select nterm, resi', '+'.join(str(a) for a in sorted_df[-30:].index))
# print()

# print(sasa_df)
# sys.exit()

mutated_residue_mask = sasa_df.index.isin(mutations['residue'])


gemme_mean = np.nanmean(gemme, axis=0)
residues_inv = {res: index for index, res in enumerate(residues)}

a = pd.Series(gemme[[residues_inv[res] for res in mutations['new_residue']], mutations['residue'] - 1], index=mutations.index)
b = pd.Series(gemme_mean[mutations['residue'] - 1], index=mutations.index)
c = sasa_df['total'].loc[mutations['residue']]

threshold = -0.779

fig, ax = plt.subplots(figsize=(10, 8))
# fig, ax = plt.subplots()

ax.axline((0, threshold), color='gray', linestyle='--', slope=0)
ax.axline((threshold, 0), (threshold, 1), color='gray', linestyle='--')

scatter = ax.scatter(b, a, c=c, s=6)

ax.set_xlabel('Mean')
ax.set_ylabel('Mutation')
ax.set_title('Hospital mutations')

fig.colorbar(scatter, ax=ax)


with (output_path / 'gemme_sasa.png').open('wb') as file:
  fig.savefig(file, dpi=300)


# def plot1():
#   fig, ax = plt.subplots()

#   ax.scatter(sasa_df['main_chain'], sasa_df['side_chain'], c=np.array(['b', 'r'])[mutated_residue_mask.astype(int)], s=1)

#   ax.set_xlabel(r'Main chain ($\AA^2$)')
#   ax.set_ylabel(r'Side chain ($\AA^2$)')

#   ax.legend((
#     ax.scatter([], [], c='b', s=1),
#     ax.scatter([], [], c='r', s=1)
#   ), ('Non-mutated residue', 'Mutated residue'))

#   fig.savefig('output/1.png', dpi=300)


# def plot2():
#   fig, ax = plt.subplots()

#   ax.scatter(sasa_df['apolar'], sasa_df['polar'], c=np.array(['b', 'r'])[mutated_residue_mask.astype(int)], s=1)

#   ax.set_xlabel(r'Apolar ($\AA^2$)')
#   ax.set_ylabel(r'Polar ($\AA^2$)')

#   ax.legend((
#     ax.scatter([], [], c='b', s=1),
#     ax.scatter([], [], c='r', s=1)
#   ), ('Non-mutated residue', 'Mutated residue'))

#   fig.savefig('output/2.png', dpi=300)


# def plot3():
#   fig, ax = plt.subplots()

#   ax.boxplot([
#     sasa_df[mutated_residue_mask]['total'],
#     sasa_df[~mutated_residue_mask]['total']
#   ], labels=['Mutated residues', 'Non-mutated residues'])

#   ax.set_ylabel('Surface (Å²)')

#   fig.savefig('output/3.png', dpi=300)



# Path('output').mkdir(exist_ok=True)

# plot1()
# plot2()
# plot3()
# # plt.show()
