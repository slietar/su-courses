from pathlib import Path
import pickle
import pandas as pd


with Path('../structure/output/data.pkl').open('rb') as file:
  data = pickle.load(file)

domains = pd.DataFrame(data['domains'])
cwd = Path(__file__).parent


commands = list[str]()

commands.append(f'load {cwd / "../drive/FBN1_AlphaFold.pdb"}, FBN1')

for domain_index, domain in domains.iterrows():
  name = f'{domain["kind"]}{domain["number"]}'
  commands.append(f'load {cwd / f"../esmfold-pruning/output/{domain_index:04}.pdb"}, {name}_co')
  commands.append(f'load {cwd / f"../esmfold-output/isolated/domains/{domain_index:04}/structure.pdb"}, {name}_is')
  commands.append(f'align {name}_co, FBN1')
  commands.append(f'align {name}_is, FBN1')


print('\n'.join(commands))


# with Path('../esmfold-output/contextualized/metadata.pkl').open('rb') as file:
#   metadata = pickle.load(file)

# print(metadata[0]['aligned_confidence_probs'].shape)
