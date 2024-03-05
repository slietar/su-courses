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
  commands.append(f'load {cwd / f"esmfold_output/domains/{domain_index:04}/structure.pdb"}, {name}')
  commands.append(f'align {name}, FBN1')


print('\n'.join(commands))
