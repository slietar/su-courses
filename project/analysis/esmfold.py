import pickle
from xml import dom

from matplotlib import pyplot as plt
import pandas as pd

from . import data, shared


with (shared.root_path / 'esmfold-output/contextualized/metadata.pkl').open('rb') as f:
  metadata = pickle.load(f)

# x = pd.DataFrame.from_records(metadata)
# print(x)

plddt_positions = list[int]()
plddt_values = list[float]()

for domain_index, (domain, domain_metadata) in enumerate(zip(data.domains.itertuples(), metadata)):
  offset = domain.start_position - (data.domains.iloc[domain_index - 1].start_position if domain_index > 0 else 0)

  plddt_positions += range(domain.start_position, domain.end_position + 1)
  plddt_values += list(domain_metadata['plddt'][offset:(offset + domain.end_position - domain.start_position + 1)])

plddt = pd.Series(plddt_values, index=pd.Series(plddt_positions, name='position'), name='plddt_esmfold_contextualized')


# pLDDT of:
#  - AlphaFold model
#  - ESMFold isolated domains
#  - ESMFold mixed domains

# RMSD comparison of:
#  - ESMFold isolated vs mixed domains
#  -
