import json

import pandas as pd

from . import shared


with (shared.root_path / 'resources/amino_acids.json').open() as file:
  amino_acids = pd.DataFrame.from_records(json.load(file))

# print(amino_acids)
