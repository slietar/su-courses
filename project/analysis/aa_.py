import json
from pathlib import Path
from typing import Optional, overload

import numpy as np


aa_map: Optional[dict[str, str]] = None

def get_aa_map():
  global aa_map

  if aa_map is None:
    with (Path(__file__).parent / '../resources/amino_acid_map.json').open('rb') as file:
      loaded_aa_map: dict[str, str] = json.load(file)
      aa_map = loaded_aa_map

  return aa_map


# @overload
# def aa_long_to_short(long: np.ndarray, /) -> np.ndarray:
#   ...

def aa_short_to_long(short: np.ndarray | list[str] | str, /):
  aa_map = get_aa_map()

  match short:
    case np.ndarray:
      return np.vectorize(lambda x: aa_map[x])(short)
    case list():
      return [aa_map[x] for x in short]
    case str():
      return aa_map[short]
