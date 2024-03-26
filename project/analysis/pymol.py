from dataclasses import dataclass

import numpy as np


@dataclass
class PymolAlignment:
  atom_count: int
  cycle_count: int
  rmsd: float
  rmsd_before: float
  score: float
  atom_count_before: int
  residue_count: int

  def __init__(self, data: tuple, /):
    # https://pymolwiki.org/index.php/Align
    self.rmsd, self.atom_count, self.cycle_count, self.rmsd_before, self.atom_count_before, self.score, self.residue_count = data


@dataclass
class PymolTransformation:
  post_translation: np.ndarray
  pre_translation: np.ndarray
  rotation: np.ndarray

  def __init__(self, data: tuple[float, ...], /):
    arr = np.asarray(data).reshape(4, 4)

    self.rotation = arr[:3, :3]
    self.pre_translation = arr[:3, 3]
    self.post_translation = arr[3, :3]

  def apply(self, arr: np.ndarray, /):
    # (N x 3) @ (3 x 3) -> (N x 3)
    return (arr + self.post_translation) @ np.linalg.inv(self.rotation) + self.pre_translation
