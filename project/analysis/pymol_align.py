from dataclasses import dataclass


@dataclass
class PymolAlignment:
  aligned_atom_count: int
  cycle_count: int
  rmsd: float
  score: float
  total_atom_acount: int

  def __init__(self, data: tuple):
    self.rmsd, self.aligned_atom_count, self.cycle_count, _, self.total_atom_acount, self.score, _ = data



# align() return values
# 1 RMSD
# 2 Atom count in RMSD
# 3 Cycle count
# 4 Initial RMSD?
# 5 Aligned atom count
# 6 Score
# 7 Number of rejected atoms?
