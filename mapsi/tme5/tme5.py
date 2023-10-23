# Simon LIETAR
# Antoine GRISLAIN

from typing import Sequence
from scipy.stats import chi2
import numpy as np

import utils


def sufficient_statistics(data: np.ndarray, dico: np.ndarray, x: int, y: int, z: Sequence[int]):
  table = utils.create_contingency_table(data, dico, x, y, z)

  result = 0.0
  z_count = 0

  for n, t in table:
    if n > 0.0:
      exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
      result += np.divide((t - exp) ** 2, exp, out=np.zeros_like(t), where=(exp != 0.0)).sum()
      z_count += 1

  return result, (len(dico[x]) - 1) * (len(dico[y]) - 1) * z_count


def indep_score(data: np.ndarray, dico: np.ndarray, x: int, y: int, z: Sequence[int]):
  if data.shape[1] < 5 * len(dico[x]) * len(dico[y]) * np.prod([len(dico[i]) for i in z]):
    return -1.0, 1

  chi, df = sufficient_statistics(data, dico, x, y, z)
  return float(chi2.sf(chi, df)), df

def best_candidate(data: np.ndarray, dico: np.ndarray, x: int, z: Sequence[int], alpha: float):
  scores = [indep_score(data, dico, x, y, z)[0] for y in range(x)]

  if not scores:
    return []

  y = int(np.argmin(scores))

  return [y] if scores[y] < alpha else []

def create_parents(data: np.ndarray, dico: np.ndarray, x: int, alpha: float):
  parents = list[int]()

  while True:
    y = best_candidate(data, dico, x, parents, alpha)
    parents += y

    if not y:
      break

  return parents

def learn_BN_structure(data: np.ndarray, dico: np.ndarray, alpha: float):
  return np.array([create_parents(data, dico, x, alpha) for x in range(dico.shape[0])], dtype=object)
