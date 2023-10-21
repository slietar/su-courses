from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any, Callable, Collection, Sequence
import pickle

from matplotlib import cm, colors
from matplotlib import pyplot as plt
import numpy as np

from . import algorithms
from .graph import Graph


@dataclass(frozen=True, slots=True)
class Algorithm:
  function: Callable[[Graph], frozenset[int]]
  label: str
  color: Any

# algorithms = [
#   Algorithm(cover_from_coupling, "Coupling", cm.Greens),
#   Algorithm(cover_greedy, "Greedy", cm.Blues),
#   Algorithm(cover_optimal1, "Optimal (4.1)", cm.Reds),
#   Algorithm(cover_optimal2, "Optimal (4.2)", cm.Purples),
# ]

def benchmark(algorithms: list[Algorithm], sample_count: int, n_values: Collection[int], p_values: list[float]):
  algorithm_count = len(algorithms)

# sample_count = 2
# p_values = [0.25] # [0, 0.25, 0.5, 0.75, 1]
# n_values = np.linspace(10, 20, 5, dtype=int)

  # (coupling, greedy), sample, n, p
  cover_size = np.zeros((algorithm_count, sample_count, len(n_values), len(p_values)), dtype=int)
  exec_time = np.zeros((algorithm_count, sample_count, len(n_values), len(p_values)))

  for sample_index in range(sample_count):
    for p_index, p in enumerate(p_values):
      for n_index, n in enumerate(n_values):
        graph = Graph.random(n, p)

        for algorithm_index, algorithm in enumerate(algorithms):
          t0 = time_ns()
          x = algorithm.function(graph)
          t1 = time_ns()

          index = algorithm_index, sample_index, n_index, p_index

          # cover_size[*index] = ...
          exec_time[*index] = (t1 - t0) * 1e-6

#   with Path("out.pickle").open("wb") as file:
#     pickle.dump((cover_size, exec_time), file)
# else:
#   with Path("out.pickle").open("rb") as file:
#     cover_size, exec_time = pickle.load(file)

# print(cover_size)

# avg_cover_size = np.average(cover_size, axis=1)
  avg_exec_time = np.average(exec_time.clip(min=1e-3), axis=1)


  fig1, ax1 = plt.subplots()
  fig2, ax2 = plt.subplots()

  p_normalize = colors.Normalize(vmin=min(p_values), vmax=max(p_values))

  for algorithm_index, algorithm in enumerate(algorithms):
    for p_index, p in enumerate(p_values):
      ax1.plot(n_values, avg_exec_time[algorithm_index, :, p_index], color=algorithm.color(p_normalize(p) * 0.8 + 0.2), label=f"{algorithm.label} (p={p})")
      # ax2.scatter(n_values, avg_cover_size[0, :, p_index], color=cm.autumn(p_normalize(p)), label=f"Coupling (p={p})")

  ax1.set_yscale('log')
  ax1.set_xlabel('Nombre de sommets $n$')
  ax1.set_ylabel('''Temps d'exécution (ms)''')
  ax1.grid(color='whitesmoke')
  ax1.set_title('''Temps d'éxécution des algorithmes couplage et gluton''')
  ax1.legend()

  # r_cover_size = cover_size.reshape((2, -1))
  # j = range(r_cover_size.shape[1])

  # # print(r_cover_size)
  # ax2.scatter(j, r_cover_size[0, :])
  # ax2.scatter(j, r_cover_size[1, :])

  # # ax2.set_xlabel("Nombre de sommets (n)")
  # # ax2.set_ylabel("Nombre de sommets dans la couverture ($|C|$)")
  # ax2.legend()

  fig1.savefig('out1.png', dpi=200)
  fig2.savefig('out2.png', dpi=200)


if __name__ == '__main__':
  plt.rcParams['font.family'] = 'Helvetica Neue'

  benchmark([
    Algorithm(lambda graph: algorithms.cover_from_coupling(graph)[1], 'Couplage', cm.Greens),
    Algorithm(lambda graph: algorithms.cover_greedy(graph), 'Glouton', cm.Reds),
  ], sample_count=20, n_values=np.linspace(0, 200, 10, dtype=int), p_values=[0.25, 0.5, 0.75])
