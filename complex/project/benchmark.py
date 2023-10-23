from dataclasses import dataclass
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from pathlib import Path
from time import time_ns
from tqdm import tqdm
from typing import Any, Callable, Collection, Sequence
import math
import numpy as np

from . import algorithms
from .graph import Graph


@dataclass(frozen=True, slots=True)
class Algorithm:
  function: Callable[[Graph], tuple[int, frozenset[int]]]
  label: str
  color: Any

def benchmark(algorithms: list[Algorithm], sample_count: int, n_values: Collection[int], p_values: Collection[float]):
  # algorithm, sample, n, p
  exec_time = np.zeros((len(algorithms), sample_count, len(n_values), len(p_values)))
  cover_size = np.zeros_like(exec_time, dtype=int)
  explored_node_count = np.zeros_like(exec_time, dtype=int)

  for sample_index in tqdm(range(sample_count)):
    for p_index, p in enumerate(p_values):
      for n_index, n in enumerate(n_values):
        graph = Graph.random(n, (1.0 / math.sqrt(n) if n > 0 else 0) if p < 0 else p)

        for algorithm_index, algorithm in enumerate(algorithms):
          index = algorithm_index, sample_index, n_index, p_index

          t0 = time_ns()
          explored_node_count[*index], cover = algorithm.function(graph)
          t1 = time_ns()

          cover_size[*index] = len(cover)
          exec_time[*index] = (t1 - t0) * 1e-6

  return cover_size, exec_time.clip(min=1e-3), explored_node_count


if __name__ == '__main__':
  plt.rcParams['font.family'] = 'Helvetica Neue'
  plt.rcParams['mathtext.fontset'] = 'custom'
  plt.rcParams['mathtext.sf'] = 'Helvetica Neue'
  plt.rcParams['figure.figsize'] = 6.0, 5.0
  plt.rcParams['figure.dpi'] = 280
  plt.rcParams['grid.color'] = 'whitesmoke'


  output_dir_path = Path(__file__).parent.parent / 'output'
  output_dir_path.mkdir(exist_ok=True)


  def create_normalize(values: Collection[float], /):
    norm = colors.Normalize(vmin=min(values), vmax=max(values))
    return lambda x: norm(x) * 0.4 + 0.4


  def suboptimal_benchmark():
    n_values = np.linspace(0, 150, 10, dtype=int)
    p_values = [0.25, 0.5, 0.75, 1.0]

    benchmark_algorithms = [
      Algorithm(lambda graph: (0, algorithms.cover_from_coupling(graph)[1]), 'Couplage', cm.Greens),
      Algorithm(lambda graph: (0, algorithms.cover_greedy(graph)), 'Glouton', cm.Reds),
    ]

    cover_size, exec_time, _ = benchmark(benchmark_algorithms, sample_count=20, n_values=n_values, p_values=p_values)

    avg_exec_time = exec_time.mean(axis=1)
    normalize_p = create_normalize(p_values[:-1])


    def plot1():
      fig, ax = plt.subplots()

      for algorithm_index, algorithm in enumerate(benchmark_algorithms):
        for p_index, p in enumerate(p_values):
          ax.plot(n_values, avg_exec_time[algorithm_index, :, p_index], color=algorithm.color(normalize_p(p)), label=f"{algorithm.label} (p = {p})")

      ax.set_yscale('log')
      ax.set_xlabel('Nombre de sommets n')
      ax.set_ylabel('''Temps d'exécution (ms)''')
      ax.xaxis.get_major_locator().set_params(integer=True)
      ax.grid()
      ax.set_title('''Temps d'éxécution des algorithmes de couplage et glouton''')
      ax.legend()

      fig.savefig(str(output_dir_path / '3-2a.png'))

    def plot2():
      fig, ax = plt.subplots()

      with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.nanmean(cover_size[0, :, :, :] / cover_size[1, :, :, :], axis=0)

      for p_index, p in enumerate(p_values):
        ax.plot(n_values, ratio[:, p_index], color=cm.Blues(normalize_p(p)), label=f'p = {p}')

      ax.set_xlabel('Nombre de sommets n')
      ax.set_ylabel(r'''Rapport d'approximation $\frac{\mathsf{couplage}}{\mathsf{glouton}}$''')
      ax.set_title('''Rapport d'approximation entre l'algorithme de couplage et l'algorithme glouton''')
      ax.xaxis.get_major_locator().set_params(integer=True)
      ax.grid()
      ax.legend()

      fig.savefig(str(output_dir_path / '3-2b.png'))

    plot1()
    plot2()


  def optimal_benchmark():
    n_values = np.linspace(0, 20, 10, dtype=int)
    p_values = [0.1, 0.2, 0.3, -1.0]
    p_labels = ['0.1', '0.2', '0.3', r'$\frac{1}{\sqrt{n}}$']

    benchmark_algorithms = [
      Algorithm(lambda graph: ((result := algorithms.cover_optimal1(graph))[0], next(iter(result[1]))), 'Sans élagage', cm.Reds),
      Algorithm(lambda graph: algorithms.cover_optimal2(graph), 'Avec élagage', cm.Blues),
      Algorithm(lambda graph: algorithms.cover_optimal3(graph), 'Avec branchement amélioré 1', cm.Greens),
      Algorithm(lambda graph: algorithms.cover_optimal4(graph), 'Avec branchement amélioré 2', cm.Purples),
      Algorithm(lambda graph: algorithms.cover_optimal5(graph), 'Avec branchement amélioré 3', cm.Reds)
    ]

    _, exec_time, explored_node_count = benchmark(benchmark_algorithms, sample_count=20, n_values=n_values, p_values=p_values)
    normalize_p = create_normalize(p_values[:-1])


    def plot(algorithm_indices: Sequence[int], p_indices: Sequence[int], name: str):
      def plot1():
        fig, ax = plt.subplots()

        dummy_line, = ax.plot([0], marker='None', linestyle='None')
        lines = list()

        for algorithm_index in algorithm_indices:
          algorithm = benchmark_algorithms[algorithm_index]
          lines.append(dummy_line)

          for p_index in p_indices:
            p = p_values[p_index]
            line, = ax.plot(n_values, exec_time[algorithm_index, :, :, p_index].mean(axis=0), color=algorithm.color(normalize_p(max(p_values) if p < 0 else p)), label=f"{algorithm.label} (p = {p_labels[p_index]})", linestyle=('dashed' if p < 0 else 'solid'))
            lines.append(line)

        ax.set_yscale('log')
        ax.set_xlabel('Nombre de sommets n')
        ax.set_ylabel('''Temps d'exécution (ms)''')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid()
        ax.set_title('''Temps d'éxécution des algorithmes de branch and bound''')
        ax.legend(lines, [label for algorithm_index in algorithm_indices for label in [rf'{benchmark_algorithms[algorithm_index].label}', *(f"p = {p_labels[p_index]}" for p_index in p_indices)]])

        fig.savefig(str(output_dir_path / f'{name}a.png'))

      def plot2():
        fig, ax = plt.subplots()

        dummy_line, = ax.plot([0], marker='None', linestyle='None')
        lines = list()

        for algorithm_index in algorithm_indices:
          algorithm = benchmark_algorithms[algorithm_index]
          lines.append(dummy_line)

          for p_index in p_indices:
            p = p_values[p_index]
            line, = ax.plot(n_values, explored_node_count[algorithm_index, :, :, p_index].mean(axis=0), color=algorithm.color(normalize_p(max(p_values) if p < 0 else p)), label=f"{algorithm.label} (p = {p_labels[p_index]})", linestyle=('dashed' if p < 0 else 'solid'))
            lines.append(line)

        ax.set_yscale('log')
        ax.set_xlabel('Nombre de sommets n')
        ax.set_ylabel('Nombre de nœuds explorés')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid()
        ax.set_title('''Nombre de nœuds explorés par les algorithmes de branch and bound''')
        ax.legend(lines, [label for algorithm_index in algorithm_indices for label in [rf'{benchmark_algorithms[algorithm_index].label}', *(f"p = {p_labels[p_index]}" for p_index in p_indices)]])

        fig.savefig(str(output_dir_path / f'{name}b.png'))

      plot1()
      plot2()

    plot([0], [0, 1, 2, 3], name='4-1')
    plot([0, 1], [0, 2], name='4-2')
    plot([1, 2, 3], [0, 2], name='4-3')
    plot([3, 4], [0, 2], name='4-3x')

  def comparison_benchmark():
    n_values = np.linspace(0, 75, 10, dtype=int)
    p_values = [0.25, 0.5]

    benchmark_algorithms = [
      Algorithm(lambda graph: (0, algorithms.cover_from_coupling(graph)[1]), 'Couplage', cm.Greens),
      Algorithm(lambda graph: (0, algorithms.cover_greedy(graph)), 'Glouton', cm.Reds),
      Algorithm(lambda graph: algorithms.cover_optimal4(graph), '...', cm.Purples)
    ]

    cover_size, _, _ = benchmark(benchmark_algorithms, sample_count=20, n_values=n_values, p_values=p_values)
    normalize_p = create_normalize(p_values)

    fig, ax = plt.subplots()

    with np.errstate(divide='ignore', invalid='ignore'):
      ratios = np.nanmean(cover_size[0:2, :, :, :] / cover_size[-1, :, :, :], axis=1)

    print(np.nanmax(ratios, axis=(1, 2)))

    for algorithm_index, algorithm in enumerate(benchmark_algorithms[0:2]):
      for p_index, p in enumerate(p_values):
        ax.plot(n_values, ratios[algorithm_index, :, p_index], color=algorithm.color(normalize_p(p)), label=f'{algorithm.label} (p = {p})')

    ax.set_xlabel('Nombre de sommets n')
    ax.set_ylabel('''Rapport d'approximation''')
    ax.set_title('''Rapport d'approximation des algorithmes de couplage et glouton''')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid()
    ax.legend()

    fig.savefig(str(output_dir_path / '4-4.png'))


  suboptimal_benchmark()
  optimal_benchmark()
  comparison_benchmark()
