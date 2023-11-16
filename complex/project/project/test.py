from pathlib import Path
from tqdm import tqdm

from . import algorithms
from .graph import Graph


if __name__ == '__main__':
  with Path('exempleinstance.txt').open('rt') as file:
    graph = Graph.parse(file)

  with Path('out.svg').open('wt') as file:
    file.write(graph.draw())


  for _ in tqdm(range(200)):
    graph = Graph.random(12, 0.3)

    cover_greedy = algorithms.cover_greedy(graph)
    _, cover_coupling = algorithms.cover_from_coupling(graph)

    _, covers = algorithms.cover_optimal1(graph)
    _, cover2 = algorithms.cover_optimal2(graph)
    _, cover3 = algorithms.cover_optimal3(graph)
    _, cover4 = algorithms.cover_optimal4(graph)

    assert len(cover_greedy) >= len(cover2)
    assert len(cover_coupling) >= len(cover2)

    assert cover2 in covers
    assert cover3 in covers
    assert cover4 in covers
