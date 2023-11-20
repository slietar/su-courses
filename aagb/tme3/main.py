from dataclasses import dataclass
import math
from ete3 import Tree
import numpy as np


tree1 = Tree('((((Electrode,Magnezone),Porygon-Z),((((Aggron,Bastiodon),Forretress),Ferrothorn),((((Regirock,Regice),Registeel),Metagross),Klinklang),Genesect)),Probopass);')
tree2 = Tree('(((((Regirock,Regice),Registeel),((Metagross,Klinklang),Genesect)),(((Aggron,Bastiodon),(Forretress,Ferrothorn)),Probopass)),(Porygon-Z,(Magnezone,Electrode)));')
# tree = Tree('((A, B), (C, (D, E)));')


mutation_matrix = np.array([
  [0, 3, 4, 9],
  [3, 0, 2, 4],
  [4, 2, 0, 4],
  [9, 4, 4, 0]
])

pokemons = {
  'Probopass': 'A',
  'Aggron': 'T',
  'Bastiodon': 'T',
  'Regirock': 'G',
  'Registeel': 'G',
  'Regice': 'G',
  'Klinklang': 'G',
  'Metagross': 'C',
  'Genesect': 'A',
  'Porygon-Z': 'C',
  'Magnezone': 'C',
  'Forretress': 'T',
  'Electrode': 'A',
  'Ferrothorn': 'G'
}

# pokemons = {
#   'A': 'C',
#   'B': 'A',
#   'C': 'C',
#   'D': 'A',
#   'E': 'G'
# }

nucleotides = ['A', 'C', 'G', 'T']


@dataclass
class NodeMetadata:
  distribution: np.ndarray
  origin: np.ndarray
  value: int = -1

def sankoff(tree: Tree):
  metadata = dict[Tree, NodeMetadata]()

  for node in tree.traverse('postorder'): # type: ignore
    if node.children:
      # axis 0: child
      # axis 1: possible nucleotide for child node
      d = np.array([metadata[child].distribution for child in node.children])

      # axis 2: possible nucleotide for current node
      m = d[..., np.newaxis] + mutation_matrix[np.newaxis, ...]

      metadata[node] = NodeMetadata(
        m.min(axis=1).sum(axis=0),
        m.argmin(axis=1)
      )
    else:
      metadata[node] = NodeMetadata(
        np.array([0 if pokemons[node.name] == nucleotide else math.inf for nucleotide in nucleotides]),
        np.array([])
      )

  for node in tree.traverse('preorder'): # type: ignore
    if node.up:
      child_index = node.up.children.index(node)
      nucleotide = metadata[node.up].origin[child_index, metadata[node.up].value]
    else:
      nucleotide = metadata[node].distribution.argmin()

    metadata[node].value = int(nucleotide)
    node.name = (f'{node.name} ' if node.name else str()) + f'[{nucleotides[nucleotide]}]'

  return metadata[tree].distribution[metadata[tree].value]

print(sankoff(tree1))
print(sankoff(tree2))

print(tree1.get_ascii(show_internal=True))
print(tree2.get_ascii(show_internal=True))
