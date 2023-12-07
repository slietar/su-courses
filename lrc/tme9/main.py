# TME9
# Simon LIÉTAR

from dataclasses import dataclass, field
from typing import Literal, NewType


# On utilise un typage plus fort que str pour les éviter les erreurs.
Relation = Literal['<', '>', 'e', 's', 'et', 'st', 'd', 'm', 'dt', 'mt', 'o', 'ot', '=']
Relations = set[Relation]


transpose: dict[Relation, Relation] = {
  '<':'>',
  '>':'<',
  'e':'et',
  's':'st',
  'et':'e',
  'st':'s',
  'd':'dt',
  'm':'mt',
  'dt':'d',
  'mt':'m',
  'o':'ot',
  'ot':'o',
  '=':'='
}

# symetrie : dict[str:str]
symetrie: dict[Relation, Relation] = {
  '<':'>',
  '>':'<',
  'e':'s',
  's':'e',
  'et':'st',
  'st':'et',
  'd':'d',
  'm':'mt',
  'dt':'dt',
  'mt':'m',
  'o':'ot',
  'ot':'o',
  '=':'='
}

# compositionBase : dict[tuple[str,str]:set[str]]
compositionBase: dict[tuple[Relation, Relation], Relations] = {
  ('<','<'):{'<'},
  ('<','m'):{'<'},
  ('<','o'):{'<'},
  ('<','et'):{'<'},
  ('<','s'):{'<'},
  ('<','d'):{'<','m','o','s','d'},
  ('<','dt'):{'<'},
  ('<','e'):{'<','m','o','s','d'},
  ('<','st'):{'<'},
  ('<','ot'):{'<','m','o','s','d'},
  ('<','mt'):{'<','m','o','s','d'},
  ('<','>'):{'<','>','m','mt','o','ot','e','et','s','st','d','dt','='},
  ('m','m'):{'<'},
  ('m','o'):{'<'},
  ('m','et'):{'<'},
  ('m','s'):{'m'},
  ('m','d'):{'o','s','d'},
  ('m','dt'):{'<'},
  ('m','e'):{'o','s','d'},
  ('m','st'):{'m'},
  ('m','ot'):{'o','s','d'},
  ('m','mt'):{'e','et','='},
  ('o','o'):{'<','m','o'},
  ('o','et'):{'<','m','o'},
  ('o','s'):{'o'},
  ('o','d'):{'o','s','d'},
  ('o','dt'):{'<','m','o','et','dt'},
  ('o','e'):{'o','s','d'},
  ('o','st'):{'o','et','dt'},
  ('o','ot'):{'o','ot','e','et','d','dt','st','s','='},
  ('s','et'):{'<','m','o'},
  ('s','s'):{'s'},
  ('s','d'):{'d'},
  ('s','dt'):{'<','m','o','et','dt'},
  ('s','e'):{'d'},
  ('s','st'):{'s','st','='},
  ('et','s'):{'o'},
  ('et','d'):{'o','s','d'},
  ('et','dt'):{'dt'},
  ('et','e'):{'e','et','='},
  ('d','d'):{'d'},
  ('d','dt'):{'<','>','m','mt','o','ot','e','et','s','st','d','dt','='},
  ('dt','d'):{'o','ot','e','et','d','dt','st','s','='}
}


# Renvoie l'ensemble des relations transposées qui apparaissent dans l'argument.
def transposeSet(relations: Relations, /) -> Relations:
  return {transpose[relation] for relation in relations}

# Renvoie l'ensemble des relations symétriques qui apparaissent dans l'argument.
def symetrieSet(relations: Relations, /) -> Relations:
  return {symetrie[relation] for relation in relations}

# Renvoie la composition des deux relations données en argument.
def compose(r1: Relation, r2: Relation, /) -> Relations:
  if r1 == '=':
    return {r2}

  if r2 == '=':
    return {r1}

  if (r1, r2) in compositionBase:
    return compositionBase[(r1, r2)]

  r1t = transpose[r1]
  r2t = transpose[r2]

  if (r2t, r1t) in compositionBase:
    return transposeSet(compositionBase[(r2t, r1t)])

  r1s = symetrie[r1]
  r2s = symetrie[r2]

  if (r1s, r2s) in compositionBase:
    return symetrieSet(compositionBase[(r1s, r2s)])

  r1st = transpose[r1s]
  r2st = transpose[r2s]

  if (r2st, r1st) in compositionBase:
    return symetrieSet(transposeSet(compositionBase[(r2st, r1st)]))

  raise RuntimeError

# Renvoie l'ensemble des relations que l'on peut obtenir des relations qui apparaissent dans les deux arguments.
def compositionSet(relations1: Relations, relations2: Relations, /) -> Relations:
  return {rel for rel1 in relations1 for rel2 in relations2 for rel in compose(rel1, rel2)}


# On vérifie que compose() fonctionne correctement.
assert compose('=', 'd') == {'d'}
assert compose('m', 'd') == {'d', 'o', 's'}
assert compose('ot', '>') == {'>'}
assert compose('>', 'e') == {'>'}
assert compose('ot', 'm') == {'dt', 'et', 'o'}


# On utilise un typage nominal pour éviter de confondre les str représentant des noeuds de str contenant du texte.
Node = NewType('Node', str)

@dataclass(slots=True)
class Graph:
  nodes: set[Node] = field(default_factory=set)

  # Pour éviter les redondances, on ne stocke que les relations entre les noeuds i et j tels que i < j.
  relations: dict[tuple[Node, Node], Relations] = field(default_factory=dict)

  # Renvoie les relations possible entre les deux noeuds donnés comme arguments, en prenant compte le fait que les relations puissent être stockées dans l'autre sens.
  def getRelations(self, i: Node, j: Node, /) -> Relations:
    if i == j:
      return {'='}

    if i < j:
      if (i, j) in self.relations:
        return self.relations[(i, j)]
    else:
      if (j, i) in self.relations:
        return transposeSet(self.relations[(j, i)])

    # Toutes les relations sont possibles si elles ne sont pas spécifiées dans le graphe.
    return set(transpose.keys())

  # Définit les relations entre les deux noeuds donnés comme arguments.
  def setRelations(self, i: Node, j: Node, relations: Relations, /):
    assert i != j

    if i < j:
      self.relations[(i, j)] = relations
    else:
      self.relations[(j, i)] = transposeSet(relations)

  # S'assure que les deux noeuds en argument existent, ajoute les relations entre eux, et propage ces relations.
  def add(self, ri: str, rj: str, relations: Relations, /, *, propagate: bool = True, verbose: bool = True):
    i = Node(ri)
    j = Node(rj)

    self.nodes |= {i, j}
    self.setRelations(i, j, relations)

    if verbose:
      print(f'[Add] Added {i} -> {j} with {relations}')

    if propagate:
      self.propagate(i, j, verbose=verbose)

  # Retire un noeud du graphe ainsi que ses relations.
  def remove(self, raw_node: str, /):
    node = Node(raw_node)

    for i, j in self.relations.copy():
      if node in (i, j):
        del self.relations[(i, j)]

  # Propage les relations entre les noeuds i et j aux autres noeuds du graphe.
  def propagate(self, ri: str, rj: str, /, *, verbose: bool = True):
    stack = [(Node(ri), Node(rj))]

    while stack:
      i, j = stack.pop()

      if verbose:
        print(f'[Propagation] Processing {i} -> {j}')

      for k in self.nodes:
        if k in (i, j):
          continue

        ij = self.getRelations(i, j)
        ik = self.getRelations(i, k)
        kj = self.getRelations(k, j)

        new_ik = ik & compositionSet(ij, self.getRelations(j, k))
        new_kj = kj & compositionSet(self.getRelations(k, i), ij)

        if (not new_ik) or (not new_kj):
          raise RuntimeError('Contradiction temporelle')

        if new_ik != ik:
          self.setRelations(i, k, new_ik)
          stack.append((i, k))

          if verbose:
            print(f'[Propagation] Changed {i} -> {k} from {ik} to {new_ik}')

        if new_kj != kj:
          self.setRelations(k, j, new_kj)
          stack.append((k, j))

          if verbose:
            print(f'[Propagation] Changed {k} -> {j} from {kj} to {new_kj}')


# Exercice 1 TD8

if True:
  print('Exercice 1\n')

  graph1 = Graph()
  graph1.add('tE', 'tH', {'d'})
  graph1.add('tH', 'tA', {'et'})
  graph1.add('tE', 'tL', {'<'})
  graph1.add('tL', 'tA', {'<'})

  assert graph1.getRelations(Node('tL'), Node('tH')) == {'d'}

  print()
  print(graph1)
  print('\n' + '-' * 80 + '\n')


# Exercice 3 TD8

if True:
  print('Exercice 3\n')

  graph2 = Graph()
  graph2.add('C', 'T', {'m'})
  graph2.add('R', 'T', {'s', '=', 'd', 'e'})

  assert graph2.getRelations(Node('C'), Node('R')) == {'m', '<'}

  print()
  print(graph2)
  print('\n' + '-' * 80 + '\n')


# Exercice 4 TD8

if True:
  print('Exercice 4\n')

  graph3 = Graph()
  graph3.add('J', 'D', {'='})
  graph3.add('J', 'C', {'=', 'e', 'et'})
  graph3.add('D', 'P', {'<', 'm'})
  graph3.add('C', 'D', {'s', '=', 'd', 'e'})

  print()
  print(graph3)
