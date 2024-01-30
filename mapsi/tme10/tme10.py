# Antoine GRISLAIN
# Simon LIÉTAR


import numpy as np


def exp(lam_: np.ndarray | list[float], /):
  lam = np.asarray(lam_)
  return -np.log(1.0 - np.random.rand(*lam.shape)) / np.maximum(lam, 1e-200)


Graph = tuple[dict[int, str], dict[tuple[int, int], float], dict[tuple[int, int], float]]
Preds = dict[int, list[tuple[int, float, float]]]

def simulation(graph: Graph, sources: list[int], max_time: float):
  names, k, r = graph
  nodes = list(range(len(names)))
  infectious_nodes = set(sources)
  infection_times = [0.0 if node in sources else max_time for node in nodes]

  while True:
    infectious_node = int(np.argmin([infection_times[node] if node in infectious_nodes else np.inf for node in nodes]))
    infection_time = infection_times[infectious_node]

    if (infection_time >= max_time) or not (infectious_node in infectious_nodes):
      break

    for other_node in nodes:
      pair = (infectious_node, other_node)
      other_infection_time = infection_times[other_node]

      if not (pair in k) or (other_infection_time <= infection_time):
        continue

      edge_k = k[pair]
      edge_r = r[pair]

      if np.random.random() < edge_k:
        infection_times[other_node] = min(other_infection_time, infection_time + exp([edge_r])[0])
        infectious_nodes.add(other_node)

    infectious_nodes.remove(infectious_node)

  return np.array(infection_times)


def getProbaMC(graph: Graph, sources: list[int], max_time: float, nbsimu: int):
  infection_counts = np.zeros(len(graph[0]))

  for _ in range(nbsimu):
    infection_counts += simulation(graph, sources, max_time) < max_time

  return infection_counts / nbsimu


def getPredsSuccs(graph: Graph):
  names, k, r = graph
  preds = Preds()
  succs = Preds()

  for (i, j), edge_k in k.items():
    edge_r = r[(i, j)]
    preds.setdefault(j, []).append((i, edge_k, edge_r))
    succs.setdefault(i, []).append((j, edge_k, edge_r))

  return preds, succs

def compute_ab(v: int, times: np.ndarray, preds: Preds, max_time: float, *, eps: float):
  if times[v] <= 0:
    return 1.0, 0.0

  js_, k_, r_ = zip(*preds[v])
  js, k, r = list(js_), np.array(k_), np.array(r_)

  alpha = k * r * np.exp(-r * (times[v] - times[js]))
  beta = k * np.exp(-r * (times[v] - times[js])) + 1.0 - k

  mask = times[js] < max_time
  a: float = np.maximum((alpha / beta * mask).sum(), eps) if times[v] < max_time else 1.0
  b: float = (np.log(beta) * mask).sum()

  return a, b

def compute_ll(times: np.ndarray, preds: Preds, max_time: float):
  sa_, sb_ = zip(*(compute_ab(v, times, preds, max_time, eps=1e-20) for v in range(len(times))))
  sa, sb = np.array(sa_), np.array(sb_)

  return (np.log(sa) + sb).sum(), sa, sb

def addVatT(v: int, times: np.ndarray, newt: float, preds: Preds, succs: Preds, sa: np.ndarray, sb: np.ndarray, max_time: float):
  times[v] = newt

  for succ, _, _ in succs.get(v, []):
    if times[succ] > newt:
      sa[succ], sb[succ] = compute_ab(succ, times, preds, max_time, eps=1e-20)

  sa[v], sb[v] = compute_ab(v, times, preds, max_time, eps=1e-20)

def logsumexp(x: np.ndarray):
  m = x.max(axis=-1)
  return m + np.log(np.exp(x - m[..., None]).sum(axis=-1))


# def gb(graph: Graph, infections: list[tuple[int, float]], max_time: int, *, burnin: int, period: int, ref, sampler: Callable):
#   nodes = range(len(graph[0]))

#   times = np.array([max_time] * len(nodes))

#   for infected_node, infection_time in infections:
#     times[infected_node] = infection_time

#   k = 10
#   k2 = 10

#   preds, succs = getPredsSuccs(graph)
#   _, sa, sb= compute_ll(times, preds, max_time)

#   for iteration in range(100):
#     # print(sa)
#     # print(sb)
#     # print()

#     for node in nodes:
#       sampler(node, times, preds, succs, sa, sb, max_time, k, k2)

#   print(sa)
#   print(sb)
#   print()
