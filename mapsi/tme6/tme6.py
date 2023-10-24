# Antoine GRISLAIN
# Simon LIETAR

import numpy as np
import random
from matplotlib import pyplot as plt


def discretise(xs: list[np.ndarray], d: int):
  intv = 360.0 / d
  return [np.floor(xi / intv).astype(int) for xi in xs]

def groupByLabel(ys: list[str]):
  groups = dict[str, list[int]]()

  for index, y in enumerate(ys):
    groups.setdefault(y, []).append(index)

  return groups

def learnMarkovModel(xs: list[np.ndarray], d: int):
  xs_discrete = discretise(xs, d)
  pi = np.bincount([xi[0] for xi in xs_discrete], minlength=d) / len(xs_discrete)

  a = np.zeros((d, d))

  for xi in xs_discrete:
    for x_prev, x in zip(xi[:-1], xi[1:]):
      a[x_prev, x] += 1

  a /= np.maximum(a.sum(axis=1).reshape(d, 1), 1)

  return pi, a

def learn_all_MarkovModels(xs: list[np.ndarray], ys: list[str], d: int):
  return { y: learnMarkovModel([xs[i] for i in group], d) for y, group in groupByLabel(ys).items() }

def stationary_distribution_freq(xs: list[np.ndarray], d: int):
  all_x = np.concatenate([xi[1:] for xi in xs])
  return np.bincount(all_x, minlength=d) / len(all_x)

def stationary_distribution_sampling(pi: np.ndarray, a: np.ndarray, *, N: int):
  def sample(probabilities):
    x = random.random()
    return np.argmax([p > x for p in np.cumsum(probabilities)])

  value = sample(pi)
  values = list[int]()

  for _ in range(N):
    value = int(sample(a[value, :]))
    values.append(value)

  return np.bincount(values, minlength=len(pi)) / len(values)

def stationary_distribution_fixed_point(a: np.ndarray, *, epsilon: float):
  value = np.zeros(a.shape[0])
  value[0] = 1.0

  while True:
    old_value = value
    value = value @ a

    if ((old_value - value) ** 2).mean() < epsilon:
      break

  return value

def stationary_distribution_fixed_point_VP(a: np.ndarray):
  eigenvalues, eigenvectors = np.linalg.eig(a.T)
  vector = eigenvectors[:, np.isclose(eigenvalues, 1.0)][:, 0]
  return vector / sum(vector)

def logL_Sequence(xi: np.ndarray, pi: np.ndarray, a: np.ndarray):
  return np.log(pi[xi[0]]) + sum(np.log(a[x_prev, x]) for x_prev, x in zip(xi[:-1], xi[1:]))

# Le signal Xd[0] n'est pas bien classé car il est plus vraisemblable que ce soit un Z que un A alors que Y[0] == 'a'.
# Les -inf correspondent a une probabilité de zero, qui devient -inf lorsque passé par un log.

def compute_all_ll(xs: list[np.ndarray], models: dict[str, tuple[np.ndarray, np.ndarray]]):
  return np.array([[logL_Sequence(xi, pi, a) for xi in xs] for pi, a in models.values()])

def accuracy(ll: np.ndarray, ys: list[str]):
  return (ll.argmax(axis=0) == [ord(y) - ord('a') for y in ys]).sum() / ll.shape[1]

def learn_MarkovModel_Laplace(xs: list[np.ndarray], d: int):
  xs_discrete = discretise(xs, d)
  pi = (np.ones(d) + np.bincount([xi[0] for xi in xs_discrete], minlength=d)) / (d + len(xs_discrete))

  a = np.ones((d, d))

  for xi in xs_discrete:
    for x_prev, x in zip(xi[:-1], xi[1:]):
      a[x_prev, x] += 1

  a /= np.maximum(a.sum(axis=1).reshape(d, 1), 1)

  return pi, a

def learn_all_MarkovModels_Laplace(xs: list[np.ndarray], ys: list[str], d: int):
  return { y: learn_MarkovModel_Laplace([xs[i] for i in group], d) for y, group in groupByLabel(ys).items() }


def confusion_matrix (ll,y,name):
  tab=np.zeros((ll.shape[0],ll.shape[0]))

  for i in range(ll.shape[1]):
    tab[ord(y[i]) - ord('a'), np.argmax(ll[:, i])] += 1

  fig, ax = plt.subplots()

  ax.imshow(tab)
  ax.set_title(name)
  ax.set_ylabel('Prediction')
  ax.set_xlabel('Verite')
  fig.show()
