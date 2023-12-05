# Antoine GRISLAIN
# Simon LIÉTAR


from matplotlib import pyplot as plt
import numpy as np


def labels_tobinary(y: np.ndarray, cl: int):
  return np.where(y == cl, 1, 0)

def pred_lr(x: np.ndarray, w: np.ndarray, b: np.ndarray | float):
  return 1.0 / (1.0 + np.exp(-np.einsum('ip, p... -> ...i', x, w) - np.asarray(b)[..., None]))

def classify_binary(y: np.ndarray):
  return np.where(y > 0.5, 1, 0)

def accuracy(y_pred: np.ndarray, y: np.ndarray):
  return (y_pred == y).sum() / len(y)

def rl_gradient_ascent(x: np.ndarray, y: np.ndarray, *, eta: float, niter_max: int):
  w = np.zeros(x.shape[1])
  b = 0.0

  accs = list[float]()

  # For the type checker
  iteration = 0

  for iteration in range(niter_max):
    y_pred = pred_lr(x, w, b)

    acc = accuracy(classify_binary(y_pred), y)
    accs.append(acc)

    w += eta * (y - y_pred) @ x
    b += eta * (y - y_pred).sum()

  return w, b, accs, iteration

def visualization(w: np.ndarray):
  fig, ax = plt.subplots()

  ax.imshow(w.reshape(16, 16), cmap='gray')
  fig.show()


def rl_gradient_ascent_one_against_all(x: np.ndarray, yd: np.ndarray, *, epsilon: float, niter_max: int):
  y = np.array([labels_tobinary(yd, cl) for cl in range(10)])
  w = np.zeros((x.shape[1], y.shape[0]))
  b = np.zeros(y.shape[0])

  for _ in range(niter_max):
    y_pred = pred_lr(x, w, b)

    w += epsilon * ((y - y_pred) @ x).T
    b += epsilon * (y - y_pred).sum(axis=1)

  return w, b

def classif_multi_class(y_pred: np.ndarray):
  return y_pred.argmax(axis=0)


# L'encodage des pixels noirs à 0 pousse l'algorithme à favoriser la présence d'un pixel blanc la où il attend un noir plutôt qu'un pixel noir là où il attend un blanc.

#Le fait de soustraire 1 à X permet d'avoir des X négatifs et donc de favoriser la présence de pixels noirs là où il en attend.


def normalize(x: np.ndarray):
  return x - 1.0

def pred_lr_multi_class(x: np.ndarray, w: np.ndarray, b: np.ndarray):
  a = np.exp((x @ w).T + b[:, None])
  return a / a.sum(axis=0)

def to_categorical(y: np.ndarray, k: int):
  return np.array([labels_tobinary(y, cl) for cl in range(k)]).T

def rl_gradient_ascent_multi_class(x: np.ndarray, yd: np.ndarray, *, eta: float, numEp: int, verbose: int):
  y = to_categorical(yd, 10).T

  w = np.zeros((x.shape[1], y.shape[0]))
  b = np.zeros(y.shape[0])

  for iteration in range(numEp):
    y_pred = pred_lr_multi_class(x, w, b)

    w += eta * ((y - y_pred) @ x).T / x.shape[0]
    b += eta * (y - y_pred).sum(axis=1) / x.shape[0]

    if verbose and (iteration % 100 == 0):
      acc = accuracy(classif_multi_class(pred_lr_multi_class(x, w, b)), yd)
      print(f'epoch {iteration} accuracy train={(acc * 100):.2f}%')

  return w, b


def rl_gradient_ascent_multi_class_batch(x: np.ndarray, yd: np.ndarray, *, eta: float, numEp: int, tbatch: int, verbose: int):
  y = to_categorical(yd, 10).T

  xs = np.array_split(x, range(tbatch, x.shape[0], tbatch))
  ys = np.array_split(y, range(tbatch, y.shape[1], tbatch), axis=1)

  w = np.zeros((x.shape[1], y.shape[0]))
  b = np.zeros(y.shape[0])

  for iteration in range(numEp):
    for xbatch, ybatch in zip(xs, ys):
      y_pred = pred_lr_multi_class(xbatch, w, b)

      w += eta * ((ybatch - y_pred) @ xbatch).T / xbatch.shape[0]
      b += eta * (ybatch - y_pred).sum(axis=1) / xbatch.shape[0]

    if verbose and (iteration % 20 == 0):
      acc = accuracy(classif_multi_class(pred_lr_multi_class(x, w, b)), yd)
      print(f'epoch {iteration} accuracy train={(acc * 100):.2f}%')

  return w, b
