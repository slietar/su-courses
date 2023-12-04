from matplotlib import pyplot as plt
import numpy as np


def labels_tobinary(y: np.ndarray, cl: int):
  return np.where(y == cl, 1, 0)

def pred_lr(x: np.ndarray, w: np.ndarray, b: np.ndarray | float):
  return 1.0 / (1.0 + np.exp(-np.einsum('ip, ...p -> ...i', x, w) - np.asarray(b)[..., None]))

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
  # pass
  # print(x.shape)
  # print(y)

  y = np.array([labels_tobinary(yd, cl) for cl in range(10)])
  w = np.zeros((y.shape[0], x.shape[1]))
  b = np.zeros(y.shape[0])

  for _ in range(niter_max):
    y_pred = pred_lr(x, w, b)

    w += epsilon * (y - y_pred) @ x
    b += epsilon * (y - y_pred).sum(axis=1)

  return w, b

def classif_multi_class(y_pred: np.ndarray):
  return np.argmax(y_pred, axis=0)
