from pathlib import Path
from typing import Callable, Optional

import numpy as np
from matplotlib import pyplot as plt

from .. import config
from ..mltools import gen_arti, plot_data


def perceptron_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.minimum(-y * np.dot(x, w), 0).sum()

def perceptron_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return (-y * x.T * (y * np.dot(x, w) <= 0)).sum(axis=1)


class Lineaire:
  def __init__(
    self,
    loss: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_loss,
    loss_g: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_grad,
    max_iter: int = 100,
    eps: float = 0.01
  ):
    self.max_iter = max_iter
    self.eps = eps
    self.w: Optional[np.ndarray] = None
    self.loss = loss
    self.loss_g = loss_g

  def fit(self, x: np.ndarray, y: np.ndarray, test_x: Optional[np.ndarray] = None, test_y: Optional[np.ndarray] = None):
    scores = np.zeros((self.max_iter + 1, 2))
    self.w = np.zeros(x.shape[1])
    # print(self.loss(w, x, y))

    it = 0
    scores[0, 0] = self.score(x, y)

    if (test_x is not None) and (test_y is not None):
      scores[0, 1] = self.score(test_x, test_y)

    for it in range(self.max_iter):
      self.w -= self.eps * self.loss_g(self.w, x, y)
      # print(self.loss(w, x, y))
      # print(self.loss(w, x, y), w, self.loss_g(w, x, y))
      # print(y * np.dot(x, w))

      scores[it + 1, 0] = self.score(x, y)

      if (test_x is not None) and (test_y is not None):
        scores[it + 1, 1] = self.score(test_x, test_y)

    return scores[:(it + 2), :]

  def predict(self, x: np.ndarray):
    assert self.w is not None
    return np.sign(np.dot(x, self.w))

  def score(self, x: np.ndarray, y: np.ndarray):
    return (self.predict(x) == y).sum() / len(y)


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(ax, data):
  im = ax.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
  ax.get_figure().colorbar(im, ax=ax)


# x, y = gen_arti(epsilon=0.5)

# l = Lineaire(max_iter=10)
# l.fit(x, y)
# assert l.w is not None

# # print(l.w)


# fig, ax = plt.subplots()

# plot_data(ax, x, y)
# ax.axline((0.0, 0.0), slope=(-l.w[0] / l.w[1]), color='C2', linestyle='--')

# plt.show()



output_path = Path('output/tme5')
output_path.mkdir(exist_ok=True, parents=True)

data_path = Path(__file__).parent / 'data'

train_x, train_y = load_usps(data_path / 'USPS_train.txt')
test_x, test_y = load_usps(data_path / 'USPS_test.txt')

# print(train_x.shape, train_y.shape)
# print(train_x.dtype)

# show_usps(train_x[0])
# print(train_y[0])

# plt.show()


def ex1():
  model = Lineaire(eps=1e-3)

  train_mask = (train_y == 6) #| (train_y == 9)
  train_ax = train_x[train_mask, :]
  train_ay = np.where(train_y[train_mask] == 6, 1, -1)

  test_mask = (test_y == 6) #| (test_y == 9)
  test_ax = test_x[test_mask, :]
  test_ay = np.where(test_y[test_mask] == 6, 1, -1)

  scores = model.fit(train_ax, train_ay, test_ax, test_ay)
  print(scores)

  # print(model.score(ax, ay))
  # print(model.score(test_x, test_y))

  fig, ax = plt.subplots()

  show_usps(ax, model.w)


  fig, ax = plt.subplots()

  ax.plot(np.arange(scores.shape[0]), scores[:, 0], label='Train')
  ax.plot(np.arange(scores.shape[0]), scores[:, 1], label='Test')
  ax.legend()

  plt.show()


ex1()
