import math
from pathlib import Path
from typing import Callable, Optional

from matplotlib.axes import Axes
import numpy as np
from matplotlib import pyplot as plt

from .. import config
from ..mltools import gen_arti, plot_data


def perceptron_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.minimum(-y * np.dot(x, w), 0).sum()

def perceptron_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return (-y * x.T * (y * np.dot(x, w) <= 0)).sum(axis=1)


def proj_poly(x: np.ndarray, /):
  return np.c_[x, (x[..., None, :] * x[..., :, None]).reshape((*x.shape[:-1], -1))]

def proj_biais(x: np.ndarray, /):
  return np.c_[x, np.ones((*x.shape[:-1], 1))]

def proj_identity(x: np.ndarray, /):
  return x


class Lineaire:
  def __init__(
    self,
    loss: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_loss,
    loss_g: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_grad,
    max_iter: int = 100,
    projection: Callable[[np.ndarray], np.ndarray] = proj_identity,
    eps: float = 0.01
  ):
    self.max_iter = max_iter
    self.eps = eps
    self.projection = projection
    self.w: Optional[np.ndarray] = None
    self.loss = loss
    self.loss_g = loss_g

  def fit(self, x: np.ndarray, y: np.ndarray, test_x: Optional[np.ndarray] = None, test_y: Optional[np.ndarray] = None, *, batch_size: Optional[int] = None):
    batch_size_ = batch_size if batch_size is not None else len(y)

    scores = np.zeros((self.max_iter + 1, 2))
    px = self.projection(x)

    self.w = np.zeros(px.shape[1])
    # self.w = np.random.uniform(-1.0, 1.0, x.shape[1])
    # print(self.loss(w, x, y))

    test_px = self.projection(test_x) if (test_x is not None) else None

    it = 0
    scores[0, 0] = self.score(x, y)

    # batches_x = np.array_split(x, batch_count)
    # batches_y = np.array_split(y, batch_count)

    split = np.arange(batch_size_, len(y), batch_size_)

    if (test_x is not None) and (test_y is not None):
      scores[0, 1] = self.score(test_x, test_y)

    for it in range(self.max_iter):
      random_indices = np.random.permutation(len(y))
      batches_x = np.split(px[random_indices, :], split)
      batches_y = np.split(y[random_indices], split)

      # batches_x = batches_x[:1]
      # batches_y = batches_y[:1]

      # Remove the last batch because it has a lower size
      if len(batches_y) > 1:
        batches_x = batches_x[:-1]
        batches_y = batches_y[:-1]

      # batches_x = [x]
      # batches_y = [y]

      batches_grad = np.array([self.loss_g(self.w, batch_x, batch_y) for batch_x, batch_y in zip(batches_x, batches_y)])
      self.w -= self.eps * batches_grad.mean(axis=0)

      # print(self.loss(w, x, y))
      # print(self.loss(w, x, y), w, self.loss_g(w, x, y))
      # print(y * np.dot(x, w))

      scores[it + 1, 0] = self._score_projected(px, y)

      if (test_px is not None) and (test_y is not None):
        scores[it + 1, 1] = self._score_projected(test_px, test_y)

    return scores[:(it + 2), :]

  def _predict_projected(self, px: np.ndarray):
    assert self.w is not None
    return np.sign(np.dot(px, self.w))

  def _score_projected(self, px: np.ndarray, y: np.ndarray):
    return (self._predict_projected(px) == y).sum() / len(y)

  def score(self, x: np.ndarray, y: np.ndarray):
    return self._score_projected(self.projection(x), y)


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

def show_usps(ax: Axes, data):
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


def plot1():
  def run(against_all: bool):
    model = Lineaire(eps=1e-3, max_iter=20)

    train_mask = (train_y == 6) | ((train_y == 9) if not against_all else True)
    # train_mask = [True] * len(train_y) # (train_y == 6) | ((train_y == 9) if not against_all else True)
    train_ax = train_x[train_mask, :]
    train_ay = np.where(train_y[train_mask] == 6, 1, -1)

    train_ax += np.random.normal(0, 10.0, train_ax.shape)

    test_mask = (test_y == 6) | ((test_y == 9) if not against_all else True)
    # test_mask = [True] * len(test_y)
    test_ax = test_x[test_mask, :]
    test_ay = np.where(test_y[test_mask] == 6, 1, -1)
    # print(test_y[test_mask] == 1)
    # test_ay = np.random.choice([-1, 1], len(test_y[test_mask]), p=[0.5, 0.5]) # np.where(test_y[test_mask] == 1, 1, -1)

    # test_ax += np.random.normal(0, 15.0, test_ax.shape)

    scores = model.fit(train_ax, train_ay, test_ax, test_ay, batch_size=100)
    # print(scores)

    # print((test_y[test_mask] != 6).sum())
    # print((test_y[test_mask] == 6).sum())

    # p = model.predict(test_ax)
    # print((p == 1).sum())

    # print(model.score(train_ax, train_ay))
    # print(model.score(test_ax, test_ay))

    fig1, ax = plt.subplots()

    show_usps(ax, model.w)


    fig2, ax = plt.subplots()

    ax.plot(np.arange(1, scores.shape[0]), scores[1:, 0], label='Entraînement')
    ax.plot(np.arange(1, scores.shape[0]), scores[1:, 1], label='Test')
    ax.legend(loc='lower right')

    ax.set_xlabel('Époque')
    ax.set_ylabel('Score')

    return fig1, fig2


  fig1, fig2 = run(False)

  with (output_path / '1.png').open('wb') as file:
    fig1.savefig(file)

  with (output_path / '2.png').open('wb') as file:
    fig2.savefig(file)


  fig1, fig2 = run(True)

  with (output_path / '3.png').open('wb') as file:
    fig1.savefig(file)

  with (output_path / '4.png').open('wb') as file:
    fig2.savefig(file)

  # plt.show()


def plot2():
  x, y = gen_arti(data_type=2, epsilon=0.1)
  print(x.shape)
  print(y.shape)

  model = Lineaire(eps=1e-3, max_iter=20, projection=proj_poly)

  fig, ax = plt.subplots()

  scores = model.fit(x, y)
  # print(model.w)

  ax.plot(np.arange(1, scores.shape[0]), scores[1:, 0], label='Entraînement')


  fig, ax = plt.subplots()

  plot_data(ax, x, y)

  delta = 0.025
  xrange = np.arange(-2, 2, delta)
  yrange = np.arange(-2, 2, delta)
  X, Y = np.meshgrid(xrange,yrange)

  # F is one side of the equation, G is the other
  # F = X**2
  # G = 1- (5*Y/4 - np.sqrt(np.abs(X)))**2
  # plt.contour((F - G), [0])
  # plt.show()

  ds = model.projection(np.array([X, Y]).T)
  dv = np.dot(ds, model.w)

  print(ds.shape)
  print(dv.shape)

  ax.contour(X, Y, dv, levels=[0], colors='red')

  # print(ds.shape)
  # print(np.array([xrange, yrange]).shape)


  plt.show()


plot2()
