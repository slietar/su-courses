from collections import namedtuple
import functools
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import itertools
import random
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm

from .. import config, mltools, utils


def perceptron_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.maximum(-y * np.dot(x, w), 0).sum()

def perceptron_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return (-y * x.T * (-y * np.dot(x, w) >= 0)).sum(axis=1)


def hinge_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray, *, alpha: float, lambda_: float):
  return np.maximum(alpha - y * np.dot(x, w), 0).sum() + lambda_ * np.dot(w, w).sum()

def hinge_loss_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray, *, alpha: float, lambda_: float):
  # print(x.shape)
  # print(y.shape)
  # print(w.shape)
  # print((w * x).sum(axis=1).shape)
  # return np.where(alpha - y * (w * x).sum(axis=1) > 0, -y[:, None] * x, 0).sum(axis=1) # + 2 * lambda_ * w

  return (-y * x.T * ((alpha - y * np.dot(x, w)) >= 0)).sum(axis=1) + 2.0 * lambda_ * w


def proj_poly(x: np.ndarray, /):
  return np.c_[
    np.ones((*x.shape[:-1], 1)),
    x,
    *(x[..., a, None] * x[..., b, None] for a, b in itertools.combinations_with_replacement(range(x.shape[-1]), 2))
  ]

def proj_biais(x: np.ndarray, /):
  return np.c_[np.ones((*x.shape[:-1], 1)), x]

def proj_identity(x: np.ndarray, /):
  return x

def proj_gauss(x: np.ndarray, /, *, base: np.ndarray, sigma: float):
  return np.exp(-0.5 * (np.linalg.norm(x[..., None, :] - base, axis=-1) / sigma) ** 2)


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
    self.loss = loss
    self.loss_g = loss_g

    self.w: Optional[np.ndarray] = None
    self.w0: Optional[np.ndarray] = None

  def fit(
      self,
      x: np.ndarray,
      y: np.ndarray,
      test_x: Optional[np.ndarray] = None,
      test_y: Optional[np.ndarray] = None,
      /, *,
      batch_size: Optional[int] = None,
      random_weights: bool = False
  ):
    batch_size_ = batch_size if batch_size is not None else len(y)

    scores = np.zeros((self.max_iter + 1, 2))
    px = self.projection(x)

    if random_weights:
      self.w0 = np.random.uniform(-1.0, 1.0, px.shape[1])
      self.w = self.w0.copy()
    else:
      self.w = np.zeros(px.shape[1])

    test_px = self.projection(test_x) if (test_x is not None) else None

    it = 0
    scores[0, 0] = self.score(x, y)

    split = np.arange(batch_size_, len(y), batch_size_)

    if (test_px is not None) and (test_y is not None):
      scores[0, 1] = self._score_projected(test_px, test_y)

    for it in range(self.max_iter):
      random_indices = np.random.permutation(len(y))
      batches_x = np.split(px[random_indices, :], split)
      batches_y = np.split(y[random_indices], split)

      # batches_x = batches_x[:1]
      # batches_y = batches_y[:1]

      # # Remove the last batch because it has a lower size
      # if len(batches_y) > 1:
      #   batches_x = batches_x[:-1]
      #   batches_y = batches_y[:-1]

      # print([len(batch) for batch in batches_y])

      for batch_x, batch_y in zip(batches_x, batches_y):
        self.w -= self.eps * self.loss_g(self.w, batch_x, batch_y)

      # batches_grad = np.array([self.loss_g(self.w, batch_x, batch_y) for batch_x, batch_y in zip(batches_x, batches_y)])
      # self.w -= self.eps * batches_grad.mean(axis=0)

      # print(self.loss(w, x, y))
      # print(self.loss(w, x, y), w, self.loss_g(w, x, y))
      # print(y * np.dot(x, w))

      scores[it + 1, 0] = self._score_projected(px, y)

      if (test_px is not None) and (test_y is not None):
        scores[it + 1, 1] = self._score_projected(test_px, test_y)

    return scores
    # return scores[:(it + 2), :]

  def _predict_projected(self, px: np.ndarray, /):
    assert self.w is not None
    return np.sign(np.dot(px, self.w))

  def _score_projected(self, px: np.ndarray, y: np.ndarray, /):
    return (self._predict_projected(px) == y).sum() / len(y)

  def predict(self, x: np.ndarray, /):
    return self._predict_projected(self.projection(x))

  def predict_value(self, x: np.ndarray, /):
    assert self.w is not None
    return np.dot(self.projection(x), self.w)

  def score(self, x: np.ndarray, y: np.ndarray, /):
    return self._score_projected(self.projection(x), y)


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

@functools.cache
def get_usps():
  train_x, train_y = load_usps(data_path / 'USPS_train.txt')
  test_x, test_y = load_usps(data_path / 'USPS_test.txt')

  return namedtuple('USPS', ['train_x', 'train_y', 'test_x', 'test_y'])(train_x, train_y, test_x, test_y)



def show_usps(ax: Axes, data: np.ndarray):
  abs_max = max(abs(data.min()), abs(data.max()))

  im = ax.imshow(data.reshape((16, 16)), interpolation='nearest', cmap='RdYlBu_r', vmin=-abs_max, vmax=abs_max)

  ax.get_figure().colorbar(im, ax=ax)
  ax.tick_params('both', bottom=False, left=False, labelbottom=False, labelleft=False)


output_path = Path('output/tme5')
output_path.mkdir(exist_ok=True, parents=True)

data_path = Path(__file__).parent / 'data'


def plot0():
  x, y = mltools.gen_arti(data_type=0)
  lim = mltools.get_lim_for_data_type(data_type=0)

  model = Lineaire(eps=1e-3, max_iter=20)
  model.fit(x, y)

  fig, ax = plt.subplots()

  mltools.plot_data(ax, x, y)

  assert model.w is not None
  ax.axline((0.0, 0.0), slope=(-model.w[0] / model.w[1]), color='gray', label='Frontière de décision', linestyle='--')

  ax.set_xlim(*lim)
  ax.set_ylim(*lim)

  ax.legend()

  with (output_path / '19.png').open('wb') as file:
    fig.savefig(file)


def run_usps(
  *,
  against_all: bool,
  batch_size: Optional[int] = None,
  noise_sigma: float = 1.0,
  noise_weight: float = 0.0
):
  np.random.seed(2)

  data = get_usps()
  model = Lineaire(eps=1e-4, max_iter=50)

  train_indices1 = (data.train_y == 6).nonzero()[0]
  train_indices2 = (data.train_y != 6 if against_all else data.train_y == 9).nonzero()[0]
  train_index_count = min(len(train_indices1), len(train_indices2))
  train_indices1 = train_indices1[np.random.permutation(len(train_indices1))[:train_index_count]]
  train_indices2 = train_indices2[np.random.permutation(len(train_indices2))[:train_index_count]]
  train_indices = np.r_[train_indices1, train_indices2]

  train_ax = data.train_x[train_indices, :]
  train_ay = np.where(data.train_y[train_indices] == 6, 1, -1)

  train_ax = train_ax + noise_weight * np.random.normal(0, noise_sigma, train_ax.shape)

  # fig, ax = plt.subplots(ncols=3, nrows=3)

  # for i, ax in enumerate(ax.flat):
  #   show_usps(ax, train_ax[i, :])

  # plt.show()
  # sys.exit()

  test_indices1 = (data.test_y == 6).nonzero()[0]
  test_indices2 = (data.test_y != 6 if against_all else data.test_y == 9).nonzero()[0]
  test_index_count = min(len(test_indices1), len(test_indices2))
  test_indices1 = test_indices1[:test_index_count]
  test_indices2 = test_indices2[:test_index_count]
  test_indices = np.r_[test_indices1, test_indices2]

  test_ax = data.test_x[test_indices, :]
  test_ay = np.where(data.test_y[test_indices] == 6, 1, -1)

  np.random.seed(2)
  scores = model.fit(train_ax, train_ay, test_ax, test_ay, batch_size=batch_size, random_weights=True)

  return model, scores


def plot1a():
  model1, scores1 = run_usps(against_all=False)
  model2, scores2 = run_usps(against_all=True)
  models = [model1, model2]
  all_scores = [scores1, scores2]

  for plot_index, model in enumerate(models):
    fig, axs = plt.subplots(ncols=2)
    fig.set_figheight(3.0)

    show_usps(axs[0], model.w)
    show_usps(axs[1], model.w - model.w0)

    axs[0].set_title('$w^{(T)}$')
    axs[1].set_title('$w^{(T)} - w^{(0)}$')

    with (output_path / f'{1 + plot_index}.png').open('wb') as file:
      fig.savefig(file)


  fig, axs = plt.subplots(nrows=2, squeeze=False)
  fig.set_figheight(5.0)
  fig.subplots_adjust(left=0.2, right=0.8)

  ax: Axes

  for ax_index, (model, scores, ax) in enumerate(zip(models, all_scores, axs.flat)):
    offset = 0
    ax.plot(np.arange(offset, scores.shape[0]), scores[offset:, 0], label=('Entraînement' if ax_index < 1 else None))
    ax.plot(np.arange(offset, scores.shape[0]), scores[offset:, 1], label=('Test' if ax_index < 1 else None))

    ax.set_xlabel('Époque')
    ax.set_ylabel('Score')

    ax.set_title(['6 contre 9', '6 contre tous'][ax_index])

    ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  utils.filter_axes(axs)

  fig.legend()

  with (output_path / '3.png').open('wb') as file:
    fig.savefig(file)


def plot1b():
  batch_sizes = [1, 10, 100, 500, None]

  fig, ax = plt.subplots()
  fig.subplots_adjust(bottom=0.15)

  for batch_size in batch_sizes:
    model, scores = run_usps(against_all=True, batch_size=batch_size, noise_weight=0.5)
    ax.plot(scores[:, 1], label=(f'm = {batch_size}' if batch_size is not None else 'm = 1328 (batch complet)'))

  ax.set_xlabel('Époque')
  ax.set_ylabel('Score')

  ax.grid()
  ax.legend(loc='lower right')

  with (output_path / '4.png').open('wb') as file:
    fig.savefig(file)



def plot2():
  fig, axs = plt.subplots(ncols=2, nrows=1)
  fig.set_figheight(3.2)

  for ax_index, (ax, (data_type, proj)) in enumerate(zip(axs, itertools.product([1, 2], [proj_biais, proj_poly]))):
    x, y = mltools.gen_arti(data_type=data_type, epsilon=0.1)
    lim = mltools.get_lim_for_data_type(data_type)

    model = Lineaire(max_iter=5, projection=proj)
    model.fit(x, y)

    # fig, ax = plt.subplots()

    mltools.plot_data(ax, x, y)
    utils.plot_boundary(ax, model.predict_value, label=(ax_index < 1), x_range=lim, y_range=lim)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

  fig.legend()
  fig.subplots_adjust(wspace=0.3, bottom=0.15)

  with (output_path / '5.png').open('wb') as file:
    fig.savefig(file)

  print(' + '.join(f'{a}{b}' for a, b in zip(model.w, ['', 'x', 'y', 'x^2', 'xy', 'y^2'])))


def create_grid(x_range: Sequence[float], y_range: Sequence[float], count: int = 100):
  # x = np.linspace(x_range[0], x_range[1], count)
  # y = np.linspace(y_range[0], y_range[1], count)
  x, y = np.meshgrid(
    np.linspace(x_range[0], x_range[1], count),
    np.linspace(y_range[0], y_range[1], count)
  )

  return np.array([x, y]).transpose((1, 2, 0))

def plot3():
  for plot_index, (data_type, base, sigma, ext_lim_var) in enumerate([
    (0, np.array([[0.0, 0.0], [0.5, 0.5]]), 1.0, 1.0),
    (0, np.array([[1.0, 2.0], [2.0, -1.0]]), 1.0, 1.0),
    (0, np.array([[0.0, 0.0], [0.5, 0.5]]), 5.0, 5.0),
    (1, create_grid((-2.0, 2.0), (-2.0, 2.0), 3).reshape(-1, 2), 1.5, 1.5),
    (1, create_grid((-2.0, 2.0), (-2.0, 2.0), 3).reshape(-1, 2), 0.5, 1.5),
    (2, create_grid((-4.0, 4.0), (-4.0, 4.0), 12).reshape(-1, 2), 0.5, 1.5),
    # (2, create_grid((-2.0, 2.0), (-2.0, 2.0), 12).reshape(-1, 2), 0.5, 1.5),
  ]):
    x, y = mltools.gen_arti(data_type=data_type, epsilon=(0.005 if data_type == 2 else 0.1))
    # lim = mltools.get_lim_for_data_type(data_type)
    lim = -2.5, 2.5

    # model = Lineaire(max_iter=150, projection=lambda x: proj_biais(create_proj_gauss(base, sigma=sigma)(x)))
    model = Lineaire(max_iter=150, projection=functools.partial(proj_gauss, base=base, sigma=sigma))
    model.fit(x, y)

    # fig, (ax, _) = plt.subplots(2, 1)
    fig = plt.figure(figsize=(config.fig_width, 3.2))
    ax = fig.add_subplot(1, 2, 1)

    mltools.plot_data(ax, x, y)
    utils.plot_boundary(ax, model.predict_value, x_range=lim, y_range=lim)

    ax.scatter(*base.T, color='C3', label='Base', marker='^')

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    if plot_index < 1:
      fig.legend()


    # fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ext_lim = (
      lim[0] - ext_lim_var,
      lim[1] + ext_lim_var
    )

    grid = create_grid(ext_lim, ext_lim, 150)
    grid_z = model.predict_value(grid)

    grid_z_range = max(
      abs(grid_z.min()),
      abs(grid_z.max())
    )

    if data_type == 2:
      grid_z_range *= 2.0

    mltools.plot_data_3d(ax, x, y, z=-grid_z_range)
    ax.plot_surface(grid[..., 0], grid[..., 1], grid_z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    ax.contour(grid[..., 0], grid[..., 1], grid_z, zdir='z', offset=-grid_z_range, cmap='coolwarm')
    ax.set(xlim=ext_lim, ylim=ext_lim, zlim=(-grid_z_range, grid_z_range), xlabel='X₁', ylabel='X₂', zlabel='f')

    fig.subplots_adjust(bottom=0.15)

    with (output_path / f'{6 + plot_index}.png').open('wb') as file:
      fig.savefig(file)


def plot4():
  x, y = mltools.gen_arti(data_type=2, epsilon=0.005)
  lim = mltools.get_lim_for_data_type(2)
  # lim = -2.5, 2.5

  alpha = 10.0
  lambda_ = 0.5

  base = create_grid((-4.0, 4.0), (-4.0, 4.0), 30).reshape(-1, 2)
  model = Lineaire(
    loss=functools.partial(hinge_loss, alpha=alpha, lambda_=lambda_),
    loss_g=functools.partial(hinge_loss_grad, alpha=alpha, lambda_=lambda_),
    max_iter=150,
    projection=functools.partial(proj_gauss, base=base, sigma=0.5)
  )

  model.fit(x, y)

  # fig, (ax, _) = plt.subplots(2, 1)
  fig = plt.figure(figsize=(config.fig_width, 3.2))
  ax = fig.add_subplot(1, 2, 1)

  mltools.plot_data(ax, x, y)
  utils.plot_boundary(ax, model.predict_value, x_range=lim, y_range=lim)

  # ax.scatter(*base.T, color='C3', label='Base', marker='^')

  ax.set_xlim(*lim)
  ax.set_ylim(*lim)


def plot5a():
  np.random.seed(0)

  x, y = mltools.gen_arti(data_type=0, epsilon=0.3)

  model1 = LinearRegression()
  model1.fit(x, y)

  model2 = SVC(kernel='linear')
  model2.fit(x, y)

  fig, ax = plt.subplots()

  mltools.plot_data(ax, x, y, highlight=model2.support_)

  ax.axline((0.0, 0.0), slope=(-model1.coef_[0] / model1.coef_[1]), color='C2', label='Perceptron', linestyle='--')
  ax.axline((0.0, 0.0), slope=(-model2.coef_[0, 0] / model2.coef_[0, 1]), color='C3', label='SVM', linestyle='--')

  ax.legend()

  with (output_path / '14.png').open('wb') as file:
    fig.savefig(file)


def plot5b():
  np.random.seed(1)

  x, y = mltools.gen_arti(data_type=0, epsilon=0.1)
  lim = (-1.5, 1.5)

  cs = [0.1, 1.0, 10.0]

  fig, axs = plt.subplots(ncols=len(cs))
  fig.set_figheight(2.2)

  ax: Axes

  for c, ax in zip(cs, axs):
    model = SVC(kernel='linear', C=c)
    model.fit(x, y)

    DecisionBoundaryDisplay.from_estimator(
      estimator=model,
      ax=ax,
      X=x,
      colors=['gray'],
      levels=[-1, 0, 1],
      linestyles=[':', '--', ':'],
      plot_method='contour',
      response_method='decision_function'
    )

    mltools.plot_data(ax, x, y, highlight=model.support_)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    ax.set_title(f'C = {c:.1f}')

  fig.subplots_adjust(bottom=0.2)
  utils.filter_axes(axs[None, :])

  with (output_path / '15.png').open('wb') as file:
    fig.savefig(file)



def plot6a():
  np.random.seed(0)

  x, y = mltools.gen_arti(data_type=1)
  lim = mltools.get_lim_for_data_type(data_type=1)

  fig, axs = plt.subplots(ncols=2)
  fig.set_figheight(3.0)

  ax: Axes

  for plot_index, (kernel, ax) in enumerate(zip(['rbf', 'poly'], axs)):
    model = SVC(kernel=kernel, degree=2, gamma=0.1)
    model.fit(x, y)

    DecisionBoundaryDisplay.from_estimator(
      estimator=model,
      ax=ax,
      X=x,
      colors=['gray'],
      levels=[0],
      linestyles=['--'],
      plot_method='contour',
      response_method='decision_function'
    )

    mltools.plot_data(ax, x, y, highlight=model.support_)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    ax.set_title(['Noyau gaussien (RBF)', 'Noyau polynomial de degré 2'][plot_index])

  fig.subplots_adjust(bottom=0.15)
  utils.filter_axes(axs[None, :])

  with (output_path / '16.png').open('wb') as file:
    fig.savefig(file)

def plot6b():
  np.random.seed(1)

  x, y = mltools.gen_arti(data_type=1, epsilon=0.2, nbex=100)
  lim = (-2.5, 2.5)

  gs = [0.1, 0.5, 10.0]

  fig, axs = plt.subplots(ncols=len(gs))
  fig.set_figheight(2.2)

  ax: Axes

  for gamma, ax in zip(gs, axs):
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(x, y)

    DecisionBoundaryDisplay.from_estimator(
      estimator=model,
      ax=ax,
      X=x,
      cmap='RdYlBu_r',
      alpha=0.5,
      plot_method='contourf',
      response_method='decision_function'
    )

    mltools.plot_data(ax, x, y, highlight=model.support_)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    ax.set_title(f'γ = {gamma:.1f}')

  fig.subplots_adjust(bottom=0.2)
  utils.filter_axes(axs[None, :])

  with (output_path / '17.png').open('wb') as file:
    fig.savefig(file)


def plot6c():
  import pandas as pd

  data = get_usps()

  grid = GridSearchCV(SVC(), [
    { 'kernel': ['poly'], 'degree': [2, 3, 4], 'C': [0.5, 1.0, 5.0] },
    { 'kernel': ['linear', 'rbf'], 'C': [0.5, 1.0, 5.0] }
  ], return_train_score=True)

  grid.fit(data.train_x[:10000, :], data.train_y[:10000])

  df = pd.DataFrame(grid.cv_results_)
  df.sort_values(['param_kernel', 'param_degree', 'param_C'], inplace=True)

  output = 'table(\n  align: (left, center, center),\n  columns: 4,\n  stroke: none,\n  table.header[*Noyau*][*$C$*][*Entraînement*][*Test*],\n  table.hline(),\n'

  for _, row in df.iterrows():
    match row.param_kernel:
      case 'linear':
        name = 'Linéaire'
      case 'poly':
        name = f'Polynomial de degré {row.param_degree}'
      case 'rbf':
        name = 'Gaussien'

    em = '*' if row.mean_test_score == df.mean_test_score.max() else ''
    output += f'  [{name}], [{row.param_C:.1f}], [{row.mean_train_score:.3f}], [{em}{row.mean_test_score:.3f}{em}],\n'

  output += ')\n'

  print(output)


def plot6d():
  data = get_usps()

  training_counts = np.arange(100, 3000, 100)
  scores = np.empty((len(training_counts), 2))

  for count_index, count in enumerate(training_counts):
    model = SVC(kernel='linear', C=5.0)
    model.fit(data.train_x[:count, :], data.train_y[:count])

    scores[count_index, 0] = model.score(data.train_x, data.train_y)
    scores[count_index, 1] = model.score(data.test_x, data.test_y)

  fig, ax = plt.subplots()

  ax.plot(training_counts, scores[:, 0], label='Entraînement')
  ax.plot(training_counts, scores[:, 1], label='Test')

  ax.set_xlabel('Nombre de points d\'entraînement')
  ax.set_ylabel('Score')

  ax.grid()
  ax.legend(loc='lower right')

  fig.subplots_adjust(bottom=0.15)

  with (output_path / '18.png').open('wb') as file:
    fig.savefig(file)



def tokenize(text: str, /):
  return [word for word in re.split(r'[^a-z]', utils.remove_accents(text.lower())) if len(word) > 2]

@functools.cache
def get_subsequences(word: str, k: int, *, _add_offset: bool = False):
  if k == 1:
    return { tuple[str, ...]((letter,)): [(letter_index if _add_offset else 0) + 1] for letter_index, letter in enumerate(word) }

  result = dict[tuple[str, ...], list[int]]()

  for letter_index, letter in enumerate(word):
    for subword, spans in get_subsequences(word[(letter_index + 1):], k - 1, _add_offset=True).items():
      result.setdefault((letter, *subword), []).extend([span + 1 for span in spans])

  return result

def string_kernel(a: list[str], b: list[str]):
  lambda_ = 0.8

  all_words = list(set(a) | set(b))
  subwords_per_word = {
    word: {
      subword: sum(lambda_ ** span for span in spans) for subword, spans in get_subsequences(word, 3).items()
    } for word in all_words
  }

  norms = { word: np.sqrt(sum(score ** 2 for score in subwords_per_word[word].values())) for word in all_words }
  result = np.zeros((len(a), len(b)))

  for index_a, word_a in enumerate(a):
    for index_b, word_b in enumerate(b):
      result[index_a, index_b] = sum(
        score_a * subwords_per_word[word_b].get(subword_a, 0.0) for subword_a, score_a in subwords_per_word[word_a].items()
      ) / norms[word_a] / norms[word_b]

  return result


  # Cleaner algorithm but requires more memory

  # all_subwords = list(functools.reduce(operator.or_, [set(word_subsequences.keys()) for word_subsequences in subwords_per_word.values()], set()))
  # all_subwords_inverse = { subword: index for index, subword in enumerate(all_subwords) }

  # values = np.zeros((len(all_words), len(all_subwords)))

  # for word_index, word in enumerate(all_words):
  #   for subword, value in subwords_per_word[word].items():
  #     values[word_index, all_subwords.index(subword)] = value

  # a_indices = [all_words.index(word) for word in a]
  # b_indices = [all_words.index(word) for word in b]

  # norms = np.sqrt((values ** 2).sum(axis=1))
  # return (values[None, b_indices, :] * values[a_indices, None, :]).sum(axis=2) / norms[None, b_indices] / norms[a_indices, None]


def plot7():
  fig, ax = plt.subplots()

  words = [
    'algorithm',
    'biology',
    'biorhythm',
    'competing',
    'computation',
    'logarithm',
    'rhythm',
  ]

  im = ax.imshow(string_kernel(words, words), cmap='plasma')
  ax.set_xticks(range(len(words)), words, rotation='vertical')
  ax.set_yticks(range(len(words)), words)

  ax.tick_params('x', bottom=False, labelbottom=False, labeltop=True)
  ax.tick_params('y', left=False)

  plt.setp(ax.get_xticklabels(), ha='left', rotation=45, rotation_mode='anchor')

  cbar = fig.colorbar(im, ax=ax)

  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('Similarité', rotation=270)

  fig.subplots_adjust(bottom=0.02, top=0.8)

  with (output_path / '12.png').open('wb') as file:
    fig.savefig(file)


def plot8():
  from sklearn.svm import SVC

  random.seed(0)
  np.random.seed(0)

  tokens = list[list[str]]()

  for text_index in [1, 2]:
    with (data_path / f'texts/{text_index}.txt').open('rt') as file:
      tokens.append(tokenize(file.read()))

    random.shuffle(tokens[-1])


  train0 = tokens[0][:500]
  train1 = tokens[1][:500]

  test_tokens = [
    tokens[0][500:],
    tokens[1][500:]
  ]

  model = SVC(kernel=string_kernel)
  model.fit([*train0, *train1], [0] * len(train0) + [1] * len(train1))

  repeat_count = 100
  result = np.zeros((repeat_count, 2))

  for repeat_index in tqdm(range(repeat_count)):
    for author_index in range(2):
      perm = np.random.permutation(len(test_tokens[author_index]))
      result[repeat_index, author_index] = model.predict([test_tokens[author_index][token_index] for token_index in perm[:100]]).mean()

  print(f'Correct predictions: {(((result[:, 0] < 0.5).sum() + (result[:, 1] > 0.5).sum()) / result.size * 100):.2f}%')

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)

  bins = np.linspace(0.0, 1.0, 21)

  ax.hist(result[:, 0], bins=bins, label='La Fontaine', alpha=0.5, rwidth=0.9)
  ax.hist(result[:, 1], bins=bins, label='Montaigne', alpha=0.5, rwidth=0.9)

  ax.set_xlim(0.0, 1.0)

  ax.set_xlabel('← La Fontaine', color='C0', loc='left')
  ax.set_ylabel('Nombre de prédictions')

  ax.yaxis.set_major_locator(MaxNLocator(integer=True))

  ax1 = fig.add_subplot(1, 1, 1, frame_on=False, yticks=[])
  ax1.set_xlabel('Montaigne →', color='C1', loc='right')

  plt.setp(ax1.get_xticklabels(), alpha=0.0)
  plt.setp(ax1.get_xticklines(), alpha=0.0)

  with (output_path / '13.png').open('wb') as file:
    fig.savefig(file)


# plot0()
# plot1a()
plot1b()
# plot2()
# plot3()
# plot5a()
# plot5b()
# plot6b()
# plot6d()
plt.show()
