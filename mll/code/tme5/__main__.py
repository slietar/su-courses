import functools
import itertools
import operator
import random
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
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
    self.w: Optional[np.ndarray] = None
    self.loss = loss
    self.loss_g = loss_g

  def fit(self, x: np.ndarray, y: np.ndarray, test_x: Optional[np.ndarray] = None, test_y: Optional[np.ndarray] = None, /, *, batch_size: Optional[int] = None):
    batch_size_ = batch_size if batch_size is not None else len(y)

    scores = np.zeros((self.max_iter + 1, 2))
    px = self.projection(x)

    self.w = np.zeros(px.shape[1])
    # self.w = np.random.uniform(-1.0, 1.0, px.shape[1])
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
  fig, axs = plt.subplots(1, 2, figsize=(config.fig_width, 3.2))

  # for plot_index, (data_type, proj) in enumerate(itertools.product([1, 2], [proj_biais, proj_poly])):
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


def plot5():
  from sklearn.linear_model import LinearRegression
  from sklearn.svm import SVC

  x, y = mltools.gen_arti(data_type=0, epsilon=0.3)

  model1 = LinearRegression()
  model1.fit(x, y)

  model2 = SVC(kernel='linear')
  model2.fit(x, y)

  fig, ax = plt.subplots()

  mltools.plot_data(ax, x, y, highlight=model2.support_)
  # utils.plot_boundary(ax, lambda x: np.sign(model1.predict(x)), label=True)
  # utils.plot_boundary(ax, lambda x: np.sign(model2.predict(x)), label=True)

  ax.axline((0.0, 0.0), slope=(-model1.coef_[0] / model1.coef_[1]), color='C2', label='Perceptron', linestyle='--')
  ax.axline((0.0, 0.0), slope=(-model2.coef_[0, 0] / model2.coef_[0, 1]), color='C3', label='SVM', linestyle='--')

  ax.legend()

  print(model2.dual_coef_)

  # TODO: Save


def plot6():
  from sklearn.inspection import DecisionBoundaryDisplay
  from sklearn.svm import SVC

  x, y = mltools.gen_arti(data_type=1)

  model = SVC(kernel='rbf', degree=2)
  model.fit(x, y)

  print(model.class_weight_)
  # print(model.coef_)
  print(model.dual_coef_.shape)
  print(x.shape)
  print(model.support_.shape)

  fig, ax = plt.subplots()

  DecisionBoundaryDisplay.from_estimator(
    estimator=model,
    ax=ax,
    X=x,
    colors=(['gray'] * 3),
    levels=[-1, 0, 1],
    linestyles=['--', '-', '--'],
    plot_method='contour',
    response_method='decision_function',
  )

  mltools.plot_data(ax, x, y, highlight=model.support_)

  # print(model.support_)

  ax.set_xlim(-2.5, 2.5)
  ax.set_ylim(-2.5, 2.5)


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
  # print('>', a, b)
  lambda_ = 0.8

  all_words = list(set(a) | set(b))
  subwords_per_word = {
    word: {
      subword: sum(lambda_ ** span for span in spans) for subword, spans in get_subsequences(word, 3).items()
    } for word in all_words
  }

  norms = { word: np.sqrt(sum(score ** 2 for score in subwords_per_word[word].values())) for word in all_words }
  result = np.zeros((len(a), len(b)))

  # print(np.array((norms.values() == 0)).sum())
  # print(np.min(np.array(list(norms.values()))))

  # print({word for word, a in norms.items() if a == 0})

  for index_a, word_a in enumerate(a):
    for index_b, word_b in enumerate(b):
      # subwords_a = subwords_per_word[word_a]
      # subwords_b = subwords_per_word[word_b]

      result[index_a, index_b] = sum(
        score_a * subwords_per_word[word_b].get(subword_a, 0.0) for subword_a, score_a in subwords_per_word[word_a].items()
      ) / norms[word_a] / norms[word_b]

  # print(result)
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
    with Path(f'code/tme5/data/texts/{text_index}.txt').open('rt') as file:
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

  # print(model.predict(tokens[0][500:5000]).mean())
  # print(model.predict(tokens[1][500:5000]).mean())

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

  # word_counts = np.arange(10, 100, 10)
  # res = np.zeros((len(word_counts), 2))

  # p0 = np.random.permutation(len(test0))
  # for word_count_index, word_count in enumerate(word_counts):
  #   # res[word_count_index, 0] = model.predict([test0[i] for i in p0[:word_count]]).mean()
  #   # res[word_count_index, 1] = model.predict(tokens[1][500:(500 + word_count)]).mean()

  #   res[word_count_index, 0] = model.predict(train0).mean()
  #   res[word_count_index, 1] = model.predict(train1).mean()

  # fig, ax = plt.subplots()

  # ax.plot(word_counts, res[:, 0], label='La Fontaine')
  # ax.plot(word_counts, res[:, 1], label='Montaigne')

  # ax.legend()


  # print(model.predict(['pommeau', 'voiture', 'pomme', 'voiturier']))

  # for i in range(500, 2000, 500):
  #   print(model.predict(tokens[0][i:(i + 500)]).mean())
  #   print(model.predict(tokens[1][i:(i + 500)]).mean())

  # a = kernel(train1, train1)


  # print(len(tokens[0]), len(tokens[1]))
  # a = kernel([*train1, *train2], [0] * len(train)
  # print(len(tokens[0]))

  # print(((a > 0) & (a < 1 - 1e-6)).sum())

  # print(kernel(['pommeau', 'voiture', 'pomme', 'voiturier'], ['pommeau', 'voiture']))

def _():
  words = [
    # 'bar',
    # 'bat',
    # 'car',
    # 'cat',
    # 'catfish',
    # 'barbara',
    # 'barb'

    'logarithm',
    'algorithm',
    'biorhythm',
    'rhythm',
    'biology',
    'competing',
    'computation',

    # 'abandon',
    # 'abandoned',
    # 'abandoning',
    # 'abandonment',
    # 'abandons',
    # 'abase',
    # 'abased',
    # 'abasement',
    # 'abasements',
    # 'abases',
    # 'abash',
    # 'abashed',
    # 'abashes',
    # 'abashing',
    # 'abasing',
    # 'abate',
    # 'abated',
    # 'abatement',
    # 'abatements',
    # 'abater',
    # 'abates',
    # 'abating',
    # 'limit',
    # 'limitability',
    # 'limitably',
    # 'limitation',
    # 'limitations',
    # 'limited',
    # 'limiter',
    # 'limiters',
    # 'limiting',
    # 'limitless',
    # 'limits',
    # 'limousine',
    # 'limp',
    # 'limped',
    # 'limping',
    # 'limply',
    # 'limpness',
    # 'limps',
  ]

  def comb(word: str, k: int, *, _add_offset: bool = False):
    if k == 1:
      return { tuple[str, ...]((letter,)): [(letter_index if _add_offset else 0) + 1] for letter_index, letter in enumerate(word) }

    result = dict[tuple[str, ...], list[int]]()

    for letter_index, letter in enumerate(word):
      for subword, spans in comb(word[(letter_index + 1):], k - 1, _add_offset=True).items():
        result.setdefault((letter, *subword), []).extend([span + 1 for span in spans])

    return result

  # print(comb('bar', 1, _add_offset=True))
  # print(comb('bat', 2, _add_offset=False))

  lambda_ = 0.8

  words_comb = [{ subword: sum(lambda_ ** span for span in spans) for subword, spans in comb(word, 2).items() } for word in words]
  all_subwords = list(functools.reduce(operator.or_, [set(word_comb.keys()) for word_comb in words_comb], set()))

  values = np.zeros((len(words), len(all_subwords)))

  for word_index, word_comb in enumerate(words_comb):
    for subword, value in word_comb.items():
      values[word_index, all_subwords.index(subword)] = value

  import pandas as pd

  print('SUBWORDS')
  print(pd.DataFrame(values, columns=[''.join(subword) for subword in all_subwords], index=words))


  print(values.shape)
  a = (values ** 2).sum(axis=1)
  # print(np.sqrt(a[None, :] * a[:, None]).shape)

  similarity = (values[None, :, :] * values[:, None, :]).sum(axis=2) / np.sqrt(a[None, :] * a[:, None])

  print('SIMILARITY')
  print(pd.DataFrame(similarity, columns=words, index=words))

  # from sklearn.manifold import MDS

  # embedding = MDS(dissimilarity='precomputed', metric=False, normalized_stress=True, n_init=100)
  # ts = embedding.fit_transform(-similarity)

  # print('MDS')
  # # print(ts)
  # print(ts.shape)

  # # print(embedding.dissimilarity_matrix_)
  # # print(embedding.stress_)

  # fig, ax = plt.subplots()

  # ax.scatter(*ts.T)

  # for i, word in enumerate(words):
  #   ax.annotate(word, (ts[i, 0], ts[i, 1]))


  fig, ax = plt.subplots()

  # im = ax.imshow(similarity, cmap='viridis')
  im = ax.imshow(np.maximum(similarity, 0.01), cmap='plasma', norm=LogNorm(vmin=0.01, vmax=1))
  ax.set_xticks(range(len(words)), words, rotation='vertical')
  ax.set_yticks(range(len(words)), words)

  fig.colorbar(im, ax=ax)

  # import networkx as nx
  # import string

  # dt = [('len', float)]
  # A = values.view(dt)

  # G = nx.from_numpy_matrix(A)
  # G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))

  # G = nx.to_agraph(G)

  # G.node_attr.update(color="red", style="filled")
  # G.edge_attr.update(color="blue", width="2.0")

  # G.draw('distances.png', format='png', prog='neato')


  # dist = np.empty(
  #   (len(words), len(words)),
  #   dtype=float
  # )

  # for a, b in itertools.combinations(range(len(words)), 2):
  #   # print(words_comb[a], words_comb[b])
  #   subwords = words_comb[a].keys() & words_comb[b].keys()

  #   dist[a, b] = dist[b, a] = 3.0
  #   # dist[a, b] = dist[b, a] = np.linalg.norm(np.array([len(set(words[a]) & set(words[b])), len(set(words[a]) | set(words[b]))]) / len(words[a])

  # print(dist)

  # x = {}

  # for letter_index, letter in enumerate(words[0]):
  #   pass

  # subwords = set([subword for word in words for subword in itertools.combinations(word, 2)])
  # print(subwords)



# plot1()
# plot2()
# plot3()
plot8()
# plt.show()
