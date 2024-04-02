import functools
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import IndexLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import config, mltools, utils


class BoostingClassifier:
  def __init__(self, create_classifier: Callable[[], DecisionTreeClassifier], classifier_count: int):
    self.classifiers = [create_classifier() for _ in range(classifier_count)]
    self.classifier_weights = np.zeros(classifier_count)

  def fit(self, x: np.ndarray, y: np.ndarray):
    all_sample_weights = np.empty((*x.shape[:-1], len(self.classifiers)))
    model_predictions = np.empty((*x.shape[:-1], len(self.classifiers)))
    es = np.empty(len(self.classifiers))
    zs = np.empty(len(self.classifiers))

    sample_count = np.prod(x.shape[:-1])
    sample_weights = np.ones(x.shape[:-1]) / sample_count
    z = 1.0

    for classifier_index, classifier in enumerate(self.classifiers):
      all_sample_weights[..., classifier_index] = sample_weights
      classifier.fit(x, y, sample_weights)
      y_pred = classifier.predict(x)

      error = (sample_weights * ((y_pred * y) < 0)).sum().clip(1e-10, 1.0 - 1e-10)
      classifier_weight = 0.5 * np.log(1.0 / error - 1.0)

      self.classifier_weights[classifier_index] = classifier_weight

      sample_weights = sample_weights * np.exp(-classifier_weight * y * y_pred)
      zt = sample_weights.sum()

      z *= zt
      sample_weights /= zt

      es[classifier_index] = error
      zs[classifier_index] = z

      model_predictions[..., classifier_index] = np.sign((self.classifier_weights[:(classifier_index + 1)] * np.asarray([classifier.predict(x) for classifier in self.classifiers[:(classifier_index + 1)]]).T).sum(axis=1))

    return all_sample_weights, model_predictions, es, zs

  def predict(self, x: np.ndarray, /, *, classifier_count: Optional[int] = None):
    sl = slice(None, classifier_count) if classifier_count is not None else slice(None)
    return np.sign((self.classifier_weights[sl] * np.asarray([classifier.predict(x) for classifier in self.classifiers[sl]]).T).sum(axis=1))


output_path = Path('output/tme6')
output_path.mkdir(exist_ok=True, parents=True)


def plot1():
  np.random.seed(0)

  x, y = mltools.gen_arti(data_type=1)
  lim = mltools.get_lim_for_data_type(data_type=1)

  model = RandomForestClassifier(max_depth=10, n_estimators=4)
  model.fit(x, y)

  fig, axs = plt.subplots(2, 2)
  fig.set_figheight(6.0)

  displayed_estimator_count = min(len(model.estimators_), axs.size)

  estimator = model.estimators_[0]

  for estimator, ax in zip(model.estimators_[:displayed_estimator_count], axs.flat):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    utils.plot_tree(ax, estimator.tree_, range_x=lim, range_y=lim)
    mltools.plot_data(ax, x, y)

  utils.filter_axes(axs)

  with (output_path / '5.png').open('wb') as file:
    fig.savefig(file)


  fig, ax = plt.subplots()

  utils.plot_boundary_contour(ax, model.predict, x_range=lim, y_range=lim)
  mltools.plot_data(ax, x, y)

  fig.subplots_adjust(left=0.25, right=0.75)

  with (output_path / '6.png').open('wb') as file:
    fig.savefig(file)


def plot2():
  np.random.seed(0)

  for fig_index, (gen_params, depths) in enumerate([
    (dict(data_type=1), np.arange(1, 10)),
    (dict(data_type=2, epsilon=0.005), np.arange(1, 30))
  ]):
    train_x, train_y = mltools.gen_arti(**gen_params)
    test_x, test_y = mltools.gen_arti(**gen_params, nbex=500)

    estimator_counts = np.arange(1, 30)
    repeat_count = 1

    errors = np.empty((len(depths), len(estimator_counts), repeat_count, 2))

    for depth_index, depth in enumerate(depths):
      for estimator_count_index, estimator_count in enumerate(estimator_counts):
        for repeat_index in range(repeat_count):
          model = RandomForestClassifier(max_depth=depth, n_estimators=estimator_count)
          model.fit(train_x, train_y)

          errors[depth_index, estimator_count_index, repeat_index, 0] = 1.0 - model.score(train_x, train_y)
          errors[depth_index, estimator_count_index, repeat_index, 1] = 1.0 - model.score(test_x, test_y)

    fig, axs = plt.subplots(nrows=2, squeeze=False) if fig_index == 0 else plt.subplots(ncols=2, squeeze=False)
    ax: Axes

    if fig_index == 1:
      fig.set_figheight(2.8)

    for ax_index, ax in enumerate(axs.flat):
      im = ax.imshow(
        errors.mean(axis=2)[::-1, :, ax_index],
        aspect='equal',
        cmap='RdYlBu_r',
          extent=(
          estimator_counts[0] - 0.5,
          estimator_counts[-1] + 0.5,
          depths[0] - 0.5,
          depths[-1] + 0.5,
        ),
        vmin=0.0,
        vmax=0.6
      )

      ax.set_title(['Entraînement', 'Test'][ax_index])

      ax.xaxis.set_major_locator(IndexLocator(5, -0.5))
      ax.yaxis.set_major_locator(IndexLocator(2 if fig_index == 0 else 5, -0.5))

      ax.tick_params(bottom=False, left=False)

    full_ax = fig.add_subplot(1, 1, 1, frame_on=False, xticks=[], yticks=[])

    full_ax.set_ylabel('Profondeur maximale', labelpad=[15, 20][fig_index])
    full_ax.set_xlabel('Nombre d\'estimateurs', labelpad=[15, 10][fig_index])

    cbar = fig.colorbar(im, aspect=[22, 15][fig_index], ax=[*axs.flat, full_ax])

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Erreur', rotation=270)

    utils.filter_axes(axs)

    with (output_path / f'{3 + fig_index}.png').open('wb') as file:
      fig.savefig(file)



def plot3():
  np.random.seed(1)

  x, y = mltools.gen_arti(data_type=1)
  lim = mltools.get_lim_for_data_type(data_type=1)

  m = (x[:, 0] > 0) | (x[:, 1] > 0)
  x = x[m, :]
  y = y[m]


  model = BoostingClassifier(
    classifier_count=16,
    create_classifier=(lambda: DecisionTreeClassifier(max_depth=1))
  )

  sample_weights, model_predictions, es, zs = model.fit(x, y)

  displayed_classifier_count = 3
  markers = ['.', '+']
  fig, axs = plt.subplots(displayed_classifier_count, 2, figsize=(config.fig_width, 7.0))

  for classifier_index, classifier in enumerate(model.classifiers[:displayed_classifier_count]):
    ax = axs[classifier_index, 0]

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_title(f'Classifieur {classifier_index + 1}')

    norm = Normalize()

    utils.plot_boundary_contour(ax, classifier.predict, x_range=lim, y_range=lim)
    mltools.plot_data(ax, x, y, alpha=((norm(sample_weights[:, classifier_index]) * 0.9 + 0.1) if classifier_index > 0 else 1.0))

    ax1 = axs[classifier_index, 1]

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    utils.plot_boundary_contour(ax1, functools.partial(model.predict, classifier_count=(classifier_index + 1)), x_range=lim, y_range=lim)
    mltools.plot_data(ax1, x, y)

    if classifier_index == 0:
      ax1.set_title('Classifieur 1')
    else:
      ax1.set_title(f'Classifieurs 1 à {classifier_index + 1}')

  utils.filter_axes(axs)
  fig.subplots_adjust(bottom=0.08, top=0.95)

  with (output_path / '1.png').open('wb') as file:
    plt.savefig(file)


  fig, ax1 = plt.subplots()

  ax2 = ax1.twinx()

  l1 = ax1.plot(np.arange(len(model.classifiers)) + 1, es, label=r'$\epsilon_t$')
  l2 = ax2.plot(np.arange(len(model.classifiers)) + 1, zs, color='C1', label=r'Z')

  ax1.set_xlabel('Nombre de classifieurs t')
  ax1.set_ylabel(r'$\epsilon_t$')
  ax2.set_ylabel('Z')

  lines = l1 + l2
  ax1.legend(lines, [l.get_label() for l in lines])
  ax1.grid()

  with (output_path / '2.png').open('wb') as file:
    plt.savefig(file)

# plot1()
# plot2()
plot3()
# plt.show()
