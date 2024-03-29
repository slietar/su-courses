from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

from matplotlib.patches import Rectangle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.tree import DecisionTreeClassifier

from .. import config, mltools
from .utils import filter_axes, plot_tree


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

  def predict(self, x: np.ndarray, /):
    return np.sign((self.classifier_weights * np.asarray([classifier.predict(x) for classifier in self.classifiers]).T).sum(axis=1))


output_path = Path('output/tme6')
output_path.mkdir(exist_ok=True, parents=True)


def plot1():
  from sklearn.ensemble import RandomForestClassifier

  x, y = mltools.gen_arti(data_type=1)
  lim = mltools.get_lim_for_data_type(data_type=1)

  model = RandomForestClassifier(max_depth=10, n_estimators=10)
  model.fit(x, y)

  fig, axs = plt.subplots(3, 3)

  displayed_estimator_count = min(len(model.estimators_), axs.size)

  estimator = model.estimators_[0]

  for estimator, ax in zip(model.estimators_[:displayed_estimator_count], axs.flat):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    plot_tree(ax, estimator.tree_, range_x=lim, range_y=lim)
    mltools.plot_data(ax, x, y)

  filter_axes(axs)


def plot2():
  np.random.seed(1)

  x, y = mltools.gen_arti(data_type=1)

  m = (x[:, 0] > 0) | (x[:, 1] > 0)
  x = x[m, :]
  y = y[m]


  model = BoostingClassifier(
    classifier_count=16,
    create_classifier=(lambda: DecisionTreeClassifier(max_depth=3))
  )

  sample_weights, model_predictions, es, zs = model.fit(x, y)

  displayed_classifier_count = 3
  markers = ['.', '+']
  fig, axs = plt.subplots(displayed_classifier_count, 2, figsize=(config.fig_width, 7.0))

  for classifier_index, classifier in enumerate(model.classifiers[:displayed_classifier_count]):
    ax = axs[classifier_index, 0]

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    ax.set_title(f'Classifieur {classifier_index + 1}')

    # inv = ax.transData.inverted()

    tree = classifier.tree_
    threshold = tree.threshold[0]
    left_cl_index = 0 if tree.value[1][0, 0] > 0.5 else 1
    right_cl_index = 0 if tree.value[2][0, 0] > 0.5 else 1

    match tree.feature[0]:
      case 0:
        ax.scatter([threshold - 0.3], [-2], c='gray', marker=markers[left_cl_index])
        ax.scatter([threshold + 0.3], [-2], c='gray', marker=markers[right_cl_index])
        ax.axvline(threshold, color='gray', linestyle='--')
      case 1:
        ax.scatter([-2], [threshold - 0.3], c='gray', marker=markers[left_cl_index])
        ax.scatter([-2], [threshold + 0.3], c='gray', marker=markers[right_cl_index])
        ax.axhline(threshold, color='gray', linestyle='--')

    for cl_index, cl in enumerate([-1, 1]):
      mask = (y * cl) > 0
      norm = Normalize()

      ax.scatter(
        x[mask, 0],
        x[mask, 1],
        alpha=((norm(sample_weights[mask, classifier_index]) * 0.9 + 0.1) if classifier_index > 0 else 1.0),
        c=np.array(['r', 'g'])[((classifier.predict(x[mask, :]) * y[mask]) > 0).astype(int)],
        marker=markers[cl_index]
      )


    ax1 = axs[classifier_index, 1]

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)

    if classifier_index == 0:
      ax1.set_title('Classifieur 1')
    else:
      ax1.set_title(f'Classifieurs 1 à {classifier_index + 1}')

    for cl_index, cl in enumerate([-1, 1]):
      mask = (y * cl) > 0

      ax1.scatter(
        x[mask, 0],
        x[mask, 1],
        c=np.array(['r', 'g'])[((model_predictions[:, classifier_index][mask] * y[mask]) > 0).astype(int)],
        marker=markers[cl_index]
      )

  for ax in axs[:-1, :].ravel():
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

  for ax in axs[:, 1].ravel():
    ax.tick_params(axis='y', left=False, labelleft=False)

  for ax in axs[-1, :].ravel():
    ax.set_xlabel('X₁')

  for ax in axs[:, 0].ravel():
    ax.set_ylabel('X₂')

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

  with (output_path / '2.png').open('wb') as file:
    plt.savefig(file)

plot1()
plt.show()
