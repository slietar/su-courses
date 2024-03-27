import math
from typing import Callable, Optional

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .. import mltools

class BoostingClassifier:
  def __init__(self, create_classifier: Callable[[], DecisionTreeClassifier], classifier_count: int):
    self.classifiers = [create_classifier() for _ in range(classifier_count)]
    self.classifier_weights = np.zeros(classifier_count)

  def fit(self, x: np.ndarray, y: np.ndarray):
    all_sample_weights = np.empty((*x.shape[:-1], len(self.classifiers)))
    es = np.empty(len(self.classifiers))
    zs = np.empty(len(self.classifiers))

    sample_count = np.prod(x.shape[:-1])
    sample_weights = np.ones(x.shape[:-1]) / sample_count
    z = 1.0

    for classifier_index, classifier in enumerate(self.classifiers):
      all_sample_weights[..., classifier_index] = sample_weights # * sample_count
      classifier.fit(x, y, sample_weights)
      y_pred = classifier.predict(x)

      error = (sample_weights * ~np.isclose(y_pred, y)).sum().clip(1e-10, 1.0 - 1e-10)
      classifier_weight = 0.5 * np.log(1.0 / error - 1.0)

      self.classifier_weights[classifier_index] = classifier_weight

      sample_weights = sample_weights * np.exp(-classifier_weight * y * y_pred)
      sample_weights /= sample_weights.sum()
      # sample_weights /= sample_weights.sum()

      z /= sample_weights.sum()

      es[classifier_index] = error
      zs[classifier_index] = z

    return all_sample_weights, es, zs

  def predict(self, x: np.ndarray, /):
    return np.sign((self.classifier_weights * [classifier.predict(x) for classifier in self.classifiers]).sum())

model = BoostingClassifier(
  classifier_count=12,
  create_classifier=(lambda: DecisionTreeClassifier(max_depth=1))
)

# fig, ax = plt.subplots()
# mltools.plot_data(ax, x, y)

def lighten_color(color, value: float):
  """
  Lightens the given color by multiplying (1-luminosity) by the given amount.
  Input can be matplotlib color string, hex string, or RGB tuple.

  Examples:
  >> lighten_color('g', 0.3)
  >> lighten_color('#F034A3', 0.6)
  >> lighten_color((.3,.55,.1), 0.5)
  """
  import matplotlib.colors as mc
  import colorsys
  try:
      c = mc.cnames[color]
  except:
      c = color

  c = np.array(mc.to_rgb(c))
  # print(mc.to_rgb(c))
  # c = colorsys.rgb_to_hls(*mc.to_rgb(c))
  # return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

  return c * value[..., None] + np.array([1, 1, 1]) * (1.0 - value[..., None])


def plot1():
  x, y = mltools.gen_arti(data_type=1)

  sample_weights, es, zs = model.fit(x, y)

  column_count = 3
  fig, axs = plt.subplots(math.ceil(len(model.classifiers) / column_count), column_count)

  for classifier_index, (classifier, ax) in enumerate(zip(model.classifiers, axs.ravel())):
    # mltools.plot_data(ax, x, y)
    ax.set_title(f'Classifier {classifier_index + 1}')

    match classifier.tree_.feature[0]:
      case 0:
        ax.axvline(classifier.tree_.threshold[0], color='C3', linestyle='--')
      case 1:
        ax.axhline(classifier.tree_.threshold[0], color='C3', linestyle='--')

    for cl_index, cl in enumerate([-1, 1]):
      norm = Normalize()

      mask = np.isclose(y, cl)
      # print(sample_weights[mask, cl_index])
      # ax.scatter(x[mask, 0], x[mask, 1], c=lighten_color(f'C{cl_index}', norm(sample_weights[mask, cl_index])), label=f'Class {cl}')
      # print(np.isclose(classifier.predict(x[mask, :]), y[mask]))
      ax.scatter(x[mask, 0], x[mask, 1], alpha=(norm(sample_weights[mask, classifier_index]) * 0.95 + 0.05), c=np.array(['r', 'g'])[np.isclose(classifier.predict(x[mask, :]), y[mask]).astype(int)], label=f'Class {cl}')

    # ax.axline((0.0, 0.0), slope=(-model2.coef_[0, 0] / model2.coef_[0, 1]), color='C3', label='SVM', linestyle='--')

  # mltools.plot_frontiere(x, classifier.predict, ax=axs[classifier_index // column_count, classifier_index % column_count])


  fig, ax = plt.subplots()

  ax.plot(es, label=r'$\epsilon_t$')
  ax.plot(zs, label=r'$Z_t$')
  ax.legend()

# print(axs.shape)

plot1()
plt.show()
