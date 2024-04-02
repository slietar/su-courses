import pickle
from abc import ABC, abstractmethod
from math import floor
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.rcsetup import cycler
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from .. import config, utils


POI_FILENAME = Path(__file__).parent / 'data/poi-paris.pkl'
parismap = mpimg.imread(str(Path(__file__).parent / 'data/paris-48.806-2.23--48.916-2.48.jpg'))
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]


class Density(ABC):
  @abstractmethod
  def fit(self, data: np.ndarray):
    ...

  @abstractmethod
  def predict(self, data: np.ndarray) -> np.ndarray:
    ...

  def score(self, data: np.ndarray):
    return np.log(np.maximum(self.predict(data), 1e-10)).sum()


class Histogramme(Density):
  def __init__(self, steps: int = 10):
    super().__init__()

    self.steps = steps

    self.edges: Optional[list[np.ndarray]] = None
    self.hist: Optional[np.ndarray] = None

  @property
  def _bin_volume(self):
    assert self.edges is not None
    return (self.edges[0][1] - self.edges[0][0]) * (self.edges[1][1] - self.edges[1][0])

  def fit(self, x: np.ndarray):
    self.hist, self.edges = np.histogramdd(x, bins=self.steps, density=True)
    # bin_volume = (self.edges[0][1] - self.edges[0][0]) * (self.edges[1][1] - self.edges[1][0])
    # self.hist *= self._bin_volume

  def predict(self, x: np.ndarray):
    assert self.edges is not None
    assert self.hist is not None

    # bin_indices = np.array(list(np.digitize(x[:, dim], self.edges[dim][1:], right=True) for dim in range(x.shape[1])))
    # print(bin_indices.shape)
    # print(bin_indices)

    # dims = np.arange(x.shape[1])
    # print(self.edges[0, bin_indices])

    # bin_volume = (self.edges[0][1] - self.edges[0][0]) * (self.edges[1][1] - self.edges[1][0])
    # bin_volume = np.array([self.edges[dim][bin_indices[dim] + 1] - self.edges[dim][bin_indices[dim]] for dim in range(x.shape[1])]).prod()
    # return self.hist[*bin_indices]
    return self.hist[*(np.digitize(x[:, dim], self.edges[dim][1:], right=True) for dim in range(x.shape[1]))]


def kernel_uniform(x: np.ndarray):
  return np.where((np.abs(x) <= 0.5).all(axis=-1), 1.0, 0.0)

def kernel_gaussian(x: np.ndarray):
  return np.exp(-0.5 * (x ** 2).sum(axis=-1)) / ((2 * np.pi) ** (x.shape[-1] * 0.5))


class KernelDensity(Density):
  def __init__(self, kernel: Callable[[np.ndarray], np.ndarray], sigma: float):
    super().__init__()

    self.kernel = kernel
    self.sigma = sigma
    self.x: Optional[np.ndarray] = None

  def fit(self, x: np.ndarray):
    self.x = x

  def predict(self, data: np.ndarray):
    assert self.x is not None
    return self.kernel((data[:, None, :] - self.x[None, :, :]) / self.sigma).sum(axis=1) / (self.sigma ** self.x.shape[1]) / self.x.shape[0]


class Nadaraya:
  def __init__(self, kernel: Callable[[np.ndarray], np.ndarray], sigma: float):
    self.kernel = kernel
    self.sigma = sigma

    self.x: Optional[np.ndarray] = None
    self.y: Optional[np.ndarray] = None

  def fit(self, x: np.ndarray, y: np.ndarray):
    self.x = x
    self.y = y

  def predict(self, data: np.ndarray):
    assert self.x is not None
    assert self.y is not None

    v = self.kernel((data[:, None, :] - self.x[None, :, :]) / self.sigma)
    return (v * self.y).sum(axis=1) / v.sum(axis=1)


def get_density2D(f,data,steps=100):
  """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
  """
  xmin, xmax = data[:,0].min(), data[:,0].max()
  ymin, ymax = data[:,1].min(), data[:,1].max()
  xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
  xx, yy = np.meshgrid(xlin,ylin)
  grid = np.c_[xx.ravel(), yy.ravel()]
  res = f.predict(grid).reshape(steps, steps)
  return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
  """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
  """
  res, xlin, ylin = get_density2D(f, data, steps)
  xx, yy = np.meshgrid(xlin, ylin)
  plt.figure()
  show_img()
  if log:
    res = np.log(res+1e-10)
  plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
  show_img(res)
  plt.colorbar()
  plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
  """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
  """
  origin = "lower" if len(img.shape) == 2 else "upper"
  alpha = 0.3 if len(img.shape) == 2 else 1.
  plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
  ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
  """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])

  Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store,
  clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
  """
  poidata = pickle.load(open(fn, "rb"))
  data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
  note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
  return data,note


np.random.seed(42)

# Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
# La fonction charge la localisation des POIs dans geo_mat et leur note.
geo_bars_unpermuted, notes_bars_raw_unpermuted = load_poi('bar')
geo_mat_rest, _ = load_poi('restaurant')
geo_clubs, _ = load_poi('night_club')

bars_permutation = np.random.permutation(geo_bars_unpermuted.shape[0])

geo_bars = geo_bars_unpermuted[bars_permutation, :]
notes_bars_raw = notes_bars_raw_unpermuted[bars_permutation]

bars_first_test_index = int(0.8 * len(geo_bars))
clubs_first_test_index = int(0.8 * len(geo_clubs))

geo_bars_training = geo_bars[:bars_first_test_index, :]
geo_bars_test = geo_bars[bars_first_test_index:, :]

geo_clubs_training = geo_clubs[:clubs_first_test_index, :]
geo_clubs_test = geo_clubs[clubs_first_test_index:, :]


indices_noted_bars = notes_bars_raw >= 0
notes_bars = notes_bars_raw[indices_noted_bars]
geo_noted_bars = geo_bars[indices_noted_bars, :]

noted_bars_first_test_index = int(0.8 * geo_noted_bars.shape[0])

geo_noted_bars_training = geo_noted_bars[:noted_bars_first_test_index, :]
geo_noted_bars_test = geo_noted_bars[noted_bars_first_test_index:, :]

notes_bars_training = notes_bars[:noted_bars_first_test_index]
notes_bars_test = notes_bars[noted_bars_first_test_index:]


output_path = Path('output/tme1')
output_path.mkdir(exist_ok=True, parents=True)


def format_coord(value: float, pos: float):
  deg = floor(value)
  amin = round((value - deg) * 60)

  return f'{deg}° {amin:02}′'

def plot_map(ax: Axes):
  return ax.imshow(parismap, extent=coords, aspect=1.5, origin='upper', alpha=0.8)

def plot_distrib(ax: Axes, data: np.ndarray):
  return ax.imshow(data, extent=coords, aspect=1.5, origin='lower', alpha=0.3)

def plot_density(density: Density, ax: Axes, *, bin_count: int, color_bar: bool = False):
  res, xlin, ylin = get_density2D(density, geo_bars, bin_count)
  xx, yy = np.meshgrid(xlin, ylin)

  plot_map(ax)
  ax.scatter(geo_bars[:, 0], geo_bars[:, 1], alpha=0.8, s=0.5)
  im = plot_distrib(ax, res)
  ax.contour(xx, yy, res, 20)

  if color_bar:
    ax.figure.colorbar(im)


def plot1():
  fig, ax = plt.subplots()

  ax.xaxis.set_major_formatter(format_coord)
  ax.yaxis.set_major_formatter(format_coord)

  ax.imshow(parismap, extent=coords, aspect=1.5, origin='lower', alpha=0.3)
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')

  current_color1: Optional[Any] = None
  current_color2: Optional[Any] = None

  batch_size = 100

  for batch_start in range(0, max(geo_bars.shape[0], geo_mat_rest.shape[0]), batch_size):
    sl = slice(batch_start, batch_start + batch_size)

    current_color1 = ax.scatter(
      geo_bars[sl, 0],
      geo_bars[sl, 1],
      color=current_color1,
      alpha=0.5,
      label=('Restaurants' if current_color1 is None else None),
      s=2
    ).get_facecolor()

    current_color2 = ax.scatter(
      geo_mat_rest[sl, 0],
      geo_mat_rest[sl, 1],
      color=current_color2,
      alpha=0.5,
      label=('Bars' if current_color2 is None else None),
      s=2
    ).get_facecolor()

  ax.legend()

  with (output_path / '1.png').open('wb') as file:
    fig.savefig(file)


def plot2():
  fig, axs = plt.subplots(2, 2)

  for bin_count, ax in zip([5, 10, 25, 50], axs.flat):
    ax: Axes
    ax.axis('off')
    ax.set_title(f'N = {bin_count}')

    hist = Histogramme(bin_count)
    hist.fit(geo_bars)

    # bin_volume = (hist.edges[0][1] - hist.edges[0][0]) * (hist.edges[1][1] - hist.edges[1][0])
    # print((hist.hist * bin_volume).sum())

    res, xlin, ylin = get_density2D(hist, geo_bars, bin_count)
    xx, yy = np.meshgrid(xlin, ylin)

    plot_map(ax)
    im = ax.contourf(xx, yy, res, alpha=0.6, levels=10)

    fig.colorbar(im, ax=ax)

  fig.subplots_adjust(bottom=0.02)

  with (output_path / '2.png').open('wb') as file:
    fig.savefig(file)


def plot4():
  for plot_index, (training, test) in enumerate([
    (geo_bars_training, geo_bars_test),
    (geo_clubs_training, geo_clubs_test)
  ]):
    steps_list = np.arange(1, 30, 1)
    likelihoods = np.empty((len(steps_list), 2))

    for index, steps in enumerate(steps_list):
      h = Histogramme(steps=steps)
      h.fit(training)

      likelihoods[index, 0] = h.score(training) / training.shape[0]
      likelihoods[index, 1] = h.score(test) / test.shape[0]

    fig, ax = plt.subplots()

    ax.plot(steps_list, likelihoods[:, 0], label='Entraînement')
    ax.plot(steps_list, likelihoods[:, 1], label='Test')

    ax.set_xlabel('Nombre de bins N')
    ax.set_ylabel('Vraisemblance moyenne')
    ax.grid()

    ax.legend(loc='lower left')

    with (output_path / f'{3 + plot_index}.png').open('wb') as file:
      fig.savefig(file)

    print('Bin count with maximum likelihood:', likelihoods[:, 1].argmax())


def plot5():
  for plot_index, (kernel, sigmas) in enumerate([
    (kernel_gaussian, [5e-4, 1e-3, 1e-2, 5e-1]),
    (kernel_uniform, [1e-4, 1e-3, 1e-2, 5e-2])
  ]):
    fig, axs = plt.subplots(2, 2)

    for sigma, ax in zip(sigmas, axs.flat):
      ax: Axes
      ax.axis('off')
      ax.set_title(f'σ = {utils.format_scientific(sigma, precision=0)}')

      hist = KernelDensity(kernel, sigma=sigma)
      hist.fit(geo_bars)

      res, xlin, ylin = get_density2D(hist, geo_bars)
      xx, yy = np.meshgrid(xlin, ylin)

      plot_map(ax)
      im = ax.contourf(xx, yy, res, alpha=0.6, levels=10)

      fig.colorbar(im, ax=ax)

    fig.subplots_adjust(bottom=0.02)

    with (output_path / f'{5 + plot_index}.png').open('wb') as file:
      fig.savefig(file)


def plot7():
  for plot_index, (kernel, bounds) in enumerate([
    (kernel_gaussian, (0.0005, 0.1)),
    (kernel_uniform, (0.00001, 0.1))
  ]):
    sigma_list = np.exp(np.linspace(np.log(bounds[0]), np.log(bounds[1]), 20))
    likelihoods = np.empty((len(sigma_list), 2))

    for index, sigma in tqdm(list(enumerate(sigma_list))):
      h = KernelDensity(kernel, sigma=sigma)
      h.fit(geo_bars_training)

      likelihoods[index, 0] = h.score(geo_bars_training) / geo_bars_training.shape[0]
      likelihoods[index, 1] = h.score(geo_bars_test) / geo_bars_test.shape[0]

    f = interp1d(sigma_list, likelihoods[:, 1], kind='cubic')
    sigma_max = minimize_scalar(lambda x: -f(x), bounds=(sigma_list[0], sigma_list[-1]))
    print(f'{["Gaussian", "uniform"][plot_index]} kernel sigma with maximum likelihood: {sigma_list[likelihoods[:, 1].argmax()]:.3e}, {sigma_max.x:.3e}')

    fig, ax = plt.subplots()

    ax.plot(sigma_list, likelihoods[:, 0], label='Entraînement')
    ax.plot(sigma_list, likelihoods[:, 1], label='Test')
    # ax.axvline(sigma_max.x, color='silver', linestyle='--')

    ax.set_xlabel('Hyperparamètre σ')
    ax.set_ylabel('Vraisemblance moyenne')
    ax.set_xscale('log')
    ax.grid()

    ax.legend()

    with (output_path / f'{7 + plot_index}.png').open('wb') as file:
      fig.savefig(file)


def plot9():
  fig, ax = plt.subplots()
  fig.set_figheight(3.5)
  fig.subplots_adjust(bottom=0.02)

  plot_map(ax)

  ax.axis('off')
  im = ax.scatter(geo_noted_bars[:, 0], geo_noted_bars[:, 1], c=notes_bars, cmap='RdYlGn', s=0.5, vmin=0, vmax=5)

  cbar = fig.colorbar(im, ax=ax)

  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('Note', rotation=270)

  with (output_path / '9.png').open('wb') as file:
    fig.savefig(file)


def plot10():
  for plot_name, kernel, bounds in [
    (10, kernel_gaussian, (0.0005, 0.1)),
    (11, kernel_uniform, (0.0005, 0.1))
  ]:
    sigma_list = np.exp(np.linspace(np.log(bounds[0]), np.log(bounds[1]), 30))
    errors = np.empty((len(sigma_list), 2))

    for index, sigma in tqdm(list(enumerate(sigma_list))):
      nadaraya = Nadaraya(kernel, sigma=sigma)
      nadaraya.fit(geo_noted_bars_training, notes_bars_training)

      pr_training = nadaraya.predict(geo_noted_bars_training)
      pr_test = nadaraya.predict(geo_noted_bars_test)

      errors[index, 0] = ((pr_training - notes_bars_training) ** 2).mean()
      errors[index, 1] = ((pr_test - notes_bars_test) ** 2).mean()

    f = interp1d(sigma_list, errors[:, 1], kind='cubic')
    sigma_min = minimize_scalar(f, bounds=(sigma_list[0], sigma_list[-1]))
    print(f'Gaussian/uniform kernel sigma with minimum error: {sigma_list[errors[:, 1].argmin()]:.3e}, {sigma_min.x:.3e}')

    fig, ax = plt.subplots()

    ax.plot(sigma_list, errors[:, 0], label='Entraînement')
    ax.plot(sigma_list, errors[:, 1], label='Test')

    ax.set_xlabel('Hyperparamètre σ')
    ax.set_ylabel('Erreur moyenne')
    ax.set_xscale('log')
    ax.grid()

    ax.legend()

    with (output_path / f'{plot_name}.png').open('wb') as file:
      fig.savefig(file)

  # print(((notes_bars_training.mean() - notes_bars_test) ** 2).mean())
  # print(((notes_bars_training.mean() - notes_bars_training) ** 2).mean())


# plot1()
# plot2()
# plot4()
plot5()
# plot7()
# plot9()
# plot10()
