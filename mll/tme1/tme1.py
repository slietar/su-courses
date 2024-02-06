from abc import ABC, abstractmethod
from math import floor
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence
from matplotlib.axes import Axes
from matplotlib.rcsetup import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d


plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.sf'] = 'Helvetica'
plt.rcParams['figure.figsize'] = 21.0 / 2.54 - 2.0, 4.0
plt.rcParams['font.size'] = 11.0
plt.rcParams['figure.dpi'] = 288
plt.rcParams['grid.color'] = 'whitesmoke'
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.prop_cycle'] = cycler(color=[
  '#348abd',
  '#e24a33',
  '#988ed5',
  '#777777',
  '#fbc15e',
  '#8eba42',
  '#ffb5b8'
])


POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
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

  def fit(self, x: np.ndarray):
    self.hist, self.edges = np.histogramdd(x, bins=self.steps, density=True)
    # bin_volume = (self.edges[0][1] - self.edges[0][0]) * (self.edges[1][1] - self.edges[1][0])
    # self.hist *= bin_volume

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
  return np.where(np.any(np.abs(x) <= 0.5, axis=-1), 1.0, 0.0)

def kernel_gaussian(x: np.ndarray):
  return np.exp(-0.5 * (x ** 2).sum(axis=-1)) / ((2 * np.pi) ** (x.shape[-1] * 0.5))


class KernelDensity(Density):
  def __init__(self, kernel: Optional[Callable[[np.ndarray], np.ndarray]], sigma: float = 0.1):
    super().__init__()

    self.kernel = kernel
    self.sigma = sigma
    self.x: Optional[np.ndarray] = None

  def fit(self, x: np.ndarray):
    self.x = x

  def predict(self, data: np.ndarray):
    assert self.kernel is not None
    assert self.x is not None

    return self.kernel((data[:, None, :] - self.x[None, :, :]) / self.sigma).sum(axis=1) / (self.sigma ** self.x.shape[1]) / self.x.shape[0]


class Nadaraya(Density):
  pass


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


# plt.ion()
# Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
# La fonction charge la localisation des POIs dans geo_mat et leur note.
geo_mat_bars, notes_bars = load_poi('bar')
geo_mat_rest, notes_rest = load_poi('restaurant')

first_test_index = int(0.8 * len(geo_mat_bars))
geo_mat_bars_training = geo_mat_bars[:first_test_index, :]
geo_mat_bars_test = geo_mat_bars[first_test_index:, :]


output_path = Path('output')
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
  res, xlin, ylin = get_density2D(density, geo_mat_bars, bin_count)
  xx, yy = np.meshgrid(xlin, ylin)

  plot_map(ax)
  ax.scatter(geo_mat_bars[:, 0], geo_mat_bars[:, 1], alpha=0.8, s=0.5)
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

  ax.scatter(geo_mat_rest[:, 0], geo_mat_rest[:, 1], alpha=0.5, label='Bars', s=2)
  ax.scatter(geo_mat_bars[:, 0], geo_mat_bars[:, 1], alpha=0.5, label='Restaurants', s=2)

  ax.legend()

  with (output_path / '1.png').open('wb') as file:
    fig.savefig(file)


def plot2():
  fig, axs = plt.subplots(2, 2)

  for bin_count, ax in zip([5, 10, 25, 50], axs.flatten()):
    ax: Axes
    ax.axis('off')
    ax.set_title(f'$N = {bin_count}$')

    hist = Histogramme(bin_count)
    hist.fit(geo_mat_bars)

    res, xlin, ylin = get_density2D(hist, geo_mat_bars, bin_count)
    xx, yy = np.meshgrid(xlin, ylin)

    ax.scatter(geo_mat_bars[:, 0], geo_mat_bars[:, 1], alpha=0.8, s=3)
    plot_map(ax)
    # plt.colorbar()
    ax.contour(xx, yy, res, 20)

  # fig.subplots_adjust(left=0., right=1.0)

  with (output_path / '2.png').open('wb') as file:
    fig.savefig(file)


def plot3():
  fig, ax = plt.subplots()
  bin_count = 20

  ax.axis('off')
  ax.set_title(f'$N = {bin_count}$')

  hist = Histogramme(bin_count)
  hist.fit(geo_mat_bars)

  res, xlin, ylin = get_density2D(hist, geo_mat_bars, bin_count)
  xx, yy = np.meshgrid(xlin, ylin)

  plot_map(ax)
  ax.scatter(geo_mat_bars[:, 0], geo_mat_bars[:, 1], alpha=0.8, s=0.5)
  im = plot_distrib(ax, res)
  ax.contour(xx, yy, res, 20)

  fig.colorbar(im)

  with (output_path / '3.png').open('wb') as file:
    fig.savefig(file)


def plot4():
  steps_list = np.arange(1, 30, 1)
  likelihoods = np.empty((len(steps_list), 2))

  for index, steps in enumerate(steps_list):
    h = Histogramme(steps=steps)
    h.fit(geo_mat_bars_training)

    likelihoods[index, 0] = h.score(geo_mat_bars_training) / geo_mat_bars_training.shape[0]
    likelihoods[index, 1] = h.score(geo_mat_bars_test) / geo_mat_bars_test.shape[0]

  fig, ax = plt.subplots()

  ax.plot(steps_list, likelihoods[:, 0], label='Entraînement')
  ax.plot(steps_list, likelihoods[:, 1], label='Test')

  ax.set_xlabel('Nombre de bins')
  ax.set_ylabel('Vraisemblance par point')
  ax.grid()

  fig.legend()

  with (output_path / '4.png').open('wb') as file:
    fig.savefig(file)

  print('Bin count with maximum likelihood:', likelihoods[:, 1].argmax())


def plot5():
  fig, ax = plt.subplots()
  ax.axis('off')

  density = KernelDensity(kernel_uniform, sigma=0.01)
  density.fit(geo_mat_bars)

  plot_density(density, ax, bin_count=10, color_bar=True)

  with (output_path / '5.png').open('wb') as file:
    fig.savefig(file)


def plot6():
  fig, ax = plt.subplots()
  ax.axis('off')

  density = KernelDensity(kernel_gaussian, sigma=0.01)
  density.fit(geo_mat_bars)

  plot_density(density, ax, bin_count=10, color_bar=True)

  with (output_path / '6.png').open('wb') as file:
    fig.savefig(file)


def plot7():
  for plot_name, kernel, bounds in [
    (7, kernel_gaussian, (0.0005, 0.1)),
    (8, kernel_uniform, (0.00001, 0.1))
  ]:
    sigma_list = np.exp(np.linspace(np.log(bounds[0]), np.log(bounds[1]), 20))

    # Gaussian
    # sigma_list = np.linspace(0.0005, 0.005, 15)
    # sigma_list = np.exp(np.linspace(np.log(0.0005), np.log(0.1), 20))

    # Uniform
    # sigma_list = np.exp(np.linspace(*np.log([0.00001, 0.1]), 20))

    likelihoods = np.empty((len(sigma_list), 2))

    for index, sigma in enumerate(sigma_list):
      h = KernelDensity(kernel, sigma=sigma)
      h.fit(geo_mat_bars_training)

      likelihoods[index, 0] = h.score(geo_mat_bars_training) / geo_mat_bars_training.shape[0]
      likelihoods[index, 1] = h.score(geo_mat_bars_test) / geo_mat_bars_test.shape[0]

    f = interp1d(sigma_list, likelihoods[:, 1], kind='cubic')
    sigma_max = minimize_scalar(lambda x: -f(x), bounds=(sigma_list[0], sigma_list[-1]))
    print('Gaussian kernel sigma with maximum likelihood:', sigma_list[likelihoods[:, 1].argmax()], sigma_max.x)

    fig, ax = plt.subplots()

    ax.plot(sigma_list, likelihoods[:, 0], label='Entraînement')
    ax.plot(sigma_list, likelihoods[:, 1], label='Test')
    # ax.axvline(sigma_max.x, color='silver', linestyle='--')

    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('Vraisemblance par point')
    ax.set_xscale('log')
    ax.grid()

    fig.legend()

    with (output_path / f'{plot_name}.png').open('wb') as file:
      fig.savefig(file)



plot5()
plot6()
plot7()
