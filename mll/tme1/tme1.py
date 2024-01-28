from abc import ABC, abstractmethod
from math import floor
from pathlib import Path
import sys
from typing import Callable, Optional
from matplotlib.ticker import Formatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


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

  def predict(self, x: np.ndarray):
    assert self.edges is not None
    assert self.hist is not None

    return self.hist[*(np.digitize(x[:, dim], self.edges[dim][1:], right=True) for dim in range(x.shape[1]))]


def kernel_uniform(x: np.ndarray):
  return np.where(np.any(x < 0.5, axis=-1), 1.0, 0.0)

def kernel_gaussian(x: np.ndarray, d: int = 2):
  return np.exp(-0.5 * (np.linalg.norm(x, axis=-1) ** 2)) / ((2 * np.pi) ** (d * 0.5))


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

    return self.kernel((data[:, None, :] - self.x[None, :, :]) / self.sigma).sum(axis=1) / (self.sigma ** data.shape[1]) / self.x.shape[0]


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


# # Affiche la carte de Paris
# show_img()

# # Affiche les POIs
# plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)
# plt.show()


output_path = Path('output')
output_path.mkdir(exist_ok=True, parents=True)


def format_coord(value: float, pos: float):
  deg = floor(value)
  amin = round((value - deg) * 60)

  return f'{deg}° {amin:02}′'


fig, ax = plt.subplots()

ax.xaxis.set_major_formatter(format_coord)
ax.yaxis.set_major_formatter(format_coord)

ax.imshow(parismap, extent=coords, aspect=1.5, origin='lower', alpha=0.3)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.scatter(geo_mat_rest[:, 0], geo_mat_rest[:, 1], alpha=0.5, color='r', label='Bars', s=2)
ax.scatter(geo_mat_bars[:, 0], geo_mat_bars[:, 1], alpha=0.5, color='b', label='Restaurants', s=2)

ax.legend()

with (output_path / '1.png').open('wb') as file:
  fig.savefig(file, bbox_inches='tight', dpi=300)


sys.exit()


h = KernelDensity(kernel_uniform, sigma=0.1)
h.fit(geo_mat_bars)
# print(h.predict(geo_mat[:100, :]).shape)

show_density(h, geo_mat_bars, log=False)
plt.show()

sys.exit()


steps = 15
h = Histogramme(steps=steps)
h.fit(geo_mat_bars[:, 0])

# show_density(h, geo_mat, steps=steps, log=False)
# plt.show()


first_test_index = int(0.8 * len(geo_mat_bars))
geo_mat_training = geo_mat_bars[:first_test_index, :]
geo_mat_test = geo_mat_bars[first_test_index:, :]


steps_list = np.arange(1, 50, 1)
likelihoods = np.empty((len(steps_list), 2))

for index, steps in enumerate(steps_list):
  h = Histogramme(steps=steps)
  h.fit(geo_mat_training)

  likelihoods[index, 0] = h.score(geo_mat_training)
  likelihoods[index, 1] = h.score(geo_mat_test)


fig, ax1 = plt.subplots()
ax1.plot(steps_list, likelihoods[:, 0])

ax2 = ax1.twinx()
ax2.plot(steps_list, likelihoods[:, 1])

plt.show()
