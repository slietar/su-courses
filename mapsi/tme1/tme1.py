import operator

from matplotlib import pyplot as plt
import numpy as np


def analyse_rapide(data: np.ndarray, /):
  print(f"mean: {data.mean()}")
  print(f"std: {data.std()}")
  print("quantiles:", np.quantile(data, np.linspace(0, 1, 10, endpoint=False)))


def discretisation_histogramme(data: np.ndarray, /, n: int):
  data_min = data.min()
  bin_width = (data.max() - data_min) / n
  bin_edges = [data_min + index * bin_width for index in range(n + 1)]
  bin_values = [
    np.where(
      (data >= bin_edges[index]) &
      (operator.lt if index < (n - 1) else operator.le)(data, bin_edges[index + 1]), # Include the max value in the last bin
      1, 0
    ).sum() for index in range(n)
  ]

  print("bornes: ", bin_edges)
  print("effectifs: ", bin_values)

  plt.bar(bin_edges[:-1], bin_values, align='edge', width=bin_width)
  plt.title("Histogramme à la main")
  plt.show()

  plt.hist(data, bins=n)
  plt.title("Histogramme avec $\\tt{plt.hist}$")
  plt.show()

  bin_values_ref, bin_edges_ref = np.histogram(data, bins=n)

  print("diff bornes:", np.abs(bin_edges - bin_edges_ref).sum())
  print("diff effectifs:", np.abs(bin_values - bin_values_ref).sum())


def discretisation_prix_au_km(data: np.ndarray, /, n: int):
  discretisation_histogramme(data[:, 10] / data[:, 13], n=n)


def loi_jointe_distance_marque(data: np.ndarray, /, n: int, brand_names: dict[str, int]):
  distances = data[:, 13]
  brands = data[:, 11]

  bin_edges = np.linspace(distances.min(), distances.max(), n + 1)
  bin_edges[-1] += 1e-6 # Add epsilon to include the max value in the last bin

  bin_indices = np.digitize(distances, bins=bin_edges) - 1
  assert np.all(bin_indices >= 0) and np.all(bin_indices < n) # Check that all values were correctly digitized

  print("distance discrétisée:", bin_indices)

  p_dm = np.array([[
    np.where((bin_indices == bin_index) & (brands == brand), 1, 0).sum() for brand in range(len(brand_names))
  ] for bin_index in range(n)]) / data.shape[0]

  fig, ax = plt.subplots(1, 1)

  ax.imshow(p_dm, interpolation='nearest')
  ax.set_xticks(np.arange(len(brand_names)))
  ax.set_xticklabels(brand_names.keys(), fontsize=8, rotation=90)

  return p_dm


def loi_conditionnelle(p_dm: np.ndarray):
  # axis 0 = distance (30)
  # axis 1 = brand (54)

  c_dm = p_dm / p_dm.sum(axis=0)

  plt.imshow(c_dm, interpolation='nearest')
  plt.show()

  return c_dm


def check_conditionnelle(c_dm: np.ndarray):
  # Use np.allclose() to test equality with a tolerance
  return np.allclose(c_dm.sum(axis=0), 1)


def trace_trajectoires(data: np.ndarray, /):
  start_coords = data[:, 6:8]
  end_coords = data[:, 8:10]

  fig, ax = plt.subplots(1, 1)
  ax.quiver(*(start_coords.T), *(end_coords - start_coords).T, data[:, 5], angles='xy', cmap='tab10', scale=1, scale_units='xy')

  # Change x and y limits, otherwise the arrow heads arent't visible
  ax.set_xlim(30, 60)
  ax.set_ylim(-10, 20)


# Calculate the distance between two points on Earth's surface, in kilometers
def haversine(points: np.ndarray, ref: np.ndarray):
  lon1, lat1, lon2, lat2 = map(np.radians, [points[:, 1], points[:, 0], ref[1], ref[0]])

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

  return 6378.137 * 2 * np.arcsin(np.sqrt(a))


def calcule_matrice_distance(data: np.ndarray, city_coords_list: np.ndarray, /):
  return np.array([haversine(data[:, 6:8], city_coords) for city_coords in city_coords_list])

def calcule_coord_plus_proche(distances: np.ndarray, /):
  return np.argmin(distances, axis=0)

def trace_ville_coord_plus_proche(data: np.ndarray, city_colors: np.ndarray, /):
  start_coords = data[:, 6:8]
  end_coords = data[:, 8:10]

  plt.quiver(*start_coords.T, *(end_coords - start_coords).T, angles='xy', color=city_colors, scale_units='xy', scale=1)
  plt.show()


def test_correlation_distance_prix(data: np.ndarray, /):
  distances = data[:, 13]
  prices = data[:, 10]

  plt.scatter(distances, prices)
  plt.show()

  return np.corrcoef(distances, prices)[1, 0]


def test_correlation_distance_confort(data: np.ndarray, /):
  filtered_data = data[data[:, 12] >= 0, :] # Only keep rows with a score >= 0
  distances = filtered_data[:, 13]
  score = filtered_data[:, 12]

  plt.scatter(distances, score)
  plt.show()

  return np.corrcoef(distances, score)[1, 0]


def calcule_prix_km_seuillée(data: np.ndarray, /, quantile: float):
  price_per_distance = data[:, 10] / data[:, 13]
  return price_per_distance.clip(None, np.quantile(price_per_distance, quantile))

def discretisation(pmk: np.ndarray, /, eps: float, nintervalles: int):
  bin_edges = np.linspace(pmk.min(), pmk.max(), nintervalles)
  bin_edges[-1] += eps

  return np.digitize(pmk, bins=bin_edges)

def loi_jointe(pmk_discrete: np.ndarray, city_indices: np.ndarray, /):
  return np.histogramdd(
    (pmk_discrete, city_indices),
    bins=(
      pmk_discrete.max() + 1,
      city_indices.max() + 1
    ),
    density=True
  )[0]
