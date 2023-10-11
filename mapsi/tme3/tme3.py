# Antoine GRISLAIN
# Simon LIÉTAR

import numpy as np
from matplotlib import pyplot as plt


def learnML_parameters(x: np.ndarray, y: np.ndarray):
  def map_digit(digit: int):
    xs = x[y == digit, :]
    return xs.mean(axis=0), xs.std(axis=0)

  return np.moveaxis(np.array([map_digit(digit) for digit in range(10)]), 0, 1)


def log_likelihood(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, defeps: float):
  bounded_sigma = np.maximum(sigma, defeps)

  with np.errstate(divide='ignore', invalid='ignore'):
    return -0.5 * np.where(
      (sigma == 0.0) & (defeps < 0.0),
      0.0,
      np.log(2.0 * np.pi * bounded_sigma ** 2) + ((x - mu) / bounded_sigma) ** 2
    ).sum()


def classify_image(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, defeps: float):
  return np.argmax([log_likelihood(x, mu[digit, :], sigma[digit, :], defeps) for digit in range(10)])

def classify_all_images(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, defeps: float):
  return np.array([classify_image(x[index, :], mu, sigma, defeps) for index in range(x.shape[0])])

def matrice_confusion(y: np.ndarray, y_pred: np.ndarray):
  result = np.zeros((10, 10))

  for expected, predicted in zip(y, y_pred):
    result[expected, predicted] += 1

  return result

def classificationRate(y: np.ndarray, y_pred: np.ndarray):
  confusion = matrice_confusion(y, y_pred)
  return confusion.trace() / len(y)

def classifTest(x: np.ndarray, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, defeps: float):
  y_pred = classify_all_images(x, mu, sigma, defeps)

  print("1- Classify all test images ...")
  print(f"2- Classification rate :  {classificationRate(y, y_pred)}")
  print("3- Matrice de confusion :")

  plt.figure(figsize=(3,3))
  plt.imshow(matrice_confusion(y, y_pred))

  return (y != y_pred).nonzero()


def binarisation(x: np.ndarray):
  return x > 0.0


def learnBernoulli(x: np.ndarray, y: np.ndarray):
  return np.array([x[y == digit, :].mean(axis=0) for digit in range(10)])

def logpobsBernoulli(x: np.ndarray, theta: np.ndarray, epsilon: float):
  bounded_theta = theta.clip(epsilon, 1.0 - epsilon)
  return (x * np.log(bounded_theta) + (1.0 - x) * np.log(1.0 - bounded_theta)).sum(axis=1)

  """
  On a une valeur positive donc le produit des probabilités est supérieur à 1, ce qui ne parait pas normal. Ceci est du au fait que l'on n'a pas normalisé.
  """

def classifBernoulliTest(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
  y_pred = np.argmax([logpobsBernoulli(x[index, :], theta, epsilon=1e-5) for index in range(x.shape[0])], axis=1)

  print("1- Classify all test images ...")
  print(f"2- Classification rate :  {classificationRate(y, y_pred)}")
  print("3- Matrice de confusion :")

  plt.figure(figsize=(3,3))
  plt.imshow(matrice_confusion(y, y_pred))


def learnGeom(x: np.ndarray, y: np.ndarray, seuil: float):
  return 1.0 / np.array([x[y == digit, :].mean(axis=0) for digit in range(10)])

def logpobsGeom(x: np.ndarray, theta: np.ndarray):
  # prod P(X = k | p) = prod (1 - p)^(k - 1) * p
  # sum log P(X = k | p) = sum (k - 1) * log(1 - p) + log(p)

  return ((x - 1.0) * np.log(1 - theta) + np.log(theta)).sum(axis=1)

def classifyGeom(x: np.ndarray, theta: np.ndarray):
  return logpobsGeom(x, theta).argmax()
