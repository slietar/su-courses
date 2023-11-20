import numpy as np
from matplotlib import pyplot as plt


def gen_data_lin(a: float, b: float, sig: float, n_train: int, n_test: int):
  x_train = np.sort(np.random.rand(n_train))
  x_test = np.sort(np.random.rand(n_test))

  y_train = a * x_train + b + np.random.normal(scale=sig, size=n_train)
  y_test = a * x_test + b + np.random.normal(scale=sig, size=n_test)

  return x_train, y_train, x_test, y_test

def modele_lin_analytique(x: np.ndarray, y: np.ndarray):
  # ddof?

  cov = np.cov(x, y)
  a = cov[0, 1] / cov[0, 0]

  return a, np.mean(y) - a * np.mean(x)

def calcul_prediction_lin(x: np.ndarray, a: float, b: float):
  return a * x + b

def erreur_mc(y: np.ndarray, y_pred: np.ndarray):
  return np.mean((y - y_pred) ** 2)

def dessine_reg_lin(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, a_pred: float, b_pred: float):
  # erreur dans le notebook sur a et b

  # a_pred, b_pred = modele_lin_analytique(x_train, y_train)
  y_pred = calcul_prediction_lin(x_test, a_pred, b_pred)
  err = erreur_mc(y_test, y_pred)

  fig, ax = plt.subplots()

  ax.plot(x_test, y_test, 'r.', alpha=0.2, label='test')
  ax.plot(x_train, y_train, 'b.', label='train')
  ax.plot([0, 1], [b_pred, a_pred + b_pred], 'g', label='prediction', linewidth=3)
  ax.legend()

def make_mat_lin_biais(x: np.ndarray):
  return np.c_[x, np.ones(x.shape)]

def reglin_matriciel(xe: np.ndarray, y: np.ndarray):
  return np.linalg.inv(xe.T @ xe) @ xe.T @ y

def calcul_prediction_matriciel(x: np.ndarray, w: np.ndarray):
  return x @ w

def gen_data_poly2(a: int, b: int, c: int, sig: float, N: int, Ntest: int):
  x_train = np.sort(np.random.rand(N))
  x_test = np.sort(np.random.rand(Ntest))
  w = np.array([a, b, c])

  y_train = np.polyval(w, x_train) + np.random.normal(scale=sig, size=N)
  y_test = np.polyval(w, x_test) + np.random.normal(scale=sig, size=Ntest)

  return x_train, y_train, x_test, y_test

def make_mat_poly_biais(x: np.ndarray):
  return np.c_[x ** 2, x, np.ones(x.shape)]

def dessine_poly_matriciel(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, w: np.ndarray):
  fig, ax = plt.subplots()

  x_curve = np.linspace(0, 1, 100)

  ax.plot(x_test, y_test, 'r.', alpha=0.2, label='test')
  ax.plot(x_train, y_train, 'b.', label='train')
  ax.plot(x_curve, np.polyval(w, x_curve), 'g', label='prediction', linewidth=3)
  ax.legend()
