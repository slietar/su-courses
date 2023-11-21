# Antoine GRISLAIN
# SIMON LIÉTAR


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

def descente_grad_mc(x: np.ndarray, y: np.ndarray, *, eps: float, nIterations: int):
  w = np.zeros((nIterations, 2))

  for t in range(1, nIterations):
    w[t, 0] = w[t-1, 0] - eps * (-2 * x[:, 0] * (-w[t-1, 0] * x[:, 0] - w[t-1, 1] + y)).sum()
    w[t, 1] = w[t-1, 1] - eps * (-2 * (-w[t-1, 0] * x[:, 0] - w[t-1, 1] + y).sum())

  return w, w

def application_reelle(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
  w = reglin_matriciel(x_train, y_train)

  yhat_train = x_train @ w
  yhat_test = x_test @ w

  print(f"{w=}")
  print(f"{erreur_mc(y_train, yhat_train)=:.4f}")
  print(f"{erreur_mc(y_test, yhat_test)=:.4f}")

  return w, yhat_train, yhat_test

def normalisation(x_train: np.ndarray, x_test: np.ndarray):
  normalize = lambda x: np.c_[(x[:, 0:-1] - x[:, 0:-1].mean(axis=0)) / x[:, 0:-1].std(axis=0), np.ones(x.shape[0])]

  return normalize(x_train), normalize(x_test)

# Seules les variables 0, 1, 3 et 5 sont importantes, les autres sont proches de zéro.


# Fonctions importées du notebook

def plot_y(y_train, y_test, yhat, yhat_t):
    # tracé des prédictions:
    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=2, sharex='col', subplot_kw=dict(frameon=False))

    ax1[0].set_title('En test')
    ax1[0].plot(y_test, label="GT")
    ax1[0].plot(yhat_t, label="pred")
    ax1[0].legend()

    ax1[1].set_title('En train')
    ax1[1].plot(y_train, label="GT")
    ax1[1].plot(yhat, label="pred")

    ax2[0].set_title('$sorted(|err|)$ en test')
    ax2[0].plot(sorted(abs(y_test-yhat_t)), label="diff")

    ax2[1].set_title('$sorted(|err|)$ en train')
    ax2[1].plot(sorted(abs(y_train-yhat)), label="diff")
    return

def separation_train_test(X, y, pc_train=0.75):
    index = np.arange(len(y))
    np.random.shuffle(index) # liste mélangée
    napp = int(len(y)*pc_train)
    X_train, y_train = X[index[:napp]], y[index[:napp]]
    X_test, y_test   = X[index[napp:]], y[index[napp:]]
    return X_train, y_train, X_test, y_test


# Questions d'ouverture

def test_variables_en_moins(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
  cols = [0, 1, 3, 5, -1]

  w, yhat, yhat_t = application_reelle(x_train[:, cols], y_train, x_test[:, cols], y_test)
  plot_y(y_train, y_test, yhat, yhat_t)

  # Ça ne change pas grand chose, c'était prévisible.

def test_origine(data):
  origine = data.values[:, -2]
  new_variables = np.array([np.where(origine == i, 1, 0) for i in range(origine.max() + 1)])

  # ...
