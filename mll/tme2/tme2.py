import sys
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.sf'] = 'Helvetica'
plt.rcParams['figure.figsize'] = 21.0 / 2.54 - 2.0, 4.0
plt.rcParams['font.size'] = 11.0
# plt.rcParams['figure.dpi'] = 288
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


def mse(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  # print('>>', x.shape, y.shape, ((w.T @ x.T - y) ** 2).shape)
  # print('>>>', np.dot(x, w).shape)
  # print(x.shape)
  # print(w.shape)
  # print('>>>', (((x @ w) - y) ** 2).shape)
  # print('>>>', (((x @ w) - y[..., None])[..., 0] ** 2).shape)
  # print('!', w.shape)
  # print('!', x.shape)
  # print()
  # print('!', (x @ w).shape)
  # return (((x @ w) - y[..., None])[..., 0] ** 2).mean(axis=-1)
  # return (((x @ w)[..., 0] - y) ** 2).mean(axis=-1)
  # return (((x @ w) - y) ** 2).mean(axis=-1)
  return ((np.einsum('...i, ...ji -> ...j', w, x) - y) ** 2).mean(axis=-1)

def mse_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return 2 * x.T @ (x @ w - y) / x.shape[0]

def reglog(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.log(1.0 + np.exp(-y * (x @ w))).mean()

def reglog_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return -(y[:, None] * x / (1.0 + 1.0 / np.exp(-y * (x @ w)))[:, None]).mean(axis=0)


def grad_check(f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], f_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], *, N: int = 100):
  x, y = gen_arti(epsilon=0.1, nbex=N)
  w = np.random.randn(x.shape[1], 1)
  w0 = w + 0.1 # np.random.randn(x.shape[1], 1)

  # print(x.shape)
  # print((w - w0).shape)
  # print(f_grad(w, x, y).shape)

  print(f(w, x, y) - f(w0, x, y))
  print((w - w0).T @ f_grad(w0, x, y))

# grad_check(mse, mse_grad)


def descente_gradient(datax: np.ndarray, datay: np.ndarray, f_loss: Callable, f_grad: Callable, eps: float, iter: int):
  np.random.seed(0)

  w = np.random.randn(datax.shape[1])
  losses = np.empty(iter)
  ws = np.empty((iter, w.shape[0]))

  # print(f_loss(w, datax, datay).shape)
  losses[0] = f_loss(w, datax, datay)
  ws[0, :] = w

  for index in range(1, iter):
    w = w - eps * f_grad(w, datax, datay)
    losses[index] = f_loss(w, datax, datay)
    ws[index, :] = w

  return w, ws, losses


def check_fonctions():
  ## On fixe la seed de l'aléatoire pour vérifier les fonctions
  np.random.seed(0)
  datax, datay = gen_arti(epsilon=0.1)
  datay = datay[:, 0]
  wrandom = np.random.randn(datax.shape[1])
  assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
  assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
  assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
  assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
  np.random.seed()


check_fonctions()


def plot():
  np.random.seed(0)

  # Shape: x (1000, 2)
  #        y (1000, 1)
  x, y = gen_arti(epsilon=0.1)
  y = y[:, 0]

  w, ws, losses = descente_gradient(x, y, mse, mse_grad, eps=0.1, iter=200)

  # print(mse(np.array(w), x, y))
  # print(mse(np.array([w, w, w * 2]), x, y))
  # return

  fig, ax = plt.subplots()

  ax.plot(np.arange(len(losses)), losses)

  ax.grid()
  ax.set_xlabel('Itération')
  ax.set_ylabel('Coût')

  fig, ax = plt.subplots()
  # ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
  # w  = np.random.randn(x.shape[1])
  # print(w.shape)
  # plot_frontiere(x,lambda x : np.sign(x.dot(w)),step=100)

  # a * x + b * y = 0
  # y = -a/b * x

  # lx = np.array([-2, 2])
  # ly = -w[0] / w[1] * lx

  ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='C2', label='Frontière de décision', linestyle='--')
  # ax.plot(lx, ly, color='silver', linestyle='--')

  # a = np.array([-5, 5])
  # b = w[:, None] * a

  # ax.plot(b[0, :], b[1, :])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')


  # for iter, w in enumerate(ws):
  #   # a = np.array([-2, 1])
  #   # b = w[:, None] * a
  #   # ax.plot(b[0, :], b[1, :], 'r-', alpha=(iter / len(ws)) ** 2)

  #   lx = np.array([-2, 2])
  #   ly = -w[0] / w[1] * lx

  #   ax.plot(lx, ly, color='silver', linestyle='--')

  # ax.plot(w[:, None] * a)
  # ax.plot(@ w)

  plot_data(x,y)

  grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

  ## Visualisation de la fonction de coût en 2D
  fig, ax = plt.subplots()

  im = ax.contourf(x_grid, y_grid, mse(grid, x, y).reshape(x_grid.shape), levels=20)
  # ax.contourf(x_grid, y_grid, np.array([mse(w,x,y).mean() for w in grid]).reshape(x_grid.shape), alpha=1.0, levels=20)
  # ax.imshow(mse(grid, x, y).reshape(x_grid.shape), extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), origin='lower', cmap='viridis')
  ax.plot(ws[:, 0], ws[:, 1], color='C1')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')

  fig.colorbar(im)


  plt.show()


plot()
