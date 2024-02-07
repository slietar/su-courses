from pathlib import Path
import sys
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
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
  '#fbc15e',
  '#777777',
  '#8eba42',
  '#ffb5b8'
])


def mse(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return ((np.einsum('...ji, ...i -> ...j', x, w) - y) ** 2).mean(axis=-1)

def mse_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return 2 * x.T @ (x @ w - y) / x.shape[0]

def reglog(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.log(1.0 + np.exp(-y * np.einsum('...ji, ...i -> ...j', x, w))).mean(axis=-1)
  # return np.log(1.0 + np.exp(-y * (x @ w))).mean()

def reglog_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  # print()
  # print(w.shape)
  # print(x.shape)
  # print(y.shape)
  # print((y[:, None] * x).shape)
  # print((1.0 + 1.0 / np.exp(-y * np.einsum('...ji, ...i -> ...j', x, w))).shape)
  # sys.exit()

  # return -(y[..., None] * x / (1.0 + 1.0 / np.exp(-y * np.einsum('...ji, ...i -> ...j', x, w)))[..., None]).mean(axis=0)
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

output_path = Path('output')
output_path.mkdir(exist_ok=True, parents=True)

def plot1():
  np.random.seed(0)

  # Shape: x (1000, 2)
  #        y (1000, 1)
  x, y = gen_arti(epsilon=0.1)
  y = y[:, 0]


  # Loss as a function of the epoch

  fig, ax = plt.subplots()
  ax.grid()

  lrs = np.array([4e-1, 3e-1, 1e-1, 1e-2])

  for lr in lrs:
    _, _, losses = descente_gradient(x, y, mse, mse_grad, eps=lr, iter=200)
    ax.plot(np.arange(len(losses)), losses, label=rf'$\epsilon = {lr}$')

  ax.set_xlabel('Itération')
  ax.set_ylabel('Coût')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.legend()

  with (output_path / '1.png').open('wb') as file:
    fig.savefig(file)


  # Data points and decision boundary

  fig, ax = plt.subplots()

  w, _, _ = descente_gradient(x, y, mse, mse_grad, eps=0.1, iter=200)

  plot_data(x,y)
  ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='C2', label='Frontière de décision', linestyle='--')

  ax.set_xlabel('$X_1$')
  ax.set_ylabel('$X_2$')

  with (output_path / '2.png').open('wb') as file:
    fig.savefig(file)


  # Loss landscape

  fig, axs = plt.subplots(2, 2)
  grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

  for ax, lr in zip(axs.flat, lrs):
    ax: Axes
    ax.contourf(x_grid, y_grid, mse(grid, x, y).reshape(x_grid.shape), levels=20)

    _, ws, _ = descente_gradient(x, y, mse, mse_grad, eps=lr, iter=100)
    ax.plot(ws[:, 0], ws[:, 1])

    # f = interp1d(np.arange(), ws[:, 1], kind='cubic')
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')

    ax.set_title(rf'$\epsilon = {lr}$')

  with (output_path / '3.png').open('wb') as file:
    fig.savefig(file)


def plot2():
  np.random.seed(0)

  x, y = gen_arti(epsilon=0.1)
  y = y[:, 0]


  # Loss as a function of the epoch

  fig, ax = plt.subplots()
  ax.grid()

  lrs = np.array([1e1, 1e0, 1e-1, 1e-2])

  for lr in lrs:
    _, _, losses = descente_gradient(x, y, reglog, reglog_grad, eps=lr, iter=500)
    ax.plot(np.arange(len(losses)), losses, label=rf'$\epsilon = {lr}$')

  ax.set_xlabel('Itération')
  ax.set_ylabel('Coût')
  ax.legend()

  with (output_path / '4.png').open('wb') as file:
    fig.savefig(file)


  # Data points and decision boundary

  fig, ax = plt.subplots()

  w, _, _ = descente_gradient(x, y, reglog, reglog_grad, eps=0.1, iter=500)

  grid, x_grid, y_grid = make_grid(
    xmin=-2,
    xmax=2,
    ymin=-2,
    ymax=2,
    step=100
  )

  im = ax.imshow(-(1.0 / (1 + np.exp(-grid @ w))).reshape(100, 100), extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), alpha=0.6, cmap='RdBu', origin='lower')

  plot_data(x,y)
  ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='C2', label='Frontière de décision', linestyle='--')

  ax.set_xlabel('$X_1$')
  ax.set_ylabel('$X_2$')

  ax.set_xlim(-2, 2)
  ax.set_ylim(-2, 2)

  fig.colorbar(im, ax=ax)

  with (output_path / '5.png').open('wb') as file:
    fig.savefig(file)


  # Loss landscape

  wss = np.asarray([descente_gradient(x, y, reglog, reglog_grad, eps=lr, iter=10000)[1] for lr in lrs])
  center = wss[0, -1, :]

  ws_max = wss.max(axis=(0, 1))
  ws_min = wss.min(axis=(0, 1))
  half_ranges = np.maximum(
    np.abs(ws_max - center),
    np.abs(ws_min - center)
  ) + 0.5

  bounds = center + half_ranges * np.array([-1, 1])[:, None]

  fig, axs = plt.subplots(2, 2)
  grid, x_grid, y_grid = make_grid(
    xmin=bounds[0, 0],
    xmax=bounds[1, 0],
    ymin=bounds[0, 1],
    ymax=bounds[1, 1],
    step=100
  )

  for ax, lr, ws in zip(axs.flat, lrs, wss):
    ax: Axes
    ax.contourf(x_grid, y_grid, np.log(reglog(grid, x, y).reshape(x_grid.shape)), levels=20)
    ax.plot(ws[:, 0], ws[:, 1])
    # ax.plot(center[0], center[1], 'o', color='C3', label='Minimum global')

    # f = interp1d(np.arange(), ws[:, 1], kind='cubic')
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')

    ax.set_title(rf'$\epsilon = {lr}$')

  with (output_path / '6.png').open('wb') as file:
    fig.savefig(file)


def plot3():
  np.random.seed(0)

  x, y = gen_arti(data_type=1, epsilon=0.1)
  y = y[:, 0]

  w, ws, losses = descente_gradient(x, y, mse, mse_grad, eps=0.1, iter=1000)

  fig, ax = plt.subplots()

  plot_data(x, y)

  grid, x_grid, y_grid = make_grid(
    xmin=-2,
    xmax=2,
    ymin=-2,
    ymax=2,
    step=100
  )

  im = ax.imshow((grid @ w).reshape(100, 100), extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), alpha=0.6, cmap='RdBu', origin='lower')

  ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='C2', label='Frontière de décision', linestyle='--')

  ax.set_xlabel('$X_1$')
  ax.set_ylabel('$X_2$')

  ax.set_xlim(-2, 2)
  ax.set_ylim(-2, 2)

  fig.colorbar(im, ax=ax)

  plt.show()


plot1()
plot2()
# plot3()
