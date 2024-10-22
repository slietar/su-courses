from pathlib import Path
from typing import Callable

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .. import config, mltools, utils


def mse(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return ((np.einsum('...ji, ...i -> ...j', x, w) - y) ** 2).mean(axis=-1)

def mse_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return 2.0 * x.T @ (x @ w - y) / x.shape[0]

def reglog(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.log(1.0 + np.exp(-y * np.einsum('...ji, ...i -> ...j', x, w))).mean(axis=-1)

def reglog_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return -(y[:, None] * x / (1.0 + 1.0 / np.exp(-y * (x @ w)))[:, None]).mean(axis=0)


def grad_check(f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], f_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], *, N: int = 100):
  x, y = mltools.gen_arti(epsilon=0.1, nbex=N)
  w = np.random.randn(x.shape[1], 1)
  w0 = w + 0.1 # np.random.randn(x.shape[1], 1)

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
  datax, datay = mltools.gen_arti(epsilon=0.1)
  wrandom = np.random.randn(datax.shape[1])
  assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
  assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
  assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
  assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
  np.random.seed()


def plot_decision(ax: Axes, w: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]):
  step = 100
  grid, x_grid, y_grid = mltools.make_grid(
    xmin=-2.5,
    xmax=2.5,
    ymin=-2.5,
    ymax=2.5,
    step=step
  )

  # im = ax.imshow(fn(grid).reshape(step, step), extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), alpha=0.5, cmap='RdYlBu', origin='lower') #, vmin=-0.05, vmax=0.05)
  im = ax.contourf(fn(grid).reshape(step, step), extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), alpha=0.5, cmap='RdYlBu', origin='lower', levels=20) #, vmin=-0.05, vmax=0.05)

  ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='gray', label='Frontière de décision', linestyle='--')

  ax.set_xlim(-2.1, 2.1)
  ax.set_ylim(-2.1, 2.1)

  # ax.get_figure().colorbar(im, ax=ax)
  return im


check_fonctions()

output_path = Path('output/tme2')
output_path.mkdir(exist_ok=True, parents=True)

def plot1():
  np.random.seed(0)

  # Shape: x (1000, 2)
  #        y (1000, 1)
  x, y = mltools.gen_arti(epsilon=0.1)


  # Loss as a function of the epoch

  fig, ax = plt.subplots()
  ax.grid()

  lrs = np.array([4e-1, 3e-1, 1e-1, 1e-2])

  for lr in lrs:
    _, _, losses = descente_gradient(x, y, mse, mse_grad, eps=lr, iter=125)
    ax.plot(np.arange(len(losses)), losses, label=f'ε = {lr}')

  ax.set_xlabel('Itération')
  ax.set_ylabel('Coût')
  # ax.set_xscale('log')
  ax.set_yscale('log')
  ax.legend()

  with (output_path / '1.png').open('wb') as file:
    fig.savefig(file)


  # Data points and decision boundary

  fig, ax = plt.subplots()

  w, _, _ = descente_gradient(x, y, mse, mse_grad, eps=0.1, iter=200)

  plot_decision(ax, w, lambda v: -v @ w)
  mltools.plot_data(ax, x, y)

  # ax.axline((0.0, 0.0), slope=(-w[0] / w[1]), color='gray', label='Frontière de décision', linestyle='--')
  ax.legend()

  with (output_path / '2.png').open('wb') as file:
    fig.savefig(file)


  # Loss landscape

  fig, axs = plt.subplots(ncols=2, nrows=2)
  fig.set_figheight(4.5)

  grid, x_grid, y_grid = mltools.make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

  for ax, lr in zip(axs.flat, lrs):
    ax: Axes
    ax.contourf(x_grid, y_grid, mse(grid, x, y).reshape(x_grid.shape), alpha=0.5, cmap='RdYlBu_r', levels=20)

    _, ws, _ = descente_gradient(x, y, mse, mse_grad, eps=lr, iter=100)
    ax.plot(ws[:, 0], ws[:, 1])

    # f = interp1d(np.arange(), ws[:, 1], kind='cubic')
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)
    # ax.arrow(ws[50, 0], ws[50, 1], 0.01, 0.02, shape='full', lw=0, length_includes_head=True, head_width=.05)

    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')

    ax.set_title(f'ε = {lr}')

  fig.subplots_adjust(hspace=0.3)
  utils.filter_axes(axs)

  with (output_path / '3.png').open('wb') as file:
    fig.savefig(file)


def plot2():
  np.random.seed(0)

  x, y = mltools.gen_arti(epsilon=0.1)


  # Loss as a function of the epoch

  fig, ax = plt.subplots()
  ax.grid()

  lrs = np.array([1e1, 1e0, 1e-1, 1e-2])

  for lr in lrs:
    _, _, losses = descente_gradient(x, y, reglog, reglog_grad, eps=lr, iter=500)
    ax.plot(np.arange(len(losses)), losses, label=f'ε = {lr}')

  ax.set_xlabel('Itération')
  ax.set_ylabel('Coût')
  ax.set_yscale('log')
  ax.legend()

  with (output_path / '4.png').open('wb') as file:
    fig.savefig(file)


  # Data points and decision boundary

  w, _, _ = descente_gradient(x, y, reglog, reglog_grad, eps=0.1, iter=500)

  fig, ax = plt.subplots()
  mltools.plot_data(ax, x, y)

  im = plot_decision(ax, w, lambda v: -1.0 / (1 + np.exp(-v @ w)))
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

  fig, axs = plt.subplots(ncols=2, nrows=2)
  fig.set_figheight(4.5)

  grid, x_grid, y_grid = mltools.make_grid(
    xmin=bounds[0, 0],
    xmax=bounds[1, 0],
    ymin=bounds[0, 1],
    ymax=bounds[1, 1],
    step=100
  )

  for ax, lr, ws in zip(axs.flat, lrs, wss):
    ax: Axes
    im = ax.contourf(x_grid, y_grid, reglog(grid, x, y).reshape(x_grid.shape), alpha=0.5, cmap='RdYlBu_r', levels=5, norm=LogNorm())
    ax.plot(ws[:, 0], ws[:, 1])

    fig.colorbar(im, ax=ax)

    # ax.plot(center[0], center[1], 'o', color='C3', label='Minimum global')

    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')

    ax.set_title(f'ε = {lr}')

  fig.subplots_adjust(hspace=0.3)
  utils.filter_axes(axs)

  with (output_path / '6.png').open('wb') as file:
    fig.savefig(file)


def plot3():
  np.random.seed(0)

  x1, y1 = mltools.gen_arti(data_type=1, epsilon=0.1, nbex=200)
  x2, y2 = mltools.gen_arti(data_type=2, epsilon=0.005)

  w1, _, _ = descente_gradient(x1, y1, mse, mse_grad, eps=0.1, iter=1000)
  w2, _, _ = descente_gradient(x2, y2, mse, mse_grad, eps=0.1, iter=1000)

  fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True)
  fig.set_figheight(3.0)

  im1 = plot_decision(ax1, w1, lambda v: v @ w1)
  im2 = plot_decision(ax2, w2, lambda v: v @ w2)
  # ax.legend()

  mltools.plot_data(ax1, x1, y1)
  mltools.plot_data(ax2, x2, y2)

  ax1.set_title('Type 1')
  ax2.set_title('Type 2')

  utils.filter_axes(np.array([[ax1, ax2]]))

  # fig.colorbar(im1, ax=[ax1, ax2])

  fig.subplots_adjust(bottom=0.15)

  with (output_path / '7.png').open('wb') as file:
    fig.savefig(file)




plot1()
plot2()
# plot3()
# plt.show()
