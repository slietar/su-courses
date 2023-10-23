# Antoine GRISLAIN
# Simon LIÉTAR

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def normale_bidim(x: np.ndarray, mu: np.ndarray, sig: np.ndarray):
  return np.exp(-0.5 * (x - mu) @ np.linalg.inv(sig) @ (x - mu).T) / np.sqrt(np.linalg.det(sig) * (2.0 * np.pi) ** len(x))

def estimation_nuage_haut_gauche():
  return [4.25, 80], [[0.2, 2], [0, 50]]

def init(x: np.ndarray):
  return (
    np.array([0.5, 0.5]),
    (x.mean(axis=0)[:, None] + [1.0, -1.0]).T,
    np.cov(x.T)[None, :, :].repeat(2, axis=0)
  )

def Q_i(x: np.ndarray, pi: np.ndarray, mu: np.ndarray, sig: np.ndarray):
  centered_x = x[:, None, :] - mu # (sample, class, distrib)
  p = (np.exp(-0.5 * np.einsum('abi, bij, abj -> ab', centered_x, np.linalg.inv(sig), centered_x)) / np.sqrt(np.linalg.det(sig) * (2.0 * np.pi) ** x.shape[1]) * pi).T

  return p / p.sum(axis=0)

def update_param(x: np.ndarray, q: np.ndarray, pi: np.ndarray, mu: np.ndarray, sig: np.ndarray):
  pi_u = q.sum(axis=1) / q.sum()
  mu_u = np.einsum('ba, ai, b -> bi', q, x, 1.0 / q.sum(axis=1))

  centered_x = x[:, None, :] - mu_u
  sig_u = np.einsum('abi, abj, ba, b -> bij', centered_x, centered_x, q, 1.0 / q.sum(axis=1))

  return pi_u, mu_u, sig_u


def EM(x: np.ndarray, initFunc: Callable = init, nIterMax: int = 100, saveParam: Optional[str] = None):
  pi, mu, sig = initFunc(x)
  nIter = -1

  save_path = Path(saveParam) if saveParam else None

  if save_path:
    save_path.parent.mkdir(parents=True, exist_ok=True)

  for nIter in range(nIterMax):
    old_mu = mu
    pi, mu, sig = update_param(x, Q_i(x, pi, mu, sig), pi, mu, sig)

    if save_path:
      with (save_path.parent / f"{save_path.name}{nIter}.pkl").open("wb") as file:
        pickle.dump({
          'pi': pi,
          'mu': mu,
          'Sig': sig
        }, file)

    if np.allclose(old_mu, mu):
      break

  return nIter, pi, mu, sig

def init_4(x: np.ndarray):
  return (
    np.array([0.25, 0.25, 0.25, 0.25]),
    (x.mean(axis=0) + [[1, 1], [1, -1], [-1, 1], [-1, -1]]),
    np.cov(x.T)[None, :, :].repeat(4, axis=0)
  )

def bad_init_4(x: np.ndarray):
  return (
    np.array([0.25, 0.25, 0.25, 0.25]),
    (x.mean(axis=0) + [[4, 2], [3, 4], [0, 0], [-5, 0]]),
    np.cov(x.T)[None, :, :].repeat(4, axis=0)
  )

def init_B(x: np.ndarray):
  return np.array([0.1] * 10), x[0:30, :].reshape(10, 3, 256).mean(axis=1)

def logpobsBernoulli(x: np.ndarray, theta: np.ndarray):
  epsilon = 1e-5
  bounded_theta = theta.clip(epsilon, 1.0 - epsilon)
  return (x * np.log(bounded_theta) + (1.0 - x) * np.log(1.0 - bounded_theta)).sum(axis=-1)

def Q_i_B(x: np.ndarray, pi: np.ndarray, theta: np.ndarray):
  result = np.zeros((theta.shape[0], x.shape[0]))

  for image_index in range(x.shape[0]):
    a = logpobsBernoulli(x[image_index, :], theta) + np.log(pi)
    s = np.max(a)
    b = a - s - np.log(np.exp(a - s).sum())

    digit_index = np.argmax(b)
    result[digit_index, image_index] = 1.0

  return result

def update_param_B(x: np.ndarray, q: np.ndarray, pi: np.ndarray, theta: np.ndarray):
  pi_u = q.sum(axis=1) / q.sum()
  theta_u = np.einsum('ba, ai, b -> bi', q, x, 1.0 / q.sum(axis=1))

  return pi_u, theta_u

def EM_B(x: np.ndarray):
  pi, theta = init_B(x)
  nIter = -1

  for nIter in range(100):
    old_theta = theta
    pi, theta = update_param_B(x, Q_i_B(x, pi, theta), pi, theta)

    if np.allclose(old_theta, theta):
      break

  return nIter, pi, theta

def calcul_purete(x: np.ndarray, y: np.ndarray, pi: np.ndarray, theta: np.ndarray):
  result = Q_i_B(x, pi, theta).astype(int)

  # Find the class of each cluster

  clusters_y = np.zeros(result.shape[0], dtype=int)

  for cluster_index in range(result.shape[0]):
    a = y[np.where(result[cluster_index, :] == 1)]
    b = np.bincount(a, minlength=result.shape[0])

    clusters_y[cluster_index] = np.argmax(b)

  # Compute the purity of each cluster

  y_hat = clusters_y[np.argmax(result, axis=0)]
  output = np.zeros((2, result.shape[0]))

  for cluster_index, cluster_y in enumerate(clusters_y):
    y_hat_c = y[y_hat == cluster_y]
    count = np.bincount(y_hat_c, minlength=result.shape[0])

    output[0, cluster_index] = count[cluster_y] / count.sum()
    output[1, cluster_index] = count[cluster_y]

  return (*output,)
