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

  # p = np.einsum('ab, b, b -> ba', np.exp(-0.5 * np.einsum('abi, bij, abj -> ab', centered_x, np.linalg.inv(sig), centered_x)), 1 / np.sqrt(np.linalg.det(sig) * (2.0 * np.pi) ** x.shape[1]), pi)

  # p = (np.exp(-0.5 * np.einsum('abi, bij, abj -> ab', centered_x, np.linalg.inv(sig), centered_x)) / np.sqrt(np.linalg.det(sig) * (2.0 * np.pi) ** x.shape[1]) * pi).T
  # return p / p.sum(axis=0)

  # a = np.einsum('...ai, aij, ...aj -> ...a', centered_x, np.linalg.inv(sig), centered_x)
  # a = np.einsum('abi, bij, abj -> ab', centered_x, np.linalg.inv(sig), centered_x)
  # b = np.exp(-0.5 * a) / np.sqrt(np.linalg.det(sig) * (2 * np.pi) ** x.shape[1])
  # return b / b.sum(axis=1)

  # print(normale_bidim(x[0, :], mu[1, :], sig[1, :, :]))
  # return b

  # print(x[:, :].shape)
  # print(mu[0, :].shape)
  # print((x - mu[0, :]).shape)

  # 1 x 2 @ 2 x 2 @ 2 x 1
  # 272 x [1] x 2 @ 2 x 2
  # 272 x  1          x 2 @ 272 x 2 x 1
  # return (x - mu[0, :])[:, None, :].shape
  # return ((x - mu[0, :]) @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :]).T)
  # return ((x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :]).T).shape
  # return ((x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :])[:, :, None]).shape
  # return (x - mu[0, :])[:, :, None].shape
  # return ((x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :])).shape

  # mu  = np.array([1.,2])[None, :]
  # sig = np.array([[3., 0.],[0., 3.]])[None, :, :]
  # x = np.array([1.,2])[None, :]

  # print(x.shape)
  # print(x.shape)

  # [272 x (1) x 2] @ [2 x 2] @ [272 x 2].T
  # a = np.einsum('...i, ij -> ...j', (x - mu[0, :]), np.linalg.inv(sig[0, :, :])) #, (x - mu[0, :]))
  # b = (x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :])
  # print(a.shape, b.squeeze().shape)
  # print(a - b.squeeze())
  # print(a)
  # print(b.squeeze())
  # print((x[0, :] - mu[0, :]) @ np.linalg.inv(sig))
  # print(np.einsum('i, ik -> k', (x[0, :] - mu[0, :]), np.linalg.inv(sig[0, :, :])))

  # a = np.einsum('...i, ij, ...j -> ...', (x - mu[0, :]), np.linalg.inv(sig[0, :, :]), (x - mu[0, :]))
  b = (x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :])[:, :, None]
  c = (x - mu[1, :])[:, None, :] @ np.linalg.inv(sig[1, :, :]) @ (x - mu[1, :])[:, :, None]

  # print(a.shape, b.squeeze().shape)
  # print(a)
  print(b.squeeze())
  print(c.squeeze())

  # a = np.einsum('...ai, aij, ...aj -> ...a', x[:, None, :] - mu, np.linalg.inv(sig), x[:, None, :] - mu)

  # mu = mu[0, :]
  # sig = sig[0, :, :]

  a = np.einsum('...ai, aij, ...aj -> ...a', x[:, None, :] - mu, np.linalg.inv(sig), x[:, None, :] - mu)
  print(a.shape)
  print(a)

  # return a
  return

  a = ((x - mu[0, :])[:, None, :] @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :])[:, :, None])
  b = ((x - mu[1, :])[:, None, :] @ np.linalg.inv(sig[1, :, :]) @ (x - mu[1, :])[:, :, None])
  # a = ((x - mu[0, :]) @ np.linalg.inv(sig[0, :, :]) @ (x - mu[0, :]).T)
  # b = ((x - mu[1, :]) @ np.linalg.inv(sig[1, :, :]) @ (x - mu[1, :]).T)
  print((a / (a + b)))

  # return (normale_bidim(x, mu[0, :], sig[0, :, :]) * pi[0]).shape
  # return normale_bidim(x[0, :], mu[0, :], sig[0, :, :]) * pi[0]

  # print(x[:, 0].shape)
  # / (normale_bidim(x[0, :], mu[0, :], sig[0, :, :]) * pi[0] + normale_bidim(x[0, :], mu[1, :], sig[1, :, :]) * pi[1])
  # print(mu)

def update_param(x: np.ndarray, q: np.ndarray, pi: np.ndarray, mu: np.ndarray, sig: np.ndarray):
  # print(q.shape) # (class, sample)
  # print(x.shape) # (sample, distrib)
  # print(mu.shape) # (class, distrib) axis 0 = i of µ_i

  pi_u = q.sum(axis=1) / q.sum()
  mu_u = np.einsum('ba, ai, b -> bi', q, x, 1.0 / q.sum(axis=1))

  # mu_u = ((q * x.T[:, None, :]).sum(axis=2) / q.sum(axis=1)).T
  #
  # sig_u = x[:, None, :] - mu_u # (sample, class, distrib)
  # print(">", sig_u.shape)
  # sig_u = np.einsum('abi, abj -> ijba', sig_u, sig_u) # (cov 1, cov 2, class, sample)
  # sig_u = np.einsum('abi, abj -> abij', sig_u, sig_u) # (sample, class, cov 1, cov 2)
  # sig_u = np.einsum('abc, ade -> ceabd', sig_u, sig_u)
  # print(">", sig_u.shape)
  # sig_u = sig_u * q
  # print(">", sig_u.shape)
  # sig_u = sig_u.sum(axis=3) # sum over sample -> (cov 1, cov 2, class)
  # print(">", sig_u.shape)

  # # sig_u = (sig_u ** 2).sum(axis=2)
  # # sig_u = (sig_u.T * q).sum(axis=1)
  # sig_u = sig_u / q.sum(axis=1) # divide by sum over sample -> (class)
  # # print(">", sig_u.shape)

  # print(">", sig_u.shape)

  # sig_u = ((np.einsum('abi, abj -> ijba', centered_x, centered_x) * q).sum(axis=3) / q.sum(axis=1)).T # sum over sample -> (cov 1, cov 2, class)
  # sig_u = (np.einsum('abi, abj, ba -> ijb', centered_x, centered_x, q) / q.sum(axis=1)).T # sum over sample -> (cov 1, cov 2, class)

  centered_x = x[:, None, :] - mu_u
  sig_u = np.einsum('abi, abj, ba, b -> bij', centered_x, centered_x, q, 1.0 / q.sum(axis=1))

  return pi_u, mu_u, sig_u


def EM(x: np.ndarray, nIterMax = 100):
  pi, mu, sig = init(x)
  nIter = -1

  for nIter in range(nIterMax):
    old_mu = mu
    pi, mu, sig = update_param(x, Q_i(x, pi, mu, sig), pi, mu, sig)

    if np.allclose(old_mu, mu):
      break

  return nIter, pi, mu, sig

  # print(pi, mu, sig)
