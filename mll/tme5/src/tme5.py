from typing import Callable, Optional
from matplotlib import pyplot as plt
import numpy as np


def perceptron_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return np.minimum(-y * np.dot(x, w), 0).sum()

def perceptron_grad(w: np.ndarray, x: np.ndarray, y: np.ndarray):
  return (-y * x.T * (y * np.dot(x, w) <= 0)).sum(axis=1)


class Lineaire:
  def __init__(
    self,
    loss: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_loss,
    loss_g: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = perceptron_grad,
    max_iter: int = 100,
    eps: float = 0.01
  ):
    self.max_iter = max_iter
    self.eps = eps
    self.w: Optional[np.ndarray] = None
    self.loss = loss
    self.loss_g = loss_g

  def fit(self, x: np.ndarray, y: np.ndarray, test_x: Optional[np.ndarray] = None, test_y: Optional[np.ndarray] = None):
    scores = np.zeros((self.max_iter + 1, 2))
    self.w = np.zeros(x.shape[1])
    # print(self.loss(w, x, y))

    it = 0
    scores[0, 0] = self.score(x, y)

    if (test_x is not None) and (test_y is not None):
      scores[0, 1] = self.score(test_x, test_y)

    for it in range(self.max_iter):
      self.w -= self.eps * self.loss_g(self.w, x, y)
      # print(self.loss(w, x, y))
      # print(self.loss(w, x, y), w, self.loss_g(w, x, y))
      # print(y * np.dot(x, w))

      scores[it + 1, 0] = self.score(x, y)

      if (test_x is not None) and (test_y is not None):
        scores[it + 1, 1] = self.score(test_x, test_y)

    return scores[:(it + 2), :]

  def predict(self, x: np.ndarray):
    assert self.w is not None
    return np.sign(np.dot(x, self.w))

  def score(self, x: np.ndarray, y: np.ndarray):
    return (self.predict(x) == y).sum() / len(y)



def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y

def plot_data(ax, data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        ax.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        ax.scatter(data[labels==l,0],data[labels==l,1],c=f'C{i}',marker=marks[i])


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(ax, data):
  im = ax.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
  ax.get_figure().colorbar(im, ax=ax)


# x, y = gen_arti(epsilon=0.5)

# l = Lineaire(max_iter=10)
# l.fit(x, y)
# assert l.w is not None

# # print(l.w)


# fig, ax = plt.subplots()

# plot_data(ax, x, y)
# ax.axline((0.0, 0.0), slope=(-l.w[0] / l.w[1]), color='C2', linestyle='--')

# plt.show()



train_x, train_y = load_usps('../data/USPS_train.txt')
test_x, test_y = load_usps('../data/USPS_test.txt')

# print(train_x.shape, train_y.shape)
# print(train_x.dtype)

# show_usps(train_x[0])
# print(train_y[0])

# plt.show()


def ex1():
  model = Lineaire(eps=1e-3)

  train_mask = (train_y == 6) #| (train_y == 9)
  train_ax = train_x[train_mask, :]
  train_ay = np.where(train_y[train_mask] == 6, 1, -1)

  test_mask = (test_y == 6) #| (test_y == 9)
  test_ax = test_x[test_mask, :]
  test_ay = np.where(test_y[test_mask] == 6, 1, -1)

  scores = model.fit(train_ax, train_ay, test_ax, test_ay)
  print(scores)

  # print(model.score(ax, ay))
  # print(model.score(test_x, test_y))

  fig, ax = plt.subplots()

  show_usps(ax, model.w)


  fig, ax = plt.subplots()

  ax.plot(np.arange(scores.shape[0]), scores[:, 0], label='Train')
  ax.plot(np.arange(scores.shape[0]), scores[:, 1], label='Test')
  ax.legend()

  plt.show()


ex1()
