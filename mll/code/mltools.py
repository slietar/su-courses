from typing import Optional, Sequence
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(ax: Axes, data, labels=None, *, alpha: np.ndarray | float = 1.0, highlight: Optional[Sequence[bool]] = None):
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

    for i, l in enumerate(sorted(list(set(labels.flatten())))):
        ax.scatter(
            data[labels == l, 0],
            data[labels == l, 1],
            alpha=(alpha[labels == l] if isinstance(alpha, np.ndarray) else alpha),
            c=f'C{i}',
            marker=marks[i],
        )

    if highlight is not None:
        for i, l in enumerate(sorted(list(set(labels.flatten())))):
            ax.scatter(
                data[highlight, 0][labels[highlight] == l],
                data[highlight, 1][labels[highlight] == l],
                alpha=(alpha[labels == l] if isinstance(alpha, np.ndarray) else alpha),
                facecolors='none',
                edgecolors='black',
                s=[86.0, 30.0][i],
                marker=['.', 'P'][i]
            )

    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')

def plot_data_3d(ax: Axes, x: np.ndarray, y: np.ndarray, z: float):
    for index, yp in enumerate(sorted(np.unique(y))):
        ax.scatter(x[y == yp, 0], x[y == yp, 1], z, c=f'C{index}', marker=['.', '+'][index])

def plot_frontiere(data,f,step=20):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def make_grid(data=None,xmin: float = -5,xmax: float = 5,ymin: float = -5,ymax: float = 5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    # x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    x, y =np.meshgrid(np.linspace(xmin,xmax,step), np.linspace(ymin,ymax,step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

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
    return data, y

def get_lim_for_data_type(data_type: int):
    match data_type:
        case 0 | 1:
            return -2.5, 2.5
        case 2:
            return -4.5, 4.5
        case _:
            raise RuntimeError
